"""python -m src.gates.serve_check --models models [--tiles 4] [--memory-limit-gib 8] [--max-request-s 30]

Resource check of the PRODUCTION inference path: every .onnx in --models is loaded into ONE process, as the Chisel
server does, and requests of --tiles tiles of 1280 x 1280 run through the same pattern:
  * onnxruntime CPU sessions with the server's options (2 intra-op threads, 1 inter-op, memory arena off)
  * a thread pool of --workers (server default 2), one lock per model, at most max(2*workers, n_models) jobs in flight
  * per tile and model: preprocess (RGB/255, CHW) -> session.run -> serving/postprocess_semantic.py
It reports latency, peak memory and the average number of busy CPU cores, and fails (exit code 2) above the memory or
latency limit. Only meaningful inside a container that enforces the real limits (docker --cpus/--memory, a k8s pod).
Not covered: image decoding, alpha masking, winner-take-all, colouring and PNG encoding of the real server.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from serving.postprocess_semantic import postprocess_semantic_masks

from ..common.envinfo import cgroup_limits, peak_rss_gib


class Model:
    def __init__(self, path: Path, intra: int):
        o = ort.SessionOptions()
        o.enable_mem_pattern, o.enable_cpu_mem_arena = True, False      # as in the server: memory goes back at once
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        o.intra_op_num_threads, o.inter_op_num_threads = intra, 1
        self.name = path.stem
        self.session = ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.outputs = [x.name for x in self.session.get_outputs()]
        side = path.with_suffix(".onnx.json")
        self.threshold = (json.loads(side.read_text()).get("threshold") if side.exists() else None) or 0.5
        self.lock, self.seconds = threading.Lock(), []

    def predict(self, tensor: np.ndarray, size: int):
        with self.lock:                                                 # one run at a time per model, like the server
            t0 = time.perf_counter()
            out = self.session.run(self.outputs, {self.input: tensor})
            self.seconds.append(time.perf_counter() - t0)
        return postprocess_semantic_masks(out, (size, size), 1.0, (0.0, 0.0), size, self.threshold)


def request(models, tiles: int, size: int, pool, max_inflight: int) -> float:
    """One request = `tiles` tiles x every model, submitted with the server's in-flight limit."""
    rng = np.random.default_rng(0)
    t0, inflight = time.perf_counter(), set()
    for _ in range(tiles):
        tile = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)                       # BGR like cv2
        tensor = np.ascontiguousarray(tile[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
        for m in models:
            inflight.add(pool.submit(m.predict, tensor, size))
            if len(inflight) >= max_inflight:
                done, inflight = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
                for f in done:
                    f.result()
    for f in cf.as_completed(inflight):
        f.result()
    return time.perf_counter() - t0


def measure(files: list[Path], tiles: int = 1, size: int = 1280, repeats: int = 3, workers: int = 2,
            intra: int = 2) -> dict:
    models = [Model(f, intra) for f in files]
    print(f"[serve] loaded {[m.name for m in models]}; peak RSS after load {peak_rss_gib():.2f} GiB", flush=True)
    limit = max(workers * 2, len(models))
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        request(models, 1, size, pool, limit)                                               # warm-up
        for m in models:
            m.seconds.clear()
        walls, cpu0, w0 = [], time.process_time(), time.perf_counter()
        for i in range(repeats):
            walls.append(request(models, tiles, size, pool, limit))
            print(f"[serve] request {i + 1}/{repeats} ({tiles} tile(s) x {len(models)} models): {walls[-1]:.2f}s", flush=True)
        busy = (time.process_time() - cpu0) / (time.perf_counter() - w0)
    cg_cpu, cg_mem = cgroup_limits()
    return {"models": [m.name for m in models], "tiles_per_request": tiles, "size": size,
            "request_mean_s": float(np.mean(walls)), "request_p95_s": float(np.percentile(walls, 95)),
            "per_model_mean_s": {m.name: float(np.mean(m.seconds)) for m in models},
            "peak_rss_gib": peak_rss_gib(), "cpu_cores_busy": busy, "threads_in_use": min(workers, len(models)) * intra,
            "cgroup_cpus": cg_cpu, "cgroup_mem_gib": cg_mem, "host_cpus": os.cpu_count()}


def evaluate(res: dict, cpu_limit: float, memory_limit_gib: float, max_request_s: float | None) -> dict:
    """Add `violations`, `warnings` and `passed` to a measure() result."""
    warnings, violations = [], []
    if res["cgroup_cpus"] is None:
        warnings.append("no CPU cgroup limit detected: not inside the production envelope")
    elif res["cgroup_cpus"] < cpu_limit:
        warnings.append(f"container limited to {res['cgroup_cpus']:.1f} CPUs < cpu_limit {cpu_limit}")
    if res["cgroup_mem_gib"] is None:
        warnings.append("no memory cgroup limit detected: peak RSS is measured, the limit is not enforced")
    if res["peak_rss_gib"] > memory_limit_gib:
        violations.append(f"peak_rss {res['peak_rss_gib']:.2f} GiB > {memory_limit_gib} GiB")
    if max_request_s is not None and res["request_p95_s"] > max_request_s:
        violations.append(f"request p95 {res['request_p95_s']:.2f}s > {max_request_s}s for {res['tiles_per_request']} tile(s)")
    return {**res, "passed": not violations, "violations": violations, "warnings": warnings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models", help="folder with the .onnx files (or a single file)")
    ap.add_argument("--tiles", type=int, default=1, help="1280x1280 tiles per request")
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2, help="server THREAD_POOL_SIZE")
    ap.add_argument("--intra", type=int, default=2, help="onnxruntime threads per model (server: 2)")
    ap.add_argument("--cpu-limit", type=float, default=8.0)
    ap.add_argument("--memory-limit-gib", type=float, default=8.0)
    ap.add_argument("--max-request-s", type=float, default=None)
    ap.add_argument("--mode", choices=["gate", "diagnostic"], default="gate")
    ap.add_argument("--out", default=None, help="write the result json here")
    a = ap.parse_args()

    p = Path(a.models)
    if not (files := sorted(p.glob("*.onnx")) if p.is_dir() else [p]):
        raise SystemExit(f"no .onnx files in {p}")
    res = evaluate(measure(files, a.tiles, a.size, a.repeats, a.workers, a.intra),
                   a.cpu_limit, a.memory_limit_gib, a.max_request_s)
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
    if res["violations"] and a.mode == "gate":
        print(f"SERVE GATE FAILED: {'; '.join(res['violations'])}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
