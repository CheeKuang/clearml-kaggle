"""python -m src.resource_check [hydra overrides]

Can this model run inference inside the production envelope (cpu_request / cpu_limit / memory_limit)?
Needs no data: the model is built from the config with random weights (cost is identical, nothing is
downloaded) and a dummy tile is timed at `cpu_request` and `cpu_limit` threads. Run it on a queue whose
agent enforces the production limits (k8s pod requests/limits or docker --cpus/--memory), otherwise the
numbers describe the host, not the envelope."""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .clearml_utils import Reporter, init_clearml
from .model_factory import build_model

GIB = 1024 ** 3


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cgroup_limits() -> tuple[float | None, float | None]:
    """(cpus, mem_gib) the container is actually limited to; None = unlimited / not in a cgroup."""
    cpu = mem = None
    cm = _read("/sys/fs/cgroup/cpu.max")                       # cgroup v2: "<quota|max> <period>"
    if cm:
        quota, period = cm.split()
        cpu = None if quota == "max" else int(quota) / int(period)
    else:                                                      # cgroup v1
        q, p = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q and p and int(q) > 0:
            cpu = int(q) / int(p)
    mm = _read("/sys/fs/cgroup/memory.max") or _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if mm and mm != "max" and int(mm) < 1 << 60:
        mem = int(mm) / GIB
    return cpu, mem


def peak_rss_gib() -> float:
    """Peak resident memory of this process (includes the torch import itself)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss     # bytes on macOS, KiB on Linux
    return r / (1 if sys.platform == "darwin" else 1024) / GIB


@torch.inference_mode()
def bench(model, x, threads: int, warmup: int, iters: int) -> np.ndarray:
    torch.set_num_threads(threads)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        times.append(time.perf_counter() - t0)
    return np.asarray(times)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    cfg.clearml.task_name = "resource-check"                    # stable name -> pipeline finds the base task
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    task, cfg = init_clearml(cfg)
    reporter = Reporter(task)
    r = cfg.resource

    build_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    build_cfg.model.encoder_weights = None                      # weights don't change cost; no downloads
    build_cfg.pretrain = None
    model = build_model(build_cfg).eval()

    x = torch.randn(r.batch_size, cfg.data.in_channels, r.tile_size, r.tile_size)
    scores = {"params_m": sum(p.numel() for p in model.parameters()) / 1e6}
    for name, threads in (("request", r.cpu_request), ("limit", r.cpu_limit)):
        t = bench(model, x, threads, r.warmup, r.iters)
        scores[f"latency_mean_s_{name}"] = float(t.mean())
        scores[f"latency_p95_s_{name}"] = float(np.percentile(t, 95))
    scores["peak_rss_gib"] = peak_rss_gib()

    cg_cpu, cg_mem = cgroup_limits()
    host_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    warnings = []
    if cg_cpu is not None:
        scores["cgroup_cpus"] = cg_cpu
        if cg_cpu < r.cpu_limit:
            warnings.append(f"container is limited to {cg_cpu:.1f} CPUs < cpu_limit {r.cpu_limit}")
    elif host_cpus < r.cpu_limit:
        warnings.append(f"host exposes {host_cpus} CPUs < cpu_limit {r.cpu_limit}: limit-thread numbers are optimistic")
    else:
        warnings.append("no CPU cgroup limit detected: not running inside the production envelope")
    if cg_mem is not None:
        scores["cgroup_mem_gib"] = cg_mem
        if cg_mem > r.memory_limit_gib:
            warnings.append(f"container memory {cg_mem:.1f} GiB > memory_limit_gib {r.memory_limit_gib}")
    else:
        warnings.append("no memory cgroup limit detected: peak RSS is measured, the limit is not enforced")

    violations = []
    if scores["peak_rss_gib"] > r.memory_limit_gib:
        violations.append(f"peak_rss {scores['peak_rss_gib']:.2f} GiB > {r.memory_limit_gib} GiB")
    if r.max_latency_s is not None and scores["latency_p95_s_limit"] > r.max_latency_s:
        violations.append(f"p95 latency {scores['latency_p95_s_limit']:.2f}s > {r.max_latency_s}s at {r.cpu_limit} threads")

    passed = not violations
    for key, val in scores.items():
        reporter.scalar("resource", key, val, 0)
        reporter.single(f"resource/{key}", val)
    reporter.scalar("resource", "passed", float(passed), 0)
    result = {"mode": r.mode, "passed": passed, "scores": scores, "violations": violations, "warnings": warnings,
              "envelope": dict(OmegaConf.to_container(r)), "model": f"{cfg.model.arch}/{cfg.model.encoder_name}"}
    (out_dir / "resource_check.json").write_text(json.dumps(result, indent=2))
    if task:
        task.upload_artifact("resource_check", artifact_object=result)
    print(json.dumps(result, indent=2))

    if not passed and r.mode == "gate":
        if task:
            task.mark_failed(status_reason="resource_gate", status_message="; ".join(violations)[:500])
        print(f"RESOURCE GATE FAILED: {'; '.join(violations)}")
        sys.exit(2)


if __name__ == "__main__":
    main()
