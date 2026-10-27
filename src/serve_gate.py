"""python -m src.serve_gate [hydra overrides]   (conf/serve/default.yaml)

ClearML Task around src/serve_check.py: can ALL the production models run together inside the container limits
(4 vCPU request / 8 vCPU limit / 8 GiB for the whole container)? It reproduces the Chisel server's inference path.
`serve.source` chooses the models:
  random  untrained models of `serve.spec` ("class=architecture:encoder,..."), exported to ONNX at 1280. Same cost as
          trained ones, so this runs BEFORE training and stops a pipeline that could never fit.
  tasks   the ONNX artifacts of the sibling `train_<class>` steps of the same pipeline (found through the
          "pipe: <controller id>" tag of this Task). Runs AFTER training on the real models.
  folder  the .onnx files in serve.folder (local use).
Exit code 2 and Task marked failed on a violation, like the other gates. Register the base task by running this with
NO extra arguments (see docs/GUIDE.md).
"""
from __future__ import annotations

import faulthandler
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import hydra
import onnxruntime  # noqa: F401  (heavy native libraries are imported BEFORE Task.init, like src.train / resource_check)
import segmentation_models_pytorch  # noqa: F401
import torch  # noqa: F401
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .clearml_utils import Reporter, init_clearml
from .export_onnx import export_untrained


def models_from_spec(spec: str, workdir: Path, size: int) -> list[Path]:
    files = []
    for item in [x.strip() for x in spec.split(",") if x.strip()]:
        cls, _, arch = item.partition("=")
        architecture, _, encoder = arch.partition(":")
        export_untrained(cls, architecture, encoder, workdir / f"{cls}.onnx", size)
        files.append(workdir / f"{cls}.onnx")
    return files


def models_from_pipeline(task, workdir: Path) -> list[Path]:
    """ONNX artifacts of the train_<class> steps that belong to the same pipeline run as `task`."""
    from clearml import Task

    tag = next((t for t in task.get_tags() if t.startswith("pipe: ")), None)
    if tag is None:
        raise SystemExit("serve.source=tasks needs to run as a pipeline step (no 'pipe: <id>' tag on this Task)")
    files = []
    for t in Task.get_tasks(tags=[tag], task_filter={"page_size": 50}):
        if not t.name.startswith("train_"):
            continue
        cls = t.name[len("train_"):]
        if t.status not in ("completed", "published") or "onnx" not in t.artifacts:
            raise SystemExit(f"{t.name} is {t.status} and has artifacts {list(t.artifacts)}: no ONNX model to check")
        shutil.copy(t.artifacts["onnx"].get_local_copy(), workdir / f"{cls}.onnx")
        shutil.copy(t.artifacts["onnx_meta"].get_local_copy(), workdir / f"{cls}.onnx.json")
        files.append(workdir / f"{cls}.onnx")
    if not files:
        raise SystemExit(f"no train_<class> tasks found with tag {tag!r}")
    return sorted(files)


def measure_isolated(model_dir: Path, s, out_json: Path) -> dict:
    """Measure in a FRESH process: the exporting / downloading above would otherwise count in the peak memory."""
    cmd = [sys.executable, "-m", "src.serve_check", "--models", str(model_dir), "--tiles", str(s.tiles),
           "--size", str(s.tile_size), "--repeats", str(s.repeats), "--workers", str(s.workers), "--intra", str(s.intra),
           "--cpu-limit", str(s.cpu_limit), "--memory-limit-gib", str(s.memory_limit_gib), "--mode", "diagnostic",
           "--out", str(out_json)] + (["--max-request-s", str(s.max_request_s)] if s.max_request_s is not None else [])
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    return json.loads(out_json.read_text())


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    cfg.clearml.task_name = "serve-gate"                        # stable name -> pipeline finds the base task
    faulthandler.dump_traceback_later(300, repeat=True)         # a stall prints its stack into the Task console
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    task, cfg = init_clearml(cfg)
    reporter = Reporter(task)
    s = cfg.serve

    workdir = Path(tempfile.mkdtemp(prefix="serve_gate_"))
    if s.source == "random":
        files = models_from_spec(s.spec, workdir, s.tile_size)
    elif s.source == "tasks":
        files = models_from_pipeline(task, workdir)
    else:
        files = sorted(Path(s.folder).glob("*.onnx"))
        workdir = Path(s.folder)
    print(f"[serve] source={s.source}: {[f.stem for f in files]}", flush=True)

    res = measure_isolated(workdir, s, out_dir / "serve_check.json")
    res["source"] = s.source
    for key in ("request_mean_s", "request_p95_s", "peak_rss_gib", "cpu_cores_busy", "threads_in_use"):
        reporter.scalar("serve", key, res[key], 0)
        reporter.single(f"serve/{key}", res[key])
    for name, sec in res["per_model_mean_s"].items():
        reporter.scalar("serve/per_model_s", name, sec, 0)
    reporter.scalar("serve", "passed", float(res["passed"]), 0)
    (out_dir / "serve_gate.json").write_text(json.dumps(res, indent=2))
    if task:
        task.upload_artifact("serve_gate", artifact_object=res)
    print(json.dumps(res, indent=2))

    if not res["passed"] and s.mode == "gate":
        if task:
            task.mark_failed(status_reason="serve_gate", status_message="; ".join(res["violations"])[:500])
        print(f"SERVE GATE FAILED: {'; '.join(res['violations'])}")
        sys.exit(2)


if __name__ == "__main__":
    main()
