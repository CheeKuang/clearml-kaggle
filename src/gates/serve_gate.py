"""python -m src.gates.serve_gate [hydra overrides]   (conf/serve/default.yaml)

ClearML Task around src/gates/serve_check.py: can ALL production models run together inside the container limits
(4 vCPU request / 8 vCPU limit / 8 GiB for the whole container)? `serve.source` chooses the models:
  random  untrained models of `serve.spec` ("class=architecture:encoder,..."), exported to ONNX. Same cost as trained
          ones, so this runs BEFORE training and stops a pipeline that could never fit.
  tasks   the ONNX artifacts of the sibling `train_<class>` steps (found by this Task's "pipe: <id>" tag). AFTER training.
  folder  the .onnx files in serve.folder (local use).
Exit code 2 + Task failed on a violation. Register the base task with NO extra arguments.
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
# heavy native libraries BEFORE Task.init: imported lazily after ClearML's import hooks, smp hung in `import triton`
import onnxruntime  # noqa: F401
import segmentation_models_pytorch  # noqa: F401
import torch  # noqa: F401
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from ..common.clearml_utils import Reporter, fail_gate, init_clearml
from ..export_onnx import export_untrained

REPO = Path(__file__).resolve().parents[2]


def models_from_spec(spec: str, workdir: Path, size: int) -> list[Path]:
    files = []
    for item in filter(None, (x.strip() for x in spec.split(","))):
        cls, _, arch = item.partition("=")
        architecture, _, encoder = arch.partition(":")
        export_untrained(cls, architecture, encoder, workdir / f"{cls}.onnx", size)
        files.append(workdir / f"{cls}.onnx")
    return files


def models_from_pipeline(task, workdir: Path) -> list[Path]:
    """ONNX artifacts of the train_<class> steps of the same pipeline run as `task`."""
    from clearml import Task

    tag = next((t for t in task.get_tags() if t.startswith("pipe: ")), None)
    if tag is None:
        raise SystemExit("serve.source=tasks must run as a pipeline step (no 'pipe: <id>' tag on this Task)")
    files = []
    for t in Task.get_tasks(tags=[tag], task_filter={"page_size": 50}):
        if not t.name.startswith("train_"):
            continue
        if t.status not in ("completed", "published") or "onnx" not in t.artifacts:
            raise SystemExit(f"{t.name} is {t.status} and has artifacts {list(t.artifacts)}: no ONNX model to check")
        cls = t.name[len("train_"):]
        shutil.copy(t.artifacts["onnx"].get_local_copy(), workdir / f"{cls}.onnx")
        shutil.copy(t.artifacts["onnx_meta"].get_local_copy(), workdir / f"{cls}.onnx.json")
        files.append(workdir / f"{cls}.onnx")
    if not files:
        raise SystemExit(f"no train_<class> tasks found with tag {tag!r}")
    return sorted(files)


def measure_isolated(model_dir: Path, s, out_json: Path) -> dict:
    """Measure in a FRESH process: the export / download above would otherwise count in the peak memory."""
    args = {"models": model_dir, "tiles": s.tiles, "size": s.tile_size, "repeats": s.repeats, "workers": s.workers,
            "intra": s.intra, "cpu-limit": s.cpu_limit, "memory-limit-gib": s.memory_limit_gib,
            "max-request-s": s.max_request_s, "mode": "diagnostic", "out": out_json}
    cmd = [sys.executable, "-m", "src.gates.serve_check"]
    cmd += [x for k, v in args.items() if v is not None for x in (f"--{k}", str(v))]
    subprocess.run(cmd, check=True, cwd=REPO)
    return json.loads(out_json.read_text())


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    cfg.clearml.task_name = "serve-gate"                        # stable name: the pipeline finds the base task by it
    faulthandler.dump_traceback_later(300, repeat=True)         # a stall prints its stack into the Task console
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    task, cfg = init_clearml(cfg)
    reporter, s = Reporter(task), cfg.serve

    workdir = Path(s.folder) if s.source == "folder" else Path(tempfile.mkdtemp(prefix="serve_gate_"))
    files = {"random": lambda: models_from_spec(s.spec, workdir, s.tile_size),
             "tasks": lambda: models_from_pipeline(task, workdir),
             "folder": lambda: sorted(workdir.glob("*.onnx"))}[s.source]()
    print(f"[serve] source={s.source}: {[f.stem for f in files]}", flush=True)

    res = {**measure_isolated(workdir, s, out_dir / "serve_check.json"), "source": s.source}
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
        fail_gate(task, "serve_gate", "; ".join(res["violations"]))


if __name__ == "__main__":
    main()
