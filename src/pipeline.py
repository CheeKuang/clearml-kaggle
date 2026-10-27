"""ClearML pipeline with ONE MODEL PER CLASS (production serves one model per class).

    domain_check ──┐                 ┌─> train_<class> ─┐
                   ├─> both gates ───┤        ...       ├─> serve_gate_post
    serve_gate_pre ┘                 └─> train_<class> ─┘
    (untrained models of the configured architectures, all together, in the production limits)
    serve_gate_post: the real exported ONNX models, all together

  python -m src.pipeline --dry-run                # print the steps and their overrides, touch nothing
  python -m src.pipeline --local --queue kaggle --train-param train.epochs=2 --train-param train.batch_size=4
  python -m src.pipeline                          # controller on the `services` queue
  python -m src.pipeline --local --classes road,water   # only some classes (the gates then check only those)

Follows the ClearML "PipelineController from tasks" pattern: PipelineController -> add_step (one per existing Task)
-> start. A step that exits non-zero (our gates use exit code 2) fails, and its children never run.
"""
from __future__ import annotations

import argparse

from omegaconf import OmegaConf


def overrides(*param_dicts) -> dict:
    """{"architecture": "unet"} -> {"Overrides/architecture": "unet"}: the section written by Task.connect(..., name="Overrides")."""
    return {f"Overrides/{k}": str(v).lower() if isinstance(v, bool) else str(v)
            for d in param_dicts for k, v in d.items() if v is not None}


def plan(c, classes: list[str] | None = None) -> list[dict]:
    """The pipeline as plain data: one dict of add_step() arguments per step, in dependency order."""
    names = classes or list(c.classes)
    unknown = [n for n in names if n not in c.classes]
    if unknown:
        raise SystemExit(f"unknown classes {unknown}; pipeline.yaml has {list(c.classes)}")
    spec = ",".join(f"{n}={c.classes[n].architecture}:{c.classes[n].encoder}" for n in names)

    def step(name, kind, parents=None, extra=None):
        return {"name": name, "parents": parents or [], "base_task_project": c.project,
                "base_task_name": c.steps[kind].task, "execution_queue": c.steps[kind].queue,
                "parameter_override": overrides(c.params, extra or {})}

    steps = [
        step("domain_check", "domain_check", extra={"data": "loveda"}),
        step("serve_gate_pre", "serve_gate", extra={**c.serve_params, "serve.source": "random", "serve.spec": spec}),
    ]
    for n in names:                                         # one training step per class
        steps.append(step(f"train_{n}", "train", ["domain_check", "serve_gate_pre"],
                          {**c.train_params, "data": n, "architecture": c.classes[n].architecture,
                           "encoder": c.classes[n].encoder}))
    steps.append(step("serve_gate_post", "serve_gate", [f"train_{n}" for n in names],
                      {**c.serve_params, "serve.source": "tasks"}))
    return steps


def build(c, classes: list[str] | None = None):
    from clearml import PipelineController

    pipe = PipelineController(name=c.name, project=c.project, version=c.version, add_pipeline_tags=True)
    for s in plan(c, classes):
        pipe.add_step(**s)                                  # name, parents, base task, queue, parameter_override
    return pipe


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline/pipeline.yaml")
    ap.add_argument("--dry-run", action="store_true", help="print the steps and exit (no ClearML calls)")
    ap.add_argument("--local", action="store_true", help="run the controller in this process")
    ap.add_argument("--steps-local", action="store_true", help="with --local: also run the steps in this process")
    ap.add_argument("--queue", help="send every step to this queue instead of the ones in the config")
    ap.add_argument("--classes", help="comma separated subset of the classes in the config")
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="override on every step")
    ap.add_argument("--train-param", action="append", default=[], metavar="KEY=VALUE", help="override on the train steps")
    ap.add_argument("--serve-param", action="append", default=[], metavar="KEY=VALUE", help="override on the serve gates")
    a = ap.parse_args()
    cfg = OmegaConf.load(a.config)
    if a.queue:
        for step in cfg.steps.values():
            step.queue = a.queue
    cfg.params = OmegaConf.merge(cfg.params, dict(kv.split("=", 1) for kv in a.param))
    cfg.train_params = OmegaConf.merge(cfg.train_params, dict(kv.split("=", 1) for kv in a.train_param))
    cfg.serve_params = OmegaConf.merge(cfg.serve_params, dict(kv.split("=", 1) for kv in a.serve_param))
    chosen = a.classes.split(",") if a.classes else None

    if a.dry_run:
        for s in plan(cfg, chosen):
            print(f"{s['name']:18} after {', '.join(s['parents']) or '-':58} base={s['base_task_name']:14} queue={s['execution_queue']}")
            print(f"{'':18}    {s['parameter_override']}")
        raise SystemExit(0)
    if "Overrides/data.clearml_dataset_id" not in overrides(cfg.params):
        raise SystemExit("pipeline.yaml: params.data.clearml_dataset_id is not set. Upload the multi-class dataset "
                         "(python -m src.dataset --path data/loveda_multi --name loveda-multi) and put its id there.")

    pipe = build(cfg, chosen)
    # start: controller on an agent queue, or in this process
    if a.local:
        pipe.start_locally(run_pipeline_steps_locally=a.steps_local)
    else:
        pipe.start(queue=cfg.controller_queue)
