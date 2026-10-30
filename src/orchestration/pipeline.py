"""ClearML pipeline, ONE MODEL PER CLASS (production serves one model per class).

    domain_check ──┐                 ┌─> train_<class> ─┐
                   ├─> both gates ───┤        ...       ├─> serve_gate_post
    serve_gate_pre ┘                 └─> train_<class> ─┘
    pre: untrained models of the configured architectures; post: the real exported ONNX models, all together

  python -m src.orchestration.pipeline --dry-run          # print the steps and their overrides, touch nothing
  python -m src.orchestration.pipeline --local --queue kaggle --param data.clearml_dataset_id=<id> \
      --train-param train.epochs=2 --train-param train.batch_size=4
  python -m src.orchestration.pipeline --local --classes road,water ...   # some classes (the gates check only those)

PipelineController -> add_step (one per base Task) -> start. A step that exits non-zero (gates use 2) fails, and its
children are never created.
"""
from __future__ import annotations

import argparse

from omegaconf import OmegaConf


def overrides(*param_dicts) -> dict:
    """{"architecture": "unet"} -> {"Overrides/architecture": "unet"} (the section Task.connect(name="Overrides") made)."""
    return {f"Overrides/{k}": str(v).lower() if isinstance(v, bool) else str(v)
            for d in param_dicts for k, v in d.items() if v is not None}


def plan(c, classes: list[str] | None = None) -> list[dict]:
    """The pipeline as plain data: one dict of add_step() arguments per step, in dependency order."""
    names = classes or list(c.classes)
    if unknown := [n for n in names if n not in c.classes]:
        raise SystemExit(f"unknown classes {unknown}; pipeline.yaml has {list(c.classes)}")
    spec = ",".join(f"{n}={c.classes[n].architecture}:{c.classes[n].encoder}" for n in names)

    def step(name, kind, parents=(), extra=None):
        return {"name": name, "parents": list(parents), "base_task_project": c.project,
                "base_task_name": c.steps[kind].task, "execution_queue": c.steps[kind].queue,
                "parameter_override": overrides(c.params, extra or {})}

    trains = [step(f"train_{n}", "train", ["domain_check", "serve_gate_pre"],
                   {**c.train_params, "data": n, "architecture": c.classes[n].architecture,
                    "encoder": c.classes[n].encoder, "loss": c.classes[n].get("loss")}) for n in names]
    return [step("domain_check", "domain_check", extra={"data": "loveda"}),
            step("serve_gate_pre", "serve_gate", extra={**c.serve_params, "serve.source": "random", "serve.spec": spec}),
            *trains,
            step("serve_gate_post", "serve_gate", [t["name"] for t in trains], {**c.serve_params, "serve.source": "tasks"})]


def newest_base_task_id(project: str, name: str) -> str:
    """The most recently CREATED non-archived task with this exact name, whatever its status.
    PipelineController's own lookup by name also returns archived tasks and skips `failed` ones, so after
    re-registering it can silently pick an old base task pinned to an old commit. Creation time, not last_update:
    stopping or editing an old base task bumps its last_update and made it look newest."""
    from clearml import Task

    if not (tasks := Task.get_tasks(project_name=project, task_name=f"^{name}$", allow_archived=False)):
        raise SystemExit(f"no non-archived base task named {name!r} in project {project!r}")
    return max(tasks, key=lambda t: t.data.created).id


def build(c, classes: list[str] | None = None):
    from clearml import PipelineController

    pipe = PipelineController(name=c.name, project=c.project, version=c.version, add_pipeline_tags=True)
    for step in plan(c, classes):
        project, name = step.pop("base_task_project"), step.pop("base_task_name")   # add_step takes the id, not these
        pipe.add_step(**step, base_task_id=newest_base_task_id(project, name))
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline/pipeline.yaml")
    ap.add_argument("--dry-run", action="store_true", help="print the steps and exit (no ClearML calls)")
    ap.add_argument("--local", action="store_true", help="run the controller in this process")
    ap.add_argument("--steps-local", action="store_true", help="with --local: also run the steps in this process")
    ap.add_argument("--queue", help="send every step to this queue instead of the ones in the config")
    ap.add_argument("--classes", help="comma separated subset of the classes in the config")
    for flag, where in (("param", "every step"), ("train-param", "the train steps"), ("serve-param", "the serve gates")):
        ap.add_argument(f"--{flag}", action="append", default=[], metavar="KEY=VALUE", help=f"override on {where}")
    a = ap.parse_args()

    cfg = OmegaConf.load(a.config)
    if a.queue:
        for step in cfg.steps.values():
            step.queue = a.queue
    for key, given in (("params", a.param), ("train_params", a.train_param), ("serve_params", a.serve_param)):
        cfg[key] = OmegaConf.merge(cfg[key], dict(kv.split("=", 1) for kv in given))
    chosen = a.classes.split(",") if a.classes else None

    if a.dry_run:
        for s in plan(cfg, chosen):
            print(f"{s['name']:18} after {', '.join(s['parents']) or '-':58} base={s['base_task_name']:14} queue={s['execution_queue']}")
            print(f"{'':18}    {s['parameter_override']}")
        return
    if "Overrides/data.clearml_dataset_id" not in overrides(cfg.params):
        raise SystemExit("data.clearml_dataset_id is not set: pass --param data.clearml_dataset_id=<id> "
                         "(the id printed by python -m src.data.upload)")
    pipe = build(cfg, chosen)
    if a.local:
        pipe.start_locally(run_pipeline_steps_locally=a.steps_local)
    else:
        pipe.start(queue=cfg.controller_queue)


if __name__ == "__main__":
    main()
