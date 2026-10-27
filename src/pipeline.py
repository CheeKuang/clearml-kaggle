"""ClearML pipeline: resource_check + domain_check run in parallel, train starts only if both pass.

    resource_check ─┐
                    ├─> train
    domain_check   ─┘

  python -m src.pipeline                          # controller on the `services` queue, steps on their queues
  python -m src.pipeline --local                  # controller in this process, steps still go to queues
  python -m src.pipeline --local --steps-local    # everything on this machine (base tasks must exist)
  python -m src.pipeline --local --queue kaggle --train-param train.epochs=2 --param resource.memory_limit_gib=0.3

Follows the ClearML "PipelineController from tasks" pattern: PipelineController -> add_step (one per
existing Task) -> start. A step that exits non-zero (our gates use exit code 2) fails, and its children
never run.
"""
from __future__ import annotations

import argparse

from clearml import PipelineController
from omegaconf import OmegaConf


def overrides(*param_dicts) -> dict:
    """{"architecture": "unet"} -> {"Overrides/architecture": "unet"}: the section written by Task.connect(..., name="Overrides")."""
    return {f"Overrides/{k}": str(v).lower() if isinstance(v, bool) else str(v)
            for d in param_dicts for k, v in d.items() if v is not None}


def build(c) -> PipelineController:
    # 1) the controller
    pipe = PipelineController(name=c.name, project=c.project, version=c.version, add_pipeline_tags=True)

    # 2) steps = existing Tasks (cloned at run time); parameter_override edits the clone's hyperparameters
    pipe.add_step(
        name="resource_check",
        base_task_project=c.project,
        base_task_name=c.steps.resource_check.task,
        parameter_override=overrides(c.params),
        execution_queue=c.steps.resource_check.queue,
    )
    pipe.add_step(
        name="domain_check",
        base_task_project=c.project,
        base_task_name=c.steps.domain_check.task,
        parameter_override=overrides(c.params),
        execution_queue=c.steps.domain_check.queue,
    )
    pipe.add_step(
        name="train",
        parents=["resource_check", "domain_check"],       # runs only after both parents completed
        base_task_project=c.project,
        base_task_name=c.steps.train.task,
        parameter_override=overrides(c.params, c.train_params),
        execution_queue=c.steps.train.queue,
    )
    return pipe


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pipeline/pipeline.yaml")
    ap.add_argument("--local", action="store_true", help="run the controller in this process")
    ap.add_argument("--steps-local", action="store_true", help="with --local: also run the steps in this process")
    ap.add_argument("--queue", help="send every step to this queue instead of the ones in the config")
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="override on every step")
    ap.add_argument("--train-param", action="append", default=[], metavar="KEY=VALUE", help="override on the train step")
    a = ap.parse_args()
    cfg = OmegaConf.load(a.config)
    if a.queue:
        for step in cfg.steps.values():
            step.queue = a.queue
    cfg.params = OmegaConf.merge(cfg.params, dict(kv.split("=", 1) for kv in a.param))
    cfg.train_params = OmegaConf.merge(cfg.train_params, dict(kv.split("=", 1) for kv in a.train_param))
    pipe = build(cfg)

    # 3) start: controller on an agent queue, or in this process
    if a.local:
        pipe.start_locally(run_pipeline_steps_locally=a.steps_local)
    else:
        pipe.start(queue=cfg.controller_queue)
