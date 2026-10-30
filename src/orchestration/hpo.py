"""Optuna HPO over clones of the train base task. UNTESTED.

  python -m src.orchestration.hpo --mode clearml   # HyperParameterOptimizer + OptimizerOptuna (controller Task)
  python -m src.orchestration.hpo --mode manual    # explicit Optuna ask/tell + Task.clone / Task.enqueue loop
Trial values go to "Overrides/<hydra key>" on each clone (keys must be in clearml.tunable). Search space: hpo/space.yaml.
"""
from __future__ import annotations

import argparse
import math
import time

import optuna
from clearml import Task
from omegaconf import OmegaConf

from .pipeline import newest_base_task_id

SECTION = "Overrides"


def base_task_of(cfg) -> Task:
    return Task.get_task(task_id=newest_base_task_id(cfg.base_task.project, cfg.base_task.name))


def make_study(cfg) -> optuna.Study:
    return optuna.create_study(direction="maximize" if "max" in cfg.objective.sign else "minimize",
                                sampler=optuna.samplers.TPESampler(seed=cfg.seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))


def run_clearml_optimizer(cfg):
    from clearml.automation import (DiscreteParameterRange, HyperParameterOptimizer, LogUniformParameterRange,
                                    UniformParameterRange)
    from clearml.automation.optuna import OptimizerOptuna

    ctl = Task.init(project_name=cfg.base_task.project, task_name=cfg.controller_task_name,
                    task_type=Task.TaskTypes.optimizer, reuse_last_task_id=False)

    def param_range(s):
        name = f"{SECTION}/{s.name}"
        if s.type == "log_uniform":             # ClearML takes exponents: value = 10 ** x
            return LogUniformParameterRange(name, min_value=math.log10(s.min), max_value=math.log10(s.max))
        if s.type == "uniform":
            return UniformParameterRange(name, min_value=s.min, max_value=s.max)
        return DiscreteParameterRange(name, values=list(s["values"]))

    opt = HyperParameterOptimizer(
        base_task_id=base_task_of(cfg).id, hyper_parameters=[param_range(s) for s in cfg.space],
        objective_metric_title=cfg.objective.title, objective_metric_series=cfg.objective.series,
        objective_metric_sign=cfg.objective.sign, optimizer_class=OptimizerOptuna, execution_queue=cfg.queue,
        max_number_of_concurrent_tasks=cfg.max_concurrent, total_max_jobs=cfg.n_trials, pool_period_min=1.0,
        save_top_k_tasks_only=5, optuna_sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        optuna_pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    opt.set_report_period(1.0)
    opt.start()
    opt.wait()
    for t in opt.get_top_experiments(top_k=3):
        print("top:", t.id, t.name, t.get_last_scalar_metrics().get(cfg.objective.title, {}))
    opt.stop()
    ctl.close()


def suggest(trial: optuna.Trial, s):
    if s.type == "log_uniform":
        return trial.suggest_float(s.name, s.min, s.max, log=True)
    if s.type == "uniform":
        return trial.suggest_float(s.name, s.min, s.max)
    return trial.suggest_categorical(s.name, list(s["values"]))


def run_manual(cfg):
    ctl = Task.init(project_name=cfg.base_task.project, task_name=cfg.controller_task_name + "-manual",
                    task_type=Task.TaskTypes.optimizer, reuse_last_task_id=False)
    base, study = base_task_of(cfg), make_study(cfg)
    maximize = "max" in cfg.objective.sign
    running, seen, launched, finished = {}, {}, 0, 0

    while finished < cfg.n_trials:
        while len(running) < cfg.max_concurrent and launched < cfg.n_trials:
            trial = study.ask()
            params = {f"{SECTION}/{s.name}": suggest(trial, s) for s in cfg.space}
            clone = Task.clone(source_task=base, name=f"trial-{trial.number}", parent=base.id)
            clone.update_parameters(params)     # NOT set_parameters: that replaces every parameter of the clone
            clone.add_tags(["optuna", f"trial-{trial.number}"])
            Task.enqueue(clone, queue_name=cfg.queue)
            running[trial.number], seen[trial.number] = (trial, clone), 0
            launched += 1
            print(f"launched trial {trial.number}: {params}")

        time.sleep(30)
        for n, (trial, t) in list(running.items()):
            status = t.get_status()
            ys = t.get_reported_scalars().get(cfg.objective.title, {}).get(cfg.objective.series, {}).get("y", [])
            for step in range(seen[n], len(ys)):        # stream intermediate values to the pruner
                trial.report(float(ys[step]), step)
            seen[n] = len(ys)
            if status == "in_progress" and ys and trial.should_prune():
                t.mark_stop_request(force=True, status_message="pruned by optuna")
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            elif status in ("completed", "published", "closed") and ys:
                study.tell(trial, float(max(ys) if maximize else min(ys)))
            elif status in ("failed", "stopped"):
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            else:
                continue
            running.pop(n)
            finished += 1
        ctl.get_logger().report_scalar("optuna", "best_value", study.best_value if study.best_trials else 0.0, finished)

    print("best:", study.best_value, study.best_params)
    ctl.upload_artifact("best_params", artifact_object=study.best_params)
    ctl.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["clearml", "manual"], default="clearml")
    ap.add_argument("--space", default="hpo/space.yaml")
    a = ap.parse_args()
    (run_clearml_optimizer if a.mode == "clearml" else run_manual)(OmegaConf.load(a.space))
