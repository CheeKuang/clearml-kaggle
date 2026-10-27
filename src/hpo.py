"""Optuna HPO over ClearML Tasks.

  python -m src.hpo --mode clearml   # HyperParameterOptimizer + OptimizerOptuna (controller Task)
  python -m src.hpo --mode manual    # explicit Optuna ask/tell + Task.clone / Task.enqueue loop
Trial parameters are written to "Overrides/<hydra key>" on each cloned Task (see Task.connect in train).
"""
from __future__ import annotations

import argparse
import math
import time

import optuna
from clearml import Task
from omegaconf import OmegaConf

SECTION = "Overrides"


def base_task_of(cfg) -> Task:
    t = Task.get_task(project_name=cfg.base_task.project, task_name=cfg.base_task.name)
    if t is None:
        raise SystemExit("Base task not found: run `python -m src.train` once first (or with clearml.execute_remotely=true).")
    return t


# ----------------------------------------------------------------------------- mode 1: clearml-managed
def run_clearml_optimizer(cfg):
    from clearml.automation import (DiscreteParameterRange, HyperParameterOptimizer, LogUniformParameterRange,
                                    UniformParameterRange)
    from clearml.automation.optuna import OptimizerOptuna

    ctl = Task.init(project_name=cfg.base_task.project, task_name=cfg.controller_task_name,
                    task_type=Task.TaskTypes.optimizer, reuse_last_task_id=False)
    base = base_task_of(cfg)

    ranges = []
    for s in cfg.space:
        name = f"{SECTION}/{s.name}"
        if s.type == "log_uniform":       # ClearML takes exponents: value = 10 ** x
            ranges.append(LogUniformParameterRange(name, min_value=math.log10(s.min), max_value=math.log10(s.max)))
        elif s.type == "uniform":
            ranges.append(UniformParameterRange(name, min_value=s.min, max_value=s.max))
        else:
            ranges.append(DiscreteParameterRange(name, values=list(s["values"])))

    opt = HyperParameterOptimizer(
        base_task_id=base.id,
        hyper_parameters=ranges,
        objective_metric_title=cfg.objective.title,
        objective_metric_series=cfg.objective.series,
        objective_metric_sign=cfg.objective.sign,
        optimizer_class=OptimizerOptuna,
        execution_queue=cfg.queue,
        max_number_of_concurrent_tasks=cfg.max_concurrent,
        total_max_jobs=cfg.n_trials,
        pool_period_min=1.0,
        save_top_k_tasks_only=5,
        optuna_sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        optuna_pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    # ctl.execute_remotely(queue_name="services")   # uncomment to run the controller itself on an agent
    opt.set_report_period(1.0)
    opt.start()
    opt.wait()
    for t in opt.get_top_experiments(top_k=3):
        print("top:", t.id, t.name, t.get_last_scalar_metrics().get(cfg.objective.title, {}))
    opt.stop()
    ctl.close()


# ----------------------------------------------------------------------------- mode 2: manual ask/tell
def suggest(trial: optuna.Trial, s):
    if s.type == "log_uniform":
        return trial.suggest_float(s.name, s.min, s.max, log=True)
    if s.type == "uniform":
        return trial.suggest_float(s.name, s.min, s.max)
    return trial.suggest_categorical(s.name, list(s["values"]))


def run_manual(cfg):
    ctl = Task.init(project_name=cfg.base_task.project, task_name=cfg.controller_task_name + "-manual",
                    task_type=Task.TaskTypes.optimizer, reuse_last_task_id=False)
    base = base_task_of(cfg)
    study = optuna.create_study(direction="maximize" if "max" in cfg.objective.sign else "minimize",
                                sampler=optuna.samplers.TPESampler(seed=cfg.seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    running, launched, finished = {}, 0, 0
    seen_steps: dict[int, int] = {}
    t_, s_ = cfg.objective.title, cfg.objective.series

    while finished < cfg.n_trials:
        while len(running) < cfg.max_concurrent and launched < cfg.n_trials:
            trial = study.ask()
            params = {f"{SECTION}/{s.name}": suggest(trial, s) for s in cfg.space}
            clone = Task.clone(source_task=base, name=f"trial-{trial.number}", parent=base.id)  # -> Draft
            clone.set_parameters(params)
            clone.add_tags(["optuna", f"trial-{trial.number}"])
            Task.enqueue(clone, queue_name=cfg.queue)                                           # -> Pending
            running[trial.number] = (trial, clone); seen_steps[trial.number] = 0
            launched += 1
            print(f"launched trial {trial.number}: {params}")

        time.sleep(30)
        for n, (trial, t) in list(running.items()):
            status = t.get_status()
            ys = t.get_reported_scalars().get(t_, {}).get(s_, {}).get("y", [])
            for step in range(seen_steps[n], len(ys)):       # stream intermediate values -> pruner
                trial.report(float(ys[step]), step)
            seen_steps[n] = len(ys)
            if status in ("in_progress",) and ys and trial.should_prune():
                t.mark_stop_request(force=True, status_message="pruned by optuna")
                study.tell(trial, state=optuna.trial.TrialState.PRUNED); running.pop(n); finished += 1
            elif status in ("completed", "published", "closed") and ys:
                study.tell(trial, float(max(ys) if "max" in cfg.objective.sign else min(ys)))
                running.pop(n); finished += 1
            elif status in ("failed", "stopped"):
                study.tell(trial, state=optuna.trial.TrialState.FAIL); running.pop(n); finished += 1
        ctl.get_logger().report_scalar("optuna", "best_value", study.best_value if study.best_trials else 0.0, finished)

    print("best:", study.best_value, study.best_params)
    ctl.upload_artifact("best_params", artifact_object=study.best_params)
    ctl.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["clearml", "manual"], default="clearml")
    ap.add_argument("--space", default="hpo/space.yaml")
    a = ap.parse_args()
    c = OmegaConf.load(a.space)
    run_clearml_optimizer(c) if a.mode == "clearml" else run_manual(c)
