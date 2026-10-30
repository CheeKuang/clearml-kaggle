"""ClearML <-> Hydra glue: Task.init, the editable "Overrides", recompose, a no-op-safe logger, gate failure.

How remote runs get their settings:
  * Hydra is the only config authority; ClearML never edits the YAML files.
  * The keys in `clearml.tunable` (conf/config.yaml) are connected as the task's "Overrides". When an agent runs a
    clone whose Overrides were edited (UI, HPO, pipeline), the new values come back from task.connect() and the
    config is recomposed. Only keys in that list can be changed remotely.
Traps handled here (do not undo):
  * connect() updates the dict you pass IN PLACE on remote runs -> compare against a copy.
  * connect() casts remote values to the local default's type (0.3 over an int 8 is dropped) -> connect text.
  * ClearML stores None as "None" or "" -> _fmt() maps both to Hydra's `null`.
"""
from __future__ import annotations

import re
import sys
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from ..data.tiles import class_names
from .envinfo import git_state

try:
    from clearml import Task
except ImportError:  # runs without clearml installed
    Task = None

# Imported lazily (Hydra _target_ strings, inside functions), so ClearML's static analysis misses them.
FORCED_REQUIREMENTS = ["lightning", "pandas", "segmentation_models_pytorch", "torchgeo", "hydra-core", "omegaconf",
                       "rasterio", "scikit-learn", "scipy", "onnxruntime", "opencv-python-headless", "onnx-graphsurgeon"]


class Reporter:
    """Task.get_logger() wrapper; every call is a no-op when ClearML is disabled."""

    def __init__(self, task: Any | None):
        self.task = task
        self.logger = task.get_logger() if task is not None else None

    def scalar(self, title: str, series: str, value: float, iteration: int = 0):
        if self.logger:
            self.logger.report_scalar(title, series, float(value), iteration)
        print(f"[{title}/{series}] it={iteration} {float(value):.5f}")

    def single(self, name: str, value: float):
        if self.logger:
            self.logger.report_single_value(name, float(value))

    def histogram(self, title: str, series: str, values, iteration: int = 0, xlabels=None):
        if self.logger:
            self.logger.report_histogram(title, series, values=values, iteration=iteration, xlabels=xlabels)

    def table(self, title: str, series: str, df, iteration: int = 0):
        if self.logger:
            self.logger.report_table(title, series, iteration=iteration, table_plot=df)

    def params(self, section: str, values: dict):
        """Write `values` as their own section of the Hyperparameters tab. Display only: nothing reads them back
        (edit "Overrides" to change a run)."""
        if self.task:
            self.task.update_parameters({f"{section}/{k}": _fmt(v) for k, v in values.items()})


def flatten(d: dict, prefix: str = "") -> dict:
    """{"a": {"b": 1}} -> {"a.b": 1}"""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def fail_gate(task, reason: str, message: str):
    """A gate was violated: mark the Task failed (so a pipeline stops) and exit with code 2."""
    if task:
        task.mark_failed(status_reason=reason, status_message=message[:500])
    print(f"{reason.upper().replace('_', ' ')} FAILED: {message}")
    sys.exit(2)


def _fmt(v: Any) -> str:
    """Value -> the text of a Hydra override (`key=<this>`); also the canonical form for comparisons."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None or v in ("None", ""):
        return "null"
    return str(v)


def _quote(text: str) -> str:
    """Hydra override values containing = , : [ ] { } ( ) quotes or spaces must be quoted."""
    if not re.search(r"[=,:\[\]{}()\s'\"\\]", text):
        return text
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def recompose(hc: HydraConfig, tunables: dict[str, Any]) -> DictConfig:
    """Original CLI overrides (minus the tunable keys) + the tunable values, so the edited values win."""
    cli = [o for o in hc.overrides.task if o.lstrip("+~").split("=")[0] not in tunables]
    return hydra.compose(config_name=hc.job.config_name,
                         overrides=cli + [f"{k}={_quote(_fmt(v))}" for k, v in tunables.items()])


def init_clearml(cfg: DictConfig) -> tuple[Any | None, DictConfig]:
    """Task.init -> connect the tunable keys as "Overrides" -> recompose with remote edits. (None, cfg) when off."""
    if not cfg.clearml.enabled or Task is None:
        return None, cfg
    c, hc = cfg.clearml, HydraConfig.get()
    if untracked := git_state()["untracked"]:
        print(f"[clearml] WARNING: {len(untracked)} file(s) under src/ or conf/ are not in git, so a remote run of this "
              f"Task will not have them (commit and push first): {untracked[:10]}", flush=True)

    # Before Task.init. The agent installs exactly the versions recorded here, so an offline image must match them.
    for pkg in FORCED_REQUIREMENTS:
        Task.add_requirements(pkg)
    # torch.onnx.export imports onnx; onnx >= 1.19 needs protobuf 6 and will not load next to Kaggle's system
    # protobuf 5 (its `google.*` namespace wins over a venv copy). onnx < 1.19 has no such check.
    Task.add_requirements("onnx", "<1.19")
    Task.add_requirements("protobuf", "<6")

    task = Task.init(project_name=c.project, task_name=c.task_name, task_type=Task.TaskTypes.training,
                     tags=list(c.tags), reuse_last_task_id=False, output_uri=c.output_uri,
                     auto_connect_frameworks={"hydra": False})            # Hydra is handled below (one authority)

    choices = hc.runtime.choices                                          # config groups use the chosen option name
    state = {k: _fmt(choices[k] if k in choices else OmegaConf.select(cfg, k)) for k in c.tunable}
    original = dict(state)
    tunables = {k: _fmt(v) for k, v in task.connect(state, name="Overrides").items()}
    if tunables != original:
        cfg = recompose(hc, tunables)
        print(f"[clearml] config recomposed with remote overrides: {tunables}")

    task.connect_configuration(OmegaConf.to_container(cfg, resolve=True), name="resolved_config",
                               description="Final Hydra config after overrides", ignore_remote_overrides=True)
    # the same final config, one field per value: columns in the experiments table, side by side in Compare
    Reporter(task).params("Resolved", {k: v for k, v in flatten(OmegaConf.to_container(cfg, resolve=True)).items()
                                       if not k.startswith("clearml.")})
    task.connect_label_enumeration({"background": 0, **{n: i + 1 for i, n in enumerate(class_names(cfg.data))}})
    task.add_tags([str(cfg.model.arch), str(cfg.model.encoder_name), str(cfg.pretrain.source)])

    if c.execute_remotely:                     # send this run to an agent queue (an agent runs an existing clone)
        task.execute_remotely(queue_name=c.queue, clone=c.clone_on_remote, exit_process=True)
    return task, cfg
