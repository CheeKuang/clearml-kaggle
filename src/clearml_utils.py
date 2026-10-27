"""ClearML <-> Hydra glue: Task.init, Task.connect (editable hyperparameters), recompose, Logger wrapper.

READ THIS FIRST (how remote runs get their settings)
  * Hydra is the only config authority. ClearML never edits the YAML files.
  * `init_clearml` publishes the keys listed in conf/config.yaml `clearml.tunable` as the task's "Overrides"
    hyperparameters. When an agent runs a CLONE whose Overrides were changed (UI, HPO or a pipeline's
    parameter_override), the changed values come back through task.connect() and the config is recomposed.
  * A setting can therefore only be changed remotely if its key is in `clearml.tunable`.
Traps this file already handles (do not undo them):
  * connect() updates the dict you pass IN PLACE on remote runs -> keep a copy before calling it.
  * connect() casts remote values to the type of the local default (0.3 over an int 8 is dropped) -> every
    value is connected as text and Hydra parses it.
  * ClearML stores None as "None" or "" -> _fmt() maps both back to Hydra's `null`.
Docs: Task.init, Task.connect, Task.connect_configuration, Task.execute_remotely.
"""
from __future__ import annotations

import re
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .data import class_names

try:
    from clearml import Task
except ImportError:  # allows running without clearml installed
    Task = None

# Packages imported lazily (via Hydra _target_ strings / inside functions) are invisible to
# ClearML's static import analysis, so the agent would not install them -> add explicitly.
FORCED_REQUIREMENTS = ["lightning", "pandas", "segmentation_models_pytorch", "torchgeo", "hydra-core", "omegaconf", "rasterio",
                       "scikit-learn", "scipy", "onnx", "onnxruntime", "opencv-python-headless"]


class Reporter:
    """Thin wrapper around Task.get_logger(); every call is a no-op when ClearML is disabled."""

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

    def image(self, title: str, series: str, img, iteration: int = 0):
        if self.logger:
            self.logger.report_image(title, series, iteration=iteration, image=img)

    def text(self, msg: str):
        if self.logger:
            self.logger.report_text(msg)


# ----------------------------------------------------------------------------- hydra helpers
def _fmt(v: Any) -> str:
    """Value -> the text of a Hydra override (`key=<this>`). Also the canonical form used to compare values."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None or v in ("None", ""):      # ClearML stringifies None (as "None" or "")
        return "null"
    return str(v)


def _quote(text: str) -> str:
    """Hydra override values containing = , : [ ] { } ( ) quotes or spaces must be quoted."""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'" if re.search(r"[=,:\[\]{}()\s'\"\\]", text) else text


def tunable_state(cfg: DictConfig, hc: HydraConfig) -> dict[str, Any]:
    """Current value of each tunable key. Config-group keys use the chosen option name."""
    choices = hc.runtime.choices
    state = {}
    for key in cfg.clearml.tunable:
        state[key] = choices[key] if key in choices else OmegaConf.select(cfg, key)
    return state


def recompose(hc: HydraConfig, tunables: dict[str, Any]) -> DictConfig:
    """Rebuild the config = original CLI overrides (minus tunable keys) + tunable values.
    Tunable keys are dropped from the CLI part so the (possibly edited) tunable value wins."""
    keys = set(tunables)
    cli = [o for o in hc.overrides.task if o.lstrip("+~").split("=")[0] not in keys]
    overrides = cli + [f"{k}={_quote(_fmt(v))}" for k, v in tunables.items()]
    return hydra.compose(config_name=hc.job.config_name, overrides=overrides)


def init_clearml(cfg: DictConfig) -> tuple[Any | None, DictConfig]:
    """Task.init -> Task.connect (editable hyperparameters) -> recompose with UI/HPO overrides."""
    hc = HydraConfig.get()
    if not cfg.clearml.enabled or Task is None:
        return None, cfg
    c = cfg.clearml

    # must run before Task.init. ClearML records the package versions of THIS machine and the agent installs
    # exactly those, so an agent image / offline mirror has to provide the same versions.
    for pkg in FORCED_REQUIREMENTS:
        Task.add_requirements(pkg)
    # torch.onnx.export imports `onnx`, whose generated code needs protobuf >= 6; Kaggle's system protobuf is 5.x and
    # ClearML records no version for packages the code does not import itself, so the clash only showed on the agent.
    Task.add_requirements("protobuf", ">=6.31.1")

    # 1) Task.init -- reuse_last_task_id=False => every call makes a fresh Task (no overwritten history)
    task = Task.init(
        project_name=c.project,
        task_name=c.task_name,
        task_type=Task.TaskTypes.training,
        tags=list(c.tags),
        reuse_last_task_id=False,
        output_uri=c.output_uri,
        auto_connect_frameworks={"hydra": False},  # Hydra is handled explicitly below (single authority)
    )

    # 2) Task.connect -- the dict shows under "Overrides" in the UI. When an agent runs a clone, the
    #    values edited in the UI (or set by the HPO controller) come back here.
    # values go in as text: connect() casts remote values back to the type of the local default, which drops
    # e.g. "0.3" over an int default (8); Hydra parses the text when the config is recomposed
    state = {k: _fmt(v) for k, v in tunable_state(cfg, hc).items()}
    original = dict(state)                       # connect() updates `state` in place on remote runs
    tunables = {k: _fmt(v) for k, v in task.connect(state, name="Overrides").items()}
    if tunables != original:
        cfg = recompose(hc, tunables)
        print(f"[clearml] config recomposed with remote overrides: {tunables}")

    # 3) snapshot of the *final* resolved config (editable HOCON in the UI; local values win on remote)
    task.connect_configuration(OmegaConf.to_container(cfg, resolve=True), name="resolved_config",
                               description="Final Hydra config after overrides",
                               ignore_remote_overrides=True)
    task.connect_label_enumeration({"background": 0, **{n: i + 1 for i, n in enumerate(class_names(cfg.data))}})
    task.add_tags([str(cfg.model.arch), str(cfg.model.encoder_name), str(cfg.pretrain.source)])

    # 4) optional: ship this run to an agent queue (e.g. a clearml-agent running inside a Kaggle notebook).
    #    On the agent this branch is not used: the agent runs a clone that already exists.
    if c.execute_remotely:
        task.execute_remotely(queue_name=c.queue, clone=c.clone_on_remote, exit_process=True)
    return task, cfg
