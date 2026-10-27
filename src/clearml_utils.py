"""ClearML <-> Hydra glue: Task.init, Task.connect (editable hyperparameters), recompose, Logger wrapper."""
from __future__ import annotations

from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

try:
    from clearml import Task
except ImportError:  # allows running without clearml installed
    Task = None

# Packages imported lazily (via Hydra _target_ strings / inside functions) are invisible to
# ClearML's static import analysis, so the agent would not install them -> add explicitly.
FORCED_REQUIREMENTS = ["lightning", "pandas", "segmentation_models_pytorch", "torchgeo", "hydra-core", "omegaconf", "rasterio",
                       "scikit-learn", "scipy"]


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
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None or v in ("None", ""):      # ClearML stringifies None (as "None" or "")
        return "null"
    return str(v)


def tunable_state(cfg: DictConfig, hc: HydraConfig) -> dict[str, Any]:
    """Current value of each tunable key. Config-group keys use the chosen option name."""
    choices = hc.runtime.choices
    state = {}
    for key in cfg.clearml.tunable:
        state[key] = choices[key] if key in choices else OmegaConf.select(cfg, key)
    return state


def recompose(hc: HydraConfig, tunables: dict[str, Any]) -> DictConfig:
    """Rebuild the config = original CLI overrides (minus tunable keys) + tunable values."""
    keys = set(tunables)
    cli = [o for o in hc.overrides.task if o.lstrip("+~").split("=")[0] not in keys]
    overrides = cli + [f"{k}={_fmt(v)}" for k, v in tunables.items()]
    return hydra.compose(config_name=hc.job.config_name, overrides=overrides)


def init_clearml(cfg: DictConfig) -> tuple[Any | None, DictConfig]:
    """Task.init -> Task.connect (editable hyperparameters) -> recompose with UI/HPO overrides."""
    hc = HydraConfig.get()
    if not cfg.clearml.enabled or Task is None:
        return None, cfg
    c = cfg.clearml

    # must run before Task.init
    for pkg in FORCED_REQUIREMENTS:
        Task.add_requirements(pkg)

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
    state = tunable_state(cfg, hc)
    tunables = dict(task.connect(state, name="Overrides"))
    if {k: _fmt(v) for k, v in tunables.items()} != {k: _fmt(v) for k, v in state.items()}:
        cfg = recompose(hc, tunables)
        print(f"[clearml] config recomposed with remote overrides: {tunables}")

    # 3) snapshot of the *final* resolved config (editable HOCON in the UI; local values win on remote)
    task.connect_configuration(OmegaConf.to_container(cfg, resolve=True), name="resolved_config",
                               description="Final Hydra config after overrides",
                               ignore_remote_overrides=True)
    task.connect_label_enumeration({"background": 0, "building": 1})
    task.add_tags([str(cfg.model.arch), str(cfg.model.encoder_name), str(cfg.pretrain.source)])

    # 4) optional: ship this run to an agent queue (e.g. a clearml-agent running inside a Kaggle notebook)
    if c.execute_remotely:
        task.execute_remotely(queue_name=c.queue, clone=c.clone_on_remote, exit_process=True)
    return task, cfg
