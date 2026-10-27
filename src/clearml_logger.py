"""Lightning Logger that forwards everything to ClearML (replaces TensorBoard/MLflow loggers).

Written against Lightning 2.x's Logger interface (name, version, experiment, log_hyperparams, log_metrics,
finalize) -- verify on first run. Metric keys "a/b" become ClearML title "a", series "b".
"""
from __future__ import annotations

from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only


class ClearMLLogger(Logger):
    def __init__(self, task):
        super().__init__()
        self._task = task
        self._clearml_logger = task.get_logger()

    @property
    def name(self) -> str:
        return self._task.name

    @property
    def version(self) -> str:
        return self._task.id

    @property
    def experiment(self):
        return self._clearml_logger

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        # Hyperparameters are already owned by Task.connect / connect_configuration (Hydra stays the
        # single authority), so nothing is written here.
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        # Lightning adds an "epoch" key; use it as the x-axis so every series (train/val/preflight-sweep)
        # shares one axis and the HPO objective (val/iou) is indexed by epoch.
        it = int(metrics.get("epoch", step or 0))
        for key, value in metrics.items():
            if key == "epoch":
                continue
            title, _, series = key.partition("/")
            if not series:
                title, series = "metrics", key
            self._clearml_logger.report_scalar(title, series, float(value), it)

    # extras used by the LightningModule when this logger is active
    @rank_zero_only
    def report_image(self, title, series, image, iteration):
        self._clearml_logger.report_image(title, series, iteration=iteration, image=image)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self._task.flush(wait_for_uploads=True)
