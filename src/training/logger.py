"""Lightning Logger that sends every metric to ClearML. Key "a/b" -> ClearML title "a", series "b"."""
from __future__ import annotations

from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only


class ClearMLLogger(Logger):
    def __init__(self, task):
        super().__init__()
        self._task, self._log = task, task.get_logger()

    @property
    def name(self) -> str:
        return self._task.name

    @property
    def version(self) -> str:
        return self._task.id

    @property
    def experiment(self):
        return self._log

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        pass                                    # owned by Task.connect / connect_configuration (Hydra is the authority)

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        it = int(metrics.get("epoch", step or 0))   # epoch as the x-axis: every series (and val/iou for HPO) shares it
        for key, value in metrics.items():
            if key != "epoch":
                title, _, series = key.partition("/")
                self._log.report_scalar(title if series else "metrics", series or key, float(value), it)

    @rank_zero_only
    def report_image(self, title, series, image, iteration):
        self._log.report_image(title, series, iteration=iteration, image=image)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self._task.flush(wait_for_uploads=True)
