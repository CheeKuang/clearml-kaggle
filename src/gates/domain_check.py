"""python -m src.gates.domain_check [hydra overrides]

The data gate (src/gates/preflight.py) as its own ClearML Task, so a pipeline runs it before training.
Exit code 2 + Task failed when a `gate` check is violated. Register the base task with NO extra arguments.
"""
from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from ..common.clearml_utils import Reporter, init_clearml
from ..data.tiles import resolve_root
from .preflight import gate_preflight


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    cfg.clearml.task_name = "domain-check"                      # stable name: the pipeline finds the base task by it
    task, cfg = init_clearml(cfg)
    gate_preflight(cfg, task, Reporter(task), resolve_root(cfg), Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
