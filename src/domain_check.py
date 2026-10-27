"""python -m src.domain_check [hydra overrides]

Standalone preflight (domain shift + label balance between reference and candidate split) as its own
ClearML Task, so a pipeline can run it before training. Same checks and gate semantics as the preflight
inside src.train: exit code 2 + Task marked failed when a `gate` check is violated.
Register the base task by running this with NO extra arguments (see docs/GUIDE.md)."""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .clearml_utils import Reporter, init_clearml
from .data import TileDataset, resolve_root
from .preflight import PreflightFailed, run_preflight


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    cfg.clearml.task_name = "domain-check"                      # stable name -> pipeline finds the base task
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    task, cfg = init_clearml(cfg)
    reporter = Reporter(task)

    root, sp = resolve_root(cfg), cfg.data.splits
    ref = TileDataset(root, sp[cfg.preflight.reference_split], cfg, limit=cfg.preflight.max_samples, seed=cfg.seed)
    cand = TileDataset(root, sp[cfg.preflight.candidate_split], cfg, limit=cfg.preflight.max_samples, seed=cfg.seed)
    try:
        run_preflight(cfg, reporter, ref, cand, out_dir)
    except PreflightFailed as e:
        if task:
            task.mark_failed(status_reason="preflight_gate", status_message=str(e)[:500])
        print(f"PREFLIGHT GATE FAILED: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
