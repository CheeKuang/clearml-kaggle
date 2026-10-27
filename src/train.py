"""python -m src.train [hydra overrides]   (Lightning + ClearML as the only tracker)"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .clearml_logger import ClearMLLogger
from .clearml_utils import Reporter, init_clearml
from .data import TileDataset, resolve_root
from .lightning_module import SegModule
from .metadata import build_metadata, write_metadata
from .preflight import PreflightFailed, run_preflight
from .sweep import report_sweep, run_sweep


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    task, cfg = init_clearml(cfg)               # Task.init + Task.connect (+ execute_remotely if enabled)
    reporter = Reporter(task)
    L.seed_everything(cfg.seed, workers=True)

    # reproducibility: final resolved config + Hydra's own snapshot (original CLI overrides)
    resolved = out_dir / "resolved_config.yaml"
    OmegaConf.save(cfg, resolved, resolve=True)
    if task:
        task.upload_artifact("resolved_config", artifact_object=str(resolved))
        task.upload_artifact("hydra_snapshot", artifact_object=str(out_dir / ".hydra"))

    root = resolve_root(cfg)
    preflight = {}
    try:
        if cfg.preflight.enabled:
            sp = cfg.data.splits
            ref = TileDataset(root, sp[cfg.preflight.reference_split], cfg,
                              limit=cfg.preflight.max_samples, seed=cfg.seed)
            cand = TileDataset(root, sp[cfg.preflight.candidate_split], cfg,
                               limit=cfg.preflight.max_samples, seed=cfg.seed)
            preflight = run_preflight(cfg, reporter, ref, cand, out_dir)
    except PreflightFailed as e:
        if task:
            task.mark_failed(status_reason="preflight_gate", status_message=str(e)[:500])
        print(f"PREFLIGHT GATE FAILED: {e}")
        sys.exit(2)

    # ------------------------------------------------------------------ training (Lightning)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = TileDataset(root, cfg.data.splits.train, cfg, augment=True)
    va = TileDataset(root, cfg.data.splits.val, cfg)
    kw = dict(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, pin_memory=dev == "cuda")
    tl, vl = DataLoader(tr, shuffle=True, drop_last=True, **kw), DataLoader(va, **kw)

    module = SegModule(cfg)
    ckpt_cb = ModelCheckpoint(dirpath=out_dir / "checkpoints", filename="best", monitor="val/iou",
                              mode="max", save_top_k=1)
    trainer = L.Trainer(
        max_epochs=cfg.train.epochs, accelerator="auto", devices=1,
        precision="16-mixed" if cfg.train.amp and dev == "cuda" else 32,
        logger=ClearMLLogger(task) if task else False,
        callbacks=[ckpt_cb, EarlyStopping("val/iou", mode="max", patience=cfg.train.patience)],
        default_root_dir=out_dir, enable_progress_bar=False, num_sanity_val_steps=0,
    )
    trainer.fit(module, tl, vl)

    best_path = ckpt_cb.best_model_path
    module.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["state_dict"])
    reporter.single("best_val_iou", module.best_iou)

    # ------------------------------------------------------------------ TTA x threshold sweep -> metadata.json
    if cfg.sweep.enabled:
        sweep_loader = DataLoader(TileDataset(root, cfg.data.splits[cfg.sweep.split], cfg),
                                  batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
        df, best = run_sweep(module, sweep_loader, cfg, dev)
        report_sweep(reporter, df, best, cfg.sweep.metric)
        meta = build_metadata(cfg, task, best_path, module, preflight, df, best)
        meta_path = write_metadata(meta, out_dir, cfg.finalize.metadata_file)
        if task:
            task.upload_artifact("best_checkpoint", artifact_object=best_path)
            task.upload_artifact("metadata", artifact_object=str(meta_path))

    if task and cfg.finalize.publish_on_completion:
        task.publish_on_completion(True)        # Task becomes read-only (Published) once it completes


if __name__ == "__main__":
    main()
