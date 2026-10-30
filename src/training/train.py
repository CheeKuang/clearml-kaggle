"""python -m src.training.train [hydra overrides]   (Lightning, ClearML as the only tracker)

A run: init_clearml -> optional preflight gate -> training -> TTA x threshold sweep -> metadata.json -> ONNX + sidecar
-> artifacts. Files land in the Hydra output dir; ClearML also captures best.ckpt as an output Model.
Register the base task with NO extra arguments: they are baked into every clone (docs/GUIDE.md).
Exit code 2 = a gate failed.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ..common.clearml_utils import Reporter, init_clearml
from ..data.tiles import TileDataset, class_names, resolve_root
from ..export_onnx import export_model, threshold_from_rows
from ..gates.preflight import gate_preflight
from .logger import ClearMLLogger
from .metadata import write_metadata
from .module import SegModule
from .sweep import report_sweep, run_sweep


def upload(task, **artifacts):
    if task:
        for name, obj in artifacts.items():
            task.upload_artifact(name, artifact_object=str(obj))


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    task, cfg = init_clearml(cfg)
    reporter = Reporter(task)
    L.seed_everything(cfg.seed, workers=True)
    OmegaConf.save(cfg, out_dir / "resolved_config.yaml", resolve=True)
    upload(task, resolved_config=out_dir / "resolved_config.yaml", hydra_snapshot=out_dir / ".hydra")

    root = resolve_root(cfg)
    preflight = gate_preflight(cfg, task, reporter, root, out_dir) if cfg.preflight.enabled else {}

    dev = "cuda" if torch.cuda.is_available() else "cpu"          # AMP only on CUDA (Mac MPS runs fp32)
    kw = dict(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, pin_memory=dev == "cuda")
    train_dl = DataLoader(TileDataset(root, cfg.data.splits.train, cfg, augment=True), shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(TileDataset(root, cfg.data.splits.val, cfg), **kw)

    module = SegModule(cfg)
    ckpt = ModelCheckpoint(dirpath=out_dir / "checkpoints", filename="best", monitor="val/iou", mode="max", save_top_k=1)
    L.Trainer(max_epochs=cfg.train.epochs, accelerator="auto", devices=1,
              precision="16-mixed" if cfg.train.amp and dev == "cuda" else 32,
              logger=ClearMLLogger(task) if task else False,
              callbacks=[ckpt, EarlyStopping("val/iou", mode="max", patience=cfg.train.patience)],
              default_root_dir=out_dir, enable_progress_bar=False, num_sanity_val_steps=0).fit(module, train_dl, val_dl)
    module.load_state_dict(torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)["state_dict"])
    reporter.single("best_val_iou", module.best_iou)

    # sweep -> metadata.json. Without the sweep, best_checkpoint/metadata are not uploaded (ClearML still keeps the Model).
    df = None
    if cfg.sweep.enabled:
        sweep_dl = DataLoader(TileDataset(root, cfg.data.splits[cfg.sweep.split], cfg),
                              batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
        df, best = run_sweep(module, sweep_dl, cfg, dev)
        report_sweep(reporter, df, best, cfg.sweep.metric)
        meta_path = out_dir / cfg.finalize.metadata_file
        write_metadata(meta_path, cfg, task, ckpt.best_model_path, module, preflight, df, best)
        upload(task, best_checkpoint=ckpt.best_model_path, metadata=meta_path)

    # ONNX for the Chisel server: RGB/255 in, sigmoid probabilities out, threshold = best at TTA "none"
    if cfg.export.onnx.enabled:
        names = class_names(cfg.data)
        single = len(names) == 1
        thr = threshold_from_rows(df.to_dict("records"), cfg.sweep.metric, names[0]) if df is not None and single else None
        onnx_path = out_dir / "onnx" / f"{names[0] if single else 'model'}.onnx"
        export_model(module.model, onnx_path, names, cfg.export.onnx.size, threshold=thr,
                     architecture=f"{cfg.model.arch}/{cfg.model.encoder_name}")
        upload(task, onnx=onnx_path, onnx_meta=onnx_path.with_suffix(".onnx.json"))

    # Pipeline steps have been seen to end as "completed" rather than "published"; cause not investigated.
    if task and cfg.finalize.publish_on_completion:
        task.publish_on_completion(True)


if __name__ == "__main__":
    main()
