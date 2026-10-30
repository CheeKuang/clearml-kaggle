"""python -m src.training.train [hydra overrides]   (Lightning, ClearML as the only tracker)

A run: init_clearml -> optional preflight gate -> training -> TTA x threshold sweep -> ONNX + shrunk ONNX + sidecar
-> check both files through the server's path -> metadata.json -> artifacts. Files land in the Hydra output dir;
ClearML also captures best.ckpt as an output Model.
What the config cannot tell (data and weight hashes, tile counts, device, file hashes) goes to Hyperparameters -> Run;
the final config goes to Hyperparameters -> Resolved (both display only; edit "Overrides" to change a run).
Register the base task with NO extra arguments: they are baked into every clone (docs/GUIDE.md).
Exit code 2 = a gate failed.
"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ..common.clearml_utils import Reporter, flatten, init_clearml
from ..common.envinfo import git_state
from ..data.tiles import TileDataset, class_names, resolve_root
from ..export_onnx import export_model, shrink_and_record, threshold_from_rows, validate_onnx
from ..gates.preflight import gate_preflight
from .logger import ClearMLLogger
from .metadata import data_fingerprint, sha256_state, write_metadata
from .module import SegModule
from .sweep import report_sweep, run_sweep


def upload(task, **artifacts):
    if task:
        for name, obj in artifacts.items():
            task.upload_artifact(name, artifact_object=str(obj))


def export_for_server(cfg, task, reporter, module, df, root: Path, out_dir: Path) -> dict:
    """ONNX for the Chisel server: RGB/255 in, sigmoid probabilities out, threshold = best at TTA "none".
    Then the shrunk copy (the production file), provenance in the sidecar, and the IoU of both through the server's
    path. Returns the sidecar's "files" block ({} when export is off)."""
    e, names = cfg.export.onnx, class_names(cfg.data)
    single = len(names) == 1
    thr = threshold_from_rows(df.to_dict("records"), cfg.sweep.metric, names[0]) if df is not None and single else None
    onnx_path = out_dir / "onnx" / f"{names[0] if single else 'model'}.onnx"
    export_model(module.model, onnx_path, names, e.size, threshold=thr,
                 architecture=f"{cfg.model.arch}/{cfg.model.encoder_name}")
    git = git_state()
    source = {"task_id": task.id if task else None, "task_url": task.get_output_log_web_page() if task else None,
              "commit": git["commit"], "uncommitted_changes": any(not s.startswith("??") for s in git["status"])}
    files = shrink_and_record(onnx_path, e.shrink, source)
    side_path = onnx_path.with_suffix(".onnx.json")
    side = json.loads(side_path.read_text())
    reporter.params("Run", {f"{k}_sha256": v["sha256"] for k, v in side["files"].items()})

    if e.validate.enabled and single and cfg.data.mean is None:                  # the server only scales by 1/255
        ds = TileDataset(root, cfg.data.splits[cfg.sweep.split], cfg, limit=e.validate.tiles, seed=cfg.seed)
        ious = validate_onnx(files, ds, thr if thr is not None else 0.5, e.size)
        if "onnx_shrunk_iou" in ious:
            ious["onnx_shrunk_iou_drop"] = ious["onnx_iou"] - ious["onnx_shrunk_iou"]
        for k, v in ious.items():
            reporter.single(k, v)
        print(f"[export] server-path IoU on {len(ds)} {cfg.sweep.split} tiles: {ious}", flush=True)
        side["server_path_iou"] = {"tiles": len(ds), **ious}
        side_path.write_text(json.dumps(side, indent=2))
    upload(task, **files, onnx_meta=side_path)
    return side["files"]


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
    data = data_fingerprint(root, cfg.data.splits.values())                  # which exact files (per-file in metadata)
    split = json.loads((root / "split.json").read_text()) if (root / "split.json").exists() else None

    dev = "cuda" if torch.cuda.is_available() else "cpu"          # AMP only on CUDA (Mac MPS runs fp32)
    kw = dict(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, pin_memory=dev == "cuda")
    train_dl = DataLoader(TileDataset(root, cfg.data.splits.train, cfg, augment=True), shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(TileDataset(root, cfg.data.splits.val, cfg), **kw)

    module = SegModule(cfg)
    precision = "16-mixed" if cfg.train.amp and dev == "cuda" else 32
    ckpt = ModelCheckpoint(dirpath=out_dir / "checkpoints", filename="best", monitor="val/iou", mode="max", save_top_k=1)
    trainer = L.Trainer(max_epochs=cfg.train.epochs, accelerator="auto", devices=1, precision=precision,
                        deterministic="warn" if cfg.train.deterministic else None,
                        logger=ClearMLLogger(task) if task else False,
                        callbacks=[ckpt, EarlyStopping("val/iou", mode="max", patience=cfg.train.patience)],
                        default_root_dir=out_dir, enable_progress_bar=False, num_sanity_val_steps=0)
    run = {**{f"{s}_tiles": d["tiles"] for s, d in data.items()}, **{f"{s}_sha256": d["sha256"] for s, d in data.items()},
           **({f"split.{k}": v for k, v in flatten(split).items()} if split else
              {"split": "not recorded (dataset made before split.json); see the per-split sha256"}),
           "steps_per_epoch": len(train_dl), "model_params": sum(p.numel() for p in module.parameters()),
           "encoder_init_sha256": sha256_state(module.model.encoder.state_dict()),     # the pretrained starting point
           "model_init_sha256": sha256_state(module.model.state_dict()),
           "scheduler": f"CosineAnnealingLR(T_max={cfg.train.epochs})",
           "device": str(trainer.strategy.root_device), "precision": str(precision),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    reporter.params("Run", run)
    print(f"[run] {json.dumps(run)}", flush=True)
    trainer.fit(module, train_dl, val_dl)
    module.load_state_dict(torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)["state_dict"])
    reporter.single("best_val_iou", module.best_iou)

    df = best = None
    if cfg.sweep.enabled:
        sweep_dl = DataLoader(TileDataset(root, cfg.data.splits[cfg.sweep.split], cfg),
                              batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
        df, best = run_sweep(module, sweep_dl, cfg, dev)
        report_sweep(reporter, df, best, cfg.sweep.metric)

    onnx_files = export_for_server(cfg, task, reporter, module, df, root, out_dir) if cfg.export.onnx.enabled else {}

    # metadata.json needs the sweep. Without it, best_checkpoint/metadata are not uploaded (ClearML still keeps the Model).
    if df is not None:
        meta_path = out_dir / cfg.finalize.metadata_file
        write_metadata(meta_path, cfg, task, ckpt.best_model_path, module, preflight, df, best,
                       run=run, data=data, split=split, onnx=onnx_files)
        upload(task, best_checkpoint=ckpt.best_model_path, metadata=meta_path)

    # Pipeline steps have been seen to end as "completed" rather than "published"; cause not investigated.
    if task and cfg.finalize.publish_on_completion:
        task.publish_on_completion(True)


if __name__ == "__main__":
    main()
