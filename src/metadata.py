"""metadata.json: frozen record of the run (config, versions, best checkpoint, sweep, preflight)."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_metadata(cfg, task, ckpt_path, module, preflight, sweep_df, sweep_best) -> dict:
    ckpt = Path(ckpt_path)
    return {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "clearml": None if task is None else {
            "task_id": task.id, "project": task.get_project_name(), "name": task.name,
            "url": task.get_output_log_web_page(), "script": dict(task.get_script() or {}),
        },
        "checkpoint": {"path": ckpt.name, "sha256": _sha256(ckpt),
                       "best_epoch": module.best_epoch, "best_val_iou": module.best_iou},
        "inference_defaults": {"tta": sweep_best["tta"], "threshold": sweep_best["threshold"],
                               "classes": list(module.names), "class_thresholds": sweep_best["class_thresholds"]},
        "sweep": {"best": sweep_best, "results": sweep_df.to_dict(orient="records")},
        "preflight": preflight,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "hydra_overrides": list(HydraConfig.get().overrides.task),
        "versions": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": dict(sorted((d.metadata["Name"], d.version) for d in md.distributions())),
        },
    }


def write_metadata(meta: dict, out_dir: Path, name: str) -> Path:
    p = out_dir / name
    p.write_text(json.dumps(meta, indent=2, default=str))
    return p
