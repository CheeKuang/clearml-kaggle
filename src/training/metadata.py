"""metadata.json: a frozen record of the run (config, versions, git, data and weight hashes, checkpoint, sweep,
preflight), plus the hashing helpers the Task's "Run" parameters use."""
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

from ..common.envinfo import git_state


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_state(state: dict) -> str:
    """Hash of a state_dict (names + tensor bytes): the same weights give the same hash on any machine."""
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode())
        h.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def data_fingerprint(root: Path, splits) -> dict:
    """Per split: number of tiles, one sha256 over every file (path + content, sorted) and the per-file hashes."""
    out = {}
    for split in dict.fromkeys(splits):
        files = sorted(p for p in (root / split).rglob("*") if p.is_file() and not p.name.startswith("."))
        per_file = {str(p.relative_to(root)): sha256_file(p) for p in files}
        out[split] = {"tiles": len(list((root / split / "images").glob("*.*"))),
                      "sha256": hashlib.sha256("".join(f"{k}\t{v}\n" for k, v in per_file.items()).encode()).hexdigest(),
                      "files": per_file}
    return out


def write_metadata(path: Path, cfg, task, ckpt_path, module, preflight, sweep_df, sweep_best, **extra) -> dict:
    """`extra` (data fingerprint, run facts, split.json, ...) is stored as top-level keys."""
    ckpt = Path(ckpt_path)
    meta = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "clearml": None if task is None else {
            "task_id": task.id, "project": task.get_project_name(), "name": task.name,
            "url": task.get_output_log_web_page(), "script": dict(task.get_script() or {})},
        "git": git_state(),
        "checkpoint": {"path": ckpt.name, "sha256": sha256_file(ckpt),
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
            "packages": dict(sorted((d.metadata["Name"], d.version) for d in md.distributions()))},
        **extra,
    }
    Path(path).write_text(json.dumps(meta, indent=2, default=str))
    return meta
