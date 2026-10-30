"""Tile dataset: <root>/<split>/images/<name> and <root>/<split>/masks/<name> (same names, any rasterio format).

Two mask conventions, chosen by the data config:
  * `data.classes` set (conf/data/<class>.yaml, loveda.yaml): masks hold class ids; each entry of `classes` becomes
    one binary channel (pixel id in that entry's list -> 1). Multi-label: the classes are independent.
  * `data.classes` null (conf/data/buildings.yaml): masks MUST already be binary; a pixel counts where mask > 0.
Masks are returned as (C, H, W), one channel per class.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def resolve_root(cfg) -> Path:
    """data.root, or a cached download of the ClearML Dataset data.clearml_dataset_id (~2.25 GiB, ~20 s on Kaggle)."""
    if ds_id := cfg.data.get("clearml_dataset_id"):
        from clearml import Dataset as ClearMLDataset

        return Path(ClearMLDataset.get(dataset_id=ds_id).get_local_copy())
    return Path(cfg.data.root)


def class_names(data_cfg) -> list[str]:
    """Names of the output channels (a single binary mask is called "building")."""
    return list(data_cfg.classes) if data_cfg.get("classes") else ["building"]


def _read(path, bands=None) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read()
    return arr[:bands] if bands else arr


class TileDataset(Dataset):
    def __init__(self, root: Path, split: str, cfg, augment: bool = False, limit: int | None = None, seed: int = 0):
        self.imgs = sorted((root / split / "images").glob("*.*"))
        if not self.imgs:
            raise FileNotFoundError(f"No images under {root / split / 'images'}")
        if limit and len(self.imgs) > limit:
            pick = np.random.default_rng(seed).choice(len(self.imgs), limit, replace=False)
            self.imgs = [self.imgs[i] for i in sorted(pick)]
        self.mask_dir, self.c, self.augment = root / split / "masks", cfg.data, augment
        self.class_names = class_names(self.c)
        if self.c.get("classes"):
            self._check_class_ids()

    def _check_class_ids(self, n: int = 8):
        """Fail loudly if none of the class ids occur (e.g. binary 0/255 masks with a class list): every target would
        silently be empty."""
        picks = np.linspace(0, len(self.imgs) - 1, min(n, len(self.imgs))).astype(int)
        seen = set().union(*(np.unique(_read(self.mask_dir / self.imgs[i].name)[0]).tolist() for i in picks))
        wanted = {i for ids in self.c.classes.values() for i in ids}
        if not seen & wanted:
            raise ValueError(f"None of the class ids {sorted(wanted)} occur in {len(picks)} sampled masks (values seen: "
                             f"{sorted(seen)[:12]}). Use data=buildings for binary masks, or a dataset made with "
                             f"`python -m src.data.prepare --mode labels`.")

    def __len__(self):
        return len(self.imgs)

    def image(self, i) -> np.ndarray:
        x = _read(self.imgs[i], self.c.in_channels).astype(np.float32) / float(self.c.divide_by)
        if self.c.mean is not None:
            x = (x - np.asarray(self.c.mean, np.float32)[:, None, None]) / np.asarray(self.c.std, np.float32)[:, None, None]
        return x

    def mask(self, i) -> np.ndarray:
        """(C, H, W) float32, one binary channel per class."""
        m = _read(self.mask_dir / self.imgs[i].name)[0]
        if self.c.get("classes"):
            return np.stack([np.isin(m, list(ids)) for ids in self.c.classes.values()]).astype(np.float32)
        return (m > 0)[None].astype(np.float32)

    def class_ratios(self) -> np.ndarray:
        """(N tiles, C) share of positive pixels per tile and class."""
        return np.array([self.mask(i).mean((1, 2)) for i in range(len(self))])

    def __getitem__(self, i):
        x, y = torch.from_numpy(self.image(i)), torch.from_numpy(self.mask(i))
        if self.augment:                        # random flips + 90-degree rotation, the same for image and mask
            if torch.rand(1) < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1) < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            k = int(torch.randint(0, 4, (1,)))
            x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
        return x, y
