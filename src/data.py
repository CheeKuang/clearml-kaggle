"""Tile dataset + helpers (images/masks as GeoTIFF or any rasterio-readable file)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def resolve_root(cfg) -> Path:
    ds_id = cfg.data.get("clearml_dataset_id")
    if ds_id:
        from clearml import Dataset as ClearMLDataset

        return Path(ClearMLDataset.get(dataset_id=ds_id).get_local_copy())
    return Path(cfg.data.root)


def _read(path, bands=None) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read()
    return arr[:bands] if bands else arr


class TileDataset(Dataset):
    def __init__(self, root: Path, split: str, cfg, augment: bool = False, limit: int | None = None, seed: int = 0):
        self.imgs = sorted((root / split / "images").glob("*.*"))
        if limit and len(self.imgs) > limit:
            rng = np.random.default_rng(seed)
            self.imgs = [self.imgs[i] for i in sorted(rng.choice(len(self.imgs), limit, replace=False))]
        self.mask_dir = root / split / "masks"
        self.c = cfg.data
        self.augment = augment
        if not self.imgs:
            raise FileNotFoundError(f"No images under {root / split / 'images'}")

    def __len__(self):
        return len(self.imgs)

    def image(self, i) -> np.ndarray:
        x = _read(self.imgs[i], self.c.in_channels).astype(np.float32) / float(self.c.divide_by)
        if self.c.mean is not None:
            m = np.asarray(self.c.mean, np.float32)[:, None, None]
            s = np.asarray(self.c.std, np.float32)[:, None, None]
            x = (x - m) / s
        return x

    def mask(self, i) -> np.ndarray:
        return (_read(self.mask_dir / self.imgs[i].name)[0] > 0).astype(np.float32)

    def building_ratios(self) -> np.ndarray:
        return np.array([self.mask(i).mean() for i in range(len(self))])

    def __getitem__(self, i):
        x, y = torch.from_numpy(self.image(i)), torch.from_numpy(self.mask(i))[None]
        if self.augment:
            if torch.rand(1) < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1) < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            k = int(torch.randint(0, 4, (1,)))
            x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
        return x, y
