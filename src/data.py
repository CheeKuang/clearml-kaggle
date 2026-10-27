"""Tile dataset + helpers (images/masks as GeoTIFF or any rasterio-readable file).

Expected layout:  <root>/<split>/images/<name>  and  <root>/<split>/masks/<name>  (same file names).
Two mask conventions, chosen by the data config:
  * `data.classes` set (conf/data/loveda.yaml): masks hold class ids; every entry of `classes` becomes one binary
    output channel (pixel in that entry's id list -> 1). Multi-label: classes are independent.
  * `data.classes` absent (conf/data/buildings.yaml): masks MUST already be binary; a pixel counts where mask > 0.
    Raw LoveDA masks (values 0-7) would count almost every pixel as positive.
Masks are returned as (C, H, W), one channel per class (C = 1 in the binary case).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def resolve_root(cfg) -> Path:
    """Local data.root, or (when data.clearml_dataset_id is set) a cached download of that ClearML Dataset.
    The first call on a new agent downloads the whole dataset (~2.25 GiB, ~20 s on Kaggle)."""
    ds_id = cfg.data.get("clearml_dataset_id")
    if ds_id:
        from clearml import Dataset as ClearMLDataset

        return Path(ClearMLDataset.get(dataset_id=ds_id).get_local_copy())
    return Path(cfg.data.root)


def class_names(data_cfg) -> list[str]:
    """Names of the output channels (a single binary mask is called "building")."""
    return list(data_cfg.classes.keys()) if data_cfg.get("classes") else ["building"]


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
        if self.c.get("classes"):
            self._check_class_ids()

    def _check_class_ids(self, n: int = 8):
        """Fail loudly if none of the configured class ids occur: e.g. data.classes is set but the masks are the old
        binary 0/255 ones, which would silently give all-zero targets for every class."""
        picks = np.linspace(0, len(self.imgs) - 1, min(n, len(self.imgs))).astype(int)
        seen = set()
        for i in picks:
            seen.update(np.unique(_read(self.mask_dir / self.imgs[i].name)[0]).tolist())
        wanted = {i for ids in self.c.classes.values() for i in ids}
        if not seen & wanted:
            raise ValueError(f"None of the class ids {sorted(wanted)} occur in {len(picks)} sampled masks (values seen: "
                             f"{sorted(seen)[:12]}). Use data=buildings for binary masks, or a dataset made with "
                             f"`python -m src.prepare_data --mode labels`.")

    def __len__(self):
        return len(self.imgs)

    def image(self, i) -> np.ndarray:
        x = _read(self.imgs[i], self.c.in_channels).astype(np.float32) / float(self.c.divide_by)
        if self.c.mean is not None:
            m = np.asarray(self.c.mean, np.float32)[:, None, None]
            s = np.asarray(self.c.std, np.float32)[:, None, None]
            x = (x - m) / s
        return x

    @property
    def class_names(self) -> list[str]:
        return class_names(self.c)

    def mask(self, i) -> np.ndarray:
        """(C, H, W) float32, one binary channel per class."""
        m = _read(self.mask_dir / self.imgs[i].name)[0]
        if self.c.get("classes"):
            return np.stack([np.isin(m, list(ids)) for ids in self.c.classes.values()]).astype(np.float32)
        return (m > 0)[None].astype(np.float32)

    def class_ratios(self) -> np.ndarray:
        """(N tiles, C) fraction of positive pixels per tile and class."""
        return np.array([self.mask(i).mean((1, 2)) for i in range(len(self))])

    def __getitem__(self, i):
        x, y = torch.from_numpy(self.image(i)), torch.from_numpy(self.mask(i))
        if self.augment:
            if torch.rand(1) < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1) < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            k = int(torch.randint(0, 4, (1,)))
            x, y = torch.rot90(x, k, (-2, -1)), torch.rot90(y, k, (-2, -1))
        return x, y
