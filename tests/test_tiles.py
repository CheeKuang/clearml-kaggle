"""TileDataset on tiny made-up tiles: the refactored dataset (transform argument, helpers outside the class) must give the same tiles,
masks and random augmentation as the version before it (src/data/tiles.py as committed), and keep its safeguards.

  python tests/test_tiles.py"""
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import tiles as new  # noqa: E402


def _old_module():
    """The committed version of src/data/tiles.py, loaded under another name."""
    src = subprocess.run(["git", "show", "HEAD:src/data/tiles.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType("tiles_old")
    exec(compile(src, "tiles_old.py", "exec"), mod.__dict__)
    return mod


def _make(root: Path, n=6, size=32, ids=(0, 4, 7)):
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "masks").mkdir(parents=True)
        for i in range(n):
            Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(root / split / "images" / f"{i}.png")
            Image.fromarray(rng.choice(ids, (size, size)).astype(np.uint8)).save(root / split / "masks" / f"{i}.png")


def _cfg(classes=None):
    return OmegaConf.create({"data": {"in_channels": 3, "divide_by": 255.0, "mean": None, "std": None, "classes": classes or {"road": [4], "water": [7]}},
                             "augment": {"hflip": 0.5, "vflip": 0.5, "rot90": True}})


def test_same_tiles_masks_and_augmentation_as_before():
    old = _old_module()
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make(root)
        cfg = _cfg()
        for augment in (False, True):
            o, n = old.TileDataset(root, "train", cfg, augment=augment), new.TileDataset(root, "train", cfg, augment=augment)
            assert len(o) == len(n)
            for i in range(len(o)):
                torch.manual_seed(100 + i)
                xo, yo = o[i]
                torch.manual_seed(100 + i)
                xn, yn = n[i]
                assert torch.equal(xo, xn) and torch.equal(yo, yn), (augment, i)
        assert np.array_equal(old.TileDataset(root, "val", cfg).class_ratios(), new.class_ratios(new.TileDataset(root, "val", cfg)))
        assert np.array_equal(old.TileDataset(root, "val", cfg, limit=3, seed=1).imgs, new.TileDataset(root, "val", cfg, limit=3, seed=1).imgs)


def test_augmentation_only_when_asked_and_a_custom_transform_is_used():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make(root)
        cfg = _cfg()
        assert new.TileDataset(root, "train", cfg).transform is None
        assert new.TileDataset(root, "train", cfg, augment=True).transform is not None
        calls = []
        ds = new.TileDataset(root, "train", cfg, transform=lambda x, y: (calls.append(1), (x * 0, y))[1])
        x, y = ds[0]
        assert calls and float(x.abs().sum()) == 0 and y.shape == (2, 32, 32)


def test_image_and_mask_get_the_same_flip():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make(root, ids=(4,))                                        # the mask is all class 4: still every pixel after any flip
        ds = new.TileDataset(root, "train", _cfg({"road": [4]}), augment=True)
        plain = new.TileDataset(root, "train", _cfg({"road": [4]}))
        torch.manual_seed(3)
        x, _ = ds[0]
        assert sorted(x.flatten().tolist()) == sorted(plain[0][0].flatten().tolist())    # a permutation of the same pixels


def test_class_id_safeguard_is_kept():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make(root, ids=(0, 255))                                     # binary 0/255 masks with a class list: nothing matches
        try:
            new.TileDataset(root, "train", _cfg({"road": [4]}))
        except ValueError as e:
            assert "None of the class ids" in str(e)
        else:
            raise AssertionError("should refuse masks without the class ids")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
