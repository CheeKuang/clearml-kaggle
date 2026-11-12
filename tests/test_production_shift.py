"""Unit tests for src/gates/production_shift.py on tiny synthetic images (no OpenEarthMap, no ClearML).

  python tests/test_production_shift.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gates.preflight import knn_interleaving, stats_features  # noqa: E402
from src.gates.production_shift import ImageSet, opaque_windows, production_set, split_set  # noqa: E402


def _png(path: Path, rgb, alpha=255, size=(64, 64), noise=8, seed=0):
    rng = np.random.default_rng(seed)
    px = np.clip(np.array(rgb) + rng.integers(-noise, noise + 1, (*size, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(np.dstack([px, np.full(size, alpha, np.uint8)]), "RGBA").save(path)


def test_only_fully_opaque_windows_are_cut():
    rgba = np.full((128, 128, 4), 255, np.uint8)
    rgba[..., :3] = 120
    rgba[:64, :, 3] = 0                                    # the top half is transparent (outside the polygon)
    wins = opaque_windows(rgba, 64, per_image=10, rng=np.random.default_rng(0))
    assert len(wins) == 2                                  # only the two windows of the bottom row
    assert wins[0].shape == (3, 64, 64) and 0 <= wins[0].min() and wins[0].max() <= 1


def test_black_windows_are_skipped_and_the_cap_is_respected():
    rgba = np.zeros((128, 128, 4), np.uint8)
    rgba[..., 3] = 255                                     # opaque but black
    assert opaque_windows(rgba, 64, 10, np.random.default_rng(0)) == []
    rgba[..., :3] = 100
    assert len(opaque_windows(rgba, 64, per_image=3, rng=np.random.default_rng(0))) == 3


def test_a_different_looking_production_set_scores_high_and_a_matching_one_low():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for split in ("train", "val"):
            (d / "root" / split / "images").mkdir(parents=True)
            for i in range(40):
                _png(d / "root" / split / "images" / f"{i}.png", (90, 120, 80), seed=i)          # greenish tiles
        for name, colour in (("same", (90, 120, 80)), ("other", (200, 190, 170))):
            (d / name).mkdir()
            for i in range(10):
                _png(d / name / f"{i}.png", colour, size=(256, 256), seed=100 + i)
        ref = split_set(d / "root", "train", 40, 0, 255.0)
        assert len(ref) == 40 and ref.image(0).shape == (3, 64, 64)
        shifts = {}
        for name in ("same", "other"):
            prod, used, skipped = production_set(d / name, tile=64, n=40, per_image=4, seed=0)
            assert used == 10 and skipped == 0 and len(prod) == 40
            shifts[name] = knn_interleaving(stats_features(ref), stats_features(prod), 5)["shift_score"]
        assert shifts["other"] > 0.9, shifts                # clearly separated
        assert shifts["same"] < 0.3, shifts                 # mixed


def test_image_set_matches_what_the_feature_function_needs():
    ds = ImageSet([np.random.default_rng(i).random((3, 32, 32)).astype(np.float32) for i in range(4)])
    assert len(ds) == 4 and stats_features(ds).shape == (4, 12)         # 3 channels x (mean, std, p5, p95)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
