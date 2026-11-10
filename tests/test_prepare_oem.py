"""Unit tests for src/data/prepare_oem.py (pure functions, no OpenEarthMap needed).

  python tests/test_prepare_oem.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.prepare_oem import choose_val_regions, crop_starts, gsd_from_transform, region_of, resample  # noqa: E402


def test_gsd_projected_uses_pixel_width():
    assert gsd_from_transform(0.3, -0.3, geographic=False) == 0.3


def test_gsd_geographic_uses_north_south_degrees():
    assert abs(gsd_from_transform(2.7e-6, -2.7e-6, geographic=True) - 0.30) < 0.01      # as in OEM paris / khartoum
    assert abs(gsd_from_transform(4.49e-6, -4.49e-6, geographic=True) - 0.50) < 0.01    # as in OEM rio


def test_crop_starts_cover_without_gaps():
    assert crop_starts(511, 512) == []
    assert crop_starts(512, 512) == [0]
    assert crop_starts(1024, 512) == [0, 512]
    assert crop_starts(833, 512) == [0, 321]                    # evenly spaced, overlapping
    for n in (512, 600, 833, 1000, 1500):                       # no leftover sliver: every pixel is in some crop
        covered = np.zeros(n, bool)
        for s in crop_starts(n, 512):
            covered[s:s + 512] = True
        assert covered.all()


def test_crop_starts_drops_a_sliver_instead_of_repeating_a_neighbour():
    assert crop_starts(833, 416) == [0, 417]                    # was [0, 416, 417]: two crops one pixel apart
    assert crop_starts(417, 416) == [0]                         # was [0, 1]
    assert crop_starts(450, 416) == [0]                         # 34 px left over (under 10%): dropped
    assert crop_starts(470, 416) == [0, 54]                     # 54 px left over: kept, with an overlapping crop


def test_resample_shapes_and_mask_ids():
    img = np.random.default_rng(0).integers(0, 255, (1024, 1024, 3), dtype=np.uint8)
    assert resample(img, 0.5, nearest=False).shape == (512, 512, 3)
    assert resample(img, 0.8333, nearest=False).shape == (853, 853, 3)
    mask = np.random.default_rng(1).integers(0, 9, (1000, 1000), dtype=np.uint8)
    out = resample(mask, 0.6 / 0.5 * 0.5, nearest=True)         # any factor
    assert out.dtype == np.uint8 and set(np.unique(out)) <= set(range(9))        # nearest never invents an id
    assert resample(mask, 1.0, nearest=True) is mask


def test_region_split_is_by_whole_region():
    names = [f"{r}_{i}" for r in ("aachen", "accra", "tokyo", "rio") for i in range(5)]
    val = choose_val_regions(sorted({region_of(n) for n in names}), 0.25, seed=42)
    assert val == choose_val_regions(sorted({region_of(n) for n in names}), 0.25, seed=42)      # reproducible
    assert len(val) == 1 and region_of("rio_12") == "rio" and region_of("pointenoire_3") == "pointenoire"
    assert region_of("port_a_piment_7") == "port_a_piment"       # region names may contain underscores


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
