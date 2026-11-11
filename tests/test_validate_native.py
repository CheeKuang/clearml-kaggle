"""The native tile used to score the production ONNX: the crop at its own scale, top left, padded with 0.

  python tests/test_validate_native.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.export_onnx import to_native_tile  # noqa: E402


def test_native_tile_keeps_pixels_and_pads_with_zero():
    img = np.random.default_rng(0).integers(1, 255, (416, 300, 3), dtype=np.uint8)
    t = to_native_tile(img, 1280)
    assert t.shape == (1280, 1280, 3) and t.dtype == np.uint8
    assert (t[:416, :300] == img).all()                            # not resized, not moved
    assert t[416:].max() == 0 and t[:, 300:].max() == 0            # everything else is 0, like the server's edge padding


def test_native_tile_of_a_full_size_tile_is_the_tile():
    img = np.random.default_rng(1).integers(0, 255, (1280, 1280, 3), dtype=np.uint8)
    assert (to_native_tile(img, 1280) == img).all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
