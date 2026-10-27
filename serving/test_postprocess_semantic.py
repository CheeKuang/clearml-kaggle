"""Geometry test: letterbox a known mask with the SERVER's own preprocess, run it through the semantic
post-processing and check that the original mask comes back.

  CHISEL_SERVER_DIR=/path/to/lectern-models/Chisel python serving/test_postprocess_semantic.py
(the server's libs/ are imported read-only for `letterbox`/`preprocess_image`)."""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
server = os.environ.get("CHISEL_SERVER_DIR", "/Users/khoocheekuang/Desktop/chisel - sitmulate /lectern-models/Chisel")
sys.path.insert(0, server)
from libs.preprocess import letterbox, preprocess_image          # noqa: E402  the server's own code

from postprocess_semantic import postprocess_semantic_masks       # noqa: E402

S = 1280
rng = np.random.default_rng(0)
worst = 1.0
for h, w in [(1280, 1280), (700, 900), (300, 300), (1000, 400), (1279, 1281), (64, 2000)]:
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    truth = np.zeros((h, w), np.uint8)
    truth[h // 5: h // 2, w // 4: (3 * w) // 4] = 255          # a rectangle
    truth[(2 * h) // 3: (5 * h) // 6, w // 10: w // 5] = 255      # and a second blob
    _, ratio, pad = preprocess_image(img, input_size=S)         # server letterbox parameters for this tile
    fake_probs, _, _ = letterbox(np.dstack([truth] * 3), new_shape=(S, S), colour=(0, 0, 0))
    probs = (fake_probs[:, :, 0].astype(np.float32) / 255.0)[None, None]   # model output in letterboxed space
    pred = postprocess_semantic_masks([probs], (h, w), ratio, pad, input_size=S, conf_threshold=0.5)
    inter, union = ((pred > 0) & (truth > 0)).sum(), ((pred > 0) | (truth > 0)).sum()
    iou = inter / union
    worst = min(worst, iou)
    print(f"tile {h:>4}x{w:<4} ratio {ratio:.3f}  output {pred.shape} {pred.dtype}  IoU with truth {iou:.4f}")
assert worst > 0.97, f"geometry off: worst IoU {worst:.4f}"
print("OK: semantic post-processing inverts the server's letterbox (worst IoU %.4f)" % worst)
