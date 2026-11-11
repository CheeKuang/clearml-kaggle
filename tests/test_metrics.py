"""IoU, F1, precision and recall from tp / fp / fn counts.

  python tests/test_metrics.py"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.module import confusion, iou_f1, precision_recall  # noqa: E402


def test_known_counts():
    counts = torch.tensor([[60.0, 20.0, 40.0]])                    # tp 60, fp 20, fn 40
    iou, f1 = iou_f1(counts)
    p, r = precision_recall(counts)
    assert abs(p.item() - 0.75) < 1e-6 and abs(r.item() - 0.6) < 1e-6          # 60/80 and 60/100
    assert abs(iou.item() - 0.5) < 1e-6                                           # 60/120
    assert abs(f1.item() - 2 * 0.75 * 0.6 / (0.75 + 0.6)) < 1e-6                  # F1 is the harmonic mean of the two


def test_from_masks_and_no_division_by_zero():
    y = torch.zeros(1, 1, 4, 4); y[..., :2, :] = 1                  # 8 true pixels
    pred = torch.zeros(1, 1, 4, 4); pred[..., :1, :] = 1; pred[..., 3, :] = 1     # 4 right, 4 wrong
    p, r = precision_recall(confusion(pred, y, (0, 2, 3)))
    assert abs(p.item() - 0.5) < 1e-6 and abs(r.item() - 0.5) < 1e-6
    empty = torch.zeros(1, 3)                                        # nothing predicted and nothing there: 0, not NaN
    assert torch.isfinite(torch.stack(precision_recall(empty))).all()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
