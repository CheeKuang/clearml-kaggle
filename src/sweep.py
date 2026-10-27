"""Post-training TTA x threshold sweep on the best checkpoint."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

# (k rot90 steps, horizontal flip) pairs
TTA_SETS = {
    "none": [(0, False)],
    "flip": [(0, False), (0, True), (2, True), (2, False)],               # id, h-flip, v-flip, h+v-flip
    "d4": [(k, f) for k in range(4) for f in (False, True)],              # full dihedral group
}


def _fwd(x, k, f):
    x = x.flip(-1) if f else x
    return torch.rot90(x, k, (-2, -1))


def _inv(x, k, f):
    x = torch.rot90(x, -k, (-2, -1))
    return x.flip(-1) if f else x


@torch.no_grad()
def run_sweep(module, loader, cfg, device) -> tuple[pd.DataFrame, dict]:
    s = cfg.sweep
    thr = torch.as_tensor(np.arange(s.thresholds.start, s.thresholds.stop + 1e-9, s.thresholds.step),
                          dtype=torch.float32, device=device)
    module.to(device).eval()
    rows = []
    for tta in s.tta:
        counts = torch.zeros(len(thr), 3, device=device)                  # tp, fp, fn per threshold
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = torch.stack([_inv(module(_fwd(x, k, f)).sigmoid(), k, f)
                                 for k, f in TTA_SETS[tta]]).mean(0)
            pred = (probs[None] > thr.view(-1, 1, 1, 1, 1)).float()       # (T,B,1,H,W)
            counts[:, 0] += (pred * y).sum((1, 2, 3, 4))
            counts[:, 1] += (pred * (1 - y)).sum((1, 2, 3, 4))
            counts[:, 2] += ((1 - pred) * y).sum((1, 2, 3, 4))
        tp, fp, fn = counts.unbind(1)
        iou, f1 = tp / (tp + fp + fn + 1e-8), 2 * tp / (2 * tp + fp + fn + 1e-8)
        for t, i, f in zip(thr.tolist(), iou.tolist(), f1.tolist()):
            rows.append({"tta": tta, "threshold": round(t, 4), "iou": i, "f1": f})
    df = pd.DataFrame(rows)
    best = df.loc[df[s.metric].idxmax()].to_dict()
    return df, best


def report_sweep(reporter, df, best, metric):
    reporter.table("sweep", "tta x threshold", df)
    for tta, g in df.groupby("tta"):
        for _, r in g.iterrows():
            reporter.scalar("sweep/iou", tta, r["iou"], int(round(r["threshold"] * 100)))
    reporter.single(f"sweep_best_{metric}", best[metric])
    reporter.single("sweep_best_threshold", best["threshold"])
