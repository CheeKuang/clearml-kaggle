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
    """Rows: one per (tta, threshold) with the MEAN over classes in `iou`/`f1` and per-class `iou_<name>`/`f1_<name>`.
    best = the row with the best mean; best["class_thresholds"] = per-class threshold at the best tta."""
    s = cfg.sweep
    names = module.names
    thr = torch.as_tensor(np.arange(s.thresholds.start, s.thresholds.stop + 1e-9, s.thresholds.step),
                          dtype=torch.float32, device=device)
    module.to(device).eval()
    rows = []
    for tta in s.tta:
        counts = torch.zeros(len(thr), len(names), 3, device=device)      # tp, fp, fn per threshold and class
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = torch.stack([_inv(module(_fwd(x, k, f)).sigmoid(), k, f)
                                 for k, f in TTA_SETS[tta]]).mean(0)
            pred = (probs[None] > thr.view(-1, 1, 1, 1, 1)).float()       # (T,B,C,H,W)
            counts[:, :, 0] += (pred * y).sum((1, 3, 4))
            counts[:, :, 1] += (pred * (1 - y)).sum((1, 3, 4))
            counts[:, :, 2] += ((1 - pred) * y).sum((1, 3, 4))
        tp, fp, fn = counts.unbind(2)                                     # each (T, C)
        iou, f1 = tp / (tp + fp + fn + 1e-8), 2 * tp / (2 * tp + fp + fn + 1e-8)
        for k, t in enumerate(thr.tolist()):
            row = {"tta": tta, "threshold": round(t, 4), "iou": iou[k].mean().item(), "f1": f1[k].mean().item()}
            for c, n in enumerate(names):
                row[f"iou_{n}"], row[f"f1_{n}"] = iou[k, c].item(), f1[k, c].item()
            rows.append(row)
    df = pd.DataFrame(rows)
    best = df.loc[df[s.metric].idxmax()].to_dict()
    at_best_tta = df[df.tta == best["tta"]]
    best["class_thresholds"] = {n: float(at_best_tta.loc[at_best_tta[f"{s.metric}_{n}"].idxmax(), "threshold"])
                                for n in names}
    return df, best


def report_sweep(reporter, df, best, metric):
    reporter.table("sweep", "tta x threshold", df)
    for tta, g in df.groupby("tta"):
        for _, r in g.iterrows():
            reporter.scalar("sweep/iou", tta, r["iou"], int(round(r["threshold"] * 100)))
    reporter.single(f"sweep_best_{metric}", best[metric])
    reporter.single("sweep_best_threshold", best["threshold"])
    for n, t in best["class_thresholds"].items():                 # per-class best threshold (at the best tta)
        reporter.single(f"sweep_best_threshold_{n}", t)
