"""After training: every test-time augmentation x threshold on the sweep split, to pick a threshold per class."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .module import confusion, iou_f1

TTA_SETS = {                                                    # (k rot90 steps, horizontal flip)
    "none": [(0, False)],
    "flip": [(0, False), (0, True), (2, True), (2, False)],     # id, h-flip, v-flip, h+v-flip
    "d4": [(k, f) for k in range(4) for f in (False, True)],    # the full dihedral group
}


def _fwd(x, k, f):
    return torch.rot90(x.flip(-1) if f else x, k, (-2, -1))


def _inv(x, k, f):
    x = torch.rot90(x, -k, (-2, -1))
    return x.flip(-1) if f else x


@torch.no_grad()
def run_sweep(module, loader, cfg, device) -> tuple[pd.DataFrame, dict]:
    """One row per (tta, threshold): mean `iou`/`f1` over classes plus `iou_<class>`/`f1_<class>`.
    best = the row with the best mean; best["class_thresholds"] = the best threshold per class at that tta."""
    s, names = cfg.sweep, module.names
    thr = torch.as_tensor(np.arange(s.thresholds.start, s.thresholds.stop + 1e-9, s.thresholds.step),
                          dtype=torch.float32, device=device)
    module.to(device).eval()
    rows = []
    for tta in s.tta:
        counts = torch.zeros(len(thr), len(names), 3, device=device)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = torch.stack([_inv(module(_fwd(x, k, f)).sigmoid(), k, f) for k, f in TTA_SETS[tta]]).mean(0)
            counts += confusion((probs[None] > thr.view(-1, 1, 1, 1, 1)).float(), y, (1, 3, 4))   # (T, C, 3)
        iou, f1 = iou_f1(counts)
        for k, t in enumerate(thr.tolist()):
            row = {"tta": tta, "threshold": round(t, 4), "iou": iou[k].mean().item(), "f1": f1[k].mean().item()}
            for c, n in enumerate(names):
                row[f"iou_{n}"], row[f"f1_{n}"] = iou[k, c].item(), f1[k, c].item()
            rows.append(row)
    df = pd.DataFrame(rows)
    best = df.loc[df[s.metric].idxmax()].to_dict()
    at_tta = df[df.tta == best["tta"]]
    best["class_thresholds"] = {n: float(at_tta.loc[at_tta[f"{s.metric}_{n}"].idxmax(), "threshold"]) for n in names}
    return df, best


def report_sweep(reporter, df, best, metric):
    reporter.table("sweep", "tta x threshold", df)
    for r in df.itertuples():
        reporter.scalar("sweep/iou", r.tta, r.iou, int(round(r.threshold * 100)))
    reporter.single(f"sweep_best_{metric}", best[metric])
    reporter.single("sweep_best_threshold", best["threshold"])
    for n, t in best["class_thresholds"].items():
        reporter.single(f"sweep_best_threshold_{n}", t)
