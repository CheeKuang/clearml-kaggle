"""Data gate: domain shift and label balance between two splits. Each check is `gate` or `diagnostic`; every score
is logged either way. Used by src.gates.domain_check and (optionally) at the start of src.training.train."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ..common.clearml_utils import fail_gate
from ..data.tiles import TileDataset


class PreflightFailed(RuntimeError):
    pass


def knn_interleaving(fa: np.ndarray, fb: np.ndarray, k: int) -> dict:
    """How often a sample's k nearest neighbours come from the OTHER split, vs. what a perfect mix would give.
    shift_score: 0 = interleaved, 1 = fully separated."""
    X = StandardScaler().fit_transform(np.vstack([fa, fb]))
    split = np.r_[np.zeros(len(fa), int), np.ones(len(fb), int)]          # 0 = reference, 1 = candidate
    k = min(k, len(X) - 1)
    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]  # drop itself
    other = (split[neighbours] != split[:, None]).mean(1)                  # share of neighbours from the other split
    expected = np.where(split == 0, len(fb), len(fa)) / (len(X) - 1)       # that share if the splits were mixed

    def shift(rows) -> float:
        return float(np.clip(1 - other[rows].mean() / expected[rows].mean(), 0, 1))

    return {"other_domain_neighbour_frac": float(other.mean()), "expected_if_mixed": float(expected.mean()),
            "shift_score": shift(slice(None)), "shift_score_ref": shift(split == 0),
            "shift_score_cand": shift(split == 1), "k": int(k)}


def label_balance(ra: np.ndarray, rb: np.ndarray) -> dict:
    ma, mb = float(ra.mean()), float(rb.mean())
    return {"wasserstein": float(wasserstein_distance(ra, rb)), "mean_ratio_ref": ma, "mean_ratio_cand": mb,
            "ratio_gap_rel": float(abs(ma - mb) / max(ma, mb, 1e-12)),
            "empty_tile_frac_ref": float((ra == 0).mean()), "empty_tile_frac_cand": float((rb == 0).mean())}


def stats_features(ds) -> np.ndarray:
    """Per-channel mean, std, 5th and 95th percentile of every image."""
    feats = []
    for i in range(len(ds)):
        x = ds.image(i).reshape(ds.c.in_channels, -1)
        feats.append(np.r_[x.mean(1), x.std(1), np.percentile(x, 5, 1), np.percentile(x, 95, 1)])
    return np.asarray(feats)


def encoder_features(ds, cfg) -> np.ndarray:
    """Pooled features of the deepest encoder stage."""
    import torch

    from ..model_factory import build_model

    enc = build_model(cfg).encoder.eval()
    with torch.no_grad():
        return np.asarray([enc(torch.from_numpy(ds.image(i))[None])[-1].mean((2, 3)).squeeze(0).numpy()
                           for i in range(len(ds))])


def run_preflight(cfg, reporter, ref_ds, cand_ds, out_dir: Path) -> dict:
    p = cfg.preflight
    results, failures = {}, []

    def finish(name, check_cfg, scores, violations):
        passed = not violations
        for key, val in scores.items():
            reporter.scalar(f"preflight/{name}", key, val, 0)
            reporter.single(f"preflight/{name}/{key}", val)
        reporter.scalar(f"preflight/{name}", "passed", float(passed), 0)
        results[name] = {"mode": check_cfg.mode, "passed": passed, "scores": scores, "violations": violations}
        print(f"[preflight] {name}: mode={check_cfg.mode} passed={passed} {violations or ''}")
        if not passed and check_cfg.mode == "gate":
            failures.append(f"{name}: {'; '.join(violations)}")

    if (d := p.domain_shift).enabled:
        feats = (lambda ds: encoder_features(ds, cfg)) if d.feature == "encoder" else stats_features
        s = knn_interleaving(feats(ref_ds), feats(cand_ds), d.k)
        finish("domain_shift", d, s, [f"shift_score {s['shift_score']:.3f} > {d.max_shift}"] if s["shift_score"] > d.max_shift else [])

    if (b := p.label_balance).enabled:
        RA, RB = ref_ds.class_ratios(), cand_ds.class_ratios()           # (tiles, classes)
        names = ref_ds.class_names
        s, v = {}, []
        for c, cname in enumerate(names):                                # every class on its own
            ra, rb = RA[:, c], RB[:, c]
            sc = label_balance(ra, rb)
            one = len(names) == 1
            s.update({(k if one else f"{cname}/{k}"): val for k, val in sc.items()})
            tag = "" if one else f"[{cname}] "
            if sc["wasserstein"] > b.max_wasserstein:
                v.append(f"{tag}wasserstein {sc['wasserstein']:.4f} > {b.max_wasserstein}")
            if b.max_ratio_gap is not None and sc["ratio_gap_rel"] > b.max_ratio_gap:
                v.append(f"{tag}ratio_gap_rel {sc['ratio_gap_rel']:.3f} > {b.max_ratio_gap}")
            bins = np.linspace(0, max(ra.max(), rb.max(), 1e-6), 21)
            for split, r in (("reference", ra), ("candidate", rb)):
                reporter.histogram(f"preflight/label_balance/{cname}", split, np.histogram(r, bins)[0], 0,
                                   xlabels=[f"{x:.3f}" for x in bins[:-1]])
        finish("label_balance", b, s, v)

    (out_dir / "preflight.json").write_text(json.dumps(results, indent=2))
    if reporter.task is not None:
        reporter.task.upload_artifact("preflight", artifact_object=results)
    if failures:
        raise PreflightFailed(" | ".join(failures))
    return results


def gate_preflight(cfg, task, reporter, root: Path, out_dir: Path) -> dict:
    """Sample both splits and run the checks; a violated `gate` fails the Task and exits with code 2."""
    p, sp = cfg.preflight, cfg.data.splits
    ref, cand = (TileDataset(root, sp[s], cfg, limit=p.max_samples, seed=cfg.seed)
                 for s in (p.reference_split, p.candidate_split))
    try:
        return run_preflight(cfg, reporter, ref, cand, out_dir)
    except PreflightFailed as e:
        fail_gate(task, "preflight_gate", str(e))
    