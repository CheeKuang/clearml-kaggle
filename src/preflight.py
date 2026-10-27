"""Config-driven preflight: two independent checks, each `gate` or `diagnostic`.
Every score is logged to ClearML regardless of mode or pass/fail."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors


class PreflightFailed(RuntimeError):
    pass


# --------------------------------------------------------------------------- scores
def knn_interleaving(fa: np.ndarray, fb: np.ndarray, k: int) -> dict:
    """Fraction of k-NN (in the pooled set) that belong to the *other* domain, vs. the fraction
    expected if domains were perfectly mixed. shift_score: 0 = interleaved, 1 = fully separated."""
    X = np.vstack([fa, fb])
    dom = np.r_[np.zeros(len(fa), int), np.ones(len(fb), int)]
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    k = min(k, len(X) - 1)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    other = (dom[idx] != dom[:, None]).mean(1)
    n = len(X)
    expected = np.where(dom == 0, len(fb) / (n - 1), len(fa) / (n - 1))
    return {
        "other_domain_neighbour_frac": float(other.mean()),
        "expected_if_mixed": float(expected.mean()),
        "shift_score": float(np.clip(1 - other.mean() / expected.mean(), 0, 1)),
        "shift_score_ref": float(np.clip(1 - other[dom == 0].mean() / expected[dom == 0].mean(), 0, 1)),
        "shift_score_cand": float(np.clip(1 - other[dom == 1].mean() / expected[dom == 1].mean(), 0, 1)),
        "k": int(k),
    }


def label_balance(ra: np.ndarray, rb: np.ndarray) -> dict:
    ma, mb = float(ra.mean()), float(rb.mean())
    return {
        "wasserstein": float(wasserstein_distance(ra, rb)),
        "mean_ratio_ref": ma,
        "mean_ratio_cand": mb,
        "ratio_gap_rel": float(abs(ma - mb) / max(ma, mb, 1e-12)),
        "empty_tile_frac_ref": float((ra == 0).mean()),
        "empty_tile_frac_cand": float((rb == 0).mean()),
    }


# --------------------------------------------------------------------------- features
def stats_features(ds) -> np.ndarray:
    feats = []
    for i in range(len(ds)):
        x = ds.image(i).reshape(ds.c.in_channels, -1)
        feats.append(np.r_[x.mean(1), x.std(1), np.percentile(x, 5, 1), np.percentile(x, 95, 1)])
    return np.asarray(feats)


def encoder_features(ds, cfg) -> np.ndarray:
    import torch

    from .model_factory import build_model

    enc = build_model(cfg).encoder.eval()
    out = []
    with torch.no_grad():
        for i in range(len(ds)):
            f = enc(torch.from_numpy(ds.image(i))[None])[-1]
            out.append(f.mean((2, 3)).squeeze(0).numpy())
    return np.asarray(out)


# --------------------------------------------------------------------------- runner
def run_preflight(cfg, reporter, ref_ds, cand_ds, out_dir: Path) -> dict:
    p = cfg.preflight
    results, failures = {}, []

    def finish(name, check_cfg, scores, violations):
        passed = not violations
        gated = check_cfg.mode == "gate"
        for key, val in scores.items():                       # always logged
            reporter.scalar(f"preflight/{name}", key, val, 0)
            reporter.single(f"preflight/{name}/{key}", val)
        reporter.scalar(f"preflight/{name}", "passed", float(passed), 0)
        results[name] = {"mode": check_cfg.mode, "passed": passed, "scores": scores, "violations": violations}
        print(f"[preflight] {name}: mode={check_cfg.mode} passed={passed} {violations or ''}")
        if not passed and gated:
            failures.append(f"{name}: {'; '.join(violations)}")

    if p.domain_shift.enabled:
        d = p.domain_shift
        if d.feature == "encoder":
            fa, fb = encoder_features(ref_ds, cfg), encoder_features(cand_ds, cfg)
        else:
            fa, fb = stats_features(ref_ds), stats_features(cand_ds)
        s = knn_interleaving(fa, fb, d.k)
        v = [f"shift_score {s['shift_score']:.3f} > {d.max_shift}"] if s["shift_score"] > d.max_shift else []
        finish("domain_shift", d, s, v)

    if p.label_balance.enabled:
        b = p.label_balance
        ra, rb = ref_ds.building_ratios(), cand_ds.building_ratios()
        s = label_balance(ra, rb)
        v = []
        if s["wasserstein"] > b.max_wasserstein:
            v.append(f"wasserstein {s['wasserstein']:.4f} > {b.max_wasserstein}")
        if b.max_ratio_gap is not None and s["ratio_gap_rel"] > b.max_ratio_gap:
            v.append(f"ratio_gap_rel {s['ratio_gap_rel']:.3f} > {b.max_ratio_gap}")
        bins = np.linspace(0, max(ra.max(), rb.max(), 1e-6), 21)
        for name, r in (("reference", ra), ("candidate", rb)):
            reporter.histogram("preflight/label_balance", name, np.histogram(r, bins)[0], 0,
                               xlabels=[f"{x:.3f}" for x in bins[:-1]])
        finish("label_balance", b, s, v)

    (out_dir / "preflight.json").write_text(json.dumps(results, indent=2))
    if reporter.task is not None:
        reporter.task.upload_artifact("preflight", artifact_object=results)
    if failures:
        raise PreflightFailed(" | ".join(failures))
    return results
