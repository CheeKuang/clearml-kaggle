"""Unit tests for the rarity check in src/gates/preflight.py on made-up class shares (no dataset, no ClearML).

  python tests/test_preflight_rarity.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.clearml_utils import Reporter  # noqa: E402
from src.gates.preflight import class_rarity, run_preflight  # noqa: E402


class FakeDS:
    """What run_preflight needs for the label checks: .class_names, len() and .mask(i) -> (classes, H, W). Each mask is one pixel
    here, so its mean is the class share we want."""

    def __init__(self, names, ratios):
        self.class_names, self._r = names, np.asarray(ratios, float)

    def __len__(self):
        return len(self._r)

    def mask(self, i):
        return self._r[i][:, None, None]


def _cfg(mode="diagnostic", **rarity):
    return OmegaConf.create({"preflight": {"domain_shift": {"enabled": False}, "label_balance": {"enabled": False},
                                           "rarity": {"enabled": True, "mode": mode, "min_tile_frac": 0.10, "min_mean_ratio": 0.01, **rarity}}})


def test_class_rarity_counts_tiles_with_the_class():
    r = np.array([0, 0, 0, 0.2, 0.0, 0.4, 0, 0, 0, 0])
    sc = class_rarity(r)
    assert sc["tiles"] == 10 and sc["tiles_with"] == 2 and abs(sc["tile_frac"] - 0.2) < 1e-9 and abs(sc["mean_ratio"] - 0.06) < 1e-9


def _run(cfg, ref, cand):
    with tempfile.TemporaryDirectory() as d:
        return run_preflight(cfg, Reporter(None), ref, cand, Path(d))["rarity"]


def test_a_common_class_passes_and_a_rare_one_is_flagged_per_split():
    rng = np.random.default_rng(0)
    common = rng.uniform(0.05, 0.4, 200)                                   # every tile has some
    rare = np.where(rng.random(200) < 0.05, 0.3, 0.0)                      # about 5% of the tiles, and few pixels
    ok = _run(_cfg(), FakeDS(["road"], common[:, None]), FakeDS(["road"], common[:, None]))
    assert ok["passed"] and ok["violations"] == []
    bad = _run(_cfg(), FakeDS(["water"], common[:, None]), FakeDS(["water"], rare[:, None]))       # only val is rare
    assert not bad["passed"] and all(v.startswith("candidate") for v in bad["violations"]), bad["violations"]
    assert bad["scores"]["candidate/tiles_with"] == int((rare > 0).sum()) and bad["scores"]["reference/tile_frac"] == 1.0


def test_each_class_is_judged_on_its_own():
    r = np.stack([np.full(100, 0.2), np.zeros(100)], axis=1)               # class a everywhere, class b nowhere
    res = _run(_cfg(), FakeDS(["a", "b"], r), FakeDS(["a", "b"], r))
    assert not res["passed"] and all(v.startswith("[b]") for v in res["violations"]), res["violations"]
    assert res["scores"]["a/reference/tile_frac"] == 1.0 and res["scores"]["b/reference/tiles_with"] == 0


def test_diagnostic_does_not_stop_but_gate_mode_does():
    from src.gates.preflight import PreflightFailed
    zero = FakeDS(["water"], np.zeros((50, 1)))
    assert not _run(_cfg("diagnostic"), zero, zero)["passed"]               # reported, no exception
    try:
        _run(_cfg("gate"), zero, zero)
    except PreflightFailed as e:
        assert "rarity" in str(e)
    else:
        raise AssertionError("gate mode should raise")


def test_a_config_without_the_rarity_block_still_works():
    cfg = OmegaConf.create({"preflight": {"domain_shift": {"enabled": False}, "label_balance": {"enabled": False}}})
    with tempfile.TemporaryDirectory() as d:
        assert run_preflight(cfg, Reporter(None), FakeDS(["a"], np.ones((3, 1))), FakeDS(["a"], np.ones((3, 1))), Path(d)) == {}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
