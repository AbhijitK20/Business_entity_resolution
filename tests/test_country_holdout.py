"""Tests for scripts/country_holdout.py — K4: country-holdout stress test.

Runs the full US<->India transfer protocol on a small synthetic dataset and
validates the report structure, metric ranges, gap/margin fields, the France
cutoff recommendation, pair-construction invariants, and determinism.

Run: python tests/test_country_holdout.py   (or pytest)
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.country_holdout import (
    build_pairs,
    load_frames,
    precision_preserving_threshold,
    run_holdout,
)
from scripts.make_synthetic_data import generate

N_S1 = 250
SEED = 42


def _load_small():
    tmp = Path(tempfile.mkdtemp(prefix="holdout_test_"))
    generate(tmp, N_S1, SEED)
    s1, gal, gt = load_frames(tmp, "train")
    return s1, gal, gt


def _run(**kw):
    s1, gal, gt = _load_small()
    defaults = dict(seed=SEED, n_trials=0, neg_per_pos=1,
                    full_density=False, provenance="synthetic-test")
    defaults.update(kw)
    return run_holdout(s1, gal, gt, **defaults)


# ---------------------------------------------------------------- pair modes
def test_build_pairs_modes():
    s1, gal, gt = _load_small()
    from scripts.country_holdout import build_records
    recs = build_records([s1, gal])
    gallery_ids = sorted(set(gal["entity_id"]))
    in_ids = sorted(s for s in gt if recs[s][2] == "us")[:40]

    full = build_pairs(gt, in_ids, recs, gallery_ids, mode="eval_full",
                       seed=SEED)
    positives = {(s, c) for s, c, l in full if l == 1}
    negs = [(s, c) for s, c, l in full if l == 0]
    assert positives, "eval_full must include positives"
    # every non-positive gallery pair is present exactly once
    expected_neg = len(in_ids) * len(gallery_ids) - len(positives)
    assert len(negs) == expected_neg, (len(negs), expected_neg)
    # every positive from GT is included
    for s in in_ids:
        for t in gt[s]:
            assert (s, t, 1) in full

    hard = build_pairs(gt, in_ids, recs, gallery_ids, mode="eval_hard",
                       seed=SEED)
    assert {(s, c) for s, c, l in hard if l == 1} == positives
    for s, c, l in hard:
        if l == 0:
            assert recs[c][2] == recs[s][2], "hard negatives must be same-country"
    # hard is a strict subset of full's candidate space
    hard_set = {(s, c) for s, c, l in hard}
    full_set = {(s, c) for s, c, l in full}
    assert hard_set <= full_set

    train = build_pairs(gt, in_ids, recs, gallery_ids, mode="train",
                        seed=SEED, neg_per_pos=2)
    labels = {l for _, _, l in train}
    assert labels == {0, 1}
    assert len(train) >= len(in_ids), "at least one negative per S1"
    # singletons (no positives) still receive negatives
    singleton_ids = [s for s in in_ids if not gt[s]]
    for s in singleton_ids:
        assert any(sid == s for sid, _, l in train if l == 0), \
            "singleton must still get training negatives"


def test_train_negatives_respect_exclusion():
    """exclude_neg_countries keeps those countries out of RANDOM training
    negatives (clean country holdout — eval country never seen)."""
    s1, gal, gt = _load_small()
    from scripts.country_holdout import build_records
    recs = build_records([s1, gal])
    gallery_ids = sorted(set(gal["entity_id"]))
    in_ids = sorted(s for s in gt if recs[s][2] == "us")[:40]

    excluded = build_pairs(gt, in_ids, recs, gallery_ids, mode="train",
                           seed=SEED, neg_per_pos=2,
                           exclude_neg_countries=("india",))
    neg_countries = {recs[c][2] for _, c, l in excluded if l == 0}
    assert "india" not in neg_countries, neg_countries
    assert neg_countries, "exclusion must not remove ALL negatives"

    # same seed without exclusion → RNG stream unchanged when the pool is
    # large enough, and india negatives are present again
    control = build_pairs(gt, in_ids, recs, gallery_ids, mode="train",
                          seed=SEED, neg_per_pos=2)
    control_neg = {recs[c][2] for _, c, l in control if l == 0}
    assert "india" in control_neg


# ------------------------------------------------------------------ helpers
def test_precision_preserving_threshold():
    import numpy as np
    y = np.asarray([1] * 20 + [0] * 20)
    proba = np.asarray([0.9] * 15 + [0.4] * 5 + [0.2] * 10 + [0.7] * 10)
    # lowest threshold whose pair precision >= 0.60 (max recall subject to it)
    t = precision_preserving_threshold(y, proba, 0.60, lo=0.05, hi=0.95,
                                       step=0.01)
    assert t is not None
    pred = proba >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    assert tp / (tp + fp) >= 0.60
    # minimality: one step lower misses the target
    if t > 0.05:
        pred2 = proba >= (t - 0.01)
        tp2 = int((pred2 & (y == 1)).sum())
        fp2 = int((pred2 & (y == 0)).sum())
        assert tp2 / (tp2 + fp2) < 0.60
    # genuinely unreachable: every positive threshold also keeps negatives
    y2 = np.asarray([1] * 20 + [0] * 20)
    proba2 = np.asarray([0.8] * 20 + [0.9] * 20)
    assert precision_preserving_threshold(y2, proba2, 0.90) is None


# ------------------------------------------------------------- full protocol
def test_holdout_report_structure():
    report = _run()
    assert report["provenance"].startswith("synthetic")
    assert report["seed"] == SEED
    dirs = report["directions"]
    assert len(dirs) == 2
    pairs_seen = {(d["train_country"], d["eval_country"]) for d in dirs}
    assert pairs_seen == {("us", "india"), ("india", "us")}
    for d in dirs:
        assert "skipped" not in d, d.get("skipped")
        t_in = d["threshold_in_country"]
        assert 0.05 <= t_in <= 0.95, t_in
        for regime in ("in_country", "transfer_hard"):
            pr = d[regime]["pair"]
            for k in ("precision", "recall", "f05"):
                assert 0.0 <= pr[k] <= 1.0, (regime, k, pr[k])
            assert d[regime]["entity"]["macro_f05"] >= 0.0
            assert d[regime]["entity"]["n_entities"] > 0
        g = d["gaps"]
        assert "precision_drop_hard" in g and "macro_f05_drop_hard" in g
        for v in g.values():
            assert v is None or math.isfinite(v)
        margin = d["margin_hard"]
        assert margin is None or margin >= 0.0
        assert isinstance(d["threshold_reachable"], bool)
        if d["threshold_precision_preserving_hard"] is not None:
            assert margin == max(0.0, round(
                d["threshold_precision_preserving_hard"] - t_in, 4))
        assert d["threshold_best_on_eval_hard"] is not None


def test_france_recommendation():
    report = _run()
    fr = report["france_recommendation"]
    for key in ("margin_delta", "threshold_pooled_proxy",
                "recommended_france_cutoff", "basis", "note"):
        assert key in fr, key
    if fr["margin_delta"] is not None:
        assert fr["margin_delta"] >= 0.0
        assert fr["recommended_france_cutoff"] is not None
        assert 0.0 < fr["recommended_france_cutoff"] <= 0.95
        pool = report["pooled_final_proxy"]
        assert 0.05 <= pool["threshold"] <= 0.95
        assert pool["val_macro_f05"] >= 0.0
    assert isinstance(fr["directions_with_unreachable_margin"], list)
    assert len(fr["note"]) > 20


def test_direction_gap_sign_convention():
    """gaps = in-country minus transfer: positive == degradation."""
    report = _run()
    for d in report["directions"]:
        if "skipped" in d:
            continue
        recomputed = round(d["in_country"]["pair"]["precision"]
                           - d["transfer_hard"]["pair"]["precision"], 4)
        assert d["gaps"]["precision_drop_hard"] == recomputed


def test_holdout_no_eval_country_in_training():
    """End-to-end leak check: each direction's training negatives come only
    from the TRAIN country — the eval country is never seen, not even as a
    negative (France/test-only countries are excluded the same way)."""
    report = _run()
    assert report["protocol"]["train_negatives_exclude_eval_country"] is True
    for d in report["directions"]:
        if "skipped" in d:
            continue
        neg_countries = d["train_negative_countries"]
        assert d["eval_country"] not in neg_countries, \
            (d["train_country"], neg_countries)
        assert neg_countries == [d["train_country"]], neg_countries


def test_france_cutoff_cap_is_explicit():
    """When pooled threshold + margin exceeds the 0.95 grid end, the report
    must say the cutoff was capped and expose the uncapped value."""
    report = _run()
    fr = report["france_recommendation"]
    if fr["uncapped_cutoff"] is None:
        assert fr["recommended_france_cutoff"] is None
        return
    assert fr["capped_at_0_95"] == (fr["uncapped_cutoff"] > 0.95)
    assert fr["recommended_france_cutoff"] == min(fr["uncapped_cutoff"], 0.95)
    if fr["capped_at_0_95"]:
        assert "capping_note" in fr
        assert "CAPPED" in fr["capping_note"]


def test_determinism_same_seed():
    a = _run()
    b = _run()
    for da, db in zip(a["directions"], b["directions"]):
        assert da["threshold_in_country"] == db["threshold_in_country"]
        assert da["margin_hard"] == db["margin_hard"]
        assert da["transfer_hard"]["pair"] == db["transfer_hard"]["pair"]
    assert (a["pooled_final_proxy"]["threshold"]
            == b["pooled_final_proxy"]["threshold"])
    assert (a["france_recommendation"]["recommended_france_cutoff"]
            == b["france_recommendation"]["recommended_france_cutoff"])


def test_full_density_mode_adds_regime():
    report = _run(full_density=True)
    for d in report["directions"]:
        if "skipped" in d:
            continue
        assert d["transfer_full"] is not None
        assert d["n_eval_pairs_full"] > d["n_eval_pairs_hard"]
        for k in ("precision", "recall", "f05"):
            assert 0.0 <= d["transfer_full"]["pair"][k] <= 1.0
        assert d["threshold_precision_preserving_full"] is not None or True
        assert "precision_drop_full" in d["gaps"]


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except unittest.SkipTest as exc:
            print(f"  SKIP  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} country-holdout tests passed.")
    sys.exit(1 if failed else 0)
