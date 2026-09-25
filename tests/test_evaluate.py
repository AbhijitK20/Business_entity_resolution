"""Unit tests for scripts/evaluate.py — K1 (oracle ceiling, segments, coverage,
reduction, singleton false merges, business-group bootstrap) and K5 (error
buckets, worst-entity diagnosis).

Run: python tests/test_evaluate.py   (or pytest tests/test_evaluate.py)
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate import (
    BUCKET_ORDER,
    bootstrap_entity_metrics,
    bootstrap_mean_ci,
    build_report,
    candidate_oracle_f05,
    classify_errors,
    complete_match_coverage,
    entity_f05,
    evaluate,
    match_count_bucket,
    oracle_scores,
    reduction_ratio,
    segment_metrics,
    singleton_false_merge_rate,
    worst_entities,
)


# --------------------------------------------------------------- metric basics
def test_entity_f05_worked_example():
    """Official PS example: predict 3, 2 correct → 0.714."""
    truth = {"S2-00047", "S3-00812"}
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    assert abs(entity_f05(truth, pred) - 0.714) < 0.001


def test_evaluate_singleton_semantics():
    gt = {"S1-1": set(), "S1-2": set(), "S1-3": {"a"}}
    preds = {"S1-1": set(), "S1-2": {"x"}, "S1-3": {"a"}}
    rep = evaluate(preds, gt)
    # 1.0 (correct empty) + 0.0 (false merge) + 1.0 (perfect) / 3
    assert abs(rep["macro_f05"] - 2 / 3) < 1e-9
    assert rep["n_singletons"] == 2
    assert abs(rep["singleton_accuracy"] - 0.5) < 1e-9


# ------------------------------------------------------------------- oracle
def test_oracle_formula_exact():
    """Oracle_i = 1 if t_i==0 else 5*r_i/(4*r_i + t_i)."""
    truth = {"S1-1": {"a", "b"}, "S1-2": set(), "S1-3": {"c"}}
    cands = {"S1-1": {"a"}, "S1-2": set(), "S1-3": set()}
    scores = oracle_scores(truth, cands)
    assert scores["S1-1"] == 5 * 1 / (4 * 1 + 2)      # r=1, t=2 → 5/6
    assert scores["S1-2"] == 1.0                       # singleton → 1
    assert scores["S1-3"] == 0.0                       # r=0 → 0
    expected = (5 / 6 + 1.0 + 0.0) / 3
    assert abs(candidate_oracle_f05(truth, cands) - expected) < 1e-9


def test_oracle_perfect_candidates_ceiling_is_one():
    truth = {"S1-1": {"a", "b"}, "S1-2": {"c"}}
    assert candidate_oracle_f05(truth, truth) == 1.0


def test_oracle_grows_with_retrieval_rate():
    """r=2,t=4 > r=1,t=4 — more retrieved true matches → higher ceiling."""
    truth = {"S1-1": {"a", "b", "c", "d"}}
    lo = candidate_oracle_f05(truth, {"S1-1": {"a"}})
    hi = candidate_oracle_f05(truth, {"S1-1": {"a", "b"}})
    assert hi > lo
    assert abs(hi - 10 / 12) < 1e-9


# ------------------------------------------------------- complete-match coverage
def test_complete_match_coverage_definitions():
    gt = {
        "S1-1": {"a", "b"},   # both retrieved → complete
        "S1-2": {"c", "d"},   # only c retrieved → incomplete
        "S1-3": set(),        # singleton → NOT eligible (excluded)
    }
    cands = {"S1-1": {"a", "b", "x"}, "S1-2": {"c"}, "S1-3": set()}
    out = complete_match_coverage(gt, cands)
    assert out["n_eligible"] == 2          # singletons excluded by definition
    assert out["n_complete"] == 1
    assert abs(out["complete_match_coverage"] - 0.5) < 1e-9


def test_complete_match_coverage_empty_truth_only():
    out = complete_match_coverage({"S1-1": set()}, {"S1-1": set()})
    assert out["complete_match_coverage"] is None
    assert out["n_eligible"] == 0


# ----------------------------------------------------------- reduction ratio
def test_reduction_ratio_uses_real_counts():
    # 10 S1 × 1000 gallery = 10000 possible; 100 candidates → 99% reduction
    assert abs(reduction_ratio(100, 10, 1000) - 0.99) < 1e-12
    # unknown denominator → None (never an invented denominator)
    assert reduction_ratio(100, 10, 0) is None


# --------------------------------------------------- singleton false-merge rate
def test_singleton_false_merge_rate_exact_one_match():
    gt = {
        "S1-1": {"a"},   # correct → not a merge
        "S1-2": {"b"},   # predicted b + extra x → merge
        "S1-3": {"c"},   # predicted empty → miss, NOT a merge
        "S1-4": set(),   # t=0 → excluded from this metric
    }
    preds = {
        "S1-1": {"a"},
        "S1-2": {"b", "x"},
        "S1-3": set(),
        "S1-4": {"z"},   # false merge on t=0 belongs to the singleton bucket
    }
    out = singleton_false_merge_rate(preds, gt)
    assert out["n_single_match_entities"] == 3
    assert out["n_false_merges"] == 1
    assert abs(out["singleton_false_merge_rate"] - 1 / 3) < 1e-9


def test_singleton_false_merge_rate_no_eligible():
    out = singleton_false_merge_rate({}, {"S1-1": set()})
    assert out["singleton_false_merge_rate"] is None


# ------------------------------------------------------------------- buckets
def test_match_count_bucket_boundaries():
    assert match_count_bucket(set()) == "0"
    assert match_count_bucket({"a"}) == "1"
    assert match_count_bucket({"a", "b"}) == "2"
    assert match_count_bucket({"a", "b", "c"}) == "3-4"
    assert match_count_bucket({"a", "b", "c", "d"}) == "3-4"
    assert match_count_bucket({str(i) for i in range(5)}) == "5+"
    assert match_count_bucket({str(i) for i in range(11)}) == "5+"


# ------------------------------------------------------------ segment metrics
def test_segment_metrics_bundle():
    gt = {"S1-1": {"a", "b"}, "S1-2": set()}
    preds = {"S1-1": {"a", "z"}, "S1-2": set()}
    cands = {"S1-1": {"a", "b", "z"}, "S1-2": set()}
    m = segment_metrics(preds, gt, list(gt), cands, n_gallery=100)
    assert m["n_s1"] == 2
    assert m["n_true_matches"] == 2
    assert m["n_candidate_pairs"] == 3
    assert abs(m["candidate_recall"] - 1.0) < 1e-9       # both a,b retrieved
    assert m["oracle_f05"] == 1.0
    assert abs(m["complete_match_coverage"] - 1.0) < 1e-9
    # 3 candidates vs 2×100 possible
    assert abs(m["reduction_ratio"] - (1 - 3 / 200)) < 1e-9


def test_segment_metrics_without_candidates():
    m = segment_metrics({"S1-1": {"a"}}, {"S1-1": {"a"}}, ["S1-1"], None)
    assert m["n_candidate_pairs"] is None
    assert m["oracle_f05"] is None
    assert m["reduction_ratio"] is None


# ---------------------------------------------------------- bootstrap (K1)
def test_bootstrap_mean_ci_deterministic_seed():
    values = np.array([1.0, 0.0, 0.5, 0.75, 0.25])
    a = bootstrap_mean_ci(values, n_boot=500, confidence=0.95, seed=42)
    b = bootstrap_mean_ci(values, n_boot=500, confidence=0.95, seed=42)
    c = bootstrap_mean_ci(values, n_boot=500, confidence=0.95, seed=7)
    assert a == b, "same seed must reproduce identical CIs"
    assert a != c or a["point"] == c["point"]
    # point estimate inside CI, correct group count, documented confidence
    assert a["lower"] <= a["point"] <= a["upper"]
    assert a["n_groups"] == len(values)
    assert a["confidence"] == 0.95
    assert a["n_boot"] == 500


def test_bootstrap_zero_replicates_disables():
    out = bootstrap_mean_ci(np.array([1.0, 2.0]), n_boot=0)
    assert out["lower"] is None and out["upper"] is None
    assert out["point"] == 1.5


def test_bootstrap_entity_metrics_macro_and_oracle():
    gt = {"S1-1": {"a"}, "S1-2": set(), "S1-3": {"b", "c"}}
    preds = {"S1-1": {"a"}, "S1-2": set(), "S1-3": {"b"}}
    cands = {"S1-1": {"a"}, "S1-2": set(), "S1-3": {"b", "c"}}
    out = bootstrap_entity_metrics(gt, preds, cands, n_boot=200, seed=42)
    assert set(out) >= {"macro_f05", "oracle_f05"}
    assert out["macro_f05"]["n_groups"] == 3          # business groups, not pairs
    assert out["macro_f05"]["lower"] <= out["macro_f05"]["point"] <= out["macro_f05"]["upper"]
    assert out["oracle_f05"]["point"] == 1.0


def test_bootstrap_groups_not_pairs():
    """Sampling unit count equals number of S1 entities, never pair count."""
    gt = {f"S1-{i}": {"a"} for i in range(4)}          # 4 groups
    preds = {k: v for k, v in gt.items()}
    # pair-level would be 4 pairs too, so make candidates richer than pairs:
    cands = {f"S1-{i}": {"a", "b", "c"} for i in range(4)}   # 12 candidate pairs
    out = bootstrap_entity_metrics(gt, preds, cands, n_boot=100, seed=1)
    assert out["macro_f05"]["n_groups"] == 4
    assert out["oracle_f05"]["n_groups"] == 4


# -------------------------------------------------------- error buckets (K5)
def test_classify_errors_three_buckets_with_scores():
    gt = {"S1-1": {"t1", "t2", "t3", "t4"}}
    preds = {"S1-1": {"t1", "fp_hi"}}
    cands = {"S1-1": {"t1", "t2", "t3", "fp_hi", "fp_lo"}}
    scores = {
        ("S1-1", "t1"): 0.95,
        ("S1-1", "t2"): 0.20,     # < threshold → matching error
        ("S1-1", "t3"): 0.90,     # ≥ threshold but not predicted → decision
        # t4 not in candidates → retrieval error
        ("S1-1", "fp_hi"): 0.88,  # ≥ threshold, kept → matching error (FP)
        ("S1-1", "fp_lo"): 0.10,  # < threshold but predicted → decision (FP)
    }
    preds["S1-1"].add("fp_lo")
    out = classify_errors(preds, gt, cands, scores, threshold=0.5)
    c = out["counts"]
    assert c["retrieval_error"] == 1                 # t4
    assert c["matching_error"] == 2                   # t2 missed + fp_hi kept
    assert c["decision_policy_error"] == 2            # t3 dropped + fp_lo kept
    assert out["totals"]["retrieval_error"] == 1
    assert out["totals"]["matching_error"] == 2
    assert out["totals"]["decision_policy_error"] == 2


def test_classify_errors_without_scores_is_unattributed():
    gt = {"S1-1": {"t1", "t2"}}
    preds = {"S1-1": {"t1"}}
    cands = {"S1-1": {"t1", "t2"}}
    out = classify_errors(preds, gt, cands, scores=None, threshold=None)
    assert out["counts"]["retrieval_error"] == 0
    assert out["counts"]["matching_error_unattributed"] == 1   # t2
    assert not out["scored"]


def test_classify_errors_integrity():
    gt = {"S1-1": {"t1"}}
    preds = {"S1-1": {"t1", "ghost"}}
    cands = {"S1-1": {"t1"}}          # 'ghost' never a candidate
    out = classify_errors(preds, gt, cands, scores=None, threshold=None)
    assert out["counts"]["integrity_error"] == 1


# ------------------------------------------------------- worst entities (K5)
def test_worst_entities_ranking_and_fields():
    gt = {
        "S1-good": {"a"},
        "S1-bad": {"b1", "b2", "b3"},
        "S1-miss": {"m"},
    }
    preds = {"S1-good": {"a"}, "S1-bad": set(), "S1-miss": set()}
    cands = {"S1-good": {"a"}, "S1-bad": {"b1"}, "S1-miss": set()}
    rows = worst_entities(preds, gt, cands, top_k=2)
    assert len(rows) == 2
    assert rows[0]["s1_id"] == "S1-bad"               # f05 = 0, more truths first
    assert rows[0]["error_category"] == "retrieval_error"
    assert rows[1]["f05"] == 0.0
    for r in rows:
        assert set(r) >= {"s1_id", "country", "f05", "n_true", "n_candidates",
                          "n_retrieved_true", "missed_true", "predicted",
                          "error_category", "root_cause"}
    good = [r for r in rows if r["s1_id"] == "S1-good"]
    assert not good                                    # perfect entity not listed


def test_worst_entities_no_padding_with_correct_entities():
    """Fewer error cases than top_k → return fewer rows; correctly-predicted
    entities are never listed as 'worst' to fill the quota."""
    gt = {
        "S1-bad": {"b1", "b2"},     # error: missed
        "S1-wrong": {"w1"},         # error: wrong prediction
        "S1-good1": {"g1"},         # correct
        "S1-good2": set(),          # correct singleton
        "S1-good3": {"g3"},         # correct
    }
    preds = {"S1-bad": set(), "S1-wrong": {"not-a-match"},
             "S1-good1": {"g1"}, "S1-good2": set(), "S1-good3": {"g3"}}
    rows = worst_entities(preds, gt, None, top_k=20)
    assert len(rows) == 2, "actual error-case count, not padded to top_k"
    assert {r["s1_id"] for r in rows} == {"S1-bad", "S1-wrong"}
    for r in rows:
        assert r["f05"] < 1.0
        assert r["error_category"] != "correct"


# ---------------------------------------------------------- end-to-end CLI
def _write_tsv(path: Path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_build_report_end_to_end(tmp_path=None):
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="eval_test_"))

    gt_rows = [
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S3-1"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-3", "matched_entity_ids": "S2-3"},
        {"source1_entity_id": "S1-4", "matched_entity_ids": "S2-4,S2-5"},
    ]
    pred_rows = [
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-3", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-4", "matched_entity_ids": "S2-4,S2-5"},
    ]
    cand_rows = [
        {"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1,S3-1"},
        {"source1_entity_id": "S1-2", "candidate_entity_ids": ""},
        {"source1_entity_id": "S1-3", "candidate_entity_ids": "S2-3"},
        {"source1_entity_id": "S1-4", "candidate_entity_ids": "S2-4,S2-5"},
    ]
    s1_rows = [
        {"entity_id": "S1-1", "business_name": "Acme", "business_address": "1 A St", "country": "US"},
        {"entity_id": "S1-2", "business_name": "Beta", "business_address": "2 B St", "country": "US"},
        {"entity_id": "S1-3", "business_name": "Gamma", "business_address": "3 C Rd", "country": "India"},
        {"entity_id": "S1-4", "business_name": "Delta", "business_address": "4 D Rd", "country": "India"},
    ]
    gal_rows = [
        {"entity_id": "S2-1", "business_name": "Acme", "business_address": "1 A St", "country": "US"},
        {"entity_id": "S2-3", "business_name": "Gamma", "business_address": "3 C Rd", "country": "India"},
        {"entity_id": "S2-4", "business_name": "Delta", "business_address": "4 D Rd", "country": "India"},
        {"entity_id": "S2-5", "business_name": "Delta Ltd", "business_address": "4 D Rd", "country": "India"},
        {"entity_id": "S3-1", "business_name": "Acme Inc", "business_address": "1 A Street", "country": "US"},
        {"entity_id": "S2-9", "business_name": "Other", "business_address": "9 Z St", "country": "US"},
    ]

    _write_tsv(tmp / "gt.tsv", gt_rows, ["source1_entity_id", "matched_entity_ids"])
    _write_tsv(tmp / "pred.tsv", pred_rows, ["source1_entity_id", "matched_entity_ids"])
    _write_tsv(tmp / "cand.tsv", cand_rows, ["source1_entity_id", "candidate_entity_ids"])
    _write_tsv(tmp / "s1.tsv", s1_rows,
               ["entity_id", "business_name", "business_address", "country"])
    _write_tsv(tmp / "gal.tsv", gal_rows,
               ["entity_id", "business_name", "business_address", "country"])

    args = argparse.Namespace(
        predictions=str(tmp / "pred.tsv"),
        ground_truth=str(tmp / "gt.tsv"),
        candidate=str(tmp / "cand.tsv"),
        s1_source=str(tmp / "s1.tsv"),
        gallery_source=[str(tmp / "gal.tsv")],
        n_gallery=None,
        bootstrap=200,
        confidence=0.95,
        seed=42,
        pair_scores=None,
        threshold=None,
        top_k=3,
        feature_evidence=False,
        json_out=None,
    )
    rep = build_report(args)

    # overall macro: S1-1 partial 5/(5*1+0+1)=5/6, S1-2 correct empty 1.0,
    # S1-3 missed 0.0, S1-4 perfect 1.0 → (5/6+1+0+1)/4
    expected_macro = (5 / 6 + 1.0 + 0.0 + 1.0) / 4
    assert abs(rep["overall"]["macro_f05"] - expected_macro) < 1e-9

    # oracle: S1-1 r=2,t=2 →1 ; S1-2 →1 ; S1-3 r=1,t=1 →1 ; S1-4 →1 → 1.0
    assert abs(rep["oracle_f05"] - 1.0) < 1e-9
    assert rep["overall"]["n_entities"] == 4

    # countries present exactly as in the source file (US + India, no France)
    assert set(rep["countries"]) == {"US", "India"}
    assert rep["countries"]["US"]["n_s1"] == 2
    assert rep["countries"]["India"]["n_s1"] == 2
    assert rep["countries"]["US"]["n_true_matches"] == 2

    # buckets cover all entities (|T|: S1-1=2, S1-2=0, S1-3=1, S1-4=2)
    total_bucket_n = sum(rep["buckets"][b]["n_s1"] for b in BUCKET_ORDER)
    assert total_bucket_n == 4
    assert rep["buckets"]["0"]["n_s1"] == 1
    assert rep["buckets"]["1"]["n_s1"] == 1
    assert rep["buckets"]["2"]["n_s1"] == 2

    # coverage + reduction + singleton false merges
    assert abs(rep["complete_match_coverage"]["complete_match_coverage"] - 1.0) < 1e-9
    # 5 candidate pairs vs 4 × 6 gallery → 1 - 5/24
    assert abs(rep["reduction_ratio"] - (1 - 5 / 24)) < 1e-9
    assert rep["singleton_false_merge"]["n_single_match_entities"] == 1
    assert rep["singleton_false_merge"]["n_false_merges"] == 0

    # bootstrap present with documented confidence/unit
    assert rep["bootstrap"]["macro_f05"]["n_groups"] == 4
    assert rep["bootstrap"]["confidence"] == 0.95
    assert rep["bootstrap"]["unit"] == "business_group(S1 entity)"

    # errors + worst entities
    assert "errors" in rep and rep["errors"]["totals"]["retrieval_error"] >= 0
    assert 1 <= len(rep["worst_entities"]) <= 3

    # country-level reduction ratio uses country gallery counts
    assert rep["countries"]["US"]["reduction_ratio"] is not None


def test_no_france_label_no_france_row(tmp_path=None):
    """France must never be fabricated when the data has no France rows."""
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="eval_nofr_"))
    gt_rows = [{"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"}]
    pred_rows = [{"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"}]
    s1_rows = [{"entity_id": "S1-1", "business_name": "A",
                "business_address": "1 St", "country": "US"}]
    _write_tsv(tmp / "gt.tsv", gt_rows, ["source1_entity_id", "matched_entity_ids"])
    _write_tsv(tmp / "pred.tsv", pred_rows, ["source1_entity_id", "matched_entity_ids"])
    _write_tsv(tmp / "s1.tsv", s1_rows,
               ["entity_id", "business_name", "business_address", "country"])
    args = argparse.Namespace(
        predictions=str(tmp / "pred.tsv"), ground_truth=str(tmp / "gt.tsv"),
        candidate=None, s1_source=str(tmp / "s1.tsv"), gallery_source=None,
        n_gallery=None, bootstrap=0, confidence=0.95, seed=42,
        pair_scores=None, threshold=None, top_k=20,
        feature_evidence=False, json_out=None,
    )
    rep = build_report(args)
    assert "France" not in rep.get("countries", {})
    assert rep["countries"] == {"US": rep["countries"]["US"]}


if __name__ == "__main__":
    import unittest

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
    print(f"\n{len(tests) - failed}/{len(tests)} evaluate tests passed.")
    sys.exit(1 if failed else 0)
