"""Unit tests for the decision layer — exact metric semantics + exclusivity +
expected-F0.5 set selection.

Run: python tests/test_decision.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.decision import (
    entity_f05, macro_f05, apply_exclusivity,
    select_sets_expected_f05, select_sets_threshold, candidate_oracle_f05,
)


def test_entity_f05_worked_example():
    """Official PS example: predict 3, 2 correct → 0.714."""
    truth = {"S2-00047", "S3-00812"}
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    assert abs(entity_f05(truth, pred) - 0.714) < 0.001, entity_f05(truth, pred)


def test_singleton_semantics():
    assert entity_f05(set(), set()) == 1.0          # correct empty
    assert entity_f05(set(), {"S2-1"}) == 0.0       # false merge on singleton
    assert entity_f05({"S2-1"}, set()) == 0.0       # missed everything


def test_entity_f05_partial():
    # two true matches, one predicted correctly → 5/6
    assert abs(entity_f05({"a", "b"}, {"a"}) - 5 / 6) < 1e-9
    # both plus one false → 10/14
    assert abs(entity_f05({"a", "b"}, {"a", "b", "c"}) - 10 / 14) < 1e-9


def test_macro_f05_mixed():
    truth = {"S1-1": {"a"}, "S1-2": set(), "S1-3": {"b", "c"}}
    pred = {"S1-1": {"a"}, "S1-2": set(), "S1-3": {"b"}}
    # 1.0 + 1.0 + 5/6 = 2.8333 / 3
    expected = (1.0 + 1.0 + 5 / 6) / 3
    assert abs(macro_f05(truth, pred) - expected) < 1e-9


def test_exclusivity_higher_p_wins():
    s1 = ["S1-A", "S1-B", "S1-A"]
    cand = ["S2-1", "S2-1", "S2-2"]      # both A and B claim S2-1
    p = np.array([0.9, 0.4, 0.8])        # A wins S2-1
    keep = apply_exclusivity(s1, cand, p, p_min=0.05)
    assert keep.tolist() == [True, False, True]


def test_exclusivity_below_p_min_dropped():
    s1 = ["S1-A"]
    cand = ["S2-1"]
    p = np.array([0.01])
    keep = apply_exclusivity(s1, cand, p, p_min=0.05)
    assert keep.tolist() == [False]


def test_expected_f05_clear_single_match():
    s1 = ["S1-A"]
    cand = ["S2-1"]
    p = np.array([0.95])
    out = select_sets_expected_f05(s1, cand, p, anchor_ids=["S1-A"])
    assert out["S1-A"] == {"S2-1"}


def test_expected_f05_singleton_chooses_empty():
    s1 = ["S1-A"]
    cand = ["S2-1"]
    p = np.array([0.10])                 # weak candidate → empty wins
    out = select_sets_expected_f05(s1, cand, p, anchor_ids=["S1-A"])
    assert out["S1-A"] == set()


def test_expected_f05_multi_match_kept():
    s1 = ["S1-A", "S1-A", "S1-A"]
    cand = ["S2-1", "S2-2", "S2-3"]
    p = np.array([0.95, 0.90, 0.60])
    out = select_sets_expected_f05(s1, cand, p, anchor_ids=["S1-A"])
    assert len(out["S1-A"]) >= 2        # never forces top-1


def test_expected_f05_zero_candidate_anchor_preserved():
    s1 = ["S1-A"]
    cand = ["S2-1"]
    p = np.array([0.9])
    out = select_sets_expected_f05(s1, cand, p, anchor_ids=["S1-A", "S1-Z"])
    assert out["S1-Z"] == set()


def test_expected_f05_beats_or_matches_threshold():
    """On a clear case both agree; on a singleton expected-F0.5 must abstain."""
    s1 = ["S1-A", "S1-B"]
    cand = ["S2-1", "S2-2"]
    p = np.array([0.95, 0.10])
    exp = select_sets_expected_f05(s1, cand, p, anchor_ids=["S1-A", "S1-B"])
    thr = select_sets_threshold(s1, cand, p, threshold=0.5, anchor_ids=["S1-A", "S1-B"])
    assert exp["S1-A"] == {"S2-1"} and thr["S1-A"] == {"S2-1"}
    assert exp["S1-B"] == set() and thr["S1-B"] == set()


def test_oracle_perfect_and_lossy():
    truth = {"S1-1": {"a", "b"}, "S1-2": set()}
    perfect = {"S1-1": {"a", "b"}, "S1-2": set()}
    assert candidate_oracle_f05(truth, perfect) == 1.0
    # only 'a' available → r=1, t=2 → 5/(4+2)=0.8333 for S1-1, 1.0 for S1-2
    lossy = {"S1-1": {"a"}, "S1-2": set()}
    assert abs(candidate_oracle_f05(truth, lossy) - (5 / 6 + 1.0) / 2) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} decision tests passed.")
