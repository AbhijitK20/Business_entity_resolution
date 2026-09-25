"""Tests for scored top-K candidate pruning (cap_candidates_scored).

Run: python tests/test_scored_cap.py
"""

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blocking import cap_candidates, cap_candidates_scored

PASSED = 0
FAILED = 0


def check(name, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  PASS  {name}")
        PASSED += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()
        FAILED += 1


def test_raw_cap_loses_true_match():
    """Demonstrates the failure mode the scored cap fixes.

    A list container gives deterministic order (sets iterate hash-based).
    """
    candidates = {0: [f"g{i}" for i in range(49)] + ["g-true"]}
    raw = cap_candidates(candidates, max_per_query=10)
    assert "g-true" not in raw[0], "raw cap unexpectedly kept the last candidate"


def test_scored_cap_keeps_similar_match():
    """True match is similar to the query; unrelated fillers should be dropped.

    Names are lowercased/normalized as in the real pipeline (RapidFuzz
    scorers are case-sensitive).
    """
    query = ["acme robotics private limited"]
    targets = [f"unrelated filler co {i}" for i in range(49)] + ["acme robotics pvt ltd"]
    ids = [f"g{i}" for i in range(49)] + ["g-true"]
    candidates = {0: set(ids)}

    scored = cap_candidates_scored(candidates, query, targets, ids, max_per_query=5)
    assert "g-true" in scored[0], f"scored cap dropped the true match: {scored[0]}"
    assert len(scored[0]) == 5


def test_small_buckets_untouched():
    query = ["Alpha"]
    targets = ["Alpha", "Beta"]
    ids = ["a", "b"]
    candidates = {0: {"a", "b"}}
    scored = cap_candidates_scored(candidates, query, targets, ids, max_per_query=10)
    assert scored[0] == {"a", "b"}


def test_dense_scores_rank_semantically():
    """With orthogonal fuzzy signal, dense cosine decides the survivors."""
    query = ["q"]
    targets = ["t0", "t1", "t2"]
    ids = ["t0", "t1", "t2"]
    # Identical fuzzy scores for all (single tokens 'q' vs 'tN').
    dense_q = np.array([[1.0, 0.0]], dtype=np.float32)
    dense_t = np.array([
        [0.0, 1.0],   # t0: cosine 0.0  -> 0.5 after rescale
        [0.9, 0.1],   # t1: highest
        [0.5, 0.5],   # t2
    ], dtype=np.float32)
    # normalize target rows
    dense_t /= np.linalg.norm(dense_t, axis=1, keepdims=True)

    candidates = {0: {"t0", "t1", "t2"}}
    scored = cap_candidates_scored(
        candidates, query, targets, ids, max_per_query=1,
        dense_query_emb=dense_q, dense_target_emb=dense_t,
        dense_weight=1.0,
    )
    assert scored[0] == {"t1"}, f"dense ranking wrong: {scored[0]}"


def test_empty_query_name_safe():
    candidates = {0: {f"g{i}" for i in range(40)}}
    query = [""]
    targets = [f"name {i}" for i in range(40)]
    ids = [f"g{i}" for i in range(40)]
    scored = cap_candidates_scored(candidates, query, targets, ids, max_per_query=5)
    assert len(scored[0]) == 5


if __name__ == "__main__":
    print("=== scored cap tests ===")
    check("raw_cap_loses_true_match", test_raw_cap_loses_true_match)
    check("scored_cap_keeps_similar_match", test_scored_cap_keeps_similar_match)
    check("small_buckets_untouched", test_small_buckets_untouched)
    check("dense_scores_rank_semantically", test_dense_scores_rank_semantically)
    check("empty_query_name_safe", test_empty_query_name_safe)
    print(f"\n{PASSED}/{PASSED + FAILED} scored-cap tests passed.")
    sys.exit(1 if FAILED else 0)
