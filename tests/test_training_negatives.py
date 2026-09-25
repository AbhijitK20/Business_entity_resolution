"""Tests for scale-safe hard-negative generation.

Run: python tests/test_training_negatives.py
"""

import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training import construct_training_pairs, generate_hard_negatives

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


def make_world(n_s1=200, n_gallery=5000, seed=0):
    import random
    rng = random.Random(seed)
    s1_rows, gal_rows, gt = [], [], {}
    for i in range(n_s1):
        sid = f"S1-{i}"
        name = f"business number {i}"
        country = rng.choice(["US", "IN"])
        s1_rows.append({"entity_id": sid, "business_name_clean": name,
                        "business_address_clean": "", "country_clean": country})
        matches = [f"G-{i}-{j}" for j in range(2)]
        gt[sid] = matches
        for mid in matches:
            gal_rows.append({"entity_id": mid, "business_name_clean": f"business {i}",
                             "business_address_clean": "", "country_clean": country})
    for j in range(n_gallery):
        gal_rows.append({"entity_id": f"D-{j}", "business_name_clean": f"filler {j}",
                         "business_address_clean": "", "country_clean": rng.choice(["US", "IN"])})
    return pd.DataFrame(s1_rows), pd.DataFrame(gal_rows), gt


def test_candidates_used_for_hard_negatives():
    s1, gal, gt = make_world()
    # Candidate sets: positives + a few specific distractors per S1
    cands = {}
    for i, sid in enumerate(s1["entity_id"]):
        cands[i] = set(gt[sid]) | {f"D-{i}", f"D-{i+1}"}
    neg = generate_hard_negatives(s1, gal, gt, n_neg_per_pos=2, candidates=cands)
    assert len(neg) > 0, "no negatives generated"
    # No positive may appear as a negative
    for _, row in neg.iterrows():
        assert row["candidate_entity_id"] not in gt[row["s1_entity_id"]], \
            "positive leaked into negatives"
    # Hard negatives (label 0) should be drawn from the candidate distractors
    hard = neg[neg["candidate_entity_id"].str.startswith("D-")]
    assert len(hard) > 0, "no hard negatives from candidate pool"


def test_no_candidates_fallback_bounded():
    s1, gal, gt = make_world(n_s1=100, n_gallery=3000)
    t0 = time.time()
    neg = generate_hard_negatives(s1, gal, gt, n_neg_per_pos=2,
                                  max_fallback_pool=100)
    elapsed = time.time() - t0
    assert len(neg) > 0, "fallback generated no negatives"
    assert elapsed < 30, f"fallback too slow: {elapsed:.1f}s"


def test_scale_runtime():
    """The naive implementation needed ~6B comparisons at this scale; the new
    one must finish in seconds."""
    s1, gal, gt = make_world(n_s1=2000, n_gallery=100_000)
    cands = {}
    for i, sid in enumerate(s1["entity_id"]):
        cands[i] = set(gt[sid]) | {f"D-{i % 90_000}", f"D-{(i+7) % 90_000}"}
    t0 = time.time()
    neg = generate_hard_negatives(s1, gal, gt, n_neg_per_pos=2, candidates=cands)
    elapsed = time.time() - t0
    assert len(neg) > 0
    assert elapsed < 60, f"scale run too slow: {elapsed:.1f}s"


def test_construct_pairs_passthrough():
    s1, gal, gt = make_world(n_s1=50, n_gallery=1000)
    cands = {i: set(gt[sid]) | {f"D-{i}"} for i, sid in enumerate(s1["entity_id"])}
    pairs, tr, va = construct_training_pairs(s1, gal, gt, candidates=cands)
    assert len(pairs) > 0 and len(tr) > 0 and len(va) > 0
    assert set(pairs["label"].unique()) == {0, 1}


if __name__ == "__main__":
    print("=== training negative-generation tests ===")
    check("candidates_used_for_hard_negatives", test_candidates_used_for_hard_negatives)
    check("no_candidates_fallback_bounded", test_no_candidates_fallback_bounded)
    check("scale_runtime", test_scale_runtime)
    check("construct_pairs_passthrough", test_construct_pairs_passthrough)
    print(f"\n{PASSED}/{PASSED + FAILED} training tests passed.")
    sys.exit(1 if FAILED else 0)
