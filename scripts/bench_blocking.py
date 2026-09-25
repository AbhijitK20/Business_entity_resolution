"""Blocking benchmark on a sampled world — measures the recall ceiling on REAL data.

Usage:
    python scripts/bench_blocking.py --world tests/world_2p5
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_training_data, parse_ground_truth, combine_sources
from src.normalize import apply_normalization
from src.blocking import (
    bidirectional_tfidf, key_blocking, phonetic_blocking, initialism_blocking,
    minhash_lsh_candidates, address_tfidf_candidates, tfidf_blocking_candidates,
    union_candidates, cap_candidates, measure_blocking_quality,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="tests/world_2p5")
    ap.add_argument("--cap", type=int, default=30)
    args = ap.parse_args()

    print("=" * 60)
    print("BLOCKING BENCHMARK — REAL SAMPLED WORLD")
    print("=" * 60)

    t0 = time.time()
    data = load_training_data(args.world)
    print(f"[load] {time.time()-t0:.1f}s")

    t0 = time.time()
    s1 = apply_normalization(data["train_s1"])
    s2 = apply_normalization(data["train_s2"])
    s3 = apply_normalization(data["train_s3"])
    print(f"[normalize] {time.time()-t0:.1f}s  "
          f"(S1={len(s1):,} S2={len(s2):,} S3={len(s3):,})")

    gt = parse_ground_truth(data["train_gt"])
    s2_s3 = combine_sources(s2, s3)

    s1_names = s1["business_name_clean"].fillna("").tolist()
    s1_addrs = s1["business_address_clean"].fillna("").tolist()
    s2_names = s2_s3["business_name_clean"].fillna("").tolist()
    s2_addrs = s2_s3["business_address_clean"].fillna("").tolist()
    s2_ids = s2_s3["entity_id"].tolist()
    s1_ids = s1["entity_id"].tolist()
    total_pairs = len(s1_names) * len(s2_names)
    print(f"possible pairs: {total_pairs:,}")

    # --- legs ---------------------------------------------------------------
    t0 = time.time()
    c1, _ = bidirectional_tfidf(s1_names, s2_names, s2_ids)
    print(f"[leg1 bidirectional tfidf] {time.time()-t0:.1f}s "
          f"cands={sum(len(v) for v in c1.values()):,}")

    t0 = time.time()
    c6 = key_blocking(s1, s2_s3)
    print(f"[leg6 exact keys] {time.time()-t0:.1f}s "
          f"cands={sum(len(v) for v in c6.values()):,}")

    t0 = time.time()
    c3 = phonetic_blocking(s1_names, s2_names, s2_ids)
    c4 = initialism_blocking(s1_names, s2_names, s2_ids)
    print(f"[leg3/4 phonetic+initialism] {time.time()-t0:.1f}s "
          f"cands={sum(len(v) for v in c3.values()) + sum(len(v) for v in c4.values()):,}")

    t0 = time.time()
    s1_sorted = [" ".join(sorted(n.split())) for n in s1_names]
    s2_sorted = [" ".join(sorted(n.split())) for n in s2_names]
    c2 = tfidf_blocking_candidates(s1_sorted, s2_sorted, s2_ids, threshold=0.25, top_k=30)
    print(f"[leg2 token-sorted] {time.time()-t0:.1f}s "
          f"cands={sum(len(v) for v in c2.values()):,}")

    t0 = time.time()
    c5 = address_tfidf_candidates(s1_addrs, s2_addrs, s2_ids, threshold=0.25, top_k=30)
    print(f"[leg5 address tfidf] {time.time()-t0:.1f}s "
          f"cands={sum(len(v) for v in c5.values()):,}")

    t0 = time.time()
    c7 = minhash_lsh_candidates(s1_names, s2_names, s2_ids, threshold=0.3)
    print(f"[leg7 minhash] {time.time()-t0:.1f}s "
          f"cands={sum(len(v) for v in c7.values()):,}")

    # --- per-leg quality ----------------------------------------------------
    print("\n--- PER-LEG RECALL ---")
    for name, cands in [("bidirectional", c1), ("token_sorted", c2),
                        ("phonetic", c3), ("initialism", c4),
                        ("address_tfidf", c5), ("exact_keys", c6),
                        ("minhash", c7)]:
        m = measure_blocking_quality(cands, gt, total_pairs, s1_ids)
        print(f"  {name:15s}: recall={m['pair_recall']:.4f} "
              f"cands/S1={m['candidate_pairs']/len(s1_names):.1f}")

    # --- union + cap --------------------------------------------------------
    t0 = time.time()
    union = union_candidates(c1, c2, c3, c4, c5, c6, c7)
    m_union = measure_blocking_quality(union, gt, total_pairs, s1_ids)
    print(f"\n[union] {time.time()-t0:.1f}s")
    print(f"  recall      : {m_union['pair_recall']:.4f}")
    print(f"  candidates  : {m_union['candidate_pairs']:,} "
          f"({m_union['candidate_pairs']/len(s1_names):.1f}/S1)")
    print(f"  reduction   : {m_union['reduction_ratio']:.6f}")

    capped = cap_candidates(union, max_per_query=args.cap)
    m_cap = measure_blocking_quality(capped, gt, total_pairs, s1_ids)
    print(f"\n[capped at {args.cap}/S1]")
    print(f"  recall      : {m_cap['pair_recall']:.4f}")
    print(f"  candidates  : {m_cap['candidate_pairs']:,} "
          f"({m_cap['candidate_pairs']/len(s1_names):.1f}/S1)")
    print(f"  reduction   : {m_cap['reduction_ratio']:.6f}")
    print(f"  matches lost: {m_union['matches_retained'] - m_cap['matches_retained']:,} "
          f"of {m_union['total_matches']:,}")


if __name__ == "__main__":
    main()
