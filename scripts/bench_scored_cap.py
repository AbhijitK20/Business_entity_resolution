"""Scored top-K cap benchmark on the sampled world.

Measures how much recall survives a per-S1 candidate budget when candidates
are RANKED by similarity (fuzzy / dense / hybrid) vs the raw first-K cap.

Usage:
    python scripts/bench_scored_cap.py --world tests/world_2p5 --budget 30
    python scripts/bench_scored_cap.py --world tests/world_2p5 --budget 30 50

Writes benchmarks/scored_cap_world_2p5.json and caches the union pickle.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_training_data, parse_ground_truth, combine_sources
from src.normalize import apply_normalization
from src.blocking import (
    bidirectional_tfidf, key_blocking, phonetic_blocking, initialism_blocking,
    minhash_lsh_candidates, address_tfidf_candidates, tfidf_blocking_candidates,
    union_candidates, cap_candidates, cap_candidates_scored,
    measure_blocking_quality,
)
from src.dense_blocking import MULTILINGUAL_MODEL, encode_texts

KEEP_COLS = ["entity_id", "business_name_clean", "business_address_clean", "country_clean"]


def build_lexical_union(world: str, cache: str, verbose: bool = True):
    """Build the 7-leg lexical union, freeing memory as we go."""
    data = load_training_data(world)
    s1 = apply_normalization(data["train_s1"])[KEEP_COLS]
    s2 = apply_normalization(data["train_s2"])[KEEP_COLS]
    s3 = apply_normalization(data["train_s3"])[KEEP_COLS]
    gt = parse_ground_truth(data["train_gt"])
    s2_s3 = combine_sources(s2, s3)
    del s2, s3, data

    s1_names = s1["business_name_clean"].fillna("").tolist()
    s1_ids = s1["entity_id"].tolist()
    gallery_names = s2_s3["business_name_clean"].fillna("").tolist()
    gallery_ids = s2_s3["entity_id"].tolist()

    if Path(cache).exists():
        t0 = time.time()
        with open(cache, "rb") as f:
            union = pickle.load(f)
        print(f"[union] loaded from cache in {time.time()-t0:.1f}s")
        return s1, s1_names, s1_ids, gallery_names, gallery_ids, gt, union

    s1_addrs = s1["business_address_clean"].fillna("").tolist()
    gallery_addrs = s2_s3["business_address_clean"].fillna("").tolist()

    legs = {
        "bidirectional": lambda: bidirectional_tfidf(s1_names, gallery_names, gallery_ids)[0],
        "token_sorted": lambda: tfidf_blocking_candidates(
            [" ".join(sorted(n.split())) for n in s1_names],
            [" ".join(sorted(n.split())) for n in gallery_names],
            gallery_ids, threshold=0.25, top_k=30),
        "phonetic": lambda: phonetic_blocking(s1_names, gallery_names, gallery_ids),
        "initialism": lambda: initialism_blocking(s1_names, gallery_names, gallery_ids),
        "address_tfidf": lambda: address_tfidf_candidates(
            s1_addrs, gallery_addrs, gallery_ids, threshold=0.25, top_k=30),
        "exact_keys": lambda: key_blocking(s1, s2_s3),
        "minhash": lambda: minhash_lsh_candidates(
            s1_names, gallery_names, gallery_ids, threshold=0.3),
    }

    union = {}
    for name, fn in legs.items():
        t0 = time.time()
        cand = fn()
        union = union_candidates(union, cand) if union else cand
        print(f"  [{name}] {time.time()-t0:.1f}s  union={sum(len(v) for v in union.values()):,}")
        del cand
    del s1_addrs, gallery_addrs, s2_s3

    with open(cache, "wb") as f:
        pickle.dump(union, f)
    print(f"[union] built + cached to {cache}")
    return s1, s1_names, s1_ids, gallery_names, gallery_ids, gt, union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="tests/world_2p5")
    ap.add_argument("--budget", nargs="+", type=int, default=[30])
    ap.add_argument("--union-cache", default="benchmarks/union_world_2p5.pkl")
    ap.add_argument("--embed-cache", default="local_data/embeddings")
    ap.add_argument("--dense-weight", type=float, default=0.5)
    ap.add_argument("--out", default="benchmarks/scored_cap_world_2p5.json")
    args = ap.parse_args()

    print("=" * 60)
    print("SCORED TOP-K CAP BENCHMARK — REAL SAMPLED WORLD")
    print("=" * 60)

    s1, s1_names, s1_ids, gallery_names, gallery_ids, gt, union = build_lexical_union(
        args.world, args.union_cache)
    total_pairs = len(s1_names) * len(gallery_names)
    base = measure_blocking_quality(union, gt, total_pairs, s1_ids)
    print(f"\n[union] recall={base['pair_recall']} cands={base['candidate_pairs']:,} "
          f"({base['candidate_pairs']/len(s1_names):.1f}/S1)")

    results = {
        "world": args.world,
        "union": base,
        "budgets": {},
    }

    # Dense embeddings (cached) for semantic scoring.
    t0 = time.time()
    q_emb = encode_texts(s1_names, model_name=MULTILINGUAL_MODEL,
                         cache_dir=args.embed_cache, role="query", show_progress=False)
    g_emb = encode_texts(gallery_names, model_name=MULTILINGUAL_MODEL,
                         cache_dir=args.embed_cache, role="target", show_progress=False)
    print(f"[dense] e5 embeddings loaded in {time.time()-t0:.1f}s")

    for budget in args.budget:
        print(f"\n--- budget {budget}/S1 ---")
        entry = {}

        t0 = time.time()
        raw = cap_candidates(union, max_per_query=budget)
        m = measure_blocking_quality(raw, gt, total_pairs, s1_ids)
        print(f"  raw first-K     : recall={m['pair_recall']}  "
              f"({m['matches_retained']:,}/{m['total_matches']:,})  {time.time()-t0:.1f}s")
        entry["raw"] = m
        del raw

        t0 = time.time()
        fuzzy = cap_candidates_scored(
            union, s1_names, gallery_names, gallery_ids,
            max_per_query=budget, verbose=True)
        m = measure_blocking_quality(fuzzy, gt, total_pairs, s1_ids)
        print(f"  scored fuzzy    : recall={m['pair_recall']}  "
              f"({m['matches_retained']:,}/{m['total_matches']:,})  {time.time()-t0:.1f}s")
        entry["scored_fuzzy"] = m
        del fuzzy

        t0 = time.time()
        dense = cap_candidates_scored(
            union, s1_names, gallery_names, gallery_ids,
            max_per_query=budget,
            dense_query_emb=q_emb, dense_target_emb=g_emb,
            dense_weight=1.0)
        m = measure_blocking_quality(dense, gt, total_pairs, s1_ids)
        print(f"  scored dense    : recall={m['pair_recall']}  "
              f"({m['matches_retained']:,}/{m['total_matches']:,})  {time.time()-t0:.1f}s")
        entry["scored_dense"] = m
        del dense

        t0 = time.time()
        hybrid = cap_candidates_scored(
            union, s1_names, gallery_names, gallery_ids,
            max_per_query=budget,
            dense_query_emb=q_emb, dense_target_emb=g_emb,
            dense_weight=args.dense_weight)
        m = measure_blocking_quality(hybrid, gt, total_pairs, s1_ids)
        print(f"  scored hybrid   : recall={m['pair_recall']}  "
              f"({m['matches_retained']:,}/{m['total_matches']:,})  {time.time()-t0:.1f}s")
        entry["scored_hybrid"] = m
        del hybrid

        results["budgets"][str(budget)] = entry
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
