"""Sweep candidate budget x fusion strategy for the scored top-K cap.

Reuses the cached union pickle + cached e5 embeddings, so each variant takes
seconds.

Usage:
    python scripts/sweep_cap.py --budgets 20 30 50 100
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from bench_scored_cap import build_lexical_union
from src.blocking import cap_candidates_scored, measure_blocking_quality
from src.dense_blocking import MULTILINGUAL_MODEL, encode_texts


def measure(cands, gt, total_pairs, s1_ids):
    m = measure_blocking_quality(cands, gt, total_pairs, s1_ids)
    return m["pair_recall"], m["matches_retained"], m["total_matches"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="tests/world_2p5")
    ap.add_argument("--budgets", nargs="+", type=int, default=[20, 30, 50, 100])
    ap.add_argument("--weights", nargs="+", type=float, default=[0.5, 0.7, 0.85, 1.0])
    ap.add_argument("--union-cache", default="benchmarks/union_world_2p5.pkl")
    ap.add_argument("--embed-cache", default="local_data/embeddings")
    ap.add_argument("--out", default="benchmarks/cap_sweep_world_2p5.json")
    args = ap.parse_args()

    s1, s1_names, s1_ids, gallery_names, gallery_ids, gt, union = build_lexical_union(
        args.world, args.union_cache)
    total_pairs = len(s1_names) * len(gallery_names)
    print(f"[union] {sum(len(v) for v in union.values()):,} candidates")

    q_emb = encode_texts(s1_names, model_name=MULTILINGUAL_MODEL,
                         cache_dir=args.embed_cache, role="query", show_progress=False)
    g_emb = encode_texts(gallery_names, model_name=MULTILINGUAL_MODEL,
                         cache_dir=args.embed_cache, role="target", show_progress=False)

    results = {"budget_sweep": {}, "fusion_sweep": {}}

    print("\n--- budget sweep (pure dense ranking) ---")
    for budget in args.budgets:
        t0 = time.time()
        cands = cap_candidates_scored(
            union, s1_names, gallery_names, gallery_ids,
            max_per_query=budget, dense_query_emb=q_emb, dense_target_emb=g_emb,
            dense_weight=1.0)
        rec, kept, tot = measure(cands, gt, total_pairs, s1_ids)
        print(f"  budget={budget:4d}: recall={rec:.4f} ({kept:,}/{tot:,})  {time.time()-t0:.1f}s")
        results["budget_sweep"][str(budget)] = {"recall": rec, "matches_retained": kept}
        del cands
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    best_budget = max(args.budgets, key=lambda b: results["budget_sweep"][str(b)]["recall"])
    print(f"\n--- fusion sweep at budget={best_budget} ---")
    for w in args.weights:
        t0 = time.time()
        cands = cap_candidates_scored(
            union, s1_names, gallery_names, gallery_ids,
            max_per_query=best_budget, dense_query_emb=q_emb, dense_target_emb=g_emb,
            dense_weight=w)
        rec, kept, tot = measure(cands, gt, total_pairs, s1_ids)
        print(f"  weighted w={w:4.2f}: recall={rec:.4f}  {time.time()-t0:.1f}s")
        results["fusion_sweep"][f"weighted_{w}"] = {"recall": rec, "matches_retained": kept}
        del cands

    t0 = time.time()
    cands = cap_candidates_scored(
        union, s1_names, gallery_names, gallery_ids,
        max_per_query=best_budget, dense_query_emb=q_emb, dense_target_emb=g_emb,
        fusion="max")
    rec, kept, tot = measure(cands, gt, total_pairs, s1_ids)
    print(f"  max fusion   : recall={rec:.4f}  {time.time()-t0:.1f}s")
    results["fusion_sweep"]["max"] = {"recall": rec, "matches_retained": kept}
    del cands

    # fuzzy-only reference at the best budget
    cands = cap_candidates_scored(
        union, s1_names, gallery_names, gallery_ids, max_per_query=best_budget)
    rec, kept, tot = measure(cands, gt, total_pairs, s1_ids)
    print(f"  fuzzy only   : recall={rec:.4f}")
    results["fusion_sweep"]["fuzzy_only"] = {"recall": rec, "matches_retained": kept}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
