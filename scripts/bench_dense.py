"""Dense blocking benchmark on the sampled world — measures recall@K on REAL data.

Usage:
    python scripts/bench_dense.py --world tests/world_2p5 --models arctic e5
    python scripts/bench_dense.py --world tests/world_2p5 --models arctic --hybrid

Writes results to benchmarks/dense_world_2p5.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_training_data, parse_ground_truth, combine_sources
from src.normalize import apply_normalization
from src.dense_blocking import (
    DEFAULT_MODEL,
    MULTILINGUAL_MODEL,
    encode_texts,
    get_device,
    topk_search,
)

MODELS = {
    "arctic": DEFAULT_MODEL,
    "e5": MULTILINGUAL_MODEL,
}


def recall_at_k(indices: np.ndarray, target_ids: np.ndarray, gt_lookup: dict,
                s1_ids: list, k: int) -> dict:
    """Full-denominator recall: every ground-truth match counts, even when the
    query has zero candidates."""
    retained = 0
    total = 0
    for q_idx, s1_id in enumerate(s1_ids):
        matched = gt_lookup.get(s1_id, [])
        total += len(matched)
        if not matched:
            continue
        cand = {target_ids[int(t)] for t in indices[q_idx, :k] if t >= 0}
        retained += len(set(matched) & cand)
    return {
        "k": k,
        "matches_retained": retained,
        "total_matches": total,
        "recall": round(retained / max(total, 1), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="tests/world_2p5")
    ap.add_argument("--models", nargs="+", default=["arctic", "e5"],
                    choices=list(MODELS.keys()))
    ap.add_argument("--top-k", type=int, default=50, help="max K measured")
    ap.add_argument("--ks", nargs="+", type=int, default=[20, 30, 50])
    ap.add_argument("--hybrid", action="store_true",
                    help="also run lexical legs and measure lexical+dense union")
    ap.add_argument("--cache-dir", default="local_data/embeddings")
    ap.add_argument("--out", default="benchmarks/dense_world_2p5.json")
    args = ap.parse_args()

    print("=" * 60)
    print(f"DENSE BLOCKING BENCHMARK — device={get_device()}")
    print("=" * 60)

    t0 = time.time()
    data = load_training_data(args.world)
    s1 = apply_normalization(data["train_s1"])
    s2 = apply_normalization(data["train_s2"])
    s3 = apply_normalization(data["train_s3"])
    gt = parse_ground_truth(data["train_gt"])
    s2_s3 = combine_sources(s2, s3)
    print(f"[load+normalize] {time.time()-t0:.1f}s  "
          f"(S1={len(s1):,} gallery={len(s2_s3):,})")

    s1_names = s1["business_name_clean"].fillna("").tolist()
    s1_ids = s1["entity_id"].tolist()
    gallery_names = s2_s3["business_name_clean"].fillna("").tolist()
    gallery_ids = np.array(s2_s3["entity_id"].tolist())

    total_gt = sum(len(v) for v in gt.values())
    print(f"ground truth pairs: {total_gt:,}")

    results = {
        "world": args.world,
        "device": get_device(),
        "n_s1": len(s1_names),
        "n_gallery": len(gallery_names),
        "total_gt": total_gt,
        "models": {},
    }

    dense_indices = {}
    for key in args.models:
        model_name = MODELS[key]
        print(f"\n--- model: {key} ({model_name}) ---")
        t0 = time.time()
        query_emb = encode_texts(s1_names, model_name=model_name,
                                 cache_dir=args.cache_dir, role="query")
        target_emb = encode_texts(gallery_names, model_name=model_name,
                                  cache_dir=args.cache_dir, role="target")
        enc_time = time.time() - t0
        print(f"  encode: {enc_time:.1f}s (cached after first run)")

        t0 = time.time()
        indices, scores = topk_search(query_emb, target_emb, top_k=args.top_k)
        search_time = time.time() - t0
        print(f"  search top-{args.top_k}: {search_time:.1f}s")

        # Free embeddings immediately — only the int32 index arrays are kept
        # (55K x 50 x 4B ~= 11MB vs ~1.4GB of float32 embeddings).
        del query_emb, target_emb, scores

        per_k = []
        for k in args.ks:
            r = recall_at_k(indices, gallery_ids, gt, s1_ids, min(k, args.top_k))
            per_k.append(r)
            print(f"  recall@{r['k']}: {r['recall']:.4f} "
                  f"({r['matches_retained']:,}/{r['total_matches']:,})")

        results["models"][key] = {
            "model_name": model_name,
            "encode_seconds": round(enc_time, 1),
            "search_seconds": round(search_time, 1),
            "recall": per_k,
        }
        dense_indices[key] = indices

        # Save incrementally so partial runs are never lost.
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    if args.hybrid:
        from src.blocking import (
            bidirectional_tfidf, key_blocking, phonetic_blocking,
            initialism_blocking, minhash_lsh_candidates,
            address_tfidf_candidates, tfidf_blocking_candidates,
            union_candidates, measure_blocking_quality,
        )
        print("\n--- lexical legs (for hybrid union) ---")
        s1_addrs = s1["business_address_clean"].fillna("").tolist()
        gallery_addrs = s2_s3["business_address_clean"].fillna("").tolist()

        t0 = time.time()
        legs = {}
        legs["bidirectional"], _ = bidirectional_tfidf(s1_names, gallery_names, list(gallery_ids))
        legs["token_sorted"] = tfidf_blocking_candidates(
            [" ".join(sorted(n.split())) for n in s1_names],
            [" ".join(sorted(n.split())) for n in gallery_names],
            list(gallery_ids), threshold=0.25)
        legs["phonetic"] = phonetic_blocking(s1_names, gallery_names, list(gallery_ids))
        legs["initialism"] = initialism_blocking(s1_names, gallery_names, list(gallery_ids))
        legs["address_tfidf"] = address_tfidf_candidates(
            s1_addrs, gallery_addrs, list(gallery_ids), threshold=0.25)
        legs["exact_keys"] = key_blocking(s1, s2_s3)
        legs["minhash"] = minhash_lsh_candidates(
            s1_names, gallery_names, list(gallery_ids), threshold=0.3)
        lexical_union = union_candidates(*legs.values())
        lex_metrics = measure_blocking_quality(
            lexical_union, gt, len(s1_names) * len(gallery_names), s1_ids)
        print(f"  lexical union: recall={lex_metrics['pair_recall']} "
              f"cands={lex_metrics['candidate_pairs']:,}")

        for key, indices in dense_indices.items():
            dense_cands = {}
            for q in range(indices.shape[0]):
                ids = {gallery_ids[int(t)] for t in indices[q, :30] if t >= 0}
                if ids:
                    dense_cands[q] = ids
            hybrid = union_candidates(lexical_union, dense_cands)
            hy_metrics = measure_blocking_quality(
                hybrid, gt, len(s1_names) * len(gallery_names), s1_ids)
            print(f"  lexical + {key} top-30: recall={hy_metrics['pair_recall']} "
                  f"cands={hy_metrics['candidate_pairs']:,}")
            results["models"][key]["hybrid_with_lexical"] = hy_metrics
            del dense_cands, hybrid

        results["lexical_union"] = lex_metrics
        results["lexical_legs_seconds"] = round(time.time() - t0, 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
