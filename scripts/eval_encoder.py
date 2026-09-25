"""Evaluate encoders (base vs fine-tuned) on the same held-out entity split.

Usage:
    python scripts/eval_encoder.py --world tests/world_2p5_split \
        --models intfloat/multilingual-e5-small models/e5-er-ft-smoke models/e5-er-ft
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_training_data, parse_ground_truth, combine_sources
from src.normalize import apply_normalization
from src.dense_blocking import get_device, topk_search

sys.path.insert(0, str(Path(__file__).parent))
from finetune_encoder import recall_at_k


def build_val_split(world: str, seed: int = 42):
    import pandas as pd

    data = load_training_data(world)
    s1 = apply_normalization(data["train_s1"])
    s2 = apply_normalization(data["train_s2"])
    s3 = apply_normalization(data["train_s3"])
    gallery = combine_sources(s2, s3)
    gt = parse_ground_truth(data["train_gt"])

    entities = s1["entity_id"].to_numpy()
    rng_np = np.random.default_rng(seed)
    perm = rng_np.permutation(len(entities))
    n_val = max(1, int(len(entities) * 0.1))
    val_ids = set(entities[perm[:n_val]])
    s1_val = s1[s1["entity_id"].isin(val_ids)]
    val_matched = {m for sid in val_ids for m in gt.get(sid, [])}
    gallery_val = gallery[gallery["entity_id"].isin(val_matched)].copy()
    extra = gallery[~gallery["entity_id"].isin(val_matched)].sample(
        n=min(len(gallery_val) * 2, len(gallery) - len(gallery_val)),
        random_state=seed)
    gallery_val = pd.concat([gallery_val, extra], ignore_index=True)
    return s1_val, gallery_val, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="tests/world_2p5_split")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    s1_val, gallery_val, gt = build_val_split(args.world, args.seed)
    print(f"val: {len(s1_val):,} S1 | gallery {len(gallery_val):,}")
    print(f"device: {get_device()}")

    from sentence_transformers import SentenceTransformer
    for name in args.models:
        try:
            model = SentenceTransformer(name, device=get_device())
            metrics = recall_at_k(model, s1_val, gallery_val, gt)
            print(f"{name:45s} {metrics}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name:45s} FAILED: {exc}")


if __name__ == "__main__":
    main()
