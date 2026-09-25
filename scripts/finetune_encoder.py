"""Fine-tune a license-safe encoder for entity matching (GPU).

Competition rules allow training on the provided data — this fine-tunes the
dense encoder on competition train pairs only (no external data).

Protocol:
  - positives: (S1 name, matched gallery name) from ground truth
  - negatives: random same-country gallery records + in-batch negatives
    (MultipleNegativesRankingLoss)
  - entity-level train/val split (no leakage)
  - eval: recall@20/30 on held-out entities

Usage:
    python scripts/finetune_encoder.py --world tests/world_2p5_split \
        --base intfloat/multilingual-e5-small --epochs 2 --batch 64 \
        --out models/e5-er-ft
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_training_data, parse_ground_truth, combine_sources
from src.normalize import apply_normalization
from src.dense_blocking import get_device, topk_search


def build_training_triplets(s1, gallery, gt, rng, max_pairs=None):
    """(anchor, positive, negative) triplets with same-country random negatives."""
    g = gallery.set_index("entity_id")
    gallery_ids = gallery["entity_id"].to_numpy()
    gallery_country = gallery["country_clean"].fillna("").to_numpy()
    by_country = {}
    for i, c in enumerate(gallery_country):
        by_country.setdefault(c, []).append(i)
    by_country = {c: np.array(v, dtype=np.int64) for c, v in by_country.items()}

    triplets = []
    for _, row in s1.iterrows():
        sid = row["entity_id"]
        matched = gt.get(sid, [])
        if not matched:
            continue
        anchor = row["business_name_clean"] or ""
        if not anchor.strip():
            continue
        country = row.get("country_clean", "") or ""
        for mid in matched:
            if mid not in g.index:
                continue
            positive = g.loc[mid, "business_name_clean"] or ""
            if not positive.strip():
                continue
            # negative: random same-country record (fallback: any)
            rows = by_country.get(country, np.array([], dtype=np.int64))
            neg = None
            for _ in range(10):
                if len(rows) == 0:
                    break
                cand_id = gallery_ids[int(rng.choice(rows))]
                if cand_id not in matched:
                    neg = g.loc[cand_id, "business_name_clean"] or ""
                    break
            if neg is None or not neg.strip():
                cand_id = gallery_ids[int(rng.integers(0, len(gallery_ids)))]
                if cand_id in matched:
                    continue
                neg = g.loc[cand_id, "business_name_clean"] or ""
            triplets.append((anchor, positive, neg))
            if max_pairs and len(triplets) >= max_pairs:
                return triplets
    return triplets


def recall_at_k(model, s1_val, gallery_val, gt, ks=(20, 30), batch_size=128):
    from src.dense_blocking import encode_texts
    q_names = s1_val["business_name_clean"].fillna("").tolist()
    g_names = gallery_val["business_name_clean"].fillna("").tolist()
    # encode directly (no cache — model changes each checkpoint)
    q_emb = model.encode(q_names, batch_size=batch_size,
                         normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False).astype(np.float32)
    g_emb = model.encode(g_names, batch_size=batch_size,
                         normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False).astype(np.float32)
    gid = np.array(gallery_val["entity_id"].tolist())
    idx, _ = topk_search(q_emb, g_emb, top_k=max(ks))
    out = {}
    for k in ks:
        kept = total = 0
        for i, sid in enumerate(s1_val["entity_id"].tolist()):
            matched = gt.get(sid, [])
            total += len(matched)
            if not matched:
                continue
            cand = {gid[int(t)] for t in idx[i, :k] if t >= 0}
            kept += len(set(matched) & cand)
        out[f"recall@{k}"] = round(kept / max(total, 1), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="tests/world_2p5_split")
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="cap training triplets (debug/speed)")
    ap.add_argument("--out", default="models/e5-er-ft")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    print("=" * 60)
    print(f"FINE-TUNE ENCODER — base={args.base} device={get_device()}")
    print("=" * 60)

    data = load_training_data(args.world)
    s1 = apply_normalization(data["train_s1"])
    s2 = apply_normalization(data["train_s2"])
    s3 = apply_normalization(data["train_s3"])
    gallery = combine_sources(s2, s3)
    gt = parse_ground_truth(data["train_gt"])

    # Entity-level split
    entities = s1["entity_id"].to_numpy()
    rng_np = np.random.default_rng(args.seed)
    perm = rng_np.permutation(len(entities))
    n_val = max(1, int(len(entities) * 0.1))
    val_ids = set(entities[perm[:n_val]])
    train_ids = set(entities[perm[n_val:]])
    s1_train = s1[s1["entity_id"].isin(train_ids)]
    s1_val = s1[s1["entity_id"].isin(val_ids)]
    # Gallery for val: records matching val entities + a distractor sample
    val_matched = {m for sid in val_ids for m in gt.get(sid, [])}
    gallery_val = gallery[gallery["entity_id"].isin(val_matched)].copy()
    extra = gallery[~gallery["entity_id"].isin(val_matched)].sample(
        n=min(len(gallery_val) * 2, len(gallery) - len(gallery_val)),
        random_state=args.seed)
    gallery_val = pd.concat([gallery_val, extra], ignore_index=True)
    print(f"[data] train S1={len(s1_train):,} val S1={len(s1_val):,} "
          f"val gallery={len(gallery_val):,}")

    t0 = time.time()
    triplets = build_training_triplets(s1_train, gallery, gt, rng,
                                       max_pairs=args.max_pairs)
    print(f"[triplets] {len(triplets):,} built in {time.time()-t0:.1f}s")

    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    model = SentenceTransformer(args.base, device=get_device())
    train_examples = [
        InputExample(texts=[a, p, n]) for a, p, n in triplets
    ]
    loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch)
    loss = losses.MultipleNegativesRankingLoss(model)

    print(f"[train] {args.epochs} epochs x {len(train_examples):,} triplets "
          f"(batch={args.batch})")
    t0 = time.time()
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=int(len(loader) * args.epochs * 0.1),
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
    )
    print(f"[train] done in {time.time()-t0:.1f}s")

    print("[eval] held-out entities...")
    metrics = recall_at_k(model, s1_val, gallery_val, gt)
    print(f"[eval] {metrics}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    meta = {
        "base": args.base, "epochs": args.epochs, "batch": args.batch,
        "lr": args.lr, "triplets": len(triplets), "eval": metrics,
        "world": args.world, "seed": args.seed,
    }
    with open(out / "finetune_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[saved] {out} (+ finetune_meta.json)")


if __name__ == "__main__":
    main()
