"""Smoke test — validates the pipeline end-to-end on video-derived fixtures.

Run: python tests/test_smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_training_data, parse_ground_truth, combine_sources
from src.normalize import apply_normalization
from src.blocking import (
    tfidf_blocking_candidates, phonetic_blocking, initialism_blocking,
    minhash_lsh_candidates, address_tfidf_candidates, union_candidates,
)
from src.features import compute_all_features, FEATURE_NAMES
from src.training import construct_training_pairs, compute_pair_features

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def main():
    print("=" * 60)
    print("SMOKE TEST — Pipeline validation on video fixtures")
    print("=" * 60)

    # Step 1: Load
    print("\n[1] Loading fixture data...")
    data = load_training_data(str(FIXTURE_DIR))
    s1 = apply_normalization(data["train_s1"])
    s2 = apply_normalization(data["train_s2"])
    s3 = apply_normalization(data["train_s3"])
    gt = parse_ground_truth(data["train_gt"])
    print(f"    S1: {len(s1)} | S2: {len(s2)} | S3: {len(s3)} | GT: {len(gt)}")

    s2_s3 = combine_sources(s2, s3)
    print(f"    Combined S2+S3 pool: {len(s2_s3)}")

    # Step 2: Normalization check
    print("\n[2] Normalization check...")
    for _, row in s1.iterrows():
        print(f"    {row['business_name']:30s} -> {row['business_name_clean']}")

    # Step 3: Blocking
    print("\n[3] Blocking (union of strategies)...")
    s1_names = s1["business_name_clean"].fillna("").tolist()
    s1_addrs = s1["business_address_clean"].fillna("").tolist()
    s2_names = s2_s3["business_name_clean"].fillna("").tolist()
    s2_addrs = s2_s3["business_address_clean"].fillna("").tolist()
    s2_ids = s2_s3["entity_id"].tolist()

    c1 = tfidf_blocking_candidates(s1_names, s2_names, s2_ids, threshold=0.2)
    c3 = phonetic_blocking(s1_names, s2_names, s2_ids)
    c4 = initialism_blocking(s1_names, s2_names, s2_ids)
    c5 = address_tfidf_candidates(s1_addrs, s2_addrs, s2_ids, threshold=0.2)
    c7 = minhash_lsh_candidates(s1_names, s2_names, s2_ids, threshold=0.3)
    candidates = union_candidates(c1, c3, c4, c5, c7)

    total_pairs = len(s1_names) * len(s2_names)
    n_cand = sum(len(v) for v in candidates.values())
    print(f"    Candidates: {n_cand} / {total_pairs} possible pairs")
    print(f"    Reduction: {1 - n_cand/max(total_pairs,1):.2%}")

    # Blocking recall check
    retained, total = 0, 0
    for s1_idx, s1_id in enumerate(s1["entity_id"]):
        truth = set(gt.get(s1_id, []))
        total += len(truth)
        retained += len(truth & candidates.get(s1_idx, set()))
    print(f"    Blocking recall: {retained}/{total} = {retained/max(total,1):.2%}")
    assert retained == total, f"BLOCKING MISSED {total - retained} true matches!"

    # Step 4: Features
    print("\n[4] Feature computation...")
    feats = compute_all_features(
        "acme robotics", "acme robotics", "500 market st san jose", "500 market street san jose", "us", "us"
    )
    assert len(feats) == len(FEATURE_NAMES), f"Expected {len(FEATURE_NAMES)} features, got {len(feats)}"
    print(f"    {len(feats)} features computed OK")
    print(f"    Accurate-match pair: name_WRatio={feats['name_WRatio']:.2f}, same_country={feats['same_country']}")

    bad_feats = compute_all_features(
        "acme robotics", "acme bakery", "500 market st san jose", "500 market st san jose", "us", "us"
    )
    print(f"    Look-alike pair:     name_WRatio={bad_feats['name_WRatio']:.2f}, same_country={bad_feats['same_country']}")

    # Step 5: Training pairs
    print("\n[5] Training pair construction...")
    pairs, train_pairs, val_pairs = construct_training_pairs(s1, s2_s3, gt, n_neg_per_pos=2)
    print(f"    Total pairs: {len(pairs)} | train: {len(train_pairs)} | val: {len(val_pairs)}")
    print(f"    Positives: {pairs['label'].sum()} | Negatives: {(pairs['label']==0).sum()}")

    # Step 6: Pair features
    print("\n[6] Pair feature matrix...")
    train_feats = compute_pair_features(train_pairs, s1, s2_s3)
    print(f"    Feature matrix: {train_feats.shape}")
    assert train_feats[FEATURE_NAMES].isna().sum().sum() == 0, "NaN in features!"

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
