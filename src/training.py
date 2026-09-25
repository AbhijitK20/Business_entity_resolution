"""Training data construction — hard negative mining.

Based on research from:
- name-matching (hard negative mining)
- ted-entity-resolution (70/30 hard/random split)
- LLM4ER (hard negatives more important than more data)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from rapidfuzz import fuzz


def generate_hard_negatives(
    s1_df: pd.DataFrame,
    s2_s3_df: pd.DataFrame,
    ground_truth: Dict[str, list],
    n_neg_per_pos: int = 2,
    hard_ratio: float = 0.7,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate hard negative pairs for training.
    
    Strategy:
    - 70% hard negatives: same country, similar name, NOT in positives
    - 30% random negatives: different country or very different name
    
    From ted-entity-resolution and name-matching.
    """
    np.random.seed(random_seed)
    
    negatives = []
    
    for s1_id, matched_ids in ground_truth.items():
        # Get S1 record
        s1_mask = s1_df["entity_id"] == s1_id
        if not s1_mask.any():
            continue
        s1_record = s1_df[s1_mask].iloc[0]
        s1_name = s1_record.get("business_name_clean", "")
        s1_country = s1_record.get("country_clean", "")
        
        # Pool of candidates (exclude positives; for singletons, exclude nothing)
        positive_set = set(matched_ids)
        candidates = s2_s3_df[~s2_s3_df["entity_id"].isin(positive_set)].copy()
        
        if len(candidates) == 0:
            continue
        
        # Hard negatives: same country, compute name similarity
        n_hard = int(n_neg_per_pos * hard_ratio)
        n_random = n_neg_per_pos - n_hard
        
        same_country = candidates[candidates["country_clean"] == s1_country]
        
        if len(same_country) > 0:
            # Compute similarity scores
            sims = same_country["business_name_clean"].apply(
                lambda x: fuzz.WRatio(s1_name, x) if pd.notna(x) else 0
            )
            same_country = same_country.copy()
            same_country["sim"] = sims.values
            same_country = same_country.sort_values("sim", ascending=False)
            
            # Take top N hardest
            hard_candidates = same_country.head(n_hard)
            for _, cand in hard_candidates.iterrows():
                negatives.append({
                    "s1_entity_id": s1_id,
                    "candidate_entity_id": cand["entity_id"],
                    "label": 0,
                })
        
        # Random negatives: different country
        if n_random > 0:
            diff_country = candidates[candidates["country_clean"] != s1_country]
            if len(diff_country) >= n_random:
                random_sample = diff_country.sample(n_random, random_state=random_seed)
                for _, cand in random_sample.iterrows():
                    negatives.append({
                        "s1_entity_id": s1_id,
                        "candidate_entity_id": cand["entity_id"],
                        "label": 0,
                    })
            elif len(diff_country) > 0:
                for _, cand in diff_country.iterrows():
                    negatives.append({
                        "s1_entity_id": s1_id,
                        "candidate_entity_id": cand["entity_id"],
                        "label": 0,
                    })
    
    return pd.DataFrame(negatives)


def generate_positive_pairs(
    ground_truth: Dict[str, list],
) -> pd.DataFrame:
    """Generate positive pairs from ground truth."""
    positives = []
    for s1_id, matched_ids in ground_truth.items():
        for matched_id in matched_ids:
            positives.append({
                "s1_entity_id": s1_id,
                "candidate_entity_id": matched_id,
                "label": 1,
            })
    return pd.DataFrame(positives)


def construct_training_pairs(
    s1_df: pd.DataFrame,
    s2_s3_df: pd.DataFrame,
    ground_truth: Dict[str, list],
    n_neg_per_pos: int = 2,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct training pairs with positives and hard negatives.
    
    Returns:
        pairs_df: DataFrame with [s1_entity_id, candidate_entity_id, label]
        splits: DataFrame with train/val split information
    """
    # Generate positives
    positives = generate_positive_pairs(ground_truth)
    
    # Generate hard negatives
    negatives = generate_hard_negatives(
        s1_df, s2_s3_df, ground_truth,
        n_neg_per_pos=n_neg_per_pos,
        random_seed=random_seed,
    )
    
    # Combine
    pairs = pd.concat([positives, negatives], ignore_index=True)
    pairs = pairs.drop_duplicates(subset=["s1_entity_id", "candidate_entity_id"])
    
    # Stratified split — guard for tiny datasets (smoke tests)
    from sklearn.model_selection import train_test_split
    label_counts = pairs["label"].value_counts()
    can_stratify = (
        len(pairs) >= 10
        and len(label_counts) == 2
        and label_counts.min() >= 2
    )
    train_pairs, val_pairs = train_test_split(
        pairs,
        test_size=0.2,
        stratify=pairs["label"] if can_stratify else None,
        random_state=random_seed,
    )
    
    return pairs, train_pairs, val_pairs


def compute_pair_features(
    pairs: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_s3_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute features for all pairs."""
    from .features import compute_all_features, FEATURE_NAMES
    
    # Build lookup dicts
    s1_lookup = {}
    for _, row in s1_df.iterrows():
        s1_lookup[row["entity_id"]] = {
            "name": row.get("business_name_clean", ""),
            "addr": row.get("business_address_clean", ""),
            "country": row.get("country_clean", ""),
        }
    
    s2_s3_lookup = {}
    for _, row in s2_s3_df.iterrows():
        s2_s3_lookup[row["entity_id"]] = {
            "name": row.get("business_name_clean", ""),
            "addr": row.get("business_address_clean", ""),
            "country": row.get("country_clean", ""),
        }
    
    # Compute features
    feature_rows = []
    for _, row in pairs.iterrows():
        s1_id = row["s1_entity_id"]
        cand_id = row["candidate_entity_id"]
        
        s1 = s1_lookup.get(s1_id, {"name": "", "addr": "", "country": ""})
        cand = s2_s3_lookup.get(cand_id, {"name": "", "addr": "", "country": ""})
        
        feats = compute_all_features(
            s1["name"], cand["name"],
            s1["addr"], cand["addr"],
            s1["country"], cand["country"],
        )
        feats["label"] = row["label"]
        feats["s1_entity_id"] = s1_id
        feats["candidate_entity_id"] = cand_id
        
        feature_rows.append(feats)
    
    return pd.DataFrame(feature_rows)
