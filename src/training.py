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
    candidates: Dict[int, set] = None,
    max_fallback_pool: int = 300,
) -> pd.DataFrame:
    """Generate hard negative pairs for training.

    Strategy:
    - 70% hard negatives: non-matching candidates from the blocking output
      (or, without candidates, same-country records ranked by name similarity)
    - 30% random negatives: different country where possible

    From ted-entity-resolution and name-matching.

    Scale note: when ``candidates`` ({s1_pos_idx: set(ids)}, the blocking
    output) is provided, hard negatives are sampled from each S1's own
    candidate pool. This avoids the naive O(n_s1 x n_gallery) full scan (which
    is ~6B fuzzy comparisons on the 2.5% world — hours) and matches the
    inference-time candidate distribution.
    """
    rng = np.random.default_rng(random_seed)
    s1_ids = s1_df["entity_id"].tolist()
    s1_index = {sid: i for i, sid in enumerate(s1_ids)}
    s1_name_by_id = dict(zip(s1_ids, s1_df["business_name_clean"].fillna("").tolist()))
    s1_country_by_id = dict(zip(s1_ids, s1_df["country_clean"].fillna("").tolist()))

    gallery_ids = s2_s3_df["entity_id"].to_numpy()
    gallery_country = s2_s3_df["country_clean"].fillna("").to_numpy()
    gallery_name = s2_s3_df["business_name_clean"].fillna("").to_numpy()
    id_to_row = {eid: i for i, eid in enumerate(gallery_ids)}
    by_country = {}
    for i, c in enumerate(gallery_country):
        by_country.setdefault(c, []).append(i)
    by_country = {c: np.array(v, dtype=np.int64) for c, v in by_country.items()}

    n_hard = int(n_neg_per_pos * hard_ratio)
    n_random = n_neg_per_pos - n_hard

    negatives = []
    for s1_id, matched_ids in ground_truth.items():
        q_idx = s1_index.get(s1_id)
        if q_idx is None:
            continue
        pos = set(matched_ids)
        chosen = set()

        # --- hard negatives ---
        if candidates is not None and candidates.get(q_idx):
            pool = [cid for cid in candidates[q_idx] if cid not in pos]
            if len(pool) > 200:  # bound the fuzzy ranking cost
                pool = list(rng.choice(pool, size=200, replace=False))
            if pool and n_hard > 0:
                s1_name = s1_name_by_id.get(s1_id, "")
                sims = np.array([
                    fuzz.WRatio(s1_name, gallery_name[id_to_row[cid]])
                    for cid in pool
                ])
                for i in np.argsort(-sims)[:n_hard]:
                    cid = pool[i]
                    if cid not in chosen:
                        negatives.append({
                            "s1_entity_id": s1_id,
                            "candidate_entity_id": cid,
                            "label": 0,
                        })
                        chosen.add(cid)
        elif max_fallback_pool > 0:
            # Bounded fallback (no blocking output available): sample a
            # same-country pool and keep the most similar records.
            rows = by_country.get(s1_country_by_id.get(s1_id, ""),
                                  np.array([], dtype=np.int64))
            if len(rows) > max_fallback_pool:
                rows = rng.choice(rows, size=max_fallback_pool, replace=False)
            pool = [gallery_ids[i] for i in rows if gallery_ids[i] not in pos]
            if pool and n_hard > 0:
                s1_name = s1_name_by_id.get(s1_id, "")
                sims = np.array([
                    fuzz.WRatio(s1_name, gallery_name[id_to_row[cid]])
                    for cid in pool
                ])
                for i in np.argsort(-sims)[:n_hard]:
                    cid = pool[i]
                    negatives.append({
                        "s1_entity_id": s1_id,
                        "candidate_entity_id": cid,
                        "label": 0,
                    })
                    chosen.add(cid)

        # --- random negatives (prefer different country) ---
        if n_random > 0 and len(gallery_ids) > 0:
            s1_country = s1_country_by_id.get(s1_id, "")
            got = 0
            attempts = 0
            while got < n_random and attempts < 40:
                attempts += 1
                i = int(rng.integers(0, len(gallery_ids)))
                cid = gallery_ids[i]
                if cid in pos or cid in chosen:
                    continue
                # Prefer a different country for the first ~15 attempts.
                if gallery_country[i] == s1_country and attempts <= 15:
                    continue
                negatives.append({
                    "s1_entity_id": s1_id,
                    "candidate_entity_id": cid,
                    "label": 0,
                })
                chosen.add(cid)
                got += 1

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
    candidates: Dict[int, set] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct training pairs with positives and hard negatives.

    ``candidates`` is the blocking output ({s1_pos_idx: set(ids)}) — when
    provided, hard negatives are sampled from it (scale-safe + matches the
    inference-time candidate distribution).

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
        candidates=candidates,
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


def _dense_cosines(
    pairs: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_s3_df: pd.DataFrame,
    embedding_cache: str,
) -> np.ndarray:
    """e5 cosine similarity for explicit (s1, candidate) pairs.

    Embeddings are cached on disk (same cache as the blocking scored-cap), so
    this is cheap after the first run. Chunked to bound memory.
    """
    from .dense_blocking import MULTILINGUAL_MODEL, encode_texts

    q_emb = encode_texts(
        s1_df["business_name_clean"].fillna("").tolist(),
        model_name=MULTILINGUAL_MODEL, cache_dir=embedding_cache,
        role="query", show_progress=False,
    )
    g_emb = encode_texts(
        s2_s3_df["business_name_clean"].fillna("").tolist(),
        model_name=MULTILINGUAL_MODEL, cache_dir=embedding_cache,
        role="target", show_progress=False,
    )
    s1_pos = {e: i for i, e in enumerate(s1_df["entity_id"].tolist())}
    s2_pos = {e: i for i, e in enumerate(s2_s3_df["entity_id"].tolist())}
    q_idx = np.fromiter((s1_pos[x] for x in pairs["s1_entity_id"]),
                        dtype=np.int64, count=len(pairs))
    t_idx = np.fromiter((s2_pos[x] for x in pairs["candidate_entity_id"]),
                        dtype=np.int64, count=len(pairs))

    cos = np.empty(len(pairs), dtype=np.float32)
    step = 200_000
    for start in range(0, len(pairs), step):
        end = min(start + step, len(pairs))
        cos[start:end] = np.einsum(
            "ij,ij->i", q_emb[q_idx[start:end]], g_emb[t_idx[start:end]]
        )
    return cos


def compute_pair_features(
    pairs: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_s3_df: pd.DataFrame,
    vectorize_threshold: int = 20_000,
    use_dense: bool = True,
    embedding_cache: str = "local_data/embeddings",
) -> pd.DataFrame:
    """Compute features for all pairs.

    Dispatches to the vectorized implementation for large inputs (scale) and
    the row-wise one for small inputs (exact, easy to debug). Both produce the
    same 36 features (verified by tests). ``name_dense_cosine`` is computed
    here (e5 embeddings) when ``use_dense`` is set; otherwise it stays 0.0.
    """
    from .features import (
        compute_all_features, compute_features_vectorized, FEATURE_NAMES,
    )

    if len(pairs) > vectorize_threshold:
        # Scale path: positional arrays + vectorized computation
        s1_ids = s1_df["entity_id"].tolist()
        s2_ids = s2_s3_df["entity_id"].tolist()
        s1_pos = {e: i for i, e in enumerate(s1_ids)}
        s2_pos = {e: i for i, e in enumerate(s2_ids)}
        idx_df = pd.DataFrame({
            "s1_idx": [s1_pos[x] for x in pairs["s1_entity_id"]],
            "s2_s3_idx": [s2_pos[x] for x in pairs["candidate_entity_id"]],
        })
        feats = compute_features_vectorized(
            idx_df,
            s1_df["business_name_clean"].fillna("").tolist(),
            s1_df["business_address_clean"].fillna("").tolist(),
            s1_df["country_clean"].fillna("").tolist(),
            s2_s3_df["business_name_clean"].fillna("").tolist(),
            s2_s3_df["business_address_clean"].fillna("").tolist(),
            s2_s3_df["country_clean"].fillna("").tolist(),
            s2_ids,
        )
        feats["label"] = pairs["label"].values
        feats["s1_entity_id"] = pairs["s1_entity_id"].values
        feats["candidate_entity_id"] = pairs["candidate_entity_id"].values
        if use_dense:
            try:
                feats["name_dense_cosine"] = _dense_cosines(
                    pairs, s1_df, s2_s3_df, embedding_cache)
            except Exception as exc:  # noqa: BLE001 — never block training
                print(f"  WARNING: dense feature unavailable ({exc}); using 0.0")
        return feats

    # Small path: row-wise (exact, debuggable)
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
    
    out = pd.DataFrame(feature_rows)
    if use_dense and len(out) > 0:
        try:
            out["name_dense_cosine"] = _dense_cosines(
                pairs, s1_df, s2_s3_df, embedding_cache)
        except Exception as exc:  # noqa: BLE001 — never block training
            print(f"  WARNING: dense feature unavailable ({exc}); using 0.0")
    return out
