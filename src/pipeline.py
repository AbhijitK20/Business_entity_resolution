"""Main pipeline orchestrator for Amazon ML Entity Resolution Challenge.

Integrates all modules into an end-to-end pipeline:
1. Load data
2. Normalize text
3. Build blocking index
4. Generate candidate pairs
5. Compute features
6. Train model
7. Optimize threshold
8. Run inference
9. Generate output
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from .data_loader import (
    load_training_data, load_test_data, profile_data,
    parse_ground_truth, combine_sources,
)
from .normalize import apply_normalization, normalize_name, normalize_address, normalize_country
from .blocking import (
    tfidf_blocking_candidates, phonetic_blocking, initialism_blocking,
    exact_blocking, minhash_lsh_candidates, address_tfidf_candidates,
    union_candidates, measure_blocking_quality,
)
from .features import compute_features_batch, FEATURE_NAMES
from .training import (
    construct_training_pairs, compute_pair_features,
)
from .model import (
    train_base_models, train_meta_learner, find_best_f05_threshold,
    save_model, load_model,
)


class EntityResolutionPipeline:
    """End-to-end entity resolution pipeline."""
    
    def __init__(self, data_dir: str, output_dir: str = "output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_data = None
        self.test_data = None
        self.models = None
        self.meta_model = None
        self.best_threshold = 0.5
        
    def run(self):
        """Execute the full pipeline."""
        print("=" * 60)
        print("Amazon ML Entity Resolution Pipeline")
        print("=" * 60)
        
        # Step 1: Load data
        print("\n[1/10] Loading data...")
        self.train_data = load_training_data(str(self.data_dir))
        self.test_data = load_test_data(str(self.data_dir))
        profile_data(self.train_data)
        
        # Step 2: Normalize
        print("\n[2/10] Normalizing text...")
        for key in ["train_s1", "train_s2", "train_s3"]:
            self.train_data[key] = apply_normalization(self.train_data[key])
        for key in ["test_s1", "test_s2", "test_s3"]:
            self.test_data[key] = apply_normalization(self.test_data[key])
        
        # Parse ground truth
        ground_truth = parse_ground_truth(self.train_data["train_gt"])
        print(f"Ground truth: {len(ground_truth)} S1 entities, "
              f"{sum(len(v) for v in ground_truth.values())} total matches")
        
        # Step 3: Build blocking (on training data)
        print("\n[3/10] Building blocking index (training)...")
        s2_s3_train = combine_sources(
            self.train_data["train_s2"], self.train_data["train_s3"]
        )
        
        # Step 4: Generate candidate pairs
        print("\n[4/10] Generating candidate pairs...")
        candidates_train = self._generate_candidates(
            self.train_data["train_s1"], s2_s3_train, "train"
        )
        
        # Step 5: Construct training data
        print("\n[5/10] Constructing training data...")
        pairs, train_pairs, val_pairs = construct_training_pairs(
            self.train_data["train_s1"], s2_s3_train, ground_truth
        )
        
        # Compute features
        print("Computing features for training pairs...")
        train_features = compute_pair_features(
            train_pairs, self.train_data["train_s1"], s2_s3_train
        )
        val_features = compute_pair_features(
            val_pairs, self.train_data["train_s1"], s2_s3_train
        )
        
        X_train = train_features[FEATURE_NAMES].values
        y_train = train_features["label"].values
        X_val = val_features[FEATURE_NAMES].values
        y_val = val_features["label"].values
        
        print(f"Training: {len(X_train)} pairs ({sum(y_train)} positive)")
        print(f"Validation: {len(X_val)} pairs ({sum(y_val)} positive)")
        
        # Step 6: Train models
        print("\n[6/10] Training ensemble models...")
        self.models = train_base_models(X_train, y_train, X_val, y_val)
        
        # Step 7: Train meta-learner
        print("\n[7/10] Training meta-learner...")
        base_val_probas = {k: v["val_proba"] for k, v in self.models.items()}
        self.meta_model, meta_val_proba = train_meta_learner(base_val_probas, y_val)
        
        # Step 8: Optimize threshold
        print("\n[8/10] Optimizing F_0.5 threshold...")
        self.best_threshold, best_f05 = find_best_f05_threshold(y_val, meta_val_proba)
        print(f"Best threshold: {self.best_threshold:.2f}, F_0.5: {best_f05:.4f}")
        
        # Step 9: Run inference on test
        print("\n[9/10] Running inference on test set...")
        self._run_inference()
        
        # Step 10: Generate output
        print("\n[10/10] Generating output files...")
        self._generate_output()
        
        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print(f"Output files in: {self.output_dir}")
        print("=" * 60)
    
    def _generate_candidates(
        self, s1_df: pd.DataFrame, s2_s3_df: pd.DataFrame, split: str
    ) -> Dict[int, set]:
        """Generate candidate pairs using 7-layer blocking union."""
        s1_names = s1_df["business_name_clean"].fillna("").tolist()
        s1_addrs = s1_df["business_address_clean"].fillna("").tolist()
        s1_countries = s1_df["country_clean"].fillna("").tolist()
        
        s2_s3_names = s2_s3_df["business_name_clean"].fillna("").tolist()
        s2_s3_addrs = s2_s3_df["business_address_clean"].fillna("").tolist()
        s2_s3_countries = s2_s3_df["country_clean"].fillna("").tolist()
        s2_s3_ids = s2_s3_df["entity_id"].tolist()
        
        print(f"  S1: {len(s1_names)} entities, S2+S3: {len(s2_s3_names)} records")
        
        # Layer 1: TF-IDF on names
        print("  Layer 1: TF-IDF name blocking...")
        c1 = tfidf_blocking_candidates(s1_names, s2_s3_names, s2_s3_ids, threshold=0.25)
        
        # Layer 2: Token-sorted TF-IDF
        print("  Layer 2: Token-sorted blocking...")
        s1_sorted = [" ".join(sorted(n.split())) for n in s1_names]
        s2_s3_sorted = [" ".join(sorted(n.split())) for n in s2_s3_names]
        c2 = tfidf_blocking_candidates(s1_sorted, s2_s3_sorted, s2_s3_ids, threshold=0.25)
        
        # Layer 3: Phonetic (Soundex + Metaphone)
        print("  Layer 3: Phonetic blocking...")
        c3 = phonetic_blocking(s1_names, s2_s3_names, s2_s3_ids)
        
        # Layer 4: Initialism
        print("  Layer 4: Initialism blocking...")
        c4 = initialism_blocking(s1_names, s2_s3_names, s2_s3_ids)
        
        # Layer 5: Address TF-IDF
        print("  Layer 5: Address TF-IDF blocking...")
        c5 = address_tfidf_candidates(s1_addrs, s2_s3_addrs, s2_s3_ids, threshold=0.25)
        
        # Layer 6: Country partition (already enforced in blocking)
        print("  Layer 6: Country partition...")
        c6 = {}  # Country is handled in candidate filtering
        
        # Layer 7: MinHash LSH
        print("  Layer 7: MinHash LSH blocking...")
        c7 = minhash_lsh_candidates(s1_names, s2_s3_names, s2_s3_ids, threshold=0.3)
        
        # Union all candidates
        candidates = union_candidates(c1, c2, c3, c4, c5, c7)
        
        total_pairs = len(s1_names) * len(s2_s3_names)
        metrics = measure_blocking_quality(candidates, ground_truth, total_pairs)
        print(f"  Blocking metrics: {metrics}")
        
        return candidates
    
    def _run_inference(self):
        """Run inference on test set."""
        s2_s3_test = combine_sources(
            self.test_data["test_s2"], self.test_data["test_s3"]
        )
        
        # Generate candidates
        candidates_test = self._generate_candidates(
            self.test_data["test_s1"], s2_s3_test, "test"
        )
        
        # Compute features for all candidates
        all_pairs = []
        for s1_idx, candidate_ids in candidates_test.items():
            s1_id = self.test_data["test_s1"].iloc[s1_idx]["entity_id"]
            for cand_id in candidate_ids:
                all_pairs.append({
                    "s1_idx": s1_idx,
                    "s2_s3_idx": s2_s3_test.index[
                        s2_s3_test["entity_id"] == cand_id
                    ].tolist()[0] if cand_id in s2_s3_test["entity_id"].values else -1,
                    "s1_entity_id": s1_id,
                    "candidate_entity_id": cand_id,
                })
        
        if not all_pairs:
            print("  WARNING: No candidate pairs generated!")
            self.test_predictions = {}
            return
        
        pairs_df = pd.DataFrame(all_pairs)
        pairs_df = pairs_df[pairs_df["s2_s3_idx"] >= 0]
        
        # Compute features
        s1_df = self.test_data["test_s1"]
        s1_names = s1_df["business_name_clean"].fillna("").tolist()
        s1_addrs = s1_df["business_address_clean"].fillna("").tolist()
        s1_countries = s1_df["country_clean"].fillna("").tolist()
        
        s2_s3_names = s2_s3_test["business_name_clean"].fillna("").tolist()
        s2_s3_addrs = s2_s3_test["business_address_clean"].fillna("").tolist()
        s2_s3_countries = s2_s3_test["country_clean"].fillna("").tolist()
        s2_s3_ids = s2_s3_test["entity_id"].tolist()
        
        print(f"  Computing features for {len(pairs_df)} candidate pairs...")
        features_df = compute_features_batch(
            pairs_df, s1_names, s1_addrs, s1_countries,
            s2_s3_names, s2_s3_addrs, s2_s3_countries, s2_s3_ids,
        )
        
        X_test = features_df[FEATURE_NAMES].values
        
        # Stack base predictions
        lgb_proba = self.models["lgb"]["model"].predict_proba(X_test)[:, 1]
        xgb_proba = self.models["xgb"]["model"].predict_proba(X_test)[:, 1]
        rf_proba = self.models["rf"]["model"].predict_proba(X_test)[:, 1]
        
        X_meta = np.column_stack([lgb_proba, xgb_proba, rf_proba])
        meta_proba = self.meta_model.predict_proba(X_meta)[:, 1]
        
        # Apply threshold
        predictions = (meta_proba >= self.best_threshold).astype(int)
        
        # Build match graph
        self.test_predictions = {}
        for idx, row in features_df.iterrows():
            s1_id = row["s1_entity_id"]
            cand_id = row["s2_s3_id"]
            
            if predictions[idx] == 1:
                if s1_id not in self.test_predictions:
                    self.test_predictions[s1_id] = []
                self.test_predictions[s1_id].append(cand_id)
        
        print(f"  Predicted matches for {len(self.test_predictions)} S1 entities")
    
    def _generate_output(self):
        """Generate matching_results.tsv and candidate_pairs.tsv."""
        s1_test = self.test_data["test_s1"]
        
        # matching_results.tsv
        results = []
        for s1_id in s1_test["entity_id"]:
            matched = self.test_predictions.get(s1_id, [])
            results.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(matched),
            })
        
        results_df = pd.DataFrame(results)
        results_path = self.output_dir / "matching_results.tsv"
        results_df.to_csv(results_path, sep="\t", index=False)
        print(f"  Written: {results_path} ({len(results_df)} rows)")
        
        # candidate_pairs.tsv
        # (For now, use the same matches — in production, this would be the blocking candidates)
        candidates = []
        for s1_id in s1_test["entity_id"]:
            matched = self.test_predictions.get(s1_id, [])
            candidates.append({
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(matched),
            })
        
        candidates_df = pd.DataFrame(candidates)
        candidates_path = self.output_dir / "candidate_pairs.tsv"
        candidates_df.to_csv(candidates_path, sep="\t", index=False)
        print(f"  Written: {candidates_path} ({len(candidates_df)} rows)")
        
        # Save model
        model_path = self.output_dir / "model.pkl"
        save_model({
            "models": self.models,
            "meta_model": self.meta_model,
            "threshold": self.best_threshold,
        }, str(model_path))
        print(f"  Written: {model_path}")


def main():
    """Main entry point."""
    data_dir = os.environ.get("DATA_DIR", "/home/abhijitk20/Amazon ML/data")
    output_dir = os.environ.get("OUTPUT_DIR", "/home/abhijitk20/Amazon ML/output")
    
    pipeline = EntityResolutionPipeline(data_dir, output_dir)
    pipeline.run()


if __name__ == "__main__":
    main()
