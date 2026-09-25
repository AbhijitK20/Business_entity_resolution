"""Data loader module for Amazon ML Entity Resolution Challenge."""
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional


def load_training_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all training TSV files."""
    train_dir = Path(data_dir) / "dataset" / "train"
    
    train_s1 = pd.read_csv(train_dir / "train_source1.tsv", sep="\t")
    train_s2 = pd.read_csv(train_dir / "train_source2.tsv", sep="\t")
    train_s3 = pd.read_csv(train_dir / "train_source3.tsv", sep="\t")
    train_gt = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t")
    
    return {
        "train_s1": train_s1,
        "train_s2": train_s2,
        "train_s3": train_s3,
        "train_gt": train_gt,
    }


def load_test_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all test TSV files."""
    test_dir = Path(data_dir) / "dataset" / "test"
    
    test_s1 = pd.read_csv(test_dir / "test_source1.tsv", sep="\t")
    test_s2 = pd.read_csv(test_dir / "test_source2.tsv", sep="\t")
    test_s3 = pd.read_csv(test_dir / "test_source3.tsv", sep="\t")
    
    return {
        "test_s1": test_s1,
        "test_s2": test_s2,
        "test_s3": test_s3,
    }


def profile_data(data: Dict[str, pd.DataFrame]) -> None:
    """Print data profile statistics."""
    for name, df in data.items():
        print(f"\n=== {name} ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        print(f"\nNull counts:")
        print(df.isnull().sum())
        if "country" in df.columns:
            print(f"\nCountry distribution:")
            print(df["country"].value_counts())


def parse_ground_truth(gt_df: pd.DataFrame) -> Dict[str, list]:
    """Parse ground truth into {s1_entity_id: [matched_ids]} dict."""
    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        if pd.isna(matched) or matched == "":
            gt_dict[s1_id] = []
        else:
            gt_dict[s1_id] = [x.strip() for x in str(matched).split(",")]
    return gt_dict


def combine_sources(s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    """Combine S2 and S3 into one pool with source tracking."""
    s2_copy = s2.copy()
    s3_copy = s3.copy()
    s2_copy["source"] = "S2"
    s3_copy["source"] = "S3"
    return pd.concat([s2_copy, s3_copy], ignore_index=True)
