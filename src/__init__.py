"""Amazon ML Entity Resolution Pipeline."""
from .data_loader import load_training_data, load_test_data
from .normalize import normalize_name, normalize_address
from .blocking import union_candidates
from .features import compute_all_features, FEATURE_NAMES
from .training import construct_training_pairs
from .model import train_base_models, find_best_f05_threshold
from .pipeline import EntityResolutionPipeline
