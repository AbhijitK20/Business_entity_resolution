"""Model training module — dual-layer ensemble stacking with PROPER OOF.

Key correctness property (leak-free stacking):
  1. Optuna-tune each base model with inner CV on X_train.
  2. Generate OUT-OF-FOLD predictions on X_train (each sample predicted by
     a model that never saw it).
  3. Train the meta-learner on OOF predictions (never on validation data).
  4. Retrain base models on the FULL X_train for inference.
  5. Evaluate the whole stack on X_val — which the meta-learner has never seen.

Based on research from:
- MetaBoost (dual-layer stacking, shallow meta-learner)
- imbalance-benchmark (threshold optimization)
- ted-entity-resolution (conservative LightGBM params)
"""
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Tuple, Optional, List
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score,
)
import optuna
from .features import FEATURE_NAMES


# --------------------------------------------------------------------- utils
def make_cv_splits(y: np.ndarray, n_splits: int = 3, random_seed: int = 42):
    """Create CV splits that survive tiny datasets.

    Preference: StratifiedKFold -> KFold -> single holdout.
    """
    y = np.asarray(y)
    n = len(y)
    labels, counts = np.unique(y, return_counts=True)

    if len(labels) == 2 and counts.min() >= n_splits and n >= 2 * n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        return list(splitter.split(np.zeros(n), y))

    if n >= 2 * n_splits:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        return list(splitter.split(np.zeros(n)))

    if n >= 2:
        cut = max(1, int(n * 0.7))
        return [(np.arange(cut), np.arange(cut, n))]

    return [(np.arange(n), np.arange(n))]


def _safe_ap(y_true: np.ndarray, proba: np.ndarray) -> float:
    """average_precision_score that tolerates single-class folds."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return average_precision_score(y_true, proba)


def _oof_predictions(model_factory, X_train, y_train, n_folds, random_seed) -> np.ndarray:
    """Out-of-fold predictions on X_train."""
    oof = np.zeros(len(X_train))
    for tr_idx, va_idx in make_cv_splits(y_train, n_folds, random_seed):
        model = model_factory()
        model.fit(X_train[tr_idx], y_train[tr_idx])
        if len(va_idx):
            oof[va_idx] = model.predict_proba(X_train[va_idx])[:, 1]
    return oof


# ------------------------------------------------------- hyperparameter tuning
def _tune_lightgbm(X_train, y_train, n_trials, random_seed) -> Dict:
    def objective(trial):
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "class_weight": "balanced",
            "num_leaves": trial.suggest_int("num_leaves", 16, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 500, step=100),
            "min_child_samples": trial.suggest_int("min_child_samples", 80, 200),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
            "force_col_wise": True,
            "verbose": -1,
            "random_state": random_seed,
        }
        scores = []
        for tr, va in make_cv_splits(y_train, 3, random_seed):
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train[tr], y_train[tr])
            if len(va):
                scores.append(_safe_ap(y_train[va], model.predict_proba(X_train[va])[:, 1]))
        return float(np.mean(scores)) if scores else 0.5

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    params = study.best_params
    params.update({
        "objective": "binary", "boosting_type": "gbdt", "class_weight": "balanced",
        "force_col_wise": True, "verbose": -1, "random_state": random_seed,
    })
    return params


def _tune_xgboost(X_train, y_train, n_trials, random_seed) -> Dict:
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 500, step=100),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
            "scale_pos_weight": len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
            "verbosity": 0,
            "random_state": random_seed,
        }
        scores = []
        for tr, va in make_cv_splits(y_train, 3, random_seed):
            model = xgb.XGBClassifier(**params)
            model.fit(X_train[tr], y_train[tr])
            if len(va):
                scores.append(_safe_ap(y_train[va], model.predict_proba(X_train[va])[:, 1]))
        return float(np.mean(scores)) if scores else 0.5

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    params = study.best_params
    params.update({
        "objective": "binary:logistic", "eval_metric": "logloss", "verbosity": 0,
        "random_state": random_seed,
        "scale_pos_weight": len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
    })
    return params


def _tune_random_forest(X_train, y_train, n_trials, random_seed) -> Dict:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": random_seed,
        }
        scores = []
        for tr, va in make_cv_splits(y_train, 3, random_seed):
            model = RandomForestClassifier(**params)
            model.fit(X_train[tr], y_train[tr])
            if len(va):
                scores.append(_safe_ap(y_train[va], model.predict_proba(X_train[va])[:, 1]))
        return float(np.mean(scores)) if scores else 0.5

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=min(n_trials, 15), show_progress_bar=False)
    params = study.best_params
    params.update({
        "class_weight": "balanced_subsample", "n_jobs": -1, "random_state": random_seed,
    })
    return params


# ------------------------------------------------------------ ensemble training
def train_base_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 25,
    random_seed: int = 42,
    n_folds: int = 5,
) -> Dict:
    """Tune + train 3 base models and produce OOF + validation predictions.

    Returns dict: {name: {model, oof, val_proba, best_params, val_auc, val_ap}}
    """
    builders = {
        "lgb": (lambda p: lgb.LGBMClassifier(**p), _tune_lightgbm),
        "xgb": (lambda p: xgb.XGBClassifier(**p), _tune_xgboost),
        "rf": (lambda p: RandomForestClassifier(**p), _tune_random_forest),
    }

    results = {}
    for name, (factory, tuner) in builders.items():
        print(f"Tuning {name.upper()}...")
        params = tuner(X_train, y_train, n_trials, random_seed)

        print(f"  Generating OOF predictions ({name.upper()}, {n_folds}-fold)...")
        oof = _oof_predictions(lambda: factory(params), X_train, y_train, n_folds, random_seed)

        print(f"  Fitting final {name.upper()} on full train...")
        final_model = factory(params)
        final_model.fit(X_train, y_train)
        val_proba = final_model.predict_proba(X_val)[:, 1]

        results[name] = {
            "model": final_model,
            "oof": oof,
            "val_proba": val_proba,
            "best_params": params,
            "val_auc": roc_auc_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else 0.5,
            "val_ap": _safe_ap(y_val, val_proba),
        }
        print(f"  {name.upper()} val AP: {results[name]['val_ap']:.4f}")

    return results


def train_meta_learner(
    base_results: Dict,
    y_train: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 20,
    random_seed: int = 42,
) -> Tuple[object, np.ndarray, np.ndarray]:
    """Train the meta-learner on OOF predictions; evaluate on validation.

    The meta-learner NEVER sees validation data during training.

    Returns (meta_model, oof_meta_proba, val_meta_proba).
    """
    base_names = ["lgb", "xgb", "rf"]
    X_meta_train = np.column_stack([base_results[n]["oof"] for n in base_names])
    X_meta_val = np.column_stack([base_results[n]["val_proba"] for n in base_names])

    def objective(trial):
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "class_weight": "balanced",
            "num_leaves": trial.suggest_int("num_leaves", 4, 16),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5),
            "force_col_wise": True,
            "verbose": -1,
            "random_state": random_seed,
        }
        scores = []
        for tr, va in make_cv_splits(y_train, 3, random_seed):
            model = lgb.LGBMClassifier(**params)
            model.fit(X_meta_train[tr], y_train[tr])
            if len(va):
                scores.append(_safe_ap(y_train[va], model.predict_proba(X_meta_train[va])[:, 1]))
        return float(np.mean(scores)) if scores else 0.5

    print("Tuning meta-learner...")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    params = study.best_params
    params.update({
        "objective": "binary", "boosting_type": "gbdt", "class_weight": "balanced",
        "force_col_wise": True, "verbose": -1, "random_state": random_seed,
    })

    meta_model = lgb.LGBMClassifier(**params)
    meta_model.fit(X_meta_train, y_train)

    oof_meta_proba = meta_model.predict_proba(X_meta_train)[:, 1]
    val_meta_proba = meta_model.predict_proba(X_meta_val)[:, 1]

    print(f"  meta OOF AP : {_safe_ap(y_train, oof_meta_proba):.4f}")
    print(f"  meta val AP : {_safe_ap(y_val, val_meta_proba):.4f}")

    return meta_model, oof_meta_proba, val_meta_proba


# ---------------------------------------------------------------- thresholds
def find_best_f05_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """Find threshold that maximizes pair-level F_0.5."""
    best_thresh, best_f05 = 0.5, 0.0
    for thresh in np.arange(0.1, 0.95, 0.01):
        preds = (y_proba >= thresh).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f05 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0.0
        if f05 > best_f05:
            best_f05, best_thresh = f05, thresh
    return best_thresh, best_f05


def find_best_macro_f05_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    s1_ids: List[str],
    step: float = 0.01,
    lo: float = 0.05,
    hi: float = 0.95,
) -> Tuple[float, float]:
    """Find the threshold that maximizes MACRO-averaged F_0.5 per S1 entity.

    Mirrors the leaderboard metric: each S1 entity in the validation pair set
    is scored individually (all-negative entities score 1.0 when predicted
    empty, 0.0 otherwise), then scores are averaged across entities.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    groups: Dict[str, List[int]] = {}
    for i, sid in enumerate(s1_ids):
        groups.setdefault(sid, []).append(i)

    entity_data = []
    for sid, idxs in groups.items():
        idxs_arr = np.asarray(idxs)
        entity_data.append((idxs_arr, y_true[idxs_arr]))

    best_thresh, best_macro = 0.5, -1.0
    for thresh in np.arange(lo, hi + 1e-9, step):
        scores = []
        for idxs_arr, labels in entity_data:
            preds = (y_proba[idxs_arr] >= thresh).astype(int)
            n_pos = int(labels.sum())

            if n_pos == 0:
                scores.append(1.0 if preds.sum() == 0 else 0.0)
                continue

            tp = int(((preds == 1) & (labels == 1)).sum())
            fp = int(((preds == 1) & (labels == 0)).sum())
            fn = n_pos - tp

            if tp == 0:
                scores.append(0.0)
                continue

            p = tp / (tp + fp)
            r = tp / (tp + fn)
            scores.append((1.25 * p * r) / (0.25 * p + r))

        macro = float(np.mean(scores)) if scores else 0.0
        if macro > best_macro:
            best_macro = macro
            best_thresh = float(thresh)

    return best_thresh, best_macro


# --------------------------------------------------------------------- io
def save_model(model, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
