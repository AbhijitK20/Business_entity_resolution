"""Model training module — dual-layer ensemble stacking.

Based on research from:
- MetaBoost (dual-layer stacking)
- imbalance-benchmark (threshold optimization)
- ted-entity-resolution (conservative LightGBM params)
- entity-deduplication (calibrated MLP)
"""
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Tuple, Optional
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score,
)
import optuna
from .features import FEATURE_NAMES


def train_base_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 25,
    random_seed: int = 42,
) -> Dict:
    """Train 3 base models with Optuna tuning.
    
    Returns dict with models and their OOF predictions.
    """
    results = {}
    
    # === LightGBM ===
    def lgb_objective(trial):
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
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train[train_idx], y_train[train_idx],
                eval_set=[(X_train[val_idx], y_train[val_idx])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            preds = model.predict_proba(X_train[val_idx])[:, 1]
            scores.append(average_precision_score(y_train[val_idx], preds))
        
        return np.mean(scores)
    
    print("Tuning LightGBM...")
    lgb_study = optuna.create_study(direction="maximize", sampler=optuna.TPESampler(seed=random_seed))
    lgb_study.optimize(lgb_objective, n_trials=n_trials, show_progress_bar=True)
    
    # Train final LightGBM with best params
    lgb_params = lgb_study.best_params
    lgb_params.update({
        "objective": "binary",
        "boosting_type": "gbdt",
        "class_weight": "balanced",
        "force_col_wise": True,
        "verbose": -1,
        "random_state": random_seed,
    })
    
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    lgb_val_proba = lgb_model.predict_proba(X_val)[:, 1]
    
    results["lgb"] = {
        "model": lgb_model,
        "val_proba": lgb_val_proba,
        "best_params": lgb_params,
        "val_auc": roc_auc_score(y_val, lgb_val_proba),
        "val_ap": average_precision_score(y_val, lgb_val_proba),
    }
    
    # === XGBoost ===
    def xgb_objective(trial):
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
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train[train_idx], y_train[train_idx],
                eval_set=[(X_train[val_idx], y_train[val_idx])],
                verbose=False,
            )
            preds = model.predict_proba(X_train[val_idx])[:, 1]
            scores.append(average_precision_score(y_train[val_idx], preds))
        
        return np.mean(scores)
    
    print("Tuning XGBoost...")
    xgb_study = optuna.create_study(direction="maximize", sampler=optuna.TPESampler(seed=random_seed))
    xgb_study.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=True)
    
    # Train final XGBoost
    xgb_params = xgb_study.best_params
    xgb_params.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 0,
        "random_state": random_seed,
        "scale_pos_weight": len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
    })
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]
    
    results["xgb"] = {
        "model": xgb_model,
        "val_proba": xgb_val_proba,
        "best_params": xgb_params,
        "val_auc": roc_auc_score(y_val, xgb_val_proba),
        "val_ap": average_precision_score(y_val, xgb_val_proba),
    }
    
    # === RandomForest ===
    def rf_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": random_seed,
        }
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            model = RandomForestClassifier(**params)
            model.fit(X_train[train_idx], y_train[train_idx])
            preds = model.predict_proba(X_train[val_idx])[:, 1]
            scores.append(average_precision_score(y_train[val_idx], preds))
        
        return np.mean(scores)
    
    print("Tuning RandomForest...")
    rf_study = optuna.create_study(direction="maximize", sampler=optuna.TPESampler(seed=random_seed))
    rf_study.optimize(rf_objective, n_trials=min(n_trials, 15), show_progress_bar=True)
    
    # Train final RF
    rf_params = rf_study.best_params
    rf_params.update({
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": random_seed,
    })
    
    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)
    rf_val_proba = rf_model.predict_proba(X_val)[:, 1]
    
    results["rf"] = {
        "model": rf_model,
        "val_proba": rf_val_proba,
        "best_params": rf_params,
        "val_auc": roc_auc_score(y_val, rf_val_proba),
        "val_ap": average_precision_score(y_val, rf_val_proba),
    }
    
    return results


def train_meta_learner(
    base_val_probas: Dict[str, np.ndarray],
    y_val: np.ndarray,
    random_seed: int = 42,
) -> Tuple[object, float]:
    """Train a shallow LightGBM meta-learner on OOF predictions.
    
    From MetaBoost: deliberately SHALLOW (depth 2-5, leaves 4-16).
    """
    # Stack base predictions
    X_meta = np.column_stack([base_val_probas["lgb"], base_val_probas["xgb"], base_val_probas["rf"]])
    
    # Optuna for meta-learner
    def meta_objective(trial):
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
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_seed)
        scores = []
        for train_idx, val_idx in skf.split(X_meta, y_val):
            model = lgb.LGBMClassifier(**params)
            model.fit(X_meta[train_idx], y_val[train_idx])
            preds = model.predict_proba(X_meta[val_idx])[:, 1]
            scores.append(average_precision_score(y_val[val_idx], preds))
        
        return np.mean(scores)
    
    print("Tuning meta-learner...")
    meta_study = optuna.create_study(direction="maximize", sampler=optuna.TPESampler(seed=random_seed))
    meta_study.optimize(meta_objective, n_trials=20, show_progress_bar=True)
    
    # Train final meta-learner
    meta_params = meta_study.best_params
    meta_params.update({
        "objective": "binary",
        "boosting_type": "gbdt",
        "class_weight": "balanced",
        "force_col_wise": True,
        "verbose": -1,
        "random_state": random_seed,
    })
    
    meta_model = lgb.LGBMClassifier(**meta_params)
    meta_model.fit(X_meta, y_val)
    
    meta_val_proba = meta_model.predict_proba(X_meta)[:, 1]
    meta_auc = roc_auc_score(y_val, meta_val_proba)
    meta_ap = average_precision_score(y_val, meta_val_proba)
    
    print(f"Meta-learner AUC: {meta_auc:.4f}, AP: {meta_ap:.4f}")
    
    return meta_model, meta_val_proba


def find_best_f05_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """Find threshold that maximizes F_0.5.
    
    F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    From imbalance-benchmark: scan in 0.01 steps.
    """
    best_thresh, best_f05 = 0.5, 0
    
    for thresh in np.arange(0.1, 0.95, 0.01):
        preds = (y_proba >= thresh).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f05 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
    
    return best_thresh, best_f05


def save_model(model, path: str) -> None:
    """Save model to pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str):
    """Load model from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)
