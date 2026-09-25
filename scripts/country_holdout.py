#!/usr/bin/env python3
"""K4 — country-holdout stress test (France proxy).

Train on one country, evaluate on an UNSEEN country, report the transfer gap
(how much precision / F_0.5 drops), and recommend a stricter cutoff margin for
France — the test-only country that never appears in training.

Protocol (TASK_BREAKDOWN K4, COMPETITIVE_INTEL §12):
    direction 1: train US   -> eval India
    direction 2: train India -> eval US
    pooled model (train US+India) as the final-model proxy
    France cutoff = pooled threshold + max precision-preserving margin

Two evaluation regimes per direction:
    * full   — every gallery record is a candidate (leaderboard-like density)
    * hard   — top same-country distractors only (worst-case precision stress)

The France margin is taken from the HARD regime (conservative).

Data: the real dataset is unavailable in this environment, so the default is
the seeded synthetic generator (clearly labeled "synthetic" in the report).
Pass --data-dir with train_source{1,2,3}.tsv + train_ground_truth.tsv to run
on real data when it is present.

Usage:
    python scripts/country_holdout.py [--data-dir DIR | --synthetic] \
        [--n-s1 600 --seed 42] [--n-trials 10] [--neg-per-pos 2] \
        [--no-full-density] [--json-out report.json]
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.append(_p)

import optuna
from rapidfuzz import fuzz

optuna.logging.set_verbosity(optuna.logging.WARNING)

from scripts.evaluate import evaluate as challenge_evaluate
from scripts.make_synthetic_data import generate as generate_synthetic
from src.decision import select_sets_threshold
from src.features import compute_all_features
from src.model import find_best_macro_f05_threshold, _tune_lightgbm
from src.normalize import apply_normalization

DEFAULT_SEED = 42
FIXED_LGB_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "class_weight": "balanced",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "min_child_samples": 50,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "force_col_wise": True,
    "verbose": -1,
    "random_state": DEFAULT_SEED,
    "n_jobs": 1,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_frames(data_dir: Path, split: str = "train"):
    """Read source1/2/3 + ground-truth TSVs; returns normalized frames + GT."""
    d = Path(data_dir) / "dataset" / split
    if not (d / f"{split}_source1.tsv").exists():
        d = Path(data_dir)
    s1 = pd.read_csv(d / f"{split}_source1.tsv", sep="\t", dtype=str).fillna("")
    gal = pd.concat([
        pd.read_csv(d / f"{split}_source2.tsv", sep="\t", dtype=str).fillna(""),
        pd.read_csv(d / f"{split}_source3.tsv", sep="\t", dtype=str).fillna(""),
    ], ignore_index=True)
    gt = {}
    with open(d / f"{split}_ground_truth.tsv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            gt[row["source1_entity_id"]] = {
                t.strip() for t in row["matched_entity_ids"].split(",") if t.strip()
            }
    s1 = apply_normalization(s1)
    gal = apply_normalization(gal)
    return s1, gal, gt


def build_records(frames):
    """id -> (clean_name, clean_addr, clean_country)."""
    out = {}
    for df in frames:
        for row in df.itertuples(index=False):
            out[row.entity_id] = (
                row.business_name_clean,
                row.business_address_clean,
                row.country_clean,
            )
    return out


# ---------------------------------------------------------------------------
# Pair construction (train / eval-hard / eval-full)
# ---------------------------------------------------------------------------
def _top_hard(names_by_id, anchor_name, pool_ids, k, exclude):
    ranked = sorted(
        ((fuzz.WRatio(anchor_name, names_by_id[i]), i) for i in pool_ids
         if i not in exclude),
        key=lambda t: (-t[0], t[1]),
    )
    return [i for _, i in ranked[:k]]


def build_pairs(gt, s1_ids, recs, gallery_ids, *, mode, seed,
                neg_per_pos=2, hard_ratio=0.7, exclude_neg_countries=()):
    """Build (s1_id, cand_id, label) pairs for one country cohort.

    mode="train"      positives + per-S1 negatives (70% hard same-country,
                      30% random other-country — mirrors training.py)
    mode="eval_hard"  positives + top same-country distractors only
    mode="eval_full"  positives + every gallery record except positives

    ``exclude_neg_countries`` removes countries from the RANDOM-negative pool
    (hard negatives are always same-country). The holdout directions use it to
    keep the EVAL country's gallery out of training pairs — a clean country
    holdout must never have seen the eval country, not even as negatives.
    """
    rng = random.Random(seed)
    names = {i: recs[i][0] for i in list(s1_ids) + list(gallery_ids)}
    countries = {i: recs[i][2] for i in list(s1_ids) + list(gallery_ids)}
    excluded = set(exclude_neg_countries)
    pairs = []
    for sid in sorted(s1_ids):
        positives = sorted(gt.get(sid, set()))
        for t in positives:
            pairs.append((sid, t, 1))
        pos_set = set(positives)
        pool = [g for g in gallery_ids if g not in pos_set]
        if mode == "eval_full":
            negs = sorted(pool)
        else:
            same = [g for g in pool if countries[g] == countries[sid]]
            diff = [g for g in pool if countries[g] != countries[sid]
                    and countries[g] not in excluded]
            if mode == "eval_hard":
                k = max(6, 2 * len(positives))
                negs = _top_hard(names, names[sid], same, k, set())
            else:
                n_total = max(2, neg_per_pos * max(1, len(positives)))
                n_hard = int(round(n_total * hard_ratio))
                hard = _top_hard(names, names[sid], same, n_hard, set())
                n_rand = n_total - len(hard)
                rand = rng.sample(diff, min(n_rand, len(diff))) if n_rand else []
                if n_rand > len(rand):
                    # other-country pool exhausted (e.g. clean country holdout
                    # excludes it) → top up from same-country non-hard pool so
                    # the per-S1 negative count stays stable
                    hard_set = set(hard)
                    rest = [g for g in same if g not in hard_set]
                    need = n_rand - len(rand)
                    rand += rng.sample(rest, min(need, len(rest))) if need else []
                negs = hard + rand
        for g in sorted(set(negs)):
            pairs.append((sid, g, 0))
    return pairs


def feature_matrix(pairs, recs):
    X, y, s1_ids, cand_ids = [], [], [], []
    for sid, cid, label in pairs:
        a = recs[sid]
        b = recs.get(cid, ("", "", ""))
        feats = compute_all_features(a[0], b[0], a[1], b[1], a[2], b[2])
        X.append([feats[f] for f in _FEATURES])
        y.append(label)
        s1_ids.append(sid)
        cand_ids.append(cid)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), s1_ids, cand_ids


from src.features import FEATURE_NAMES as _FEATURES  # noqa: E402


def split_grouped(s1_ids, y, seed, val_frac=0.2):
    """Grouped 80/20 split by S1 identity (leak-safe)."""
    rng = random.Random(seed)
    groups = sorted(set(s1_ids))
    rng.shuffle(groups)
    cut = max(1, int(len(groups) * (1 - val_frac)))
    val_groups = set(groups[cut:])
    train_idx = [i for i, s in enumerate(s1_ids) if s not in val_groups]
    val_idx = [i for i, s in enumerate(s1_ids) if s in val_groups]
    if not val_idx:
        val_idx = train_idx[-max(1, len(train_idx) // 5):]
        train_idx = train_idx[:-len(val_idx)]
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def pair_metrics(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) else 0.0
    return {"precision": p, "recall": r, "f05": f,
            "tp": tp, "fp": fp, "fn": fn, "n_pairs": len(y)}


def entity_metrics(pairs, proba, threshold, gt, anchor_ids):
    s1_ids = [s for s, _, _ in pairs]
    cand_ids = [c for _, c, _ in pairs]
    preds = select_sets_threshold(s1_ids, cand_ids, proba, threshold,
                                  anchor_ids=anchor_ids)
    truth = {s: gt.get(s, set()) for s in anchor_ids}
    return challenge_evaluate(preds, truth)


def precision_preserving_threshold(y, proba, target_p, lo=0.05, hi=0.95, step=0.01):
    """Lowest threshold on the UNSEEN country whose pair precision >= the
    in-country precision. Returns None if unreachable up to hi.

    Uses an integer-stepped grid (not `t += step`) so thresholds like 0.7
    stay exact — float drift would exclude scores equal to the threshold.
    """
    n = int(round((hi - lo) / step))
    for k in range(n + 1):
        t = round(lo + k * step, 4)
        pred = proba >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        if tp + fp and tp / (tp + fp) >= target_p:
            return t
    return None


def best_threshold_for(y, proba, s1_ids, lo=0.05, hi=0.95, step=0.01):
    """Threshold maximizing macro F_0.5 on this (labelled) cohort — a
    diagnostic upper bound, not a deployable cutoff."""
    t, m = find_best_macro_f05_threshold(y, proba, s1_ids, step=step, lo=lo, hi=hi)
    return round(float(t), 4), float(m)


# ---------------------------------------------------------------------------
# Training + one direction
# ---------------------------------------------------------------------------
def fit_model(X, y, n_trials, seed):
    if n_trials and n_trials > 0:
        params = _tune_lightgbm(X, y, n_trials, seed)
        params["n_jobs"] = 1
    else:
        params = dict(FIXED_LGB_PARAMS, random_state=seed)
    import lightgbm as lgb
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    return model




# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def run_holdout(s1, gal, gt, *, seed=DEFAULT_SEED, n_trials=10, neg_per_pos=2,
                full_density=True, provenance="synthetic", train_countries=("us", "india")):
    recs = build_records([s1, gal])
    gallery_ids = sorted(set(gal["entity_id"]))
    report = {
        "provenance": provenance,
        "seed": seed,
        "n_trials": n_trials,
        "full_density": full_density,
        "directions": [],
    }

    pairs_cache = {}

    def direction_metrics(train_c, eval_c):
        tr_ids = sorted(s for s in gt if recs[s][2] == train_c)
        ev_ids = sorted(s for s in gt if recs[s][2] == eval_c)
        if len(tr_ids) < 5 or len(ev_ids) < 5:
            return {"train_country": train_c, "eval_country": eval_c,
                    "skipped": f"too few entities (train={len(tr_ids)}, "
                               f"eval={len(ev_ids)})"}

        train_key = ("train", train_c, eval_c)
        if train_key not in pairs_cache:
            # Clean holdout: training negatives come ONLY from the train
            # country — neither the eval country nor any other (e.g. test-only
            # France) may appear in this country's training pairs.
            all_countries = {recs[g][2] for g in gallery_ids}
            exclude = tuple(sorted(all_countries - {train_c}))
            pairs_cache[train_key] = build_pairs(
                gt, tr_ids, recs, gallery_ids, mode="train", seed=seed,
                neg_per_pos=neg_per_pos, exclude_neg_countries=exclude)
        train_pairs = pairs_cache[train_key]

        X, y, s1_ids, _ = feature_matrix(train_pairs, recs)
        tr_idx, va_idx = split_grouped(s1_ids, y, seed)
        model = fit_model(X[tr_idx], y[tr_idx], n_trials, seed)

        proba_val = model.predict_proba(X[va_idx])[:, 1]
        va_s1 = [s1_ids[i] for i in va_idx]
        t_in, macro_val = find_best_macro_f05_threshold(
            y[va_idx], proba_val, va_s1)
        t_in = round(float(t_in), 4)
        in_pair = pair_metrics(y[va_idx], proba_val, t_in)
        in_ent = entity_metrics([train_pairs[i] for i in va_idx],
                                proba_val, t_in, gt, sorted(set(va_s1)))

        hard_pairs = build_pairs(gt, ev_ids, recs, gallery_ids,
                                 mode="eval_hard", seed=seed)
        Xh, yh, s1h, _ = feature_matrix(hard_pairs, recs)
        ph = model.predict_proba(Xh)[:, 1]
        hard_pair = pair_metrics(yh, ph, t_in)
        hard_ent = entity_metrics(hard_pairs, ph, t_in, gt, ev_ids)
        t_pres = precision_preserving_threshold(yh, ph, in_pair["precision"])
        t_best_h, best_h_macro = best_threshold_for(yh, ph, s1h)

        full_pair = full_ent = None
        t_pres_full = t_best_f = None
        if full_density:
            full_pairs = build_pairs(gt, ev_ids, recs, gallery_ids,
                                     mode="eval_full", seed=seed)
            Xf, yf, s1f, _ = feature_matrix(full_pairs, recs)
            pf = model.predict_proba(Xf)[:, 1]
            full_pair = pair_metrics(yf, pf, t_in)
            full_ent = entity_metrics(full_pairs, pf, t_in, gt, ev_ids)
            t_pres_full = precision_preserving_threshold(
                yf, pf, in_pair["precision"])
            t_best_f, _ = best_threshold_for(yf, pf, s1f)

        gaps = {
            "precision_drop_hard": round(
                in_pair["precision"] - hard_pair["precision"], 4),
            "precision_drop_rel_hard": round(
                (in_pair["precision"] - hard_pair["precision"])
                / in_pair["precision"], 4) if in_pair["precision"] else None,
            "macro_f05_drop_hard": round(in_ent["macro_f05"]
                                         - hard_ent["macro_f05"], 4),
        }
        if full_density:
            gaps["precision_drop_full"] = round(
                in_pair["precision"] - full_pair["precision"], 4)
            gaps["macro_f05_drop_full"] = round(
                in_ent["macro_f05"] - full_ent["macro_f05"], 4)

        margin = max(0.0, round(t_pres - t_in, 4)) if t_pres is not None else None
        return {
            "train_country": train_c,
            "eval_country": eval_c,
            "n_train_entities": len(tr_ids),
            "n_eval_entities": len(ev_ids),
            "n_train_pairs": int(len(y)),
            "train_negative_countries": sorted(
                {recs[c][2] for _, c, l in train_pairs if l == 0}),
            "n_eval_pairs_hard": int(len(yh)),
            "n_eval_pairs_full": int(len(yf)) if full_density else None,
            "threshold_in_country": t_in,
            "in_country": {"pair": in_pair, "entity": in_ent,
                           "val_macro_f05_at_t": round(float(macro_val), 4)},
            "transfer_hard": {"pair": hard_pair, "entity": hard_ent},
            "transfer_full": ({"pair": full_pair, "entity": full_ent}
                              if full_density else None),
            "gaps": gaps,
            "threshold_precision_preserving_hard": t_pres,
            "threshold_precision_preserving_full": t_pres_full,
            "margin_hard": margin,
            "threshold_best_on_eval_hard": t_best_h,
            "threshold_best_macro_hard": round(float(best_h_macro), 4),
            "threshold_best_on_eval_full": t_best_f,
            "threshold_reachable": t_pres is not None,
        }

    directions = []
    pairs_a, pairs_b = train_countries
    directions.append(direction_metrics(pairs_a, pairs_b))
    directions.append(direction_metrics(pairs_b, pairs_a))
    report["directions"] = directions

    pooled = _pooled_model(gt, recs, gallery_ids, train_countries, seed,
                           n_trials, neg_per_pos)
    report["pooled_final_proxy"] = pooled

    report["protocol"] = {
        "train_negatives_exclude_eval_country": True,
        "note": ("holdout directions train only on their own country's "
                 "pairs: the eval country's gallery (and any test-only "
                 "country) never appears in training, not even as negatives"),
    }
    margins = [d.get("margin_hard") for d in directions
               if d.get("margin_hard") is not None]
    unreachable = [f"{d['train_country']}->{d['eval_country']}"
                   for d in directions if d.get("threshold_reachable") is False]
    delta = max(margins) if margins else None
    t_final = pooled.get("threshold")
    france_cutoff = None
    uncapped = None
    capped = False
    if delta is not None and t_final is not None:
        uncapped = round(t_final + delta, 4)
        capped = uncapped > 0.95
        france_cutoff = round(min(uncapped, 0.95), 4)
    report["france_recommendation"] = {
        "margin_delta": delta,
        "threshold_pooled_proxy": t_final,
        "recommended_france_cutoff": france_cutoff,
        "uncapped_cutoff": uncapped,
        "capped_at_0_95": capped,
        "basis": "max precision-preserving margin across US->India and "
                 "India->US, added to the pooled (all-country) threshold",
        "directions_with_unreachable_margin": unreachable,
        "note": "France never appears in training; this proxy assumes its "
                "difficulty lies between US and India under the current "
                "features. Re-run on real data before freezing the cutoff.",
    }
    if capped:
        report["france_recommendation"]["capping_note"] = (
            f"uncapped cutoff {uncapped} exceeds the 0.95 grid end — "
            "reported cutoff is CAPPED at 0.95 (thresholds above 0.95 "
            "are not searched by this script)"
        )
    return report


def _pooled_model(gt, recs, gallery_ids, countries, seed, n_trials, neg_per_pos):
    ids = sorted(s for s in gt if recs[s][2] in countries)
    if len(ids) < 10:
        return {"skipped": "too few entities"}
    # pooled = final-model proxy: trains on the TRAIN countries only; test-only
    # countries (France) must never leak into its negatives either
    gal_countries = {recs[g][2] for g in gallery_ids}
    exclude = tuple(sorted(gal_countries - set(countries)))
    pairs = build_pairs(gt, ids, recs, gallery_ids, mode="train", seed=seed,
                        neg_per_pos=neg_per_pos, exclude_neg_countries=exclude)
    X, y, s1_ids, _ = feature_matrix(pairs, recs)
    tr_idx, va_idx = split_grouped(s1_ids, y, seed)
    model = fit_model(X[tr_idx], y[tr_idx], n_trials, seed)
    proba = model.predict_proba(X[va_idx])[:, 1]
    va_s1 = [s1_ids[i] for i in va_idx]
    t, macro = find_best_macro_f05_threshold(y[va_idx], proba, va_s1)
    ent = entity_metrics([pairs[i] for i in va_idx], proba, t, gt,
                         sorted(set(va_s1)))
    return {
        "threshold": round(float(t), 4),
        "val_macro_f05": round(float(macro), 4),
        "val_pair": pair_metrics(y[va_idx], proba, t),
        "val_entity": ent,
        "n_pairs": int(len(y)),
        "n_entities": len(ids),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(report):
    print("=" * 72)
    print(f"K4 country-holdout stress test  [{report['provenance']} data, "
          f"seed={report['seed']}, full_density={report['full_density']}]")
    print("=" * 72)
    for d in report["directions"]:
        if "skipped" in d:
            print(f"\n  {d['train_country']} -> {d['eval_country']}: "
                  f"SKIPPED ({d['skipped']})")
            continue
        print(f"\n  direction: train {d['train_country'].upper()} -> "
              f"eval {d['eval_country'].upper()} "
              f"({d['n_train_entities']} train / {d['n_eval_entities']} eval "
              f"entities)")
        ic, th = d["in_country"], d["transfer_hard"]
        print(f"    in-country  @ t={d['threshold_in_country']:.2f}  "
              f"P={ic['pair']['precision']:.4f} R={ic['pair']['recall']:.4f} "
              f"F0.5={ic['pair']['f05']:.4f}  "
              f"entity-macro={ic['entity']['macro_f05']:.4f}")
        print(f"    transfer(hard)             "
              f"P={th['pair']['precision']:.4f} R={th['pair']['recall']:.4f} "
              f"F0.5={th['pair']['f05']:.4f}  "
              f"entity-macro={th['entity']['macro_f05']:.4f}")
        if d.get("transfer_full"):
            tf = d["transfer_full"]
            print(f"    transfer(full)             "
                  f"P={tf['pair']['precision']:.4f} R={tf['pair']['recall']:.4f} "
                  f"F0.5={tf['pair']['f05']:.4f}  "
                  f"entity-macro={tf['entity']['macro_f05']:.4f}")
        g = d["gaps"]
        rel = (f" ({g['precision_drop_rel_hard'] * 100:.1f}%)"
               if g.get("precision_drop_rel_hard") is not None else "")
        print(f"    gap: precision {g['precision_drop_hard']:+.4f}{rel}, "
              f"macro F0.5 {g['macro_f05_drop_hard']:+.4f}")
        t_pres = d["threshold_precision_preserving_hard"]
        print(f"    precision-preserving cutoff (hard): "
              f"{t_pres if t_pres is not None else 'UNREACHABLE <=0.95'}  "
              f"| best-on-eval (diagnostic): {d['threshold_best_on_eval_hard']}")
        print(f"    margin (hard): {d['margin_hard']}")
    pool = report.get("pooled_final_proxy", {})
    if "threshold" in pool:
        print(f"\n  pooled final-model proxy: t={pool['threshold']:.2f} "
              f"val-macro-F0.5={pool['val_macro_f05']:.4f}")
    fr = report["france_recommendation"]
    print("\n  France cutoff recommendation:")
    print(f"    margin delta          : {fr['margin_delta']}")
    print(f"    pooled threshold      : {fr['threshold_pooled_proxy']}")
    cap_txt = ("  <-- CAPPED from uncapped "
               f"{fr['uncapped_cutoff']}" if fr.get("capped_at_0_95") else "")
    print(f"    recommended cutoff    : {fr['recommended_france_cutoff']}{cap_txt}")
    if fr.get("capping_note"):
        print(f"    {fr['capping_note']}")
    if fr["directions_with_unreachable_margin"]:
        print(f"    UNREACHABLE margins   : "
              f"{fr['directions_with_unreachable_margin']}")
    print(f"    {fr['note']}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--data-dir", type=str, default=None,
                     help="directory with train_source{1,2,3}.tsv + "
                          "train_ground_truth.tsv (or its dataset/train parent)")
    src.add_argument("--synthetic", action="store_true", default=True,
                     help="use the seeded synthetic generator (default; "
                          "the real dataset is not available here)")
    ap.add_argument("--n-s1", type=int, default=600)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-trials", type=int, default=10,
                    help="optuna trials (0 = fixed conservative params)")
    ap.add_argument("--neg-per-pos", type=int, default=2)
    ap.add_argument("--no-full-density", dest="full_density",
                    action="store_false", default=True)
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    if args.data_dir:
        s1, gal, gt = load_frames(Path(args.data_dir), args.split)
        provenance = f"real:{args.data_dir} split={args.split}"
    else:
        tmp = Path("_holdout_synth")
        tmp.mkdir(exist_ok=True)
        generate_synthetic(tmp, args.n_s1, args.seed)
        s1, gal, gt = load_frames(tmp, args.split)
        provenance = f"synthetic n_s1={args.n_s1} split={args.split}"

    report = run_holdout(s1, gal, gt, seed=args.seed, n_trials=args.n_trials,
                         neg_per_pos=args.neg_per_pos,
                         full_density=args.full_density,
                         provenance=provenance)
    print_report(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"report written to {args.json_out}")


if __name__ == "__main__":
    main()
