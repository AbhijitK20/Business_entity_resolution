#!/usr/bin/env python3
"""K5 demo harness — reproducible error analysis without the real dataset.

Trains a LightGBM matcher on the seeded synthetic generator, simulates a
name-similarity blocking stage (top-K candidates per S1), runs the repo's
real decision layer (exclusivity + expected-F0.5 selection), then writes:

    <out>/predictions.tsv    source1_entity_id  matched_entity_ids
    <out>/candidates.tsv     source1_entity_id  candidate_entity_ids
    <out>/pair_scores.tsv    source1_entity_id  candidate_entity_id  score

and invokes scripts/evaluate.py with --pair-scores + --threshold so errors are
bucketed into retrieval / matching / decision-policy / integrity, with the
top-20 worst entities and feature evidence. Output:
    <out>/evaluation.json

All numbers are SYNTHETIC-labelled — re-run with --data-dir on real data.

Usage:
    python scripts/error_analysis_demo.py [--n-s1 600] [--seed 42]
        [--top-cands 15] [--n-trials 0] [--out output/k5_demo]
"""
import argparse
import csv
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.append(_p)

from rapidfuzz import fuzz

from scripts.country_holdout import (
    build_pairs,
    build_records,
    feature_matrix,
    fit_model,
    load_frames,
    split_grouped,
)
from scripts.make_synthetic_data import generate as generate_synthetic
from src.decision import select_sets_expected_f05
from src.model import find_best_macro_f05_threshold
from src.normalize import apply_normalization

DEFAULT_SEED = 42


def simulate_blocking(s1_df, gal_df, top_k):
    """Candidates = top-K gallery records by name similarity (deterministic).

    This stands in for src/blocking.py so the demo stays inside the
    evaluation stream's ownership and never imports a teammate's file.
    """
    names = dict(zip(gal_df["entity_id"], gal_df["business_name_clean"]))
    cands = {}
    for row in s1_df.itertuples(index=False):
        ranked = sorted(
            ((fuzz.WRatio(row.business_name_clean, nm), eid)
             for eid, nm in names.items()),
            key=lambda t: (-t[0], t[1]),
        )
        cands[row.entity_id] = [eid for _, eid in ranked[:top_k]]
    return cands


def run(args):
    if args.data_dir:
        data_dir = Path(args.data_dir)
        provenance = f"real:{args.data_dir}"
    else:
        data_dir = Path(tempfile.mkdtemp(prefix="k5_demo_"))
        generate_synthetic(data_dir, args.n_s1, args.seed)
        provenance = f"synthetic n_s1={args.n_s1}"

    s1, gal, gt = load_frames(data_dir, "train")
    recs = build_records([s1, gal])
    cands = simulate_blocking(s1, gal, args.top_cands)

    gallery_ids = sorted(set(gal["entity_id"]))
    s1_ids = sorted(gt)
    rng = random.Random(args.seed)
    rng.shuffle(s1_ids)
    cut = max(10, int(0.8 * len(s1_ids)))
    fit_ids = sorted(s1_ids[:cut])       # model trains ONLY on these
    eval_ids = sorted(s1_ids[cut:])      # error analysis runs ONLY on these
    s1_ids = eval_ids

    train_pairs = build_pairs(gt, fit_ids, recs, gallery_ids, mode="train",
                              seed=args.seed, neg_per_pos=2)
    X, y, pair_s1, _ = feature_matrix(train_pairs, recs)
    tr_idx, va_idx = split_grouped(pair_s1, y, args.seed)
    model = fit_model(X[tr_idx], y[tr_idx], args.n_trials, args.seed)

    proba_val = model.predict_proba(X[va_idx])[:, 1]
    t, _ = find_best_macro_f05_threshold(
        y[va_idx], proba_val, [pair_s1[i] for i in va_idx])
    t = round(float(t), 4)

    score_pairs, proba_all = [], []
    for sid in s1_ids:
        for cid in cands[sid]:
            score_pairs.append((sid, cid))
    Xc, _, _, _ = feature_matrix([(s, c, 0) for s, c in score_pairs], recs)
    proba_all = model.predict_proba(Xc)[:, 1]

    predictions = select_sets_expected_f05(
        [s for s, _ in score_pairs],
        [c for _, c in score_pairs],
        proba_all,
        anchor_ids=s1_ids,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "ground_truth_eval.tsv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        for sid in eval_ids:
            w.writerow([sid, ",".join(sorted(gt.get(sid, set())))])

    with open(out / "predictions.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        for sid in s1_ids:
            w.writerow([sid, ",".join(sorted(predictions.get(sid, set())))])

    with open(out / "candidates.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source1_entity_id", "candidate_entity_ids"])
        for sid in s1_ids:
            w.writerow([sid, ",".join(sorted(cands[sid]))])

    with open(out / "pair_scores.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source1_entity_id", "candidate_entity_id", "score"])
        for (sid, cid), p in zip(score_pairs, proba_all):
            w.writerow([sid, cid, f"{float(p):.6f}"])

    cmd = [
        sys.executable, str(_REPO_ROOT / "scripts" / "evaluate.py"),
        "--predictions", str(out / "predictions.tsv"),
        "--ground-truth", str(out / "ground_truth_eval.tsv"),
        "--candidate", str(out / "candidates.tsv"),
        "--s1-source", str(data_dir / "dataset" / "train" / "train_source1.tsv"),
        "--gallery-source", str(data_dir / "dataset" / "train" / "train_source2.tsv"),
        "--gallery-source", str(data_dir / "dataset" / "train" / "train_source3.tsv"),
        "--pair-scores", str(out / "pair_scores.tsv"),
        "--threshold", str(t),
        "--top-k", "20",
        "--bootstrap", str(args.bootstrap),
        "--seed", str(args.seed),
        "--feature-evidence",
        "--json-out", str(out / "evaluation.json"),
    ]
    print(f"provenance: {provenance}  |  decision threshold: {t}")
    print(f"entities: {len(fit_ids)} fit (model training) / "
          f"{len(eval_ids)} held-out eval (error analysis)")
    print(f"candidate set: {args.top_cands}/S1  |  running evaluate.py ...\n")
    proc = subprocess.run(cmd, cwd=_REPO_ROOT)
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    print(f"\nartifacts in {out}/ : predictions.tsv candidates.tsv "
          f"pair_scores.tsv evaluation.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None,
                    help="real dataset dir (default: generate synthetic)")
    ap.add_argument("--n-s1", type=int, default=600)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--top-cands", type=int, default=15,
                    help="simulated blocking candidates per S1")
    ap.add_argument("--n-trials", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default="output/k5_demo")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
