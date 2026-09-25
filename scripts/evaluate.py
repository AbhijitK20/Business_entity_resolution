"""Macro F_0.5 evaluator — mirrors the leaderboard metric exactly.

The leaderboard computes F_0.5 PER Source 1 entity, then macro-averages across
all S1 entities. Singletons are included: correctly predicting an empty list
scores 1.0, predicting any match scores 0.0.

Usage:
    python scripts/evaluate.py --predictions output/matching_results.tsv \
                               --ground-truth data/dataset/train/train_ground_truth.tsv
"""
import argparse
import csv
from pathlib import Path


def f05(precision: float, recall: float) -> float:
    """F_0.5 = (1.25 * P * R) / (0.25 * P + R)."""
    denom = 0.25 * precision + recall
    if denom <= 0:
        return 0.0
    return (1.25 * precision * recall) / denom


def parse_list(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def load_dict(path: Path, id_col: str, value_col: str):
    """Load a TSV into {id: set(values)}."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rid = (row.get(id_col) or "").strip()
            out[rid] = parse_list(row.get(value_col) or "")
    return out


def evaluate(predictions: dict, ground_truth: dict) -> dict:
    """Compute macro-averaged precision, recall, F_0.5 per S1 entity.

    Follows the challenge definition: every S1 entity in the ground truth is
    scored individually (including singletons), then averaged.
    """
    per_entity = []
    total_tp = total_fp = total_fn = 0

    for s1_id, truth in ground_truth.items():
        pred = predictions.get(s1_id, set())

        if not truth and not pred:
            # Correct singleton
            per_entity.append(1.0)
            continue
        if not truth and pred:
            # False merge on singleton
            per_entity.append(0.0)
            total_fp += len(pred)
            continue

        tp = len(pred & truth)
        fp = len(pred - truth)
        fn = len(truth - pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if not pred:
            # Missed everything for a matching entity
            per_entity.append(0.0)
            continue

        p = tp / (tp + fp)
        r = tp / (tp + fn)
        per_entity.append(f05(p, r))

    macro_f05 = sum(per_entity) / len(per_entity) if per_entity else 0.0

    # Micro-level (pair) metrics for diagnostics
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f05 = f05(micro_p, micro_r)

    n_singletons = sum(1 for t in ground_truth.values() if not t)
    singleton_correct = sum(
        1 for s1_id, t in ground_truth.items()
        if not t and not predictions.get(s1_id, set())
    )

    return {
        "macro_f05": macro_f05,
        "micro_f05": micro_f05,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "n_entities": len(ground_truth),
        "n_singletons": n_singletons,
        "singleton_accuracy": singleton_correct / n_singletons if n_singletons else 1.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ground-truth", required=True)
    args = ap.parse_args()

    preds = load_dict(Path(args.predictions), "source1_entity_id", "matched_entity_ids")
    gt = load_dict(Path(args.ground_truth), "source1_entity_id", "matched_entity_ids")

    report = evaluate(preds, gt)

    print("=" * 50)
    print("EVALUATION REPORT (leaderboard-style metric)")
    print("=" * 50)
    print(f"  MACRO F_0.5          : {report['macro_f05']:.4f}   <-- leaderboard metric")
    print(f"  micro F_0.5          : {report['micro_f05']:.4f}")
    print(f"  micro precision      : {report['micro_precision']:.4f}")
    print(f"  micro recall         : {report['micro_recall']:.4f}")
    print(f"  entities             : {report['n_entities']}")
    print(f"  singletons           : {report['n_singletons']}")
    print(f"  singleton accuracy   : {report['singleton_accuracy']:.4f}")


if __name__ == "__main__":
    main()
