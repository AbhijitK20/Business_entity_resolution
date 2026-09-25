"""Macro F_0.5 evaluator — leaderboard metric + oracle/segmented evaluation (K1)
+ structured error analysis (K5).

The leaderboard computes F_0.5 PER Source 1 entity, then macro-averages across
all S1 entities. Singletons are included: correctly predicting an empty list
scores 1.0, predicting any match scores 0.0.

Added by the evaluation stream (K1):
  * candidate-oracle F_0.5 ceiling   Oracle_i = 1 if t_i==0 else 5*r_i/(4*r_i+t_i)
  * per-country breakdown (US / India / France-proxy when present)
  * match-count buckets {0, 1, 2, 3-4, 5+} with retrieval + matching metrics
  * complete-match coverage          (T_i fully contained in C_i)
  * reduction ratio                  (vs all S1 x gallery comparisons)
  * singleton false-merge rate       (entities whose truth has exactly 1 match)
  * bootstrap confidence intervals   (resampled at BUSINESS GROUP = S1 level)

Added by K5:
  * error bucketing: retrieval / matching / decision-policy
  * top-K worst-entity diagnosis (optionally with pair-score evidence)

Usage:
    python scripts/evaluate.py --predictions output/matching_results.tsv \
                               --ground-truth data/dataset/train/train_ground_truth.tsv \
                               [--candidate output/candidate_pairs.tsv] \
                               [--s1-source .../train_source1.tsv] \
                               [--gallery-source .../train_source2.tsv] \
                               [--gallery-source .../train_source3.tsv] \
                               [--bootstrap 1000 --confidence 0.95 --seed 42] \
                               [--pair-scores scores.tsv --threshold 0.5] \
                               [--top-k 20 --json-out report.json]
"""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_SEED = 42            # repo-wide seed convention (training.py, model.py)
DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP = 1000

BUCKET_ORDER = ["0", "1", "2", "3-4", "5+"]


# ---------------------------------------------------------------------------
# Core metric helpers (leaderboard semantics — unchanged)
# ---------------------------------------------------------------------------
def f05(precision: float, recall: float) -> float:
    """F_0.5 = (1.25 * P * R) / (0.25 * P + R)."""
    denom = 0.25 * precision + recall
    if denom <= 0:
        return 0.0
    return (1.25 * precision * recall) / denom


def entity_f05(truth: set, predicted: set) -> float:
    """Per-entity F_0.5 with exact singleton semantics (5*TP/(5*TP+4*FP+FN))."""
    if not truth:
        return 1.0 if not predicted else 0.0
    tp = len(truth & predicted)
    if tp == 0:
        return 0.0
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    return 5.0 * tp / (5.0 * tp + 4.0 * fp + fn)


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


def macro_f05_by_bucket(
    predictions: dict,
    ground_truth: dict,
    key_fn,
) -> dict:
    """Macro F_0.5 sliced by an arbitrary grouping function.

    ``key_fn(s1_id, truth_set) -> group_label``.
    Returns {group: {"macro_f05", "n"}}.
    """
    buckets: dict = {}
    for s1_id, truth in ground_truth.items():
        label = key_fn(s1_id, truth)
        pred = predictions.get(s1_id, set())
        score = entity_f05(truth, pred)
        b = buckets.setdefault(label, {"scores": [], "n": 0})
        b["scores"].append(score)
        b["n"] += 1
    return {
        label: {"macro_f05": sum(b["scores"]) / b["n"], "n": b["n"]}
        for label, b in buckets.items()
    }


def match_count_bucket(truth: set) -> str:
    """Cardinality bucket matching the validation protocol: 0, 1, 2, 3-4, 5+."""
    n = len(truth)
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 4:
        return "3-4"
    return "5+"


# ---------------------------------------------------------------------------
# K1 — candidate-oracle ceiling
# ---------------------------------------------------------------------------
def oracle_scores(ground_truth: dict, candidates: dict) -> dict:
    """Per-entity candidate-oracle F_0.5 (blocking ceiling, per BLUEPRINT §2.6).

        Oracle_i = 1                     if t_i == 0
                 = 5*r_i / (4*r_i + t_i)  otherwise

    where t_i = |T_i| and r_i = |C_i ∩ T_i|.
    """
    out = {}
    for s1_id, truth in ground_truth.items():
        t = len(truth)
        if t == 0:
            out[s1_id] = 1.0
            continue
        r = len(truth & candidates.get(s1_id, set()))
        out[s1_id] = 5.0 * r / (4.0 * r + t) if r > 0 else 0.0
    return out


def candidate_oracle_f05(ground_truth: dict, candidates: dict) -> float:
    """Macro-averaged candidate-oracle F_0.5 (blocking ceiling)."""
    scores = oracle_scores(ground_truth, candidates)
    return sum(scores.values()) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# K1 — complete-match coverage / reduction ratio / singleton false merges
# ---------------------------------------------------------------------------
def complete_match_coverage(ground_truth: dict, candidates: dict) -> dict:
    """Fraction of S1 entities with a non-empty truth whose ENTIRE true-match
    set is contained in the retrieved candidate set.

    Definitions used everywhere in this file:
      * eligible entities: those with t_i >= 1 (singletons are excluded — they
        have no true match to retrieve, so "complete coverage" is undefined
        for them and including them would inflate the number).
      * complete_i = 1 if T_i ⊆ C_i else 0.
      * complete_match_coverage = mean(complete_i) over eligible entities.
    """
    eligible = [s1 for s1, t in ground_truth.items() if t]
    if not eligible:
        return {
            "n_eligible": 0,
            "n_complete": 0,
            "complete_match_coverage": None,
        }
    n_complete = sum(
        1 for s1 in eligible
        if truth_is_subset(ground_truth[s1], candidates.get(s1, set()))
    )
    return {
        "n_eligible": len(eligible),
        "n_complete": n_complete,
        "complete_match_coverage": n_complete / len(eligible),
    }


def truth_is_subset(truth: set, cand: set) -> bool:
    return all(t in cand for t in truth)


def reduction_ratio(n_candidate_pairs: int, n_s1: int, n_gallery: int):
    """Search-space reduction of blocking vs the full S1 x gallery product.

        reduction_ratio = 1 - n_candidate_pairs / (n_s1 * n_gallery)

    Uses actual candidate counts; returns None when the denominator is unknown
    (no gallery size supplied) or zero — never an invented denominator.
    """
    total = int(n_s1) * int(n_gallery)
    if total <= 0:
        return None
    return 1.0 - (n_candidate_pairs / total)


def singleton_false_merge_rate(predictions: dict, ground_truth: dict) -> dict:
    """False-merge rate restricted to SINGLE-MATCH entities (t_i == 1).

    Definition (kept separate from general false positives on purpose):
      * eligible = S1 entities whose ground truth contains EXACTLY ONE match
        (note: NOT t_i == 0; the all-empty baseline already covers those).
      * an entity counts as a false merge if the prediction contains any id
        that is not its single true match (|pred - truth| >= 1). Predicting the
        true match plus extras IS a false merge; predicting empty is a miss,
        not a merge.
      * rate = false merges / eligible.
    """
    eligible = [s1 for s1, t in ground_truth.items() if len(t) == 1]
    if not eligible:
        return {"n_single_match_entities": 0, "n_false_merges": 0,
                "singleton_false_merge_rate": None}
    n_bad = 0
    for s1 in eligible:
        truth = ground_truth[s1]
        pred = predictions.get(s1, set())
        if pred - truth:
            n_bad += 1
    return {
        "n_single_match_entities": len(eligible),
        "n_false_merges": n_bad,
        "singleton_false_merge_rate": n_bad / len(eligible),
    }


# ---------------------------------------------------------------------------
# K1 — segment (country / bucket) metric bundle
# ---------------------------------------------------------------------------
def segment_metrics(
    predictions: dict,
    ground_truth: dict,
    s1_ids,
    candidates: dict = None,
    n_gallery: int = None,
) -> dict:
    """Full metric bundle for one segment (country or match-count bucket).

    ``s1_ids`` is the list of ground-truth entities in the segment.
    ``candidates`` / ``n_gallery`` optional (reduction ratio needs n_gallery).
    """
    gt = {s1: ground_truth[s1] for s1 in s1_ids}
    base = evaluate(predictions, gt)

    n_true = sum(len(t) for t in gt.values())
    n_cand_pairs = None
    cand_recall = None
    n_retrieved_true = None
    oracle = None
    coverage = None
    if candidates is not None:
        n_cand_pairs = sum(len(candidates.get(s1, set())) for s1 in gt)
        n_retrieved_true = sum(
            len(t & candidates.get(s1, set())) for s1, t in gt.items()
        )
        cand_recall = (n_retrieved_true / n_true) if n_true else None
        oracle = candidate_oracle_f05(gt, candidates)
        coverage = complete_match_coverage(gt, candidates)

    red = None
    if candidates is not None and n_gallery is not None:
        red = reduction_ratio(n_cand_pairs, len(gt), n_gallery)

    return {
        "n_s1": len(gt),
        "n_true_matches": n_true,
        "n_candidate_pairs": n_cand_pairs,
        "n_retrieved_true_matches": n_retrieved_true,
        "candidate_recall": cand_recall,
        "macro_f05": base["macro_f05"],
        "micro_precision": base["micro_precision"],
        "micro_recall": base["micro_recall"],
        "micro_f05": base["micro_f05"],
        "oracle_f05": oracle,
        "complete_match_coverage": coverage["complete_match_coverage"]
        if coverage else None,
        "reduction_ratio": red,
        "singleton_accuracy": base["singleton_accuracy"],
        "n_singletons": base["n_singletons"],
    }


def group_segment_ids(ground_truth: dict, key_fn) -> dict:
    """Group ground-truth entity ids by key_fn(s1_id, truth_set)."""
    out = {}
    for s1_id, truth in ground_truth.items():
        out.setdefault(key_fn(s1_id, truth), []).append(s1_id)
    return out


# ---------------------------------------------------------------------------
# K1 — bootstrap confidence intervals at the BUSINESS-GROUP (S1) level
# ---------------------------------------------------------------------------
def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Percentile bootstrap CI for the mean of per-group values.

    The sampling unit is the BUSINESS GROUP (one value per S1 entity), never
    the pair — candidates from the same S1 are correlated, so pair-level
    bootstrapping would understate uncertainty.

    Returns {"point", "lower", "upper", "confidence", "n_boot", "seed",
             "n_groups"}.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    point = float(values.mean()) if n else float("nan")
    if n == 0 or n_boot <= 0:
        return {"point": point, "lower": None, "upper": None,
                "confidence": confidence, "n_boot": 0, "seed": seed,
                "n_groups": n}

    rng = np.random.default_rng(seed)
    # Chunk so we never materialize more than ~40M indices at once
    # (keeps memory bounded on the 2.2M-entity real dataset).
    chunk = max(1, min(int(n_boot), int(40_000_000 // max(n, 1))))
    means = np.empty(n_boot, dtype=float)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        means[done:done + b] = values[idx].mean(axis=1)
        done += b

    lo_q = (1.0 - confidence) / 2.0 * 100.0
    hi_q = (1.0 - (1.0 - confidence) / 2.0) * 100.0
    lower, upper = np.percentile(means, [lo_q, hi_q])
    return {
        "point": point,
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "n_groups": n,
    }


def bootstrap_entity_metrics(
    ground_truth: dict,
    predictions: dict,
    candidates: dict = None,
    n_boot: int = DEFAULT_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Bootstrap CIs for macro F_0.5 (and oracle F_0.5 when candidates given).

    Grouping unit = S1 business entity (one row per business group).
    """
    s1_ids = list(ground_truth.keys())
    f05_values = np.array(
        [entity_f05(ground_truth[s], predictions.get(s, set())) for s in s1_ids],
        dtype=float,
    )
    out = {
        "macro_f05": bootstrap_mean_ci(f05_values, n_boot, confidence, seed),
    }
    if candidates is not None:
        # compute the oracle map ONCE — recomputing it per entity would be
        # O(n^2) and infeasible on the 2.2M-entity real dataset.
        oracle_map = oracle_scores(ground_truth, candidates)
        oracle_values = np.array([oracle_map[s] for s in s1_ids], dtype=float)
        out["oracle_f05"] = bootstrap_mean_ci(
            oracle_values, n_boot, confidence, seed
        )
    return out


# ---------------------------------------------------------------------------
# K5 — error analysis (retrieval vs matching vs decision-policy)
# ---------------------------------------------------------------------------
def load_pair_scores(path) -> dict:
    """Load a TSV of per-pair scores: source1_entity_id, candidate_entity_id, score."""
    scores = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = ((row.get("source1_entity_id") or "").strip(),
                   (row.get("candidate_entity_id") or "").strip())
            try:
                scores[key] = float(row.get("score"))
            except (TypeError, ValueError):
                continue
    return scores


def classify_errors(
    predictions: dict,
    ground_truth: dict,
    candidates: dict,
    scores: dict = None,
    threshold: float = None,
) -> dict:
    """Bucket every error into exactly one of three categories.

    Retrieval error      — a true match that blocking never retrieved (T \\ C).
                           Ownership: blocking.
    Matching error       — a true match that WAS retrieved but the model scored
                           it below the decision cutoff, or a false positive the
                           model scored at/above the cutoff.
                           Ownership: features/model.
                           Without pair scores these are counted in
                           ``matching_error_unattributed`` (retrieved but not
                           predicted; cannot be split further).
    Decision-policy error— the score was fine (>= cutoff) but the final
                           policy (exclusivity / expected-F0.5 prefix) rejected
                           it, or the policy accepted a pair scored BELOW the
                           cutoff.
                           Ownership: decision layer.

    False positives not present in the candidate set at all are counted as
    ``integrity_errors`` (matches must be a subset of candidates).
    """
    scores = scores or {}
    scored = scores is not None and len(scores) > 0 and threshold is not None

    counts = Counter()
    per_entity = {}

    for s1_id, truth in ground_truth.items():
        pred = predictions.get(s1_id, set())
        cand = candidates.get(s1_id, set())
        cats = Counter()

        for miss in (truth - pred):
            if miss not in cand:
                cats["retrieval_error"] += 1
            elif scored:
                s = scores.get((s1_id, miss))
                if s is None:
                    cats["matching_error_unattributed"] += 1
                elif s >= threshold:
                    cats["decision_policy_error"] += 1
                else:
                    cats["matching_error"] += 1
            else:
                cats["matching_error_unattributed"] += 1

        for extra in (pred - truth):
            if extra not in cand:
                cats["integrity_error"] += 1
            elif scored:
                s = scores.get((s1_id, extra))
                if s is None:
                    cats["matching_error_unattributed_fp"] += 1
                elif s >= threshold:
                    cats["matching_error"] += 1
                else:
                    cats["decision_policy_error"] += 1
            else:
                cats["matching_error_unattributed_fp"] += 1

        if cats:
            per_entity[s1_id] = cats
            counts.update(cats)

    n_retrieved_misses = (
        counts["matching_error"]
        + counts["decision_policy_error"]
        + counts["matching_error_unattributed"]
    )
    canonical_keys = (
        "retrieval_error", "matching_error", "decision_policy_error",
        "integrity_error", "matching_error_unattributed",
        "matching_error_unattributed_fp",
    )
    counts_out = {k: int(counts[k]) for k in canonical_keys}
    return {
        "counts": counts_out,
        "per_entity": per_entity,
        "n_entities_with_errors": len(per_entity),
        "scored": scored,
        "threshold": threshold if scored else None,
        "totals": {
            "retrieval_error": counts["retrieval_error"],
            "matching_error": counts["matching_error"] + counts[
                "matching_error_unattributed"] + counts[
                "matching_error_unattributed_fp"],
            "matching_error_from_scores": counts["matching_error"],
            "matching_error_unattributed": counts["matching_error_unattributed"]
            + counts["matching_error_unattributed_fp"],
            "decision_policy_error": counts["decision_policy_error"],
            "integrity_error": counts["integrity_error"],
            "retrieved_but_missed": n_retrieved_misses,
        },
    }


def _dominant_category(cats: Counter) -> str:
    if not cats:
        return "correct"
    return sorted(cats.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


_ROOT_CAUSE = {
    "retrieval_error": "blocking never surfaced the true match (blocking miss)",
    "matching_error": "model scored a retrieved pair below/above the cutoff wrongly (feature/model issue)",
    "matching_error_unattributed": "retrieved but not predicted — needs pair scores to split matching vs decision",
    "matching_error_unattributed_fp": "false positive without score evidence (needs pair scores)",
    "decision_policy_error": "score was acceptable but exclusivity/expected-F0.5 policy changed the outcome",
    "integrity_error": "predicted id absent from candidates (pipeline integrity violation)",
    "unattributed_error": "no candidate set supplied — cannot split retrieval vs matching vs decision",
}


def worst_entities(
    predictions: dict,
    ground_truth: dict,
    candidates: dict = None,
    scores: dict = None,
    threshold: float = None,
    country_of: dict = None,
    top_k: int = 20,
    max_ids_shown: int = 8,
) -> list:
    """Diagnose the worst S1 entities by per-entity F_0.5 (ties: more true
    matches first, then id — deterministic).

    Only entities with an ACTUAL error (f05 < 1.0) are returned: when there
    are fewer than ``top_k`` error cases the list is simply shorter — correct
    entities are never padded in to fill the quota.

    Returns dicts with controlled summaries (ids + counts only, no raw PII).
    """
    rows = []
    for s1_id, truth in ground_truth.items():
        pred = predictions.get(s1_id, set())
        cand = candidates.get(s1_id, set()) if candidates is not None else set()
        score = entity_f05(truth, pred)
        retrieved_true = truth & cand
        missed_true = truth - pred
        cats = Counter()
        if candidates is not None:
            for miss in missed_true:
                if miss not in cand:
                    cats["retrieval_error"] += 1
                else:
                    cats["matching_error_unattributed"] += 1
            for extra in (pred - truth):
                cats["matching_error_unattributed_fp"] += 1
        # score-based refinement
        if scores and threshold is not None:
            cats = Counter()
            for miss in missed_true:
                if candidates is not None and miss not in cand:
                    cats["retrieval_error"] += 1
                else:
                    s = scores.get((s1_id, miss))
                    if s is None:
                        cats["matching_error_unattributed"] += 1
                    elif s >= threshold:
                        cats["decision_policy_error"] += 1
                    else:
                        cats["matching_error"] += 1
            for extra in (pred - truth):
                if candidates is not None and extra not in cand:
                    cats["integrity_error"] += 1
                else:
                    s = scores.get((s1_id, extra))
                    if s is None:
                        cats["matching_error_unattributed_fp"] += 1
                    elif s >= threshold:
                        cats["matching_error"] += 1
                    else:
                        cats["decision_policy_error"] += 1

        # without candidates, misses cannot be attributed to a bucket — but
        # they are still errors, never "correct"
        category = (_dominant_category(cats) if cats
                    else ("unattributed_error" if score < 1.0 else "correct"))
        rows.append({
            "s1_id": s1_id,
            "country": (country_of or {}).get(s1_id, "?"),
            "f05": score,
            "n_true": len(truth),
            "n_candidates": len(cand) if candidates is not None else None,
            "n_retrieved_true": len(retrieved_true) if candidates is not None else None,
            "missed_true": sorted(missed_true),
            "predicted": sorted(pred),
            "top_predicted_with_scores": sorted(
                (
                    (cid, scores.get((s1_id, cid))) if scores else (cid, None)
                    for cid in pred
                ),
                key=lambda t: (-(t[1] if t[1] is not None else 0.0), t[0]),
            ),
            "error_category": category,
            "error_counts": dict(cats),
            "root_cause": _ROOT_CAUSE.get(category, "?"),
            "_missed_true": len(missed_true),
            "_fp": len(pred - truth),
        })

    rows.sort(key=lambda r: (r["f05"], -r["n_true"], r["s1_id"]))
    worst = [r for r in rows if r["f05"] < 1.0][:top_k]
    for r in worst:
        r["missed_true"] = r["missed_true"][:max_ids_shown]
        r["predicted"] = r["predicted"][:max_ids_shown]
        r["top_predicted_with_scores"] = r["top_predicted_with_scores"][:max_ids_shown]
        r.pop("_missed_true", None)
        r.pop("_fp", None)
    return worst


def attach_feature_evidence(worst: list, records_by_id: dict, max_pairs: int = 3) -> None:
    """Enrich worst-entity rows with pairwise feature evidence for their top
    predicted candidates. ``records_by_id``: id -> (name, addr, country).
    """
    try:
        from src.features import compute_all_features
        from src.normalize import normalize_name, normalize_address, normalize_country
    except Exception as exc:  # pragma: no cover — evidence is best-effort
        print(f"  (feature evidence unavailable: {exc})")
        return

    for row in worst:
        src = records_by_id.get(row["s1_id"])
        if src is None:
            row["feature_evidence"] = []
            continue
        s1_name = normalize_name(src[0])
        s1_addr = normalize_address(src[1])
        s1_country = normalize_country(src[2])
        evidence = []
        for cand_id, cand_score in row["top_predicted_with_scores"][:max_pairs]:
            tgt = records_by_id.get(cand_id)
            if tgt is None:
                continue
            feats = compute_all_features(
                s1_name, normalize_name(tgt[0]),
                s1_addr, normalize_address(tgt[1]),
                s1_country, normalize_country(tgt[2]),
            )
            evidence.append({
                "cand_id": cand_id,
                "pair_score": cand_score,
                "name_WRatio": round(feats.get("name_WRatio", 0.0), 4),
                "addr_WRatio": round(feats.get("addr_WRatio", 0.0), 4),
                "name_jaccard": round(feats.get("name_jaccard", 0.0), 4),
                "same_country": feats.get("same_country"),
                "contradiction_count": feats.get("contradiction_count"),
                "both_addrs_present": feats.get("both_addrs_present"),
            })
        row["feature_evidence"] = evidence


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def load_country_map(path, keep_records: bool = True):
    """id -> country; records (name, addr, country) kept only when needed.

    Retaining records for every row costs GBs on the real dataset, so callers
    that don't need --feature-evidence pass keep_records=False.
    """
    country_of = {}
    records = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rid = (row.get("entity_id") or "").strip()
            country_of[rid] = (row.get("country") or "?").strip()
            if keep_records:
                records[rid] = (
                    row.get("business_name") or "",
                    row.get("business_address") or "",
                    row.get("country") or "",
                )
    return country_of, records


def load_gallery_stats(paths, keep_records: bool = True):
    """Stream gallery TSVs → (total_rows, {country: rows}, {id: record tuple}).

    ``keep_records=False`` skips the per-id record dict (only --feature-evidence
    needs it) — on real data that dict alone would dominate memory.
    """
    total = 0
    by_country = Counter()
    records = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                total += 1
                country = (row.get("country") or "?").strip()
                by_country[country] += 1
                if keep_records:
                    rid = (row.get("entity_id") or "").strip()
                    records[rid] = (
                        row.get("business_name") or "",
                        row.get("business_address") or "",
                        row.get("country") or "",
                    )
    return total, dict(by_country), records


def _fmt(value, digits=4, pct=False):
    if value is None:
        return "n/a"
    if pct:
        return f"{value * 100:.2f}%"
    return f"{value:.{digits}f}"


def _segment_row(label, m):
    return (
        f"  {label:<10s} | n_s1={m['n_s1']:<7d} true={m['n_true_matches']:<7d} "
        f"cand={_fmt(m['n_candidate_pairs'], 0):>8s} "
        f"cand_rec={_fmt(m['candidate_recall'], 4):>7s} "
        f"macro={m['macro_f05']:.4f} oracle={_fmt(m['oracle_f05'])} "
        f"P={_fmt(m['micro_precision'])} R={_fmt(m['micro_recall'])} "
        f"F0.5_micro={_fmt(m['micro_f05'])} "
        f"cover={_fmt(m['complete_match_coverage'], 4, pct=True)} "
        f"red={_fmt(m['reduction_ratio'], 4, pct=True)}"
    )


def build_report(args) -> dict:
    preds = load_dict(Path(args.predictions), "source1_entity_id", "matched_entity_ids")
    gt = load_dict(Path(args.ground_truth), "source1_entity_id", "matched_entity_ids")

    cands = None
    if args.candidate:
        cands = load_dict(Path(args.candidate), "source1_entity_id", "candidate_entity_ids")

    country_of, s1_records = {}, {}
    keep_records = bool(getattr(args, "feature_evidence", False))
    if args.s1_source:
        country_of, s1_records = load_country_map(
            args.s1_source, keep_records=keep_records
        )

    n_gallery = args.n_gallery
    gallery_by_country = None
    gallery_records = {}
    if args.gallery_source:
        n_gallery, gallery_by_country, gallery_records = load_gallery_stats(
            args.gallery_source, keep_records=keep_records
        )

    base = evaluate(preds, gt)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "args": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "overall": base,
    }

    # --- oracle ceiling -----------------------------------------------------
    if cands is not None:
        report["oracle_f05"] = candidate_oracle_f05(gt, cands)
        report["selector_loss"] = report["oracle_f05"] - base["macro_f05"]
        report["complete_match_coverage"] = complete_match_coverage(gt, cands)
        report["candidate_recall_micro"] = (
            sum(len(t & cands.get(s, set())) for s, t in gt.items())
            / max(sum(len(t) for t in gt.values()), 1)
        )
        report["n_candidate_pairs"] = sum(len(c) for c in cands.values())
        if n_gallery is not None:
            report["reduction_ratio"] = reduction_ratio(
                report["n_candidate_pairs"], len(gt), n_gallery
            )
            report["n_gallery"] = n_gallery

    # --- singleton false-merge rate ----------------------------------------
    report["singleton_false_merge"] = singleton_false_merge_rate(preds, gt)

    # --- match-count buckets ------------------------------------------------
    bucket_ids = group_segment_ids(gt, lambda s1, t: match_count_bucket(t))
    # Buckets partition S1 entities, not the gallery → denominator uses the
    # full gallery size (n_s1_bucket × n_gallery).
    report["buckets"] = {
        b: segment_metrics(preds, gt, bucket_ids.get(b, []), cands, n_gallery)
        for b in BUCKET_ORDER
    }

    # --- per-country --------------------------------------------------------
    if country_of:
        country_ids = group_segment_ids(gt, lambda s1, t: country_of.get(s1, "?"))
        report["countries"] = {}
        gal_fallback = gallery_by_country is None and n_gallery is not None
        for country, ids in sorted(country_ids.items()):
            gal = None
            if gallery_by_country is not None:
                gal = gallery_by_country.get(country)
            elif n_gallery is not None:
                gal = n_gallery
            report["countries"][country] = segment_metrics(
                preds, gt, ids, cands, gal
            )
        report["countries_note"] = (
            "France appears only when the evaluation data actually contains "
            "France-labelled S1 rows (test split); never fabricated."
        )
        if gal_fallback:
            report["countries_note"] += (
                " Per-country reduction ratios use the TOTAL gallery size "
                "(--n-gallery fallback) — pass --gallery-source with a "
                "country column for exact per-country denominators."
            )

    # --- bootstrap CIs (business-group level) --------------------------------
    if args.bootstrap and args.bootstrap > 0:
        report["bootstrap"] = bootstrap_entity_metrics(
            gt, preds, cands,
            n_boot=args.bootstrap,
            confidence=args.confidence,
            seed=args.seed,
        )
        report["bootstrap"]["unit"] = "business_group(S1 entity)"
        report["bootstrap"]["confidence"] = args.confidence
        report["bootstrap"]["seed"] = args.seed
        report["bootstrap"]["n_boot"] = args.bootstrap
        report["bootstrap"]["note"] = (
            "Percentile bootstrap resampling S1 business groups (never pairs); "
            f"{args.bootstrap} replicates, confidence={args.confidence}, "
            f"seed={args.seed}."
        )

    # --- error analysis (K5) -------------------------------------------------
    scores = load_pair_scores(args.pair_scores) if args.pair_scores else None
    if scores is not None and args.threshold is None:
        raise SystemExit("--threshold is required together with --pair-scores")
    if cands is not None:
        report["errors"] = classify_errors(preds, gt, cands, scores, args.threshold)
        report["worst_entities"] = worst_entities(
            preds, gt, cands, scores, args.threshold, country_of, top_k=args.top_k
        )
        if args.feature_evidence and s1_records:
            records = dict(s1_records)
            records.update(gallery_records)
            attach_feature_evidence(report["worst_entities"], records)

    return report


def print_report(report: dict, args) -> None:
    base = report["overall"]
    print("=" * 78)
    print("EVALUATION REPORT (leaderboard-style metric)")
    print("=" * 78)
    print(f"  MACRO F_0.5          : {base['macro_f05']:.4f}   <-- leaderboard metric")
    print(f"  micro F_0.5          : {base['micro_f05']:.4f}")
    print(f"  micro precision      : {base['micro_precision']:.4f}")
    print(f"  micro recall         : {base['micro_recall']:.4f}")
    print(f"  entities             : {base['n_entities']}")
    print(f"  singletons           : {base['n_singletons']}")
    print(f"  singleton accuracy   : {base['singleton_accuracy']:.4f}")

    if "oracle_f05" in report:
        print()
        print("  --- BLOCKING CEILING (candidate oracle F_0.5) ---")
        print(f"  candidate oracle F_0.5 : {report['oracle_f05']:.4f}")
        print(f"  selector loss          : {report['selector_loss']:.4f} "
              f"(achieved {base['macro_f05']:.4f})")
        print(f"  candidate recall       : {report['candidate_recall_micro']:.4f} "
              f"(micro over true matches)")
        cov = report["complete_match_coverage"]
        print(f"  complete-match coverage: "
              f"{_fmt(cov['complete_match_coverage'], 4, pct=True)} "
              f"({cov['n_complete']}/{cov['n_eligible']} S1 with t_i>=1 "
              f"have T_i ⊆ C_i)")
        if "reduction_ratio" in report:
            print(f"  reduction ratio        : "
                  f"{_fmt(report['reduction_ratio'], 4, pct=True)} "
                  f"({report['n_candidate_pairs']} candidates vs "
                  f"{base['n_entities'] * report['n_gallery']} possible S1×gallery "
                  f"pairs; n_gallery={report['n_gallery']})")

    sfm = report["singleton_false_merge"]
    print()
    print("  --- SINGLETON FALSE-MERGE RATE (ground truth has exactly 1 match) ---")
    print(f"  false merges          : {sfm['n_false_merges']}/"
          f"{sfm['n_single_match_entities']} = "
          f"{_fmt(sfm['singleton_false_merge_rate'], 4, pct=True)}")

    print()
    print("  --- BY MATCH-COUNT BUCKET (truth |T_i| per S1) ---")
    for b in BUCKET_ORDER:
        if b in report["buckets"]:
            print(_segment_row(b, report["buckets"][b]))

    if "countries" in report:
        print()
        print("  --- BY COUNTRY ---")
        for country, m in report["countries"].items():
            print(_segment_row(country, m))
        print(f"  note: {report['countries_note']}")

    if "bootstrap" in report:
        print()
        bs = report["bootstrap"]
        print(f"  --- BOOTSTRAP {int(bs['confidence'] * 100)}% CI "
              f"(unit={bs['unit']}, n_boot={bs.get('n_boot', 0)}, "
              f"seed={bs.get('seed')}) ---")
        for metric in ("macro_f05", "oracle_f05"):
            if metric in bs and bs[metric].get("lower") is not None:
                ci = bs[metric]
                print(f"  {metric:<12s}: point={ci['point']:.4f}  "
                      f"CI=[{ci['lower']:.4f}, {ci['upper']:.4f}]  "
                      f"(n_groups={ci['n_groups']})")
        if bs.get("note"):
            print(f"  note: {bs['note']}")

    if "errors" in report:
        err = report["errors"]
        t = err["totals"]
        print()
        print("  --- ERROR BUCKETS (retrieval vs matching vs decision-policy) ---")
        print(f"  retrieval errors       : {t['retrieval_error']} "
              f"(true match never in candidates)")
        print(f"  matching errors        : {t['matching_error']} "
              f"(of which unattributed without scores: "
              f"{t['matching_error_unattributed']})")
        print(f"  decision-policy errors : {t['decision_policy_error']}")
        print(f"  integrity errors       : {t['integrity_error']} "
              f"(predicted id not in candidates)")
        print(f"  entities with errors   : {err['n_entities_with_errors']}")
        if not err["scored"]:
            print("  note: no --pair-scores/--threshold given → matching vs "
                  "decision cannot be split; retrieved-but-missed pairs are "
                  "reported as matching_error_unattributed.")

    if report.get("worst_entities"):
        print()
        print(f"  --- TOP {len(report['worst_entities'])} WORST S1 ENTITIES ---")
        for r in report["worst_entities"]:
            print(f"  {r['s1_id']} country={r['country']} f05={r['f05']:.4f} "
                  f"|T|={r['n_true']} |C|={r['n_candidates']} "
                  f"retrieved_true={r['n_retrieved_true']} "
                  f"cat={r['error_category']}")
            print(f"      missed={r['missed_true']} pred={r['predicted']}")
            print(f"      likely cause: {r['root_cause']}")
            if r.get("feature_evidence"):
                for ev in r["feature_evidence"]:
                    print(f"      evidence {ev['cand_id']}: {ev}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--candidate", default=None,
                    help="Optional candidate_pairs.tsv → oracle ceiling + coverage")
    ap.add_argument("--s1-source", default=None,
                    help="source1 TSV → per-country breakdown (+ feature evidence)")
    ap.add_argument("--gallery-source", action="append", default=None,
                    help="gallery TSV(s) (repeatable) → reduction ratio per country")
    ap.add_argument("--n-gallery", type=int, default=None,
                    help="gallery size if TSVs unavailable (fallback)")
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP,
                    help="business-group bootstrap replicates (0 disables)")
    ap.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--pair-scores", default=None,
                    help="TSV source1_entity_id/candidate_entity_id/score "
                         "→ splits matching vs decision-policy errors")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--feature-evidence", action="store_true",
                    help="compute pairwise features for worst-entity diagnosis")
    ap.add_argument("--json-out", default=None, help="write full report JSON here")
    args = ap.parse_args()

    report = build_report(args)
    print_report(report, args)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report written: {out}")


if __name__ == "__main__":
    main()
