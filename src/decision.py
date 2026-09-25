"""Decision layer — converts calibrated pair probabilities into entity-level sets.

Implements the consensus approach of the top teams:
  1. Probability calibration (isotonic) on a held-out fold
  2. One-to-one exclusivity — each S2/S3 record belongs to at most one S1
     (verified on train ground truth: zero exceptions across 7.6M pairs)
  3. Expected-F0.5 prefix selection per S1 — optimizes the actual metric,
     including choosing the empty set for singletons

Ported from SABER's measured selector (dev stage-2 F0.5 0.98633, oracle 0.9982)
and the-resolvers' verified one-to-one constraint.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Metric helpers (exact leaderboard semantics)
# --------------------------------------------------------------------------
def entity_f05(truth: Set[str], predicted: Set[str]) -> float:
    """Per-entity F_0.5 with exact singleton semantics.

    Equivalent to 1.25*P*R/(0.25*P+R) but numerically stable:
        F_0.5 = 5*TP / (5*TP + 4*FP + FN)
    """
    if not truth:
        return 1.0 if not predicted else 0.0
    tp = len(truth & predicted)
    if tp == 0:
        return 0.0
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    return 5.0 * tp / (5.0 * tp + 4.0 * fp + fn)


def macro_f05(truth_by_id: Dict[str, Set[str]],
              pred_by_id: Dict[str, Set[str]]) -> float:
    """Macro-averaged F_0.5 over every S1 entity (missing predictions = empty)."""
    if not truth_by_id:
        return 0.0
    scores = [entity_f05(truth, pred_by_id.get(s1, set()))
              for s1, truth in truth_by_id.items()]
    return float(np.mean(scores))


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def fit_isotonic(proba: np.ndarray, y: np.ndarray):
    """Fit isotonic calibration on a held-out fold. Returns a transform fn."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(proba, y)
    return lambda p: iso.predict(p)


# --------------------------------------------------------------------------
# Exclusivity — each candidate record belongs to at most one S1
# --------------------------------------------------------------------------
def apply_exclusivity(
    s1_ids: List[str],
    cand_ids: List[str],
    proba: np.ndarray,
    p_min: float = 0.05,
) -> np.ndarray:
    """Keep only the highest-probability S1 for each candidate record.

    Tie-break: lower s1_id wins (deterministic).
    Returns a boolean keep-mask over the input pairs.

    Verified-safe: on train ground truth, no S2/S3 record matches >1 S1
    (zero exceptions across 7,638,365 pairs), so a candidate claimed by a
    better-scoring S1 is a false positive for every other S1.
    """
    keep = np.zeros(len(proba), dtype=bool)
    best: Dict[str, Tuple[float, str, int]] = {}  # cand -> (p, s1, idx)

    order = np.lexsort((np.asarray(s1_ids), -proba))
    for i in order:
        p = float(proba[i])
        if p < p_min:
            continue
        cid = cand_ids[i]
        sid = s1_ids[i]
        cur = best.get(cid)
        if cur is None or p > cur[0] or (p == cur[0] and sid < cur[1]):
            best[cid] = (p, sid, i)

    for _, (_, _, i) in best.items():
        keep[i] = True
    return keep


# --------------------------------------------------------------------------
# Expected-F0.5 set selection (per S1, including the empty set)
# --------------------------------------------------------------------------
def select_sets_expected_f05(
    s1_ids: List[str],
    cand_ids: List[str],
    proba: np.ndarray,
    anchor_ids: Optional[List[str]] = None,
    beta2: float = 0.25,
    p_min: float = 0.05,
    empty_mode: str = "max_p",
) -> Dict[str, Set[str]]:
    """Choose a match set per S1 by maximizing expected F_0.5.

    Steps (SABER's algorithm):
      1. Exclusivity: highest-p S1 owns each candidate record (p >= p_min)
      2. Per S1: sort surviving candidates by p descending
      3. For k = 1..n: expected_f(k) = (1+beta2) * sum(p_1..p_k)
                                       / (beta2 * E|T| + k)
         where E|T| = sum of all owned p (expected number of true matches)
      4. empty_score: "max_p" (baseline proxy 1 - max_p, SABER default) or
         "product" (∏(1-p_j) over owned candidates — higher for many weak
         candidates; note SABER warns this treats dependent edges as
         independent, so it is an ablation, not the default)
      5. Emit the top-k prefix if it beats empty_score, else the empty set

    Args:
        anchor_ids: full S1 roster; anchors with no candidates are emitted empty.
    Returns:
        {s1_id: set(candidate_ids)}
    """
    n = len(proba)
    if n == 0:
        return {sid: set() for sid in (anchor_ids or [])}

    # --- 1. exclusivity -----------------------------------------------------
    keep = apply_exclusivity(s1_ids, cand_ids, proba, p_min=p_min)

    # --- 2. group surviving pairs by S1 ------------------------------------
    by_s1: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    for i in range(n):
        if keep[i]:
            by_s1[s1_ids[i]].append((float(proba[i]), cand_ids[i]))

    results: Dict[str, Set[str]] = {}

    for s1, pairs in by_s1.items():
        pairs.sort(key=lambda t: (-t[0], t[1]))  # p desc, id asc (deterministic)
        ps = [p for p, _ in pairs]
        expected_truth = float(np.sum(ps))
        max_p = ps[0]

        # --- 3/4. expected F0.5 of each prefix vs empty --------------------
        cum = np.cumsum(ps)
        k = np.arange(1, len(ps) + 1)
        expected_f = (1.0 + beta2) * cum / (beta2 * expected_truth + k)

        if empty_mode == "product":
            empty_score = float(np.prod([1.0 - p for p in ps]))
        else:  # "max_p" — SABER default
            empty_score = 1.0 - max_p

        best_k = int(np.argmax(expected_f)) + 1
        if expected_f[best_k - 1] > empty_score:
            results[s1] = {cid for _, cid in pairs[:best_k]}
        else:
            results[s1] = set()

    # --- 5. anchors with zero candidates stay empty ------------------------
    if anchor_ids is not None:
        for sid in anchor_ids:
            results.setdefault(sid, set())

    return results


def select_sets_threshold(
    s1_ids: List[str],
    cand_ids: List[str],
    proba: np.ndarray,
    threshold: float,
    anchor_ids: Optional[List[str]] = None,
) -> Dict[str, Set[str]]:
    """Simple baseline decision: accept every pair above a global threshold.

    Kept for ablation — expected-F0.5 selection should beat this on the metric.
    """
    results: Dict[str, Set[str]] = defaultdict(set)
    for i in range(len(proba)):
        if proba[i] >= threshold:
            results[s1_ids[i]].add(cand_ids[i])
    if anchor_ids is not None:
        for sid in anchor_ids:
            results.setdefault(sid, set())
    return dict(results)


# --------------------------------------------------------------------------
# Candidate-oracle ceiling (blocking quality in F_0.5 units)
# --------------------------------------------------------------------------
def candidate_oracle_f05(
    truth_by_id: Dict[str, Set[str]],
    candidates_by_id: Dict[str, Set[str]],
) -> float:
    """Best achievable macro F_0.5 given the candidate sets.

    Oracle_i = 1                       if |T_i| == 0
             = 5*r_i / (4*r_i + t_i)   otherwise
    where t_i = |T_i| and r_i = |C_i ∩ T_i|.
    """
    if not truth_by_id:
        return 0.0
    scores = []
    for s1, truth in truth_by_id.items():
        t = len(truth)
        if t == 0:
            scores.append(1.0)
            continue
        r = len(truth & candidates_by_id.get(s1, set()))
        scores.append(5.0 * r / (4.0 * r + t) if r > 0 else 0.0)
    return float(np.mean(scores))
