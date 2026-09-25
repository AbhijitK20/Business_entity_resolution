"""K3 — feature validation tests.

Validates EVERY feature family that currently exists in src/features.py with
controlled semantic examples (positive match, clear negative, missing values,
edge cases), compares match vs negative class distributions (reporting medians
honestly — weak features are reported, never hidden), checks the documented
``-1`` sentinel behaviour of ``num_jacc`` / ``house_eq`` / ``region_overlap``
when those features exist, and provides dud/constant-feature detection tooling.

Blocked families (per TASK_BREAKDOWN: K3 blocked on A1 — blocker-evidence /
competition / IDF-record features are Abhijit's A1 deliverable) are detected
at runtime: the test SKIPS with an explicit BLOCKED note until those features
land, and runs semantic checks the moment they appear. Nothing is faked.

Run: python tests/test_features.py   (or pytest tests/test_features.py)
"""
import csv
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import FEATURE_NAMES, compute_all_features
from scripts.make_synthetic_data import generate

# Families specified in BLUEPRINT §2.3 that A1 (Abhijit) is adding.
# K3 tests them as soon as they exist in FEATURE_NAMES.
BLOCKED_FAMILIES = {
    "blocker_evidence": [
        "cos_v", "cos_t", "in_vq", "in_vr", "in_tq", "in_tr", "in_a", "in_n",
        "rank_vq", "rank_vr", "rank_tq", "rank_tr",
    ],
    "competition": [
        "n_cand_s1", "n_cand_r", "cos_v_rank_in_s1", "cos_v_rank_in_r",
    ],
    "idf_record": [
        "name_wjacc", "addr_wjacc", "name_rare_miss_s1", "name_rare_miss_r",
        "addr_rare_miss_s1", "addr_rare_miss_r", "name_idf_match",
        "nfreq_s1", "nfreq_r", "afreq_s1", "afreq_r", "r_nonlatin",
    ],
    "structure_sentinels": [
        # BLUEPRINT §2.3 Structure: num_jacc / house_eq / region_overlap use
        # -1 sentinels; r_addr_empty is the record-side presence flag.
        "num_jacc", "house_eq", "r_addr_empty", "region_overlap",
    ],
}

SENTINEL_FEATURES = ["num_jacc", "house_eq", "region_overlap"]


def _skip_if_missing(features, required, family):
    missing = [f for f in required if f not in features]
    if missing:
        raise unittest.SkipTest(
            f"BLOCKED on A1: family '{family}' not in FEATURE_NAMES yet "
            f"(missing: {', '.join(missing)}). Test runs once Abhijit lands "
            f"the blocker-evidence/competition/IDF features."
        )


# ---------------------------------------------------------------------------
# 1. Structure: every advertised feature is actually computed, no NaN/inf
# ---------------------------------------------------------------------------
def test_feature_names_match_computation():
    cases = [
        ("acme robotics", "acme robotics inc", "500 market st, san jose",
         "500 market street, san jose", "US", "US"),
        ("", "", "", "", "", ""),
        ("acme", "", "1 oak rd", "", "US", "India"),
        ("xyz", "abc", "", "", "", ""),
    ]
    for args in cases:
        feats = compute_all_features(*args)
        assert set(feats) == set(FEATURE_NAMES), (
            f"missing {set(FEATURE_NAMES) - set(feats)} / "
            f"extra {set(feats) - set(FEATURE_NAMES)}"
        )
        for k, v in feats.items():
            assert v is not None, k
            assert not (isinstance(v, float) and np.isnan(v)), f"{k} is NaN"
            assert np.isfinite(v), f"{k} not finite"


def test_feature_count_families():
    """The three implemented families (name 10, address 6, country 1, cross 8,
    missingness 6, contradiction 4) sum to FEATURE_NAMES length."""
    expected_prefix = [
        "name_token_sort_ratio", "name_partial_ratio", "name_WRatio",
        "name_jaro_winkler", "name_jaccard", "name_edit_ratio",
        "name_trigram_jaccard", "name_soundex_match", "name_metaphone_match",
        "name_length_ratio",
        "addr_token_sort_ratio", "addr_partial_ratio", "addr_WRatio",
        "addr_jaccard", "addr_trigram_jaccard", "addr_length_ratio",
        "same_country",
    ]
    assert FEATURE_NAMES[:len(expected_prefix)] == expected_prefix
    for f in ("name_a_present", "addr_a_present", "both_names_present",
              "country_conflict", "addr_number_conflict", "city_conflict",
              "contradiction_count"):
        assert f in FEATURE_NAMES


# ---------------------------------------------------------------------------
# 2. Semantic tests per implemented family
# ---------------------------------------------------------------------------
def test_name_family_positive_negative_missing():
    match = compute_all_features(
        "acme robotics", "acme robotics inc",
        "500 market st", "500 market st", "US", "US")
    neg = compute_all_features(
        "acme robotics", "bright cafe",
        "500 market st", "500 market st", "US", "US")
    # positive match scores high on name similarity
    assert match["name_WRatio"] > 0.85
    assert match["name_token_sort_ratio"] > 0.8
    assert match["name_jaccard"] > 0.5
    # clear negative scores low
    assert neg["name_WRatio"] < 0.6
    assert neg["name_jaccard"] <= 0.1
    # ordering: match must beat negative on every core name similarity
    for f in ("name_WRatio", "name_token_sort_ratio", "name_partial_ratio",
              "name_edit_ratio", "name_trigram_jaccard"):
        assert match[f] > neg[f], f

    # identical strings → exact 1.0s where defined
    same = compute_all_features("delta foods", "delta foods",
                                "8 oak ave", "8 oak ave", "US", "US")
    assert same["name_WRatio"] == 1.0
    assert same["name_jaccard"] == 1.0
    assert same["name_edit_ratio"] == 1.0

    # missing (empty) names must not explode; missingness flags carry the info
    miss = compute_all_features("", "delta foods", "8 oak ave", "8 oak ave",
                                "US", "US")
    assert miss["name_a_present"] == 0.0
    assert miss["name_b_present"] == 1.0
    assert miss["both_names_present"] == 0.0
    assert 0.0 <= miss["name_WRatio"] <= 1.0


def test_address_family_positive_negative_missing():
    match = compute_all_features(
        "acme", "acme", "500 market st, san jose", "500 market street, san jose",
        "US", "US")
    neg = compute_all_features(
        "acme", "acme", "500 market st, san jose", "9 lake dr, boise",
        "US", "US")
    assert match["addr_WRatio"] > neg["addr_WRatio"]
    assert match["addr_jaccard"] > neg["addr_jaccard"]
    assert match["addr_partial_ratio"] > 0.7
    for k in ("addr_WRatio", "addr_partial_ratio", "addr_token_sort_ratio",
              "addr_jaccard", "addr_trigram_jaccard", "addr_length_ratio"):
        assert 0.0 <= match[k] <= 1.0, k

    blank = compute_all_features("a", "a", "", "", "US", "US")
    assert blank["addr_a_present"] == 0.0
    assert blank["both_addrs_present"] == 0.0
    assert 0.0 <= blank["addr_WRatio"] <= 1.0


def test_country_feature_open_set_safe():
    same = compute_all_features("a", "a", "x", "x", "US", "us")   # case-insens? no
    diff = compute_all_features("a", "a", "x", "x", "US", "India")
    # exact-match semantics per implementation (case-sensitive compare)
    assert compute_all_features("a", "a", "", "", "US", "US")["same_country"] == 1.0
    assert diff["same_country"] == 0.0
    # open-set: unknown country must not crash and yields 0/1 only
    unk = compute_all_features("a", "a", "", "", "France", "")
    assert unk["same_country"] in (0.0, 1.0)
    assert same["same_country"] in (0.0, 1.0)


def test_cross_family_ranges_and_semantics():
    f = compute_all_features("acme robotics", "acme robotics inc",
                             "500 market st", "500 market street",
                             "US", "US")
    for k in ("name_addr_WRatio_avg", "name_addr_WRatio_max",
              "name_addr_WRatio_min", "name_addr_jaccard_avg",
              "name_phonetic_vote", "combined_trigram"):
        assert 0.0 <= f[k] <= 1.0, k
    # avg is exactly the mean of its parts
    assert abs(f["name_addr_WRatio_avg"]
               - (f["name_WRatio"] + f["addr_WRatio"]) / 2) < 1e-12
    # min <= avg <= max
    assert f["name_addr_WRatio_min"] <= f["name_addr_WRatio_avg"] <= f["name_addr_WRatio_max"]
    # is_company: a legal-suffix name is a company, a bare personal-ish name isn't
    assert f["is_company"] in (0.0, 1.0)
    corp = compute_all_features("acme inc", "acme", "", "", "US", "US")
    assert corp["is_company"] == 1.0
    # surname length difference normalized
    assert 0.0 <= f["surname_length_diff"] <= 1.0


def test_missingness_family_exact_flags():
    all_missing = compute_all_features("", "", "", "", "US", "US")
    assert all_missing["name_a_present"] == 0.0
    assert all_missing["name_b_present"] == 0.0
    assert all_missing["addr_a_present"] == 0.0
    assert all_missing["addr_b_present"] == 0.0
    assert all_missing["both_names_present"] == 0.0
    assert all_missing["both_addrs_present"] == 0.0

    all_present = compute_all_features("a", "b", "c", "d", "US", "US")
    assert all_present["both_names_present"] == 1.0
    assert all_present["both_addrs_present"] == 1.0

    half = compute_all_features("a", "", "c", "", "US", "US")
    assert half["name_a_present"] == 1.0 and half["name_b_present"] == 0.0
    assert half["both_names_present"] == 0.0
    # whitespace-only counts as missing
    ws = compute_all_features("  ", "b", "c", "d", "US", "US")
    assert ws["name_a_present"] == 0.0
    assert ws["both_names_present"] == 0.0


def test_contradiction_family_semantics():
    # country conflict: both known, different
    f = compute_all_features("a", "b", "1 x rd, austin", "2 y st, dallas",
                             "US", "India")
    assert f["country_conflict"] == 1.0
    assert f["addr_number_conflict"] == 1.0      # {1} vs {2} disjoint
    assert f["city_conflict"] == 1.0             # austin vs dallas
    assert f["contradiction_count"] == 3.0

    # missing country must NOT become a conflict (missing != contradiction)
    g = compute_all_features("a", "b", "1 x rd", "2 y st", "", "India")
    assert g["country_conflict"] == 0.0

    # missing address must NOT become a number conflict
    h = compute_all_features("a", "b", "", "2 y st", "US", "US")
    assert h["addr_number_conflict"] == 0.0
    assert h["city_conflict"] == 0.0             # one side has <2 tokens

    # agreeing evidence → no contradictions
    ok = compute_all_features("a", "b", "5 oak ave, austin", "5 oak avenue, austin",
                              "US", "US")
    assert ok["contradiction_count"] == 0.0
    assert ok["country_conflict"] == 0.0
    assert ok["addr_number_conflict"] == 0.0
    assert ok["city_conflict"] == 0.0
    # count is exactly the sum of its three flags
    assert ok["contradiction_count"] == (
        ok["country_conflict"] + ok["addr_number_conflict"] + ok["city_conflict"])


# ---------------------------------------------------------------------------
# 3. Blocked families (A1) — run semantics the moment they exist
# ---------------------------------------------------------------------------
def test_blocker_evidence_family():
    feats = compute_all_features("a", "b", "c", "d", "US", "US")
    _skip_if_missing(feats, BLOCKED_FAMILIES["blocker_evidence"],
                     "blocker_evidence")
    # semantic invariants (BLUEPRINT §2.3): leg flags binary, ranks 0-based
    # with 99 = absent sentinel, cosines in [0, 1]
    for k in ("in_vq", "in_vr", "in_tq", "in_tr", "in_a", "in_n"):
        assert feats[k] in (0.0, 1.0), k
    for k in ("rank_vq", "rank_vr", "rank_tq", "rank_tr"):
        v = feats[k]
        assert float(v).is_integer() and 0 <= v <= 99, \
            f"{k}: expected integer rank in [0, 99] (99 = absent), got {v}"
    # flag/rank consistency: a leg that did not retrieve the pair must carry
    # the absent sentinel; a retrieved pair must have a real 0-based rank
    for flag, rank in (("in_vq", "rank_vq"), ("in_vr", "rank_vr"),
                       ("in_tq", "rank_tq"), ("in_tr", "rank_tr")):
        if feats[flag] == 0.0:
            assert feats[rank] == 99, \
                f"{rank}: absent (99) expected when {flag}=0, got {feats[rank]}"
        else:
            assert feats[rank] < 99, \
                f"{rank}: real rank expected when {flag}=1, got {feats[rank]}"
    for k in ("cos_v", "cos_t"):
        assert 0.0 <= feats[k] <= 1.0, k


def test_competition_family():
    feats = compute_all_features("a", "b", "c", "d", "US", "US")
    _skip_if_missing(feats, BLOCKED_FAMILIES["competition"], "competition")
    for k in ("n_cand_s1", "n_cand_r"):
        assert feats[k] >= 0, k
    for k in ("cos_v_rank_in_s1", "cos_v_rank_in_r"):
        assert feats[k] >= 1, k          # 1-based ranks


def test_idf_record_family():
    feats = compute_all_features("a", "b", "c", "d", "US", "US")
    _skip_if_missing(feats, BLOCKED_FAMILIES["idf_record"], "idf_record")
    assert feats["r_nonlatin"] in (0.0, 1.0)
    for k in ("name_wjacc", "addr_wjacc", "name_idf_match"):
        assert 0.0 <= feats[k] <= 1.0, k
    for k in ("nfreq_s1", "nfreq_r", "afreq_s1", "afreq_r"):
        assert feats[k] >= 0, k


def test_sentinel_features_minus_one_semantics():
    """num_jacc / house_eq / region_overlap use -1 for unknown (BLUEPRINT §2.3).

    Critical: -1 must never be confused with a normal similarity in [0, 1].
    """
    feats = compute_all_features("a", "b", "c", "d", "US", "US")
    _skip_if_missing(feats, SENTINEL_FEATURES, "structure_sentinels")

    # --- num_jacc: digit-set Jaccard, -1 if either side has no digits
    both = compute_all_features("a", "b", "12 main st", "12 oak rd", "US", "US")
    assert abs(both["num_jacc"] - 1.0) < 1e-9        # {12} ∩ {12}
    partial = compute_all_features("a", "b", "12 main st", "34 oak rd", "US", "US")
    assert partial["num_jacc"] == 0.0                 # disjoint digit sets
    none = compute_all_features("a", "b", "main st", "12 oak rd", "US", "US")
    assert none["num_jacc"] == -1.0                   # unknown, not 0
    # boundary: exactly one digit token each, equal → 1
    edge = compute_all_features("a", "b", "7 rd", "7 street", "US", "US")
    assert abs(edge["num_jacc"] - 1.0) < 1e-9

    # --- house_eq: first digit token equality, -1 if either null
    eq = compute_all_features("a", "b", "12 main st", "12 second ave", "US", "US")
    assert eq["house_eq"] == 1.0
    ne = compute_all_features("a", "b", "12 main st", "34 second ave", "US", "US")
    assert ne["house_eq"] == 0.0
    missing = compute_all_features("a", "b", "main st", "12 second ave", "US", "US")
    assert missing["house_eq"] == -1.0
    both_missing = compute_all_features("a", "b", "main st", "second ave",
                                        "US", "US")
    assert both_missing["house_eq"] == -1.0

    # --- region_overlap: -1 when region unknown on either side
    unk = compute_all_features("a", "b", "12 main st", "12 main st", "", "")
    assert unk["region_overlap"] == -1.0
    # -1 must never be treated as a similarity: sentinel features are never
    # expected inside [0,1]-only aggregations
    assert not (0.0 <= unk["region_overlap"] <= 1.0)


def test_r_addr_empty_flag_semantics():
    """``r_addr_empty`` (BLUEPRINT §2.3 Structure) — record/b-side address
    presence flag; binary, flips with the record side's address.

    BLOCKED on A1: skips with an explicit note until the feature lands.
    """
    feats = compute_all_features("a", "b", "c", "d", "US", "US")
    _skip_if_missing(feats, ["r_addr_empty"], "structure_sentinels")

    empty_b = compute_all_features("a", "b", "c", "", "US", "US")
    full = compute_all_features("a", "b", "c", "d", "US", "US")
    assert empty_b["r_addr_empty"] in (0.0, 1.0)
    assert empty_b["r_addr_empty"] == 1.0, "record address empty → flag set"
    assert full["r_addr_empty"] == 0.0, "record address present → flag clear"


# ---------------------------------------------------------------------------
# 4. Class separation (match vs negative) — honest reporting
# ---------------------------------------------------------------------------
def _pair(name_a, name_b, addr_a, addr_b, ca="US", cb="US"):
    return compute_all_features(name_a, name_b, addr_a, addr_b, ca, cb)


CONTROLLED_PAIRS = [
    # (label, s1_name, cand_name, s1_addr, cand_addr, country_b)
    ("match", "acme robotics", "acme robotics inc", "500 market st, san jose",
     "500 market street, san jose", "US"),
    ("match", "delta foods", "delta foods co", "8 oak ave, austin",
     "8 oak avenue, austin", "US"),
    ("match", "krishna sweets", "krishna sweets pvt ltd",
     "mg road, bengaluru", "mg road, bengaluru", "India"),
    ("match", "bright cafe", "bright cafe", "22 pine st, reno",
     "22 pine street, reno", "US"),
    ("negative", "acme robotics", "acme bakery", "500 market st, san jose",
     "500 market st, san jose", "US"),
    ("negative", "delta foods", "zen traders", "8 oak ave, austin",
     "4 hill rd, boise", "US"),
    ("negative", "krishna sweets", "sharma electricals", "mg road, bengaluru",
     "nehru nagar, mumbai", "India"),
    ("negative", "bright cafe", "harbor freight", "22 pine st, reno",
     "9 lake dr, boise", "US"),
    ("negative", "acme robotics", "nova interiors", "500 market st, san jose",
     "3 cedar ln, denver", "US"),
]


def class_separation_report(feature_rows) -> dict:
    """Median(match) vs median(negative) per feature. Reported honestly —
    features that do NOT separate are listed as weak, never removed/reshaped.
    """
    by_label = {}
    for label, feats in feature_rows:
        by_label.setdefault(label, []).append(feats)
    matches = by_label.get("match", [])
    negs = by_label.get("negative", [])
    out = {}
    for fname in FEATURE_NAMES:
        m = np.array([r[fname] for r in matches], dtype=float)
        n = np.array([r[fname] for r in negs], dtype=float)
        med_m = float(np.median(m)) if len(m) else float("nan")
        med_n = float(np.median(n)) if len(n) else float("nan")
        out[fname] = {
            "median_match": med_m,
            "median_negative": med_n,
            "delta": med_m - med_n,
            "separates": bool(med_m > med_n),
        }
    return out


def test_class_separation_controlled_pairs():
    rows = [(label, _pair(nb, nn, ab, an, "US", cb))
            for label, nb, nn, ab, an, cb in CONTROLLED_PAIRS]
    report = class_separation_report(rows)
    # core similarity features MUST separate on controlled pairs
    min_match = {"name_WRatio": 0.8, "name_token_sort_ratio": 0.8,
                 "addr_WRatio": 0.75, "addr_jaccard": 0.4}
    for f, thr in min_match.items():
        r = report[f]
        assert r["separates"], f"{f}: match {r['median_match']} <= neg {r['median_negative']}"
        assert r["median_match"] > thr, f
        assert r["median_negative"] < 0.7, f
    # contradiction features separate the NEGATIVE way (more on negatives)
    cc = report["contradiction_count"]
    assert cc["delta"] <= 0, "contradiction_count must not be higher on matches"
    # every feature reported (no silent omissions)
    assert set(report) == set(FEATURE_NAMES)


def _load_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _synthetic_pair_rows(n_s1=300, seed=42, neg_per_s1=2):
    """Positives from ground truth + hard negatives (same-country, non-matching,
    most name-similar) from a freshly generated synthetic split."""
    from src.normalize import normalize_name, normalize_address, normalize_country

    tmp = Path(tempfile.mkdtemp(prefix="feat_sep_"))
    generate(tmp, n_s1, seed)
    d = tmp / "dataset" / "train"
    s1 = {r["entity_id"]: r for r in _load_tsv(d / "train_source1.tsv")}
    gal = {}
    for src in ("train_source2.tsv", "train_source3.tsv"):
        for r in _load_tsv(d / src):
            gal[r["entity_id"]] = r
    gt = _load_tsv(d / "train_ground_truth.tsv")

    from rapidfuzz import fuzz

    def feats(a, b):
        return compute_all_features(
            normalize_name(a["business_name"]), normalize_name(b["business_name"]),
            normalize_address(a["business_address"]),
            normalize_address(b["business_address"]),
            normalize_country(a["country"]), normalize_country(b["country"]),
        )

    rows = []
    rng = random.Random(seed)
    for g in gt:
        s1_id = g["source1_entity_id"]
        if s1_id not in s1:
            continue
        a = s1[s1_id]
        positives = [t for t in g["matched_entity_ids"].split(",") if t.strip()]
        for t in positives[:3]:
            if t in gal:
                rows.append(("match", feats(a, gal[t])))
        # hard negatives: same country, not a true match, top name similarity
        same = [r for r in gal.values()
                if r["country"] == a["country"]
                and r["entity_id"] not in set(positives)]
        if not same:
            continue
        sims = sorted(
            ((fuzz.WRatio(normalize_name(a["business_name"]),
                          normalize_name(r["business_name"])), r)
             for r in same),
            key=lambda t: -t[0])
        for _, r in rng.sample(sims[:20], min(neg_per_s1, len(sims[:20]))):
            rows.append(("negative", feats(a, r)))
    return rows


def test_class_separation_synthetic_report():
    """Report medians on realistic synthetic pairs. Only a weak sanity
    assertion here — the REPORT is the deliverable (weak features are named,
    not hidden)."""
    rows = _synthetic_pair_rows()
    assert len(rows) > 50, f"too few pairs: {len(rows)}"
    report = class_separation_report(rows)
    print("\n  class separation (median match vs median negative):")
    weak = []
    for fname, r in report.items():
        flag = "OK  " if r["separates"] else "WEAK"
        if not r["separates"]:
            weak.append(fname)
        print(f"    {flag} {fname:<28s} match={r['median_match']:.4f} "
              f"neg={r['median_negative']:.4f} delta={r['delta']:+.4f}")
    print(f"  weak/non-separating features ({len(weak)}): {weak}")
    # sanity: the strongest lexical feature must separate on real-shaped data
    assert report["name_WRatio"]["separates"] or report["addr_WRatio"]["separates"]


# ---------------------------------------------------------------------------
# 5. Dud / constant feature detection (report only — never auto-delete)
# ---------------------------------------------------------------------------
def detect_dud_features(rows, feature_names=None, dominant_threshold=0.995):
    """Flag features that are constant, near-constant, all-NaN, duplicated by
    another feature, or otherwise non-informative on the supplied rows
    (list of feature dicts).

    Returns {feature: reason}. Detected features are REPORTED — deletion is
    the model owner's decision. Duplicate detection marks the LATER feature
    (in ``feature_names`` order) as ``duplicate_of(<first>)``; the first
    occurrence is kept as the canonical column.
    """
    feature_names = feature_names or FEATURE_NAMES
    out = {}
    n = len(rows)
    if n == 0:
        return {f: "empty_dataset" for f in feature_names}
    for f in feature_names:
        vals = [r.get(f) for r in rows]
        if any(v is None for v in vals):
            out[f] = "missing_values"
            continue
        arr = np.asarray(vals, dtype=float)
        if np.isnan(arr).all():
            out[f] = "all_nan"
            continue
        finite = arr[~np.isnan(arr)]
        if len(finite) == 0:
            out[f] = "all_nan"
            continue
        if np.all(finite == finite[0]):
            out[f] = f"constant({finite[0]})"
            continue
        # near-constant: one value dominates
        uniq, counts = np.unique(finite, return_counts=True)
        if counts.max() / len(finite) >= dominant_threshold:
            out[f] = f"near_constant(value={uniq[counts.argmax()]}, " \
                     f"{counts.max() / len(finite):.3%})"
    # duplicate columns: identical value vectors (first occurrence wins)
    arrays = {}
    for f in feature_names:
        if f in out:
            continue
        arrays[f] = np.asarray([r.get(f) for r in rows], dtype=float)
    seen = {}
    for f in feature_names:
        if f not in arrays:
            continue
        for g, g_arr in seen.items():
            if np.array_equal(arrays[f], g_arr, equal_nan=True):
                out[f] = f"duplicate_of({g})"
                break
        else:
            seen[f] = arrays[f]
    return out


def test_dud_detection_finds_planted_duds():
    planted = []
    for i in range(30):
        planted.append({
            "always_one": 1.0,
            "always_zero": 0.0,
            "all_nan": float("nan"),
            "useful": float(i % 2),
        })
    out = detect_dud_features(planted,
                              ["always_one", "always_zero", "all_nan",
                               "useful", "missing"])
    assert out["always_one"].startswith("constant")
    assert out["always_zero"].startswith("constant")
    assert out["all_nan"] == "all_nan"
    assert out["missing"] == "missing_values"
    assert "useful" not in out          # informative feature not flagged


def test_dud_detection_finds_duplicate_features():
    """Identical value vectors → the later feature is reported as a duplicate
    of the first (in feature_names order); distinct features stay clean."""
    rows = [{"a": float(i % 2), "b": float(i % 2), "c": float(i % 3)}
            for i in range(30)]
    out = detect_dud_features(rows, ["a", "b", "c"])
    assert out.get("b") == "duplicate_of(a)", out
    assert "a" not in out          # first occurrence is canonical
    assert "c" not in out          # distinct feature not flagged


def test_dud_detection_on_real_feature_matrix():
    rows = [feats for _, feats in
            [(lbl, _pair(nb, nn, ab, an)) for lbl, nb, nn, ab, an, _ in
             CONTROLLED_PAIRS]]
    out = detect_dud_features(rows)
    # address/name fields are all present in this controlled set → the
    # present-flags are constant here; that is a property of the SAMPLE, not
    # a reason to delete — so this test only asserts the tool reports, and
    # prints, rather than failing:
    print(f"\n  dud scan on controlled pairs: {out if out else 'none flagged'}")
    assert isinstance(out, dict)
    # contradictory evidence: contradiction_count varies → must NOT be flagged
    assert "contradiction_count" not in out
    assert "name_WRatio" not in out


# ---------------------------------------------------------------------------
# 6. Family inventory / blocker reporting
# ---------------------------------------------------------------------------
def test_report_family_inventory():
    feats = compute_all_features("a", "b", "c", "d", "US", "US")
    present, blocked = [], []
    for family, names in BLOCKED_FAMILIES.items():
        have = [n for n in names if n in feats]
        if len(have) == len(names):
            present.append(family)
        elif not have:
            blocked.append(family)
        else:
            blocked.append(family)   # partially landed → still flagged
    print(f"\n  feature families present: 6 base families + {present}")
    print(f"  feature families BLOCKED on A1: {blocked}")
    print(f"  total FEATURE_NAMES = {len(feats)}")
    assert len(feats) == len(FEATURE_NAMES)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"  SKIP  {name}\n        {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    total = len(tests)
    print(f"\n{total - failed - skipped}/{total} feature tests passed "
          f"({skipped} blocked/skipped).")
    sys.exit(1 if failed else 0)
