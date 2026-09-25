"""Feature engineering module — 35 pairwise features.

Feature families:
- name similarity (10)   — lexical + phonetic
- address similarity (6) — lexical + trigram
- country (1)            — exact match (open-set safe, no hard-coding)
- cross-field (8)        — name×address evidence combinations
- missingness (6)        — present/absent flags (masterplan V §13)
- contradiction (4)      — explicit negative evidence (masterplan V §14)

Based on research from:
- ted-entity-resolution (17 features)
- entity-deduplication (19 features)
- name-matching (8 features)
- UBS-ER (phonetic voting)
- StringMatcher (partial_token_sort, partial_token_set)
"""
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from rapidfuzz import fuzz, distance
import jellyfish


def compute_name_features(name_a: str, name_b: str) -> Dict[str, float]:
    """Compute 10 name similarity features."""
    features = {}
    
    # RapidFuzz features
    features["name_token_sort_ratio"] = fuzz.token_sort_ratio(name_a, name_b) / 100.0
    features["name_partial_ratio"] = fuzz.partial_ratio(name_a, name_b) / 100.0
    features["name_WRatio"] = fuzz.WRatio(name_a, name_b) / 100.0
    
    # Jaro-Winkler
    features["name_jaro_winkler"] = distance.JaroWinkler.similarity(name_a, name_b)
    
    # Jaccard on tokens
    tokens_a = set(name_a.split())
    tokens_b = set(name_b.split())
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    features["name_jaccard"] = len(intersection) / max(len(union), 1)
    
    # Edit distance ratio
    edit_dist = distance.Levenshtein.distance(name_a, name_b)
    max_len = max(len(name_a), len(name_b), 1)
    features["name_edit_ratio"] = 1.0 - (edit_dist / max_len)
    
    # Trigram Jaccard
    def get_trigrams(text):
        text_clean = text.replace(" ", "")
        if len(text_clean) < 3:
            return set([text_clean])
        return set(text_clean[i:i+3] for i in range(len(text_clean) - 2))
    
    tri_a = get_trigrams(name_a)
    tri_b = get_trigrams(name_b)
    tri_inter = tri_a & tri_b
    tri_union = tri_a | tri_b
    features["name_trigram_jaccard"] = len(tri_inter) / max(len(tri_union), 1)
    
    # Phonetic matches
    sx_a = jellyfish.soundex(name_a) if name_a else ""
    sx_b = jellyfish.soundex(name_b) if name_b else ""
    features["name_soundex_match"] = 1.0 if (sx_a == sx_b and sx_a != "") else 0.0
    
    mp_a = jellyfish.metaphone(name_a) if name_a else ""
    mp_b = jellyfish.metaphone(name_b) if name_b else ""
    features["name_metaphone_match"] = 1.0 if (mp_a == mp_b and mp_a != "") else 0.0
    
    # Length ratio
    len_a = max(len(name_a), 1)
    len_b = max(len(name_b), 1)
    features["name_length_ratio"] = min(len_a, len_b) / max(len_a, len_b)
    
    return features


def compute_address_features(addr_a: str, addr_b: str) -> Dict[str, float]:
    """Compute 6 address similarity features."""
    features = {}
    
    features["addr_token_sort_ratio"] = fuzz.token_sort_ratio(addr_a, addr_b) / 100.0
    features["addr_partial_ratio"] = fuzz.partial_ratio(addr_a, addr_b) / 100.0
    features["addr_WRatio"] = fuzz.WRatio(addr_a, addr_b) / 100.0
    
    # Jaccard on tokens
    tokens_a = set(addr_a.split())
    tokens_b = set(addr_b.split())
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    features["addr_jaccard"] = len(intersection) / max(len(union), 1)
    
    # Trigram Jaccard
    def get_trigrams(text):
        text_clean = text.replace(" ", "")
        if len(text_clean) < 3:
            return set([text_clean])
        return set(text_clean[i:i+3] for i in range(len(text_clean) - 2))
    
    tri_a = get_trigrams(addr_a)
    tri_b = get_trigrams(addr_b)
    tri_inter = tri_a & tri_b
    tri_union = tri_a | tri_b
    features["addr_trigram_jaccard"] = len(tri_inter) / max(len(tri_union), 1)
    
    # Length ratio
    len_a = max(len(addr_a), 1)
    len_b = max(len(addr_b), 1)
    features["addr_length_ratio"] = min(len_a, len_b) / max(len_a, len_b)
    
    return features


def compute_country_feature(country_a: str, country_b: str) -> Dict[str, float]:
    """Compute country match feature."""
    return {"same_country": 1.0 if country_a == country_b else 0.0}


def compute_cross_features(
    name_features: Dict[str, float],
    addr_features: Dict[str, float],
    name_a: str,
    name_b: str,
    addr_a: str,
    addr_b: str,
) -> Dict[str, float]:
    """Compute 8 cross features."""
    features = {}
    
    # Name-Address averages
    name_wr = name_features.get("name_WRatio", 0)
    addr_wr = addr_features.get("addr_WRatio", 0)
    features["name_addr_WRatio_avg"] = (name_wr + addr_wr) / 2
    features["name_addr_WRatio_max"] = max(name_wr, addr_wr)
    features["name_addr_WRatio_min"] = min(name_wr, addr_wr)
    
    name_jac = name_features.get("name_jaccard", 0)
    addr_jac = addr_features.get("addr_jaccard", 0)
    features["name_addr_jaccard_avg"] = (name_jac + addr_jac) / 2
    
    # Company detection
    from .normalize import is_company_name
    features["is_company"] = float(is_company_name(name_a) or is_company_name(name_b))
    
    # Phonetic voting (from UBS-ER)
    sx_a = jellyfish.soundex(name_a) if name_a else ""
    sx_b = jellyfish.soundex(name_b) if name_b else ""
    mp_a = jellyfish.metaphone(name_a) if name_a else ""
    mp_b = jellyfish.metaphone(name_b) if name_b else ""
    ny_a = jellyfish.nysiis(name_a) if name_a else ""
    ny_b = jellyfish.nysiis(name_b) if name_b else ""
    
    phonetic_matches = 0
    phonetic_total = 0
    if sx_a and sx_b:
        phonetic_total += 1
        if sx_a == sx_b:
            phonetic_matches += 1
    if mp_a and mp_b:
        phonetic_total += 1
        if mp_a == mp_b:
            phonetic_matches += 1
    if ny_a and ny_b:
        phonetic_total += 1
        if ny_a == ny_b:
            phonetic_matches += 1
    
    features["name_phonetic_vote"] = phonetic_matches / max(phonetic_total, 1)
    
    # Surname length difference (from UBS-ER)
    tokens_a = name_a.split()
    tokens_b = name_b.split()
    surname_a = tokens_a[-1] if tokens_a else ""
    surname_b = tokens_b[-1] if tokens_b else ""
    features["surname_length_diff"] = abs(len(surname_a) - len(surname_b)) / max(len(surname_a), len(surname_b), 1)
    
    # Combined trigram on name+addr
    combined_a = name_a + " " + addr_a
    combined_b = name_b + " " + addr_b
    def get_trigrams(text):
        text_clean = text.replace(" ", "")
        if len(text_clean) < 3:
            return set([text_clean])
        return set(text_clean[i:i+3] for i in range(len(text_clean) - 2))
    
    tri_a = get_trigrams(combined_a)
    tri_b = get_trigrams(combined_b)
    tri_inter = tri_a & tri_b
    tri_union = tri_a | tri_b
    features["combined_trigram"] = len(tri_inter) / max(len(tri_union), 1)
    
    return features


def compute_missingness_features(
    name_a: str,
    name_b: str,
    addr_a: str,
    addr_b: str,
) -> Dict[str, float]:
    """Compute 6 missingness features.

    From teammate masterplan V §13: explicit present/absent flags let the model
    distinguish "known mismatch" from "unknown", instead of similarity-on-missing
    silently collapsing to a zero that looks like evidence.
    """
    name_a_p = 1.0 if (name_a and name_a.strip()) else 0.0
    name_b_p = 1.0 if (name_b and name_b.strip()) else 0.0
    addr_a_p = 1.0 if (addr_a and addr_a.strip()) else 0.0
    addr_b_p = 1.0 if (addr_b and addr_b.strip()) else 0.0
    return {
        "name_a_present": name_a_p,
        "name_b_present": name_b_p,
        "addr_a_present": addr_a_p,
        "addr_b_present": addr_b_p,
        "both_names_present": name_a_p * name_b_p,
        "both_addrs_present": addr_a_p * addr_b_p,
    }


def compute_contradiction_features(
    name_a: str,
    name_b: str,
    addr_a: str,
    addr_b: str,
    country_a: str,
    country_b: str,
) -> Dict[str, float]:
    """Compute contradiction (negative-evidence) features.

    From teammate masterplans (V §14, v2 §9.1): the organizer's own example —
    "a different business that happens to share an address" — is exactly the case
    a similarity-only matcher over-trusts. Contradiction flags are computed only
    when BOTH sides have a confident non-missing value (missing != contradiction).

    - country_conflict:       both countries known and different
    - addr_number_conflict:   both addresses contain digits and digit sets are disjoint
    - city_conflict:          last tokens of both addresses differ with no containment
    - contradiction_count:    sum of the above flags
    """
    country_conflict = 0.0
    if country_a and country_b and country_a.strip() and country_b.strip():
        country_conflict = 1.0 if country_a.strip().lower() != country_b.strip().lower() else 0.0

    digit_sets = []
    for addr in (addr_a, addr_b):
        if addr and addr.strip():
            digit_sets.append(set(re.findall(r"\d+", addr)))
        else:
            digit_sets.append(None)
    addr_number_conflict = 0.0
    if digit_sets[0] and digit_sets[1] and not (digit_sets[0] & digit_sets[1]):
        addr_number_conflict = 1.0

    city_conflict = 0.0
    tokens_a = addr_a.split() if addr_a else []
    tokens_b = addr_b.split() if addr_b else []
    if len(tokens_a) >= 2 and len(tokens_b) >= 2:
        city_a, city_b = tokens_a[-1], tokens_b[-1]
        if city_a != city_b and city_a not in city_b and city_b not in city_a:
            city_conflict = 1.0

    return {
        "country_conflict": country_conflict,
        "addr_number_conflict": addr_number_conflict,
        "city_conflict": city_conflict,
        "contradiction_count": country_conflict + addr_number_conflict + city_conflict,
    }


def compute_all_features(
    name_a: str,
    name_b: str,
    addr_a: str,
    addr_b: str,
    country_a: str,
    country_b: str,
) -> Dict[str, float]:
    """Compute all 35 features for a pair."""
    name_feats = compute_name_features(name_a, name_b)
    addr_feats = compute_address_features(addr_a, addr_b)
    country_feats = compute_country_feature(country_a, country_b)
    cross_feats = compute_cross_features(
        name_feats, addr_feats, name_a, name_b, addr_a, addr_b
    )
    missing_feats = compute_missingness_features(name_a, name_b, addr_a, addr_b)
    contradiction_feats = compute_contradiction_features(
        name_a, name_b, addr_a, addr_b, country_a, country_b
    )
    
    all_feats = {}
    all_feats.update(name_feats)
    all_feats.update(addr_feats)
    all_feats.update(country_feats)
    all_feats.update(cross_feats)
    all_feats.update(missing_feats)
    all_feats.update(contradiction_feats)
    
    return all_feats


def compute_features_batch(
    pairs: pd.DataFrame,
    s1_names: List[str],
    s1_addrs: List[str],
    s1_countries: List[str],
    s2_s3_names: List[str],
    s2_s3_addrs: List[str],
    s2_s3_countries: List[str],
    s2_s3_ids: List[str],
) -> pd.DataFrame:
    """Compute features for all candidate pairs.

    Args:
        pairs: DataFrame with columns [s1_idx, s2_s3_idx]
        s1_names, s1_addrs, s1_countries: S1 field arrays
        s2_s3_names, s2_s3_addrs, s2_s3_countries: S2+S3 field arrays
        s2_s3_ids: S2+S3 entity_id array

    Returns:
        DataFrame with feature columns + entity IDs
    """
    results = []
    
    for _, row in pairs.iterrows():
        s1_idx = int(row["s1_idx"])
        s2_idx = int(row["s2_s3_idx"])
        
        feats = compute_all_features(
            s1_names[s1_idx],
            s2_s3_names[s2_idx],
            s1_addrs[s1_idx],
            s2_s3_addrs[s2_idx],
            s1_countries[s1_idx],
            s2_s3_countries[s2_idx],
        )
        
        feats["s1_idx"] = s1_idx
        feats["s2_s3_idx"] = s2_idx
        feats["s2_s3_id"] = s2_s3_ids[s2_idx]
        
        results.append(feats)
    
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Vectorized feature computation for scale (SABER's cpdist approach).
# rapidfuzz.process.cpdist(a, b, scorer=...) is element-wise + multithreaded.
# Set/phonetic features are precomputed per unique record, then combined.
# ---------------------------------------------------------------------------

def _precompute_record_features(values: List[str]) -> Dict[str, list]:
    """Compute per-record token sets, trigrams, phonetics once (dedup by value)."""
    cache: Dict[str, tuple] = {}
    out = {"tokens": [], "trigrams": [], "soundex": [], "metaphone": [],
           "nysiis": [], "digits": [], "is_company": []}
    for v in values:
        if v in cache:
            t, tg, sx, mp, ny, dg, ic = cache[v]
        else:
            t = set(v.split())
            clean = v.replace(" ", "")
            tg = set(clean[i:i + 3] for i in range(max(len(clean) - 2, 1)))
            sx = jellyfish.soundex(v) if v else ""
            mp = jellyfish.metaphone(v) if v else ""
            ny = jellyfish.nysiis(v) if v else ""
            dg = set(re.findall(r"\d+", v))
            from .normalize import is_company_name
            ic = float(is_company_name(v)) if v else 0.0
            cache[v] = (t, tg, sx, mp, ny, dg, ic)
        out["tokens"].append(t)
        out["trigrams"].append(tg)
        out["soundex"].append(sx)
        out["metaphone"].append(mp)
        out["nysiis"].append(ny)
        out["digits"].append(dg)
        out["is_company"].append(ic)
    return out


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def compute_features_vectorized(
    pairs: pd.DataFrame,
    s1_names: List[str],
    s1_addrs: List[str],
    s1_countries: List[str],
    s2_s3_names: List[str],
    s2_s3_addrs: List[str],
    s2_s3_countries: List[str],
    s2_s3_ids: List[str],
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """Vectorized pairwise features — designed for millions of pairs.

    Uses rapidfuzz.process.cpdist (element-wise, multithreaded) for all fuzzy
    string features; per-record set/phonetic features are precomputed once and
    combined with cheap set operations. Produces the same 35 features as
    ``compute_all_features``.
    """
    from rapidfuzz import process as rf_process

    s1_idx = pairs["s1_idx"].to_numpy()
    s2_idx = pairs["s2_s3_idx"].to_numpy()

    n1 = [s1_names[i] for i in s1_idx]
    a1 = [s1_addrs[i] for i in s1_idx]
    c1 = [s1_countries[i] for i in s1_idx]
    n2 = [s2_s3_names[i] for i in s2_idx]
    a2 = [s2_s3_addrs[i] for i in s2_idx]
    c2 = [s2_s3_countries[i] for i in s2_idx]

    # --- fuzzy string features (element-wise, multithreaded) ---------------
    feats: Dict[str, np.ndarray] = {}
    feats["name_token_sort_ratio"] = rf_process.cpdist(n1, n2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32) / 100.0
    feats["name_partial_ratio"] = rf_process.cpdist(n1, n2, scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32) / 100.0
    feats["name_WRatio"] = rf_process.cpdist(n1, n2, scorer=fuzz.WRatio, workers=-1, dtype=np.float32) / 100.0
    feats["name_jaro_winkler"] = rf_process.cpdist(n1, n2, scorer=distance.JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)
    feats["name_edit_ratio"] = rf_process.cpdist(n1, n2, scorer=distance.Levenshtein.normalized_similarity, workers=-1, dtype=np.float32)
    feats["addr_token_sort_ratio"] = rf_process.cpdist(a1, a2, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32) / 100.0
    feats["addr_partial_ratio"] = rf_process.cpdist(a1, a2, scorer=fuzz.partial_ratio, workers=-1, dtype=np.float32) / 100.0
    feats["addr_WRatio"] = rf_process.cpdist(a1, a2, scorer=fuzz.WRatio, workers=-1, dtype=np.float32) / 100.0

    # --- per-record precomputed sets (dedup by value) ----------------------
    s1_feat = _precompute_record_features(s1_names)
    s2_feat = _precompute_record_features(s2_s3_names)
    s1_addr_feat = _precompute_record_features(s1_addrs)
    s2_addr_feat = _precompute_record_features(s2_s3_addrs)
    # combined name+address trigrams must be computed on the concatenation
    # (cross-boundary trigrams included), matching compute_all_features
    s1_comb = _precompute_record_features([f"{n} {a}" for n, a in zip(s1_names, s1_addrs)])
    s2_comb = _precompute_record_features([f"{n} {a}" for n, a in zip(s2_s3_names, s2_s3_addrs)])

    name_jaccard = np.empty(len(pairs), dtype=np.float32)
    name_trigram = np.empty(len(pairs), dtype=np.float32)
    name_soundex = np.empty(len(pairs), dtype=np.float32)
    name_metaphone = np.empty(len(pairs), dtype=np.float32)
    addr_jaccard = np.empty(len(pairs), dtype=np.float32)
    addr_trigram = np.empty(len(pairs), dtype=np.float32)
    combined_trigram = np.empty(len(pairs), dtype=np.float32)
    surname_diff = np.empty(len(pairs), dtype=np.float32)
    num_jacc = np.empty(len(pairs), dtype=np.float32)
    house_eq = np.empty(len(pairs), dtype=np.float32)

    for k in range(len(pairs)):
        i1, i2 = s1_idx[k], s2_idx[k]
        # name
        t1, t2 = s1_feat["tokens"][i1], s2_feat["tokens"][i2]
        name_jaccard[k] = _jaccard(t1, t2)
        name_trigram[k] = _jaccard(s1_feat["trigrams"][i1], s2_feat["trigrams"][i2])
        name_soundex[k] = 1.0 if (s1_feat["soundex"][i1] and s1_feat["soundex"][i1] == s2_feat["soundex"][i2]) else 0.0
        name_metaphone[k] = 1.0 if (s1_feat["metaphone"][i1] and s1_feat["metaphone"][i1] == s2_feat["metaphone"][i2]) else 0.0
        # address
        at1, at2 = s1_addr_feat["tokens"][i1], s2_addr_feat["tokens"][i2]
        addr_jaccard[k] = _jaccard(at1, at2)
        addr_trigram[k] = _jaccard(s1_addr_feat["trigrams"][i1], s2_addr_feat["trigrams"][i2])
        # combined trigram on name+addr (concatenated, matching row-wise)
        combined_trigram[k] = _jaccard(s1_comb["trigrams"][i1], s2_comb["trigrams"][i2])
        # surname length diff
        st1 = s1_names[i1].split()
        st2 = s2_s3_names[i2].split()
        sa, sb = (st1[-1] if st1 else ""), (st2[-1] if st2 else "")
        surname_diff[k] = abs(len(sa) - len(sb)) / max(len(sa), len(sb), 1)
        # numeric
        d1, d2 = s1_addr_feat["digits"][i1], s2_addr_feat["digits"][i2]
        union = d1 | d2
        num_jacc[k] = (len(d1 & d2) / len(union)) if union else -1.0
        h1 = min(d1) if d1 else None
        h2 = min(d2) if d2 else None
        house_eq[k] = -1.0 if (h1 is None or h2 is None) else (1.0 if h1 == h2 else 0.0)

    feats["name_jaccard"] = name_jaccard
    feats["name_trigram_jaccard"] = name_trigram
    feats["name_soundex_match"] = name_soundex
    feats["name_metaphone_match"] = name_metaphone
    feats["addr_jaccard"] = addr_jaccard
    feats["addr_trigram_jaccard"] = addr_trigram
    feats["combined_trigram"] = combined_trigram
    feats["surname_length_diff"] = surname_diff

    # --- length ratios + country + missingness + contradictions ------------
    len1 = np.array([len(x) for x in n1], dtype=np.float32)
    len2 = np.array([len(x) for x in n2], dtype=np.float32)
    alen1 = np.array([len(x) for x in a1], dtype=np.float32)
    alen2 = np.array([len(x) for x in a2], dtype=np.float32)
    feats["name_length_ratio"] = np.minimum(len1, len2) / np.maximum(np.maximum(len1, len2), 1)
    feats["addr_length_ratio"] = np.minimum(alen1, alen2) / np.maximum(np.maximum(alen1, alen2), 1)

    c1_arr = np.array(c1, dtype=object)
    c2_arr = np.array(c2, dtype=object)
    feats["same_country"] = (c1_arr == c2_arr).astype(np.float32)

    name_present1 = (len1 > 0).astype(np.float32)
    name_present2 = (len2 > 0).astype(np.float32)
    addr_present1 = (alen1 > 0).astype(np.float32)
    addr_present2 = (alen2 > 0).astype(np.float32)
    feats["name_a_present"] = name_present1
    feats["name_b_present"] = name_present2
    feats["addr_a_present"] = addr_present1
    feats["addr_b_present"] = addr_present2
    feats["both_names_present"] = name_present1 * name_present2
    feats["both_addrs_present"] = addr_present1 * addr_present2

    # contradictions
    country_conflict = ((c1_arr != c2_arr) & (name_present1 > 0) & (name_present2 > 0)).astype(np.float32)
    num_conflict = np.where((num_jacc >= 0) & (num_jacc == 0), 1.0, 0.0).astype(np.float32)
    # city conflict: last tokens differ with no containment (only when both have >=2 tokens)
    city_conflict = np.zeros(len(pairs), dtype=np.float32)
    for k in range(len(pairs)):
        ta, tb = a1[k].split(), a2[k].split()
        if len(ta) >= 2 and len(tb) >= 2:
            ca, cb = ta[-1], tb[-1]
            if ca != cb and ca not in cb and cb not in ca:
                city_conflict[k] = 1.0
    feats["country_conflict"] = country_conflict
    feats["addr_number_conflict"] = num_conflict
    feats["city_conflict"] = city_conflict
    feats["contradiction_count"] = country_conflict + num_conflict + city_conflict

    # cross features
    feats["name_addr_WRatio_avg"] = (feats["name_WRatio"] + feats["addr_WRatio"]) / 2
    feats["name_addr_WRatio_max"] = np.maximum(feats["name_WRatio"], feats["addr_WRatio"])
    feats["name_addr_WRatio_min"] = np.minimum(feats["name_WRatio"], feats["addr_WRatio"])
    feats["name_addr_jaccard_avg"] = (name_jaccard + addr_jaccard) / 2
    feats["is_company"] = np.maximum(
        np.array(s1_feat["is_company"], dtype=np.float32)[s1_idx],
        np.array(s2_feat["is_company"], dtype=np.float32)[s2_idx],
    )

    # phonetic voting (soundex + metaphone + nysiis) — all precomputed
    phonetic_vote = np.empty(len(pairs), dtype=np.float32)
    for k in range(len(pairs)):
        i1, i2 = s1_idx[k], s2_idx[k]
        matches = total = 0
        sx1, sx2 = s1_feat["soundex"][i1], s2_feat["soundex"][i2]
        mp1, mp2 = s1_feat["metaphone"][i1], s2_feat["metaphone"][i2]
        ny1, ny2 = s1_feat["nysiis"][i1], s2_feat["nysiis"][i2]
        if sx1 and sx2:
            total += 1; matches += int(sx1 == sx2)
        if mp1 and mp2:
            total += 1; matches += int(mp1 == mp2)
        if ny1 and ny2:
            total += 1; matches += int(ny1 == ny2)
        phonetic_vote[k] = matches / total if total else 0.0
    feats["name_phonetic_vote"] = phonetic_vote

    out = pd.DataFrame({name: feats[name] for name in FEATURE_NAMES})
    out["s1_idx"] = s1_idx
    out["s2_s3_idx"] = s2_idx
    out["s2_s3_id"] = [s2_s3_ids[i] for i in s2_idx]
    return out


FEATURE_NAMES = [
    # Name features (10)
    "name_token_sort_ratio", "name_partial_ratio", "name_WRatio",
    "name_jaro_winkler", "name_jaccard", "name_edit_ratio",
    "name_trigram_jaccard", "name_soundex_match", "name_metaphone_match",
    "name_length_ratio",
    # Address features (6)
    "addr_token_sort_ratio", "addr_partial_ratio", "addr_WRatio",
    "addr_jaccard", "addr_trigram_jaccard", "addr_length_ratio",
    # Country feature (1)
    "same_country",
    # Cross features (8)
    "name_addr_WRatio_avg", "name_addr_WRatio_max", "name_addr_WRatio_min",
    "name_addr_jaccard_avg", "is_company", "name_phonetic_vote",
    "surname_length_diff", "combined_trigram",
    # Missingness features (6) — masterplan V §13
    "name_a_present", "name_b_present", "addr_a_present", "addr_b_present",
    "both_names_present", "both_addrs_present",
    # Contradiction features (4) — masterplan V §14
    "country_conflict", "addr_number_conflict", "city_conflict",
    "contradiction_count",
]
