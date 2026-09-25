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
