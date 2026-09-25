"""Blocking module — 7-layer union strategy for candidate generation.

Based on research from:
- canonmap: initialism, soundex, phonetic, exact blocking
- StringMatcher: MinHash LSH blocking
- BlockingPy: ANN-based blocking
- goldenmatch: multi-pass blocking
- armory: blocking recall = ceiling concept
"""
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from rapidfuzz import fuzz, process
import jellyfish
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def soundex_key(name: str) -> str:
    """Generate Soundex key for a name."""
    if not name:
        return ""
    return jellyfish.soundex(name)


def metaphone_key(name: str) -> str:
    """Generate Metaphone key for a name."""
    if not name:
        return ""
    return jellyfish.metaphone(name)


def initialism_key(name: str) -> str:
    """Generate initialism key from name.
    
    From canonmap: bidirectional matching.
    If name is 2-6 chars alpha only, treat AS an initialism.
    """
    name_clean = name.strip().upper()
    if name_clean.isalpha() and 2 <= len(name_clean) <= 6 and " " not in name_clean:
        return name_clean
    parts = re.findall(r"[A-Za-z]+", name)
    return "".join(p[0].upper() for p in parts) if parts else ""


def token_sort_key(name: str) -> str:
    """Generate token-sorted key for blocking."""
    tokens = sorted(name.lower().split())
    return " ".join(tokens)


def tfidf_blocking_candidates(
    query_names: List[str],
    target_names: List[str],
    target_ids: List[str],
    threshold: float = 0.3,
    max_features: int = 10000,
) -> Dict[int, Set[str]]:
    """Generate candidates using TF-IDF cosine similarity.
    
    Returns {query_idx: set of target_ids}.
    """
    if not query_names or not target_names:
        return {}
    
    # Fit TF-IDF on all names
    all_names = query_names + target_names
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        analyzer="word",
    )
    tfidf_matrix = tfidf.fit_transform(all_names)
    
    # Compute similarities between query and target
    query_matrix = tfidf_matrix[:len(query_names)]
    target_matrix = tfidf_matrix[len(query_names):]
    
    candidates = defaultdict(set)
    
    # Batch computation for efficiency
    sim_matrix = cosine_similarity(query_matrix, target_matrix)
    
    for q_idx in range(len(query_names)):
        for t_idx in range(len(target_names)):
            if sim_matrix[q_idx, t_idx] >= threshold:
                candidates[q_idx].add(target_ids[t_idx])
    
    return candidates


def phonetic_blocking(
    query_names: List[str],
    target_names: List[str],
    target_ids: List[str],
) -> Dict[int, Set[str]]:
    """Generate candidates using Soundex + Metaphone blocking.
    
    From canonmap: phonetic and soundex blocking strategies.
    """
    candidates = defaultdict(set)
    
    # Build target index
    target_soundex = defaultdict(list)
    target_metaphone = defaultdict(list)
    
    for t_idx, (name, tid) in enumerate(zip(target_names, target_ids)):
        sx = soundex_key(name)
        mp = metaphone_key(name)
        if sx:
            target_soundex[sx].append(tid)
        if mp:
            target_metaphone[mp].append(tid)
    
    # Query
    for q_idx, name in enumerate(query_names):
        sx = soundex_key(name)
        mp = metaphone_key(name)
        
        if sx in target_soundex:
            candidates[q_idx].update(target_soundex[sx])
        if mp in target_metaphone:
            candidates[q_idx].update(target_metaphone[mp])
    
    return candidates


def initialism_blocking(
    query_names: List[str],
    target_names: List[str],
    target_ids: List[str],
) -> Dict[int, Set[str]]:
    """Generate candidates using initialism matching.
    
    From canonmap: bidirectional initialism matching.
    """
    candidates = defaultdict(set)
    
    # Build target index
    target_initialisms = defaultdict(list)
    for t_idx, (name, tid) in enumerate(zip(target_names, target_ids)):
        init = initialism_key(name)
        if init:
            target_initialisms[init].append(tid)
    
    # Query
    for q_idx, name in enumerate(query_names):
        init = initialism_key(name)
        if init in target_initialisms:
            candidates[q_idx].update(target_initialisms[init])
    
    return candidates


def exact_blocking(
    query_names: List[str],
    target_names: List[str],
    target_ids: List[str],
) -> Dict[int, Set[str]]:
    """Generate candidates using exact matching."""
    candidates = defaultdict(set)
    
    # Build target index
    target_exact = defaultdict(list)
    for t_idx, (name, tid) in enumerate(zip(target_names, target_ids)):
        target_exact[name].append(tid)
    
    # Query
    for q_idx, name in enumerate(query_names):
        if name in target_exact:
            candidates[q_idx].update(target_exact[name])
    
    return candidates


def minhash_lsh_candidates(
    query_names: List[str],
    target_names: List[str],
    target_ids: List[str],
    threshold: float = 0.4,
    num_perm: int = 128,
    shingle_size: int = 3,
) -> Dict[int, Set[str]]:
    """Generate candidates using MinHash LSH.
    
    From StringMatcher: character n-gram shingles + MinHash + LSH.
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        # Fallback: use RapidFuzz for fuzzy matching
        return _rapidfuzz_fallback_candidates(
            query_names, target_names, target_ids, threshold
        )
    
    candidates = defaultdict(set)
    
    def make_minhash(text):
        m = MinHash(num_perm=num_perm)
        # Clean and shingle
        text_clean = re.sub(r"[^\w]", "", text.lower())
        if len(text_clean) < shingle_size:
            m.update(text_clean.encode("utf8"))
        else:
            for i in range(len(text_clean) - shingle_size + 1):
                shingle = text_clean[i:i+shingle_size]
                m.update(shingle.encode("utf8"))
        return m
    
    # Build LSH index from targets
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    target_mhs = {}
    
    for t_idx, (name, tid) in enumerate(zip(target_names, target_ids)):
        mh = make_minhash(name)
        try:
            lsh.insert(f"t_{t_idx}", mh)
            target_mhs[t_idx] = mh
        except ValueError:
            pass  # Duplicate key
    
    # Query
    for q_idx, name in enumerate(query_names):
        mh = make_minhash(name)
        results = lsh.query(mh)
        for r in results:
            t_idx = int(r.split("_")[1])
            candidates[q_idx].add(target_ids[t_idx])
    
    return candidates


def _rapidfuzz_fallback_candidates(
    query_names: List[str],
    target_names: List[str],
    target_ids: List[str],
    threshold: float = 0.4,
) -> Dict[int, Set[str]]:
    """Fallback candidate generation using RapidFuzz when datasketch unavailable."""
    candidates = defaultdict(set)
    
    for q_idx, q_name in enumerate(query_names):
        # Use WRatio for fuzzy matching
        results = process.extract(
            q_name,
            [(t_name, t_id) for t_name, t_id in zip(target_names, target_ids)],
            scorer=fuzz.WRatio,
            score_cutoff=threshold * 100,
            limit=50,
        )
        for match_name, match_id, score in results:
            candidates[q_idx].add(match_id)
    
    return candidates


def address_tfidf_candidates(
    query_addrs: List[str],
    target_addrs: List[str],
    target_ids: List[str],
    threshold: float = 0.3,
) -> Dict[int, Set[str]]:
    """Generate candidates using TF-IDF on addresses."""
    # Filter out empty addresses
    valid_query = [(i, a) for i, a in enumerate(query_addrs) if a]
    valid_target = [(i, a, tid) for i, (a, tid) in enumerate(zip(target_addrs, target_ids)) if a]
    
    if not valid_query or not valid_target:
        return defaultdict(set)
    
    query_indices = [v[0] for v in valid_query]
    query_texts = [v[1] for v in valid_query]
    target_texts = [v[1] for v in valid_target]
    target_ids_filtered = [v[2] for v in valid_target]
    
    all_texts = query_texts + target_texts
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    tfidf_matrix = tfidf.fit_transform(all_texts)
    
    query_matrix = tfidf_matrix[:len(query_texts)]
    target_matrix = tfidf_matrix[len(query_texts):]
    
    candidates = defaultdict(set)
    sim_matrix = cosine_similarity(query_matrix, target_matrix)
    
    for q_local_idx, q_global_idx in enumerate(query_indices):
        for t_local_idx in range(len(target_texts)):
            if sim_matrix[q_local_idx, t_local_idx] >= threshold:
                candidates[q_global_idx].add(target_ids_filtered[t_local_idx])
    
    return candidates


def union_candidates(*candidate_dicts: Dict[int, Set[str]]) -> Dict[int, Set[str]]:
    """Union multiple candidate sets."""
    union = defaultdict(set)
    for cand_dict in candidate_dicts:
        for key, values in cand_dict.items():
            union[key].update(values)
    return union


def measure_blocking_quality(
    candidates: Dict[int, Set[str]],
    ground_truth: Dict[str, list],
    total_possible_pairs: int,
    s1_ids: List[str] = None,
) -> dict:
    """Measure blocking quality metrics.

    From armory: blocking recall = ceiling on fusion recall.

    Args:
        candidates: {s1_idx: set of candidate entity IDs}
        ground_truth: {s1_entity_id: [matched_ids]}
        total_possible_pairs: |S1| * |S2+S3|
        s1_ids: list of S1 entity IDs indexed by position (required to map idx -> id)
    """
    matches_retained = 0
    total_matches = 0

    for q_idx, candidate_ids in candidates.items():
        if s1_ids is None or q_idx >= len(s1_ids):
            continue
        s1_id = s1_ids[q_idx]
        matched_ids = ground_truth.get(s1_id, [])
        total_matches += len(matched_ids)
        matches_retained += len(set(matched_ids) & set(candidate_ids))

    candidate_count = sum(len(v) for v in candidates.values())
    reduction_ratio = 1 - (candidate_count / max(total_possible_pairs, 1))
    pair_recall = matches_retained / max(total_matches, 1)

    return {
        "candidate_pairs": candidate_count,
        "reduction_ratio": round(reduction_ratio, 4),
        "matches_retained": matches_retained,
        "total_matches": total_matches,
        "pair_recall": round(pair_recall, 4),
    }
