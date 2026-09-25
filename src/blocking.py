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
    top_k: int = None,
    chunk_size: int = 512,
) -> Dict[int, Set[str]]:
    """Generate candidates using TF-IDF cosine similarity.

    Scale-safe: computes similarities in query chunks using SPARSE matrices
    (never materializes a dense n_query x n_target matrix) and optionally caps
    each query to its top_k targets by cosine score.

    Args:
        threshold: minimum cosine similarity to keep a pair.
        top_k: keep at most this many targets per query (None = unlimited).
        chunk_size: number of queries processed per similarity block.
    """
    if not query_names or not target_names:
        return {}
    
    # Fit TF-IDF on all names
    all_names = query_names + target_names
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        analyzer="word",
        dtype=np.float32,
    )
    tfidf_matrix = tfidf.fit_transform(all_names)
    
    query_matrix = tfidf_matrix[:len(query_names)]
    target_matrix = tfidf_matrix[len(query_names):]
    target_matrix_t = target_matrix.T.tocsr()
    
    candidates = defaultdict(set)
    n_queries = query_matrix.shape[0]
    
    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        chunk = query_matrix[start:end]
        # Sparse cosine similarity: rows of `chunk` are L2-normalized by TfidfVectorizer,
        # target rows are normalized too, so dot product == cosine.
        sims = (chunk @ target_matrix_t).tocsr()
        
        for local_i in range(end - start):
            row = sims.getrow(local_i)
            if row.nnz == 0:
                continue
            data = row.data
            indices = row.indices
            # Prune below threshold
            keep_mask = data >= threshold
            if not keep_mask.any():
                continue
            vals = data[keep_mask]
            idxs = indices[keep_mask]
            # Top-K by score
            if top_k is not None and len(vals) > top_k:
                top_pos = np.argpartition(-vals, top_k)[:top_k]
                vals, idxs = vals[top_pos], idxs[top_pos]
            q_idx = start + local_i
            for t_pos in idxs:
                candidates[q_idx].add(target_ids[t_pos])
    
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
    top_k: int = None,
    chunk_size: int = 512,
) -> Dict[int, Set[str]]:
    """Generate candidates using TF-IDF on addresses.

    Scale-safe: chunked sparse similarity + optional top-K per query.
    """
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
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, dtype=np.float32)
    tfidf_matrix = tfidf.fit_transform(all_texts)
    
    query_matrix = tfidf_matrix[:len(query_texts)]
    target_matrix = tfidf_matrix[len(query_texts):]
    target_matrix_t = target_matrix.T.tocsr()
    
    candidates = defaultdict(set)
    n_queries = query_matrix.shape[0]
    
    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        chunk = query_matrix[start:end]
        sims = (chunk @ target_matrix_t).tocsr()
        
        for local_i in range(end - start):
            row = sims.getrow(local_i)
            if row.nnz == 0:
                continue
            keep_mask = row.data >= threshold
            if not keep_mask.any():
                continue
            vals = row.data[keep_mask]
            idxs = row.indices[keep_mask]
            if top_k is not None and len(vals) > top_k:
                top_pos = np.argpartition(-vals, top_k)[:top_k]
                vals, idxs = vals[top_pos], idxs[top_pos]
            q_global = query_indices[start + local_i]
            for t_pos in idxs:
                candidates[q_global].add(target_ids_filtered[t_pos])
    
    return candidates


def cap_candidates(
    candidates: Dict[int, Set[str]],
    max_per_query: int,
    keep_fn=None,
) -> Dict[int, Set[str]]:
    """Cap each query's candidate set to at most max_per_query entries.

    ``keep_fn(s1_idx, cand_id) -> float`` optionally scores candidates for
    prioritization; higher scores survive. Default keeps insertion order.

    Used as a final per-entity budget guard (masterplan v2 §8).
    """
    if max_per_query is None:
        return candidates
    capped = {}
    for q_idx, cands in candidates.items():
        if len(cands) <= max_per_query:
            capped[q_idx] = set(cands)
            continue
        cand_list = list(cands)
        if keep_fn is not None:
            cand_list.sort(key=lambda c: keep_fn(q_idx, c), reverse=True)
        capped[q_idx] = set(cand_list[:max_per_query])
    return capped


def union_candidates(*candidate_dicts: Dict[int, Set[str]]) -> Dict[int, Set[str]]:
    """Union multiple candidate sets."""
    union = defaultdict(set)
    for cand_dict in candidate_dicts:
        for key, values in cand_dict.items():
            union[key].update(values)
    return union


# ---------------------------------------------------------------------------
# Adaptive-K pruning + bidirectional retrieval (ported from SABER's measured
# approach: reverse legs + adaptive K lifted India union recall 0.9779 → 0.9903)
# ---------------------------------------------------------------------------

def adaptive_prune(
    idx: np.ndarray,
    sc: np.ndarray,
    kmin: int,
    kmax: int,
    gap: float,
) -> np.ndarray:
    """Row-wise adaptive-K keep mask over (n, K) candidate matrices.

    Keep rank < kmin, OR score >= top1 - gap, up to kmax.
    Exact formula from SABER's blocker.
    """
    k = idx.shape[1]
    r = np.arange(k)[None, :]
    return (idx >= 0) & (r < kmax) & ((r < kmin) | (sc >= sc[:, :1] - gap))


def _topk_from_sparse_row(data, indices, kmax):
    """Return (indices, scores) of the top-kmax entries of one sparse row."""
    if len(data) > kmax:
        pos = np.argpartition(-data, kmax)[:kmax]
        data, indices = data[pos], indices[pos]
    order = np.argsort(-data)
    return indices[order], data[order]


def tfidf_blocking_adaptive(
    query_names: List[str],
    target_names: List[str],
    target_ids: List[str],
    threshold: float = 0.10,
    kmin: int = 5,
    kmax: int = 30,
    gap: float = 0.10,
    max_features: int = 10000,
    chunk_size: int = 512,
) -> Tuple[Dict[int, Set[str]], Dict[int, Dict[str, float]]]:
    """TF-IDF blocking with SABER's adaptive-K prune.

    Returns:
        candidates: {query_idx: set(target_ids)}
        scores:     {query_idx: {target_id: cosine_score}}  (for provenance features)
    """
    if not query_names or not target_names:
        return {}, {}

    all_names = query_names + target_names
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features,
                            analyzer="word", dtype=np.float32)
    tfidf_matrix = tfidf.fit_transform(all_names)
    query_matrix = tfidf_matrix[:len(query_names)]
    target_matrix = tfidf_matrix[len(query_names):].T.tocsr()

    candidates: Dict[int, Set[str]] = defaultdict(set)
    scores: Dict[int, Dict[str, float]] = defaultdict(dict)
    n_queries = query_matrix.shape[0]

    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        sims = (query_matrix[start:end] @ target_matrix).tocsr()
        for local_i in range(end - start):
            row = sims.getrow(local_i)
            if row.nnz == 0:
                continue
            keep = row.data >= threshold
            if not keep.any():
                continue
            data, indices = row.data[keep], row.indices[keep]
            top_idx, top_sc = _topk_from_sparse_row(data, indices, kmax)
            idx_arr = top_idx[None, :]
            sc_arr = top_sc[None, :]
            mask = adaptive_prune(idx_arr, sc_arr, kmin, kmax, gap)[0]
            q_idx = start + local_i
            for t_pos, s in zip(top_idx[mask], top_sc[mask]):
                tid = target_ids[t_pos]
                candidates[q_idx].add(tid)
                scores[q_idx][tid] = float(s)

    return candidates, scores


def bidirectional_tfidf(
    s1_names: List[str],
    gallery_names: List[str],
    gallery_ids: List[str],
    threshold: float = 0.10,
    forward_kmin: int = 5, forward_kmax: int = 30, forward_gap: float = 0.10,
    reverse_kmin: int = 2, reverse_kmax: int = 5, reverse_gap: float = 0.05,
    max_features: int = 10000,
) -> Tuple[Dict[int, Set[str]], Dict[int, Dict[str, float]]]:
    """Run TF-IDF blocking in BOTH directions and union.

    Forward: S1 → gallery (top forward_kmax per S1)
    Reverse: gallery → S1 (top reverse_kmax per gallery record; because each
             S2/S3 matches at most one S1, this cheap direction is high-recall)

    Returns (candidates, scores) where scores hold the max score seen.
    """
    fwd_c, fwd_s = tfidf_blocking_adaptive(
        s1_names, gallery_names, gallery_ids,
        threshold=threshold, kmin=forward_kmin, kmax=forward_kmax, gap=forward_gap,
        max_features=max_features,
    )

    # Reverse: for each gallery record, find its top S1s, then invert
    rev_c, rev_s = tfidf_blocking_adaptive(
        gallery_names, s1_names, [str(i) for i in range(len(s1_names))],
        threshold=threshold, kmin=reverse_kmin, kmax=reverse_kmax, gap=reverse_gap,
        max_features=max_features,
    )
    candidates: Dict[int, Set[str]] = defaultdict(set)
    scores: Dict[int, Dict[str, float]] = defaultdict(dict)

    for s1_idx, cands in fwd_c.items():
        candidates[s1_idx].update(cands)
        scores[s1_idx].update(fwd_s.get(s1_idx, {}))

    gallery_id_to_pos = {gid: i for i, gid in enumerate(gallery_ids)}
    for gal_idx, s1_strs in rev_c.items():
        gal_id = gallery_ids[gal_idx]
        for s1_str in s1_strs:
            s1_idx = int(s1_str)
            candidates[s1_idx].add(gal_id)
            rev_score = rev_s.get(gal_idx, {}).get(s1_str, 0.0)
            if rev_score > scores[s1_idx].get(gal_id, 0.0):
                scores[s1_idx][gal_id] = rev_score

    return candidates, scores


def key_blocking(
    s1_df: pd.DataFrame,
    gallery_df: pd.DataFrame,
    address_key_min_len: int = 12,
    max_bucket: int = 30,
    max_addr_bucket: int = 200,
) -> Dict[int, Set[str]]:
    """Exact-key blocking legs (SABER/vaibhav/resolvers consensus).

    - address key: exact normalized address of >= address_key_min_len chars
      (bucket capped at max_addr_bucket — shared buildings/malls)
    - name key: core name (legal-stripped) + last two address tokens
      (buckets > max_bucket dropped as too generic)
    - PIN/ZIP key: 5-6 digit runs in the address (India PIN / US ZIP)
    """
    candidates: Dict[int, Set[str]] = defaultdict(set)
    gallery_ids = gallery_df["entity_id"].tolist()
    s1_ids = s1_df["entity_id"].tolist()

    def _build_index(key_fn):
        index: Dict[str, List[str]] = defaultdict(list)
        for gid, name, addr in zip(gallery_ids,
                                   gallery_df["business_name_clean"].fillna("").tolist(),
                                   gallery_df["business_address_clean"].fillna("").tolist()):
            for k in key_fn(name, addr):
                index[k].append(gid)
        return index

    addr_index = _build_index(lambda n, a: [a] if len(a) >= address_key_min_len else [])
    name_index = _build_index(_name_core_key)
    pin_index = _build_index(lambda n, a: _pin_keys(a))

    for s1_idx, name, addr in zip(range(len(s1_ids)),
                                  s1_df["business_name_clean"].fillna("").tolist(),
                                  s1_df["business_address_clean"].fillna("").tolist()):
        for k in ([addr] if len(addr) >= address_key_min_len else []):
            bucket = addr_index.get(k, [])
            if 0 < len(bucket) <= max_addr_bucket:
                candidates[s1_idx].update(bucket)
        for k in _name_core_key(name, addr):
            bucket = name_index.get(k, [])
            if 0 < len(bucket) <= max_bucket:
                candidates[s1_idx].update(bucket)
        for k in _pin_keys(addr):
            candidates[s1_idx].update(pin_index.get(k, []))

    return candidates


def _name_core_key(name: str, addr: str) -> List[str]:
    """Core name (legal-stripped) + last two address tokens; core must be >=3 chars."""
    from .normalize import LEGAL_SUFFIXES
    tokens = [t for t in name.split()
              if t and not any(re.fullmatch(p.replace(r"\b", "").rstrip("?.\\"), t)
                               for p in LEGAL_SUFFIXES)]
    core = " ".join(tokens).strip()
    if len(core) < 3:
        return []
    addr_tokens = addr.split()[-2:]
    return [f"{core}|{' '.join(addr_tokens)}"]


def _pin_keys(addr: str) -> List[str]:
    """5-6 digit runs = PIN (India) / ZIP (US) candidates."""
    return re.findall(r"\b\d{5,6}\b", addr)


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
