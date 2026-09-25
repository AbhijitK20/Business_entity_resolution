"""Dense semantic blocking leg — GPU-accelerated retrieval.

Why: lexical legs (TF-IDF / phonetic / keys) plateau around ~0.94 union recall
at very high candidate counts, and ~80% of true matches have different
normalized names (SABER finding). Dense embeddings capture semantic similarity
that character n-grams miss.

License-safe encoders (competition rules: MIT/Apache-2.0, no external data):
  - Snowflake/snowflake-arctic-embed-xs  (Apache-2.0, 22M params, 384 dims)
  - intfloat/multilingual-e5-small       (MIT, 118M params, 384 dims)

Hardware: RTX 4050 Laptop, 6 GB VRAM. Search is exact cosine top-K, chunked
and fp16 to fit VRAM; falls back to CPU transparently.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

DEFAULT_MODEL = "Snowflake/snowflake-arctic-embed-xs"
MULTILINGUAL_MODEL = "intfloat/multilingual-e5-small"

# Models that require instruction prefixes (asymmetric query/passage).
_PREFIX_RULES = (
    ("intfloat/multilingual-e5", ("query: ", "passage: ")),
    ("intfloat/e5", ("query: ", "passage: ")),
)


def _prefixes_for(model_name: str) -> tuple:
    for key, prefixes in _PREFIX_RULES:
        if model_name.startswith(key):
            return prefixes
    return "", ""


def get_device() -> str:
    """Return 'cuda' when a GPU is available, else 'cpu'."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _texts_fingerprint(texts: List[str]) -> str:
    h = hashlib.md5()
    for t in texts:
        h.update((t or "").encode("utf8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()[:10]


def _cache_file(
    cache_dir: Optional[str],
    model_name: str,
    role: str,
    texts: List[str],
) -> Optional[Path]:
    if cache_dir is None:
        return None
    slug = model_name.replace("/", "__")
    fp = _texts_fingerprint(texts)
    return Path(cache_dir) / f"{slug}__{role}__{len(texts)}__{fp}.npy"


def encode_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    role: str = "target",
    show_progress: bool = True,
) -> np.ndarray:
    """Encode texts to L2-normalized float32 embeddings.

    ``role`` selects the instruction prefix ("query" vs "target") for models
    that need asymmetric prefixes (E5 family). Results are cached to disk keyed
    by model + role + text fingerprint, so re-runs are instant.
    """
    cache_path = _cache_file(cache_dir, model_name, role, texts)
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer

    device = device or get_device()
    model = SentenceTransformer(model_name, device=device)
    q_prefix, p_prefix = _prefixes_for(model_name)
    prefix = q_prefix if role == "query" else p_prefix
    prepared = [f"{prefix}{t or ''}" for t in texts]

    emb = model.encode(
        prepared,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    ).astype(np.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
    return emb


def topk_search(
    query_emb: np.ndarray,
    target_emb: np.ndarray,
    top_k: int = 30,
    query_chunk: int = 512,
    use_fp16: bool = True,
) -> tuple:
    """Exact cosine top-K search, chunked for small-VRAM GPUs.

    Returns ``(indices, scores)`` as numpy arrays of shape (n_queries, top_k),
    with -1 / -inf padding when fewer than ``top_k`` targets exist.
    """
    import torch

    device = get_device()
    n_q, n_t = query_emb.shape[0], target_emb.shape[0]
    k = min(top_k, n_t)

    t = torch.from_numpy(target_emb)
    if device == "cuda" and use_fp16:
        t = t.half()
    t = t.to(device)

    indices = np.full((n_q, k), -1, dtype=np.int32)
    scores = np.full((n_q, k), -np.inf, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_q, query_chunk):
            end = min(start + query_chunk, n_q)
            q = torch.from_numpy(query_emb[start:end])
            if device == "cuda" and use_fp16:
                q = q.half()
            q = q.to(device)
            sims = q @ t.T
            chunk_scores, chunk_idx = torch.topk(sims, k=k, dim=1)
            indices[start:end] = chunk_idx.cpu().numpy()
            scores[start:end] = chunk_scores.float().cpu().numpy()

    return indices, scores


def dense_blocking_candidates(
    query_texts: List[str],
    target_texts: List[str],
    target_ids: List[str],
    model_name: str = DEFAULT_MODEL,
    top_k: int = 30,
    cache_dir: Optional[str] = None,
    batch_size: int = 256,
    device: Optional[str] = None,
    query_chunk: int = 512,
) -> Dict[int, Set[str]]:
    """Dense embedding blocking leg.

    Interface matches the lexical legs in ``src.blocking``: returns
    ``{query_index: {target_ids}}``.
    """
    query_emb = encode_texts(
        query_texts,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
        role="query",
    )
    target_emb = encode_texts(
        target_texts,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
        role="target",
    )

    indices, _ = topk_search(
        query_emb, target_emb, top_k=top_k, query_chunk=query_chunk
    )

    candidates: Dict[int, Set[str]] = defaultdict(set)
    for q_idx in range(indices.shape[0]):
        for t_idx in indices[q_idx]:
            if t_idx >= 0:
                candidates[q_idx].add(target_ids[int(t_idx)])
    return candidates


def dense_scores(
    query_texts: List[str],
    target_texts: List[str],
    pairs: List[tuple],
    model_name: str = DEFAULT_MODEL,
    cache_dir: Optional[str] = None,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> np.ndarray:
    """Cosine similarity for explicit (query_idx, target_idx) pairs.

    Used by scored top-K pruning to rank union candidates by dense similarity
    without a full matrix multiply.
    """
    query_emb = encode_texts(
        query_texts,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
        role="query",
    )
    target_emb = encode_texts(
        target_texts,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
        role="target",
    )

    out = np.zeros(len(pairs), dtype=np.float32)
    if not pairs:
        return out
    q_idx = np.array([p[0] for p in pairs], dtype=np.int64)
    t_idx = np.array([p[1] for p in pairs], dtype=np.int64)
    qv = query_emb[q_idx]
    tv = target_emb[t_idx]
    out = np.einsum("ij,ij->i", qv, tv).astype(np.float32)
    return out
