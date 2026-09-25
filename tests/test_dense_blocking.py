"""Tests for the dense embedding blocking leg.

Run: python tests/test_dense_blocking.py
Requires: sentence-transformers + the default model (downloads once, ~90MB).
Skips gracefully when the stack is unavailable.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.dense_blocking import (
    DEFAULT_MODEL,
    _prefixes_for,
    _texts_fingerprint,
    dense_blocking_candidates,
    get_device,
    topk_search,
)

PASSED = 0
FAILED = 0


def check(name, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  PASS  {name}")
        PASSED += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()
        FAILED += 1


def test_prefix_rules():
    assert _prefixes_for("intfloat/multilingual-e5-small") == ("query: ", "passage: ")
    assert _prefixes_for("Snowflake/snowflake-arctic-embed-xs") == ("", "")


def test_fingerprint_deterministic():
    a = _texts_fingerprint(["acme", "beta"])
    b = _texts_fingerprint(["acme", "beta"])
    c = _texts_fingerprint(["acme", "gamma"])
    assert a == b and a != c


def test_topk_search_math():
    """topk_search must return exact cosine top-K on known vectors."""
    rng = np.random.default_rng(0)
    q = rng.normal(size=(4, 16)).astype(np.float32)
    t = rng.normal(size=(10, 16)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    idx, scores = topk_search(q, t, top_k=3)
    ref = q @ t.T
    for i in range(4):
        expected = np.argsort(-ref[i])[:3]
        assert list(idx[i]) == list(expected), f"row {i}: {idx[i]} != {expected}"
        # fp16 search precision: indices exact, scores within ~2e-4
        assert np.allclose(scores[i], ref[i][expected], atol=1e-3)


def test_semantic_retrieval():
    """Near-duplicates must retrieve each other; unrelated names must not win."""
    queries = [
        "Acme Robotics Private Limited",
        "Sunrise Medical Store",
        "Green Leaf Cafe",
    ]
    targets = [
        "ACME ROBOTICS PVT LTD",        # 0: should match q0
        "Sun Rise Medicos",             # 1: should match q1
        "Greenleaf Coffee House",       # 2: should match q2
        "Bombay Hardware Traders",      # 3: unrelated
        "Dental Clinic Smile Care",     # 4: unrelated
    ]
    ids = [f"S2-{i}" for i in range(len(targets))]
    cands = dense_blocking_candidates(
        queries, targets, ids, model_name=DEFAULT_MODEL, top_k=3
    )
    assert 0 in cands and "S2-0" in cands[0], f"q0 missed: {cands.get(0)}"
    assert 1 in cands and "S2-1" in cands[1], f"q1 missed: {cands.get(1)}"
    assert 2 in cands and "S2-2" in cands[2], f"q2 missed: {cands.get(2)}"
    for q_idx in cands:
        assert len(cands[q_idx]) <= 3, "top_k not respected"


def test_interface_validity():
    queries = ["Alpha Beta Gamma"]
    targets = ["Alpha Beta Gamma", "Delta Epsilon"]
    ids = ["ID-A", "ID-B"]
    cands = dense_blocking_candidates(queries, targets, ids, top_k=2)
    assert set(cands.keys()) == {0}
    assert cands[0] <= {"ID-A", "ID-B"}


def test_device_report():
    dev = get_device()
    assert dev in ("cuda", "cpu")


if __name__ == "__main__":
    print(f"=== dense blocking tests (device={get_device()}) ===")
    check("prefix_rules", test_prefix_rules)
    check("fingerprint_deterministic", test_fingerprint_deterministic)
    check("topk_search_math", test_topk_search_math)
    check("interface_validity", test_interface_validity)
    check("device_report", test_device_report)
    check("semantic_retrieval", test_semantic_retrieval)
    print(f"\n{PASSED}/{PASSED + FAILED} dense tests passed.")
    sys.exit(1 if FAILED else 0)
