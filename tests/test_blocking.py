"""Baseline + regression unit tests for src/blocking.py — EXISTING behavior only.

No production code is changed by these tests. They document what each blocking
function does today, including current quirks, and pin the Vishwesh fixes
(recall denominator, deterministic capping, bucket safety).

Run: python tests/test_blocking.py

Dependency policy: never install, never fake. When the module cannot be
imported normally the tests fall back to loading it through a synthetic `src`
package (skipping src/__init__.py, which pulls teammate-only deps) and stub
ONLY the missing direct 3rd-party imports of blocking.py. Tests whose code
path would call a stubbed dependency report BLOCKED instead of passing.
"""
import importlib.util
import sys
import types
from pathlib import Path

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd

# Probe BEFORE any stub is installed (find_spec breaks on __spec__-less stubs).
_DEPS_ABSENT = {
    n for n in ("rapidfuzz", "jellyfish", "datasketch")
    if importlib.util.find_spec(n) is None
}

_IMPORT_ERROR = None   # set only when blocking.py cannot be loaded at all
_STUBBED = set()       # deps faked purely to satisfy module-level imports
_IMPORT_MODE = ""


class _Stub:
    """Sentinel standing in for a missing dependency. Never returns a result."""

    def __init__(self, label: str):
        self._label = label

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"BLOCKED: {self._label} is not installed (import stub only) — "
            "this code path cannot be verified without faking it"
        )

    def __getattr__(self, item):
        raise RuntimeError(
            f"BLOCKED: {self._label}.{item} is not installed (import stub only)"
        )


def _stub_module(name: str, *attrs: str) -> None:
    mod = types.ModuleType(name)
    for attr in attrs:
        setattr(mod, attr, _Stub(f"{name}.{attr}"))
    sys.modules[name] = mod


def _load_blocking():
    """Load src.blocking, preferring the real package import.

    Fallback (dependencies of src/__init__.py missing): load normalize.py and
    blocking.py as a synthetic `src` package so every function that does not
    touch a missing 3rd-party dep runs against REAL code.
    """
    global _IMPORT_ERROR, _IMPORT_MODE, _STUBBED

    try:
        import importlib
        mod = importlib.import_module("src.blocking")
        _IMPORT_MODE = "package import (src.blocking)"
        return mod
    except Exception as pkg_exc:
        first_error = f"{type(pkg_exc).__name__}: {pkg_exc}"

    try:
        for name, attrs in (
            ("rapidfuzz", ("fuzz", "process")),
            ("jellyfish", ("soundex", "metaphone")),
        ):
            if name in _DEPS_ABSENT and name not in sys.modules:
                _stub_module(name, *attrs)
                _STUBBED.add(name)

        for mod_name in [m for m in list(sys.modules)
                         if m == "src" or m.startswith("src.")]:
            del sys.modules[mod_name]

        pkg = types.ModuleType("src")
        pkg.__path__ = [str(_REPO / "src")]
        sys.modules["src"] = pkg

        for mod_name in ("normalize", "blocking"):
            spec = importlib.util.spec_from_file_location(
                f"src.{mod_name}", _REPO / "src" / f"{mod_name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"src.{mod_name}"] = module
            spec.loader.exec_module(module)

        _IMPORT_MODE = (
            "synthetic src package (src/__init__.py skipped); "
            f"package import failed with [{first_error}]; "
            f"stubbed imports: {sorted(_STUBBED) or 'none'}"
        )
        _IMPORT_ERROR = None
        return sys.modules["src.blocking"]
    except Exception as exc:
        _IMPORT_ERROR = exc
        _IMPORT_MODE = f"FAILED: package import [{first_error}]; load [{type(exc).__name__}: {exc}]"
        return None


_BLOCKING = _load_blocking()
if _BLOCKING is not None:
    adaptive_prune = _BLOCKING.adaptive_prune
    tfidf_blocking_candidates = _BLOCKING.tfidf_blocking_candidates
    tfidf_blocking_adaptive = _BLOCKING.tfidf_blocking_adaptive
    bidirectional_tfidf = _BLOCKING.bidirectional_tfidf
    key_blocking = _BLOCKING.key_blocking
    union_candidates = _BLOCKING.union_candidates
    cap_candidates = _BLOCKING.cap_candidates
    phonetic_blocking = _BLOCKING.phonetic_blocking
    initialism_blocking = _BLOCKING.initialism_blocking
    address_tfidf_candidates = _BLOCKING.address_tfidf_candidates
    minhash_lsh_candidates = _BLOCKING.minhash_lsh_candidates
    measure_blocking_quality = _BLOCKING.measure_blocking_quality


def _require() -> None:
    if _IMPORT_ERROR is not None:
        raise AssertionError(
            f"BLOCKED: src.blocking could not be loaded: "
            f"{type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}"
        )


def _require_real(*deps: str) -> None:
    """Block a test whose code path needs a dependency that is not installed."""
    missing = [d for d in deps if d in _DEPS_ABSENT]
    if missing:
        raise AssertionError(
            f"BLOCKED: real {', '.join(missing)} not installed — "
            "behavior unverifiable without faking it"
        )


def _phonetic(*args, **kwargs):
    """phonetic_blocking guarded so it never runs against a stubbed jellyfish."""
    _require()
    _require_real("jellyfish")
    return phonetic_blocking(*args, **kwargs)


def _minhash(*args, **kwargs):
    """minhash_lsh_candidates guarded: needs datasketch OR real rapidfuzz."""
    _require()
    if "datasketch" in _DEPS_ABSENT and "rapidfuzz" in _DEPS_ABSENT:
        raise AssertionError(
            "BLOCKED: neither datasketch nor rapidfuzz installed — "
            "minhash/fallback path unverifiable without faking it"
        )
    return minhash_lsh_candidates(*args, **kwargs)


def _df(rows):
    return pd.DataFrame(
        rows, columns=["entity_id", "business_name_clean", "business_address_clean"]
    )


# ---------------------------------------------------------------------------
# adaptive_prune
# ---------------------------------------------------------------------------
def test_adaptive_prune_keeps_kmin_regardless_of_gap():
    _require()
    idx = np.array([[10, 11, 12, -1]])
    sc = np.array([[0.90, 0.10, 0.05, 0.00]])
    mask = adaptive_prune(idx, sc, kmin=2, kmax=4, gap=0.10)
    assert mask.tolist() == [[True, True, False, False]], mask.tolist()


def test_adaptive_prune_gap_window_keeps_near_top1():
    _require()
    idx = np.array([[0, 1, 2]])
    sc = np.array([[0.90, 0.85, 0.20]])
    mask = adaptive_prune(idx, sc, kmin=1, kmax=3, gap=0.10)
    assert mask.tolist() == [[True, True, False]], mask.tolist()


def test_adaptive_prune_kmax_truncates_even_within_gap():
    _require()
    idx = np.array([[0, 1, 2, 3]])
    sc = np.array([[0.9, 0.9, 0.9, 0.9]])
    mask = adaptive_prune(idx, sc, kmin=1, kmax=2, gap=0.5)
    assert mask.tolist() == [[True, True, False, False]], mask.tolist()


def test_adaptive_prune_empty_row():
    _require()
    idx = np.zeros((1, 0), dtype=int)
    sc = np.zeros((1, 0))
    mask = adaptive_prune(idx, sc, kmin=1, kmax=5, gap=0.1)
    assert mask.tolist() == [[]], mask.tolist()


# ---------------------------------------------------------------------------
# tfidf_blocking_candidates
# ---------------------------------------------------------------------------
def test_tfidf_empty_input():
    _require()
    assert tfidf_blocking_candidates([], ["a"], ["S2-1"]) == {}
    assert tfidf_blocking_candidates(["a"], [], []) == {}


def test_tfidf_forward_retrieval():
    _require()
    c = tfidf_blocking_candidates(
        ["acme robotics"],
        ["acme robotics incorporated", "zzz unrelated"],
        ["S2-1", "S2-2"],
        threshold=0.3,
    )
    assert "S2-1" in c[0], c
    assert "S2-2" not in c[0], c


def test_tfidf_top_k_cap():
    _require()
    c = tfidf_blocking_candidates(
        ["alpha beta"],
        ["alpha beta one", "alpha beta two", "alpha beta three"],
        ["S2-1", "S2-2", "S2-3"],
        threshold=0.1,
        top_k=2,
    )
    assert len(c[0]) == 2, c


def test_tfidf_duplicate_ids_collapse():
    _require()
    c = tfidf_blocking_candidates(
        ["alpha"], ["alpha", "alpha"], ["S2-1", "S2-1"], threshold=0.3
    )
    assert c[0] == {"S2-1"}, c


def test_tfidf_zero_candidate_query_absent():
    _require()
    c = tfidf_blocking_candidates(
        ["acme robotics", "totally unrelated query words"],
        ["acme robotics"],
        ["S2-1"],
        threshold=0.3,
    )
    assert 0 in c and c[0] == {"S2-1"}, c
    assert 1 not in c, (
        "CURRENT CONTRACT: queries with zero matches get NO key (not an empty set)"
    )


def test_tfidf_deterministic():
    _require()
    kwargs = dict(threshold=0.2)
    a = tfidf_blocking_candidates(
        ["acme robotics", "delta foods"],
        ["acme robotics inc", "delta foods co"],
        ["S2-1", "S2-2"],
        **kwargs,
    )
    b = tfidf_blocking_candidates(
        ["acme robotics", "delta foods"],
        ["acme robotics inc", "delta foods co"],
        ["S2-1", "S2-2"],
        **kwargs,
    )
    assert dict(a) == dict(b)


# ---------------------------------------------------------------------------
# bidirectional_tfidf
# ---------------------------------------------------------------------------
def test_bidirectional_empty_input():
    _require()
    c, s = bidirectional_tfidf([], [], [])
    assert len(c) == 0 and len(s) == 0


def test_bidirectional_forward_retrieval():
    _require()
    c, s = bidirectional_tfidf(
        ["acme robotics"], ["acme robotics inc"], ["S2-1"], threshold=0.1,
        forward_kmin=1, forward_kmax=5,
    )
    assert "S2-1" in c[0], c
    assert "S2-1" in s[0] and 0.0 < s[0]["S2-1"] <= 1.0, s


def test_bidirectional_reverse_leg_adds_edges():
    _require()
    s1 = ["market street one", "zzz qqq"]
    gal = ["market street one", "market street two"]
    ids = ["S2-1", "S2-2"]

    fwd, _ = tfidf_blocking_adaptive(
        s1, gal, ids, threshold=0.1, kmin=1, kmax=1, gap=0.0
    )
    c, s = bidirectional_tfidf(
        s1, gal, ids, threshold=0.1,
        forward_kmin=1, forward_kmax=1, forward_gap=0.0,
        reverse_kmin=1, reverse_kmax=2, reverse_gap=0.0,
    )
    # forward leg keeps only top-1 per S1 -> S2-2 is excluded for S1[0]
    assert "S2-2" not in fwd.get(0, set()), fwd
    # reverse leg (gallery -> S1) recovers S2-2 for S1[0]
    assert "S2-2" in c.get(0, set()), (
        f"reverse retrieval failed; forward={dict(fwd)} bidirectional={dict(c)}"
    )
    assert "S2-2" in s.get(0, {}), s


def test_bidirectional_scores_returned():
    _require()
    _, s = bidirectional_tfidf(
        ["acme robotics"], ["acme robotics inc", "delta foods"], ["S2-1", "S2-2"],
        threshold=0.1, forward_kmin=1, forward_kmax=5,
    )
    assert "S2-1" in s[0], s
    assert len(s[0]) >= 1


def test_bidirectional_deterministic():
    _require()
    args = (
        ["market street one"],
        ["market street one", "market street two"],
        ["S2-1", "S2-2"],
    )
    kw = dict(threshold=0.1, forward_kmin=1, forward_kmax=5)
    a, sa = bidirectional_tfidf(*args, **kw)
    b, sb = bidirectional_tfidf(*args, **kw)
    assert dict(a) == dict(b)
    assert {k: dict(v) for k, v in sa.items()} == {k: dict(v) for k, v in sb.items()}


# ---------------------------------------------------------------------------
# key_blocking
# ---------------------------------------------------------------------------
def test_key_empty_frames():
    _require()
    empty = _df([])
    assert len(key_blocking(empty, empty)) == 0
    s1 = _df([("S1-1", "acme traders", "123 main street springfield")])
    assert len(key_blocking(s1, empty)) == 0


def test_key_address_key_retrieval_multiple_candidates():
    _require()
    addr = "123 main street springfield"
    s1 = _df([("S1-1", "solo trader", addr)])
    gal = _df([
        ("S2-1", "gal one", addr),
        ("S2-2", "gal two", addr),
        ("S2-3", "other", "99 other road dallas texas"),
    ])
    c = key_blocking(s1, gal)
    assert c[0] == {"S2-1", "S2-2"}, dict(c)
    assert len(c[0]) == 2, "one S1 must be allowed multiple candidates"


def test_key_short_address_skipped():
    _require()
    addr = "1 elm st"  # < 12 chars -> address key disabled by design
    s1 = _df([("S1-1", "unique name aaa", addr)])
    gal = _df([("S2-1", "totally other name bbb", addr)])
    c = key_blocking(s1, gal)
    assert 0 not in c, dict(c)


def test_key_name_bucket_cap():
    _require()
    s1 = _df([("S1-1", "acme traders", "1 alpha road paris")])
    gal = _df([
        ("S2-1", "acme traders", "2 alpha road paris"),
        ("S2-2", "acme traders", "3 alpha road paris"),
        ("S2-3", "acme traders", "4 alpha road paris"),
    ])
    dropped = key_blocking(s1, gal, max_bucket=2)  # bucket size 3 > 2
    assert 0 not in dropped, dict(dropped)
    kept = key_blocking(s1, gal, max_bucket=3)
    assert kept[0] == {"S2-1", "S2-2", "S2-3"}, dict(kept)


def test_key_address_bucket_cap():
    _require()
    addr = "77 green field lane bristol"
    s1 = _df([("S1-1", "solo trader", addr)])
    gal = _df([
        ("S2-1", "gal one", addr),
        ("S2-2", "gal two", addr),
        ("S2-3", "gal three", addr),
    ])
    dropped = key_blocking(s1, gal, max_addr_bucket=2)  # bucket size 3 > 2
    assert 0 not in dropped, dict(dropped)
    kept = key_blocking(s1, gal, max_addr_bucket=3)
    assert kept[0] == {"S2-1", "S2-2", "S2-3"}, dict(kept)


def test_key_pin_retrieval():
    _require()
    s1 = _df([("S1-1", "alpha", "call us at 560001 near tower")])
    gal = _df([
        ("S2-1", "beta", "totally different place 560001"),
        ("S2-2", "gamma", "no pin data available here"),
    ])
    c = key_blocking(s1, gal)
    assert c[0] == {"S2-1"}, dict(c)


def test_key_pin_bucket_within_limit_kept():
    # PIN bucket of 10 <= default max_pin_bucket=30 -> all kept.
    _require()
    s1 = _df([("S1-1", "alpha", "call us at 560001 near tower")])
    gal = _df([
        (f"S2-{i}", f"gal {i}", f"other place {560001} row {i}") for i in range(10)
    ])
    c = key_blocking(s1, gal)
    assert len(c[0]) == 10, dict(c)


def test_key_pin_oversized_bucket_dropped():
    # REGRESSION (bucket safety): a hot PIN bucket must not explode candidates.
    _require()
    s1 = _df([("S1-1", "alpha", "call us at 560001 near tower")])
    gal = _df([
        (f"S2-{i}", f"gal {i}", f"other place {560001} row {i}") for i in range(40)
    ])
    dropped = key_blocking(s1, gal)  # bucket 40 > default max_pin_bucket 30
    assert 0 not in dropped, f"oversized PIN bucket was not dropped: {dict(dropped)}"
    kept = key_blocking(s1, gal, max_pin_bucket=40)
    assert len(kept[0]) == 40, dict(kept)


def test_key_duplicate_ids_collapse():
    _require()
    addr = "123 main street springfield"
    s1 = _df([("S1-1", "solo trader", addr)])
    gal = _df([("S2-1", "gal one", addr), ("S2-1", "gal one", addr)])
    c = key_blocking(s1, gal)
    assert c[0] == {"S2-1"}, dict(c)


def test_key_zero_candidate_s1_has_no_key():
    _require()
    s1 = _df([
        ("S1-1", "acme traders", "1 alpha road paris"),
        ("S1-2", "unique quiet lane owner", "unique quiet lane"),
    ])
    gal = _df([
        ("S2-1", "acme traders", "2 alpha road paris"),
    ])
    c = key_blocking(s1, gal)
    assert 0 in c and c[0] == {"S2-1"}, dict(c)
    assert 1 not in c, dict(c)


def test_key_pin_no_match_documents_empty_key_quirk():
    # CURRENT BEHAVIOR (inconsistent zero-candidate contract — see audit report):
    # an S1 whose address contains a 5-6 digit run gets a PRESENT-BUT-EMPTY key,
    # while other zero-candidate S1s get no key at all.
    _require()
    s1 = _df([("S1-1", "unique quiet lane owner", "call us at 560001 near tower")])
    gal = _df([("S2-1", "other", "no matching pin anywhere")])
    c = key_blocking(s1, gal)
    assert 0 in c and c[0] == set(), dict(c)


def test_key_deterministic():
    _require()
    addr = "123 main street springfield"
    s1 = _df([("S1-1", "solo trader", addr)])
    gal = _df([("S2-1", "gal one", addr)])
    a = key_blocking(s1, gal)
    b = key_blocking(s1, gal)
    assert dict(a) == dict(b)


# ---------------------------------------------------------------------------
# union_candidates
# ---------------------------------------------------------------------------
def test_union_empty():
    _require()
    assert dict(union_candidates()) == {}
    assert dict(union_candidates({}, {})) == {}


def test_union_dedup_and_merge():
    _require()
    a = {0: {"x", "y"}}
    b = {0: {"y", "z"}, 1: {"w"}}
    u = dict(union_candidates(a, b))
    assert u == {0: {"x", "y", "z"}, 1: {"w"}}, u


def test_union_zero_candidate_key_absent():
    _require()
    u = dict(union_candidates({0: {"x"}}))
    assert 1 not in u
    assert dict(union_candidates({0: set()})) == {0: set()}


def test_union_deterministic():
    _require()
    parts = [{0: {"a"}, 1: {"b"}}, {1: {"c"}}, {0: {"d"}}]
    a = dict(union_candidates(*parts))
    b = dict(union_candidates(*parts))
    assert a == b


# ---------------------------------------------------------------------------
# cap_candidates
# ---------------------------------------------------------------------------
def test_cap_none_passthrough():
    _require()
    c = {0: {"a", "b", "c"}}
    assert dict(cap_candidates(c, None)) == {0: {"a", "b", "c"}}


def test_cap_under_limit():
    _require()
    c = {0: {"a", "b", "c"}}
    assert dict(cap_candidates(c, 5)) == {0: {"a", "b", "c"}}


def test_cap_empty():
    _require()
    assert dict(cap_candidates({}, 10)) == {}


def test_cap_over_limit_length():
    _require()
    c = {0: {f"S2-{i}" for i in range(10)}}
    out = cap_candidates(c, 4)
    assert len(out[0]) == 4
    assert out[0] <= c[0]


def test_cap_without_keep_fn_keeps_lowest_ids():
    # REGRESSION (determinism): set iteration order is hash-seed dependent, so
    # capping must sort explicitly — survivors are the lowest candidate IDs.
    _require()
    c = {0: {f"S2-{i}" for i in range(10)}}
    out = cap_candidates(c, 5)
    assert out[0] == {"S2-0", "S2-1", "S2-2", "S2-3", "S2-4"}, out


def test_cap_deterministic_repeated_output():
    _require()
    c = {0: {f"S2-{i}" for i in range(10)}, 1: {f"S2-{i}" for i in range(7)}}
    a = cap_candidates(c, 3)
    b = cap_candidates(c, 3)
    assert dict(a) == dict(b)
    assert a[0] == b[0] and a[1] == b[1]


def test_cap_keep_fn_prefers_high_score():
    _require()
    c = {0: {"a", "b", "c"}}
    scores = {"a": 1.0, "b": 9.0, "c": 5.0}
    out = cap_candidates(c, 1, keep_fn=lambda q, cand: scores[cand])
    assert out[0] == {"b"}, out


def test_cap_keep_fn_tie_breaks_by_id():
    # REGRESSION (determinism): equal scores must resolve to lowest IDs.
    _require()
    c = {0: {"c", "a", "b"}}
    out = cap_candidates(c, 2, keep_fn=lambda q, cand: 1.0)
    assert out[0] == {"a", "b"}, out


def test_cap_deterministic_with_keep_fn():
    _require()
    c = {0: {"a", "b", "c"}}
    scores = {"a": 1.0, "b": 9.0, "c": 5.0}
    fn = lambda q, cand: scores[cand]
    assert cap_candidates(c, 2, keep_fn=fn) == cap_candidates(c, 2, keep_fn=fn)


# ---------------------------------------------------------------------------
# measure_blocking_quality — recall denominator regression
# ---------------------------------------------------------------------------
def test_measure_blocking_return_structure():
    _require()
    m = measure_blocking_quality({0: {"x"}}, {"A": ["x"]}, 10, ["A"])
    assert set(m) == {
        "candidate_pairs", "reduction_ratio", "matches_retained",
        "total_matches", "pair_recall",
    }, m


def test_measure_blocking_recall_counts_zero_candidate_s1():
    # THE REGRESSION (bug fix): an S1 with truth but no generated candidates
    # must still contribute to the recall denominator.
    #   A -> {x}, B -> {y}, C -> {z}
    #   candidates: A -> {x}, B -> {wrong}, C -> absent
    #   => retained 1 / total 3 = 0.3333
    _require()
    gt = {"A": ["x"], "B": ["y"], "C": ["z"]}
    s1_ids = ["A", "B", "C"]
    cands = {0: {"x"}, 1: {"wrong"}}   # C produced nothing
    m = measure_blocking_quality(cands, gt, 9, s1_ids)
    assert m["matches_retained"] == 1, m
    assert m["total_matches"] == 3, m
    assert m["pair_recall"] == 0.3333, m


def test_measure_blocking_all_truth_retained():
    _require()
    gt = {"A": ["x"], "B": ["y", "z"]}
    s1_ids = ["A", "B"]
    cands = {0: {"x"}, 1: {"y", "z", "noise"}}
    m = measure_blocking_quality(cands, gt, 10, s1_ids)
    assert m["matches_retained"] == 3 and m["total_matches"] == 3, m
    assert m["pair_recall"] == 1.0, m


def test_measure_blocking_empty_candidates_counts_full_denominator():
    _require()
    gt = {"A": ["x"], "B": ["y"], "C": ["z"]}
    m = measure_blocking_quality({}, gt, 9, ["A", "B", "C"])
    assert m["matches_retained"] == 0, m
    assert m["total_matches"] == 3, m
    assert m["pair_recall"] == 0.0, m
    assert m["candidate_pairs"] == 0 and m["reduction_ratio"] == 1.0, m


def test_measure_blocking_zero_match_s1_contributes_nothing():
    _require()
    gt = {"A": ["x"], "B": []}   # parse_ground_truth emits [] for zero matches
    m = measure_blocking_quality({0: {"x"}}, gt, 10, ["A", "B"])
    assert m["total_matches"] == 1 and m["matches_retained"] == 1, m


def test_measure_blocking_candidate_pairs_and_reduction_ratio():
    _require()
    cands = {0: {"x"}, 1: {"wrong"}}
    m = measure_blocking_quality(cands, {"A": ["x"]}, 9, ["A", "B"])
    assert m["candidate_pairs"] == 2, m
    assert m["reduction_ratio"] == round(1 - 2 / 9, 4), m


def test_measure_blocking_without_s1_ids_reports_zero_recall():
    # CONTRACT: idx -> id mapping is impossible without s1_ids, so recall
    # evaluates to 0 (s1_ids is required for a meaningful recall number).
    _require()
    m = measure_blocking_quality({0: {"x"}}, {"A": ["x"]}, 10, None)
    assert m["total_matches"] == 1 and m["matches_retained"] == 0, m
    assert m["pair_recall"] == 0.0, m


# ---------------------------------------------------------------------------
# phonetic_blocking
# ---------------------------------------------------------------------------
def test_phonetic_empty_input():
    assert dict(_phonetic([], ["smith"], ["S2-1"])) == {}
    assert dict(_phonetic(["smith"], [], [])) == {}
    assert dict(_phonetic([""], ["smith"], ["S2-1"])) == {}


def test_phonetic_forward_retrieval():
    c = _phonetic(["smith"], ["smith"], ["S2-1"])
    assert c[0] == {"S2-1"}, dict(c)


def test_phonetic_oversized_bucket_dropped():
    # REGRESSION (bucket safety): a hot Soundex bucket must not explode
    # candidates — same "drop buckets >30" convention as the key legs.
    t = ["smith"] * 40
    ids = [f"S2-{i}" for i in range(40)]
    dropped = _phonetic(["smith"], t, ids)          # bucket 40 > default 30
    assert 0 not in dropped, f"oversized bucket not dropped: {dict(dropped)}"
    kept = _phonetic(["smith"], t, ids, max_bucket=40)
    assert len(kept[0]) == 40, dict(kept)


def test_phonetic_not_one_to_one_many_to_many():
    # Blocking must never enforce assignment: one S1 -> many candidates,
    # one candidate claimed by many S1s.
    s1 = ["smith", "smyth"]
    t = ["smith", "smyth", "smithe"]
    ids = ["S2-1", "S2-2", "S2-3"]
    c = _phonetic(s1, t, ids)
    assert c[0] == {"S2-1", "S2-2", "S2-3"}, dict(c)
    assert c[1] == {"S2-1", "S2-2", "S2-3"}, dict(c)
    assert "S2-1" in c[0] and "S2-1" in c[1], "same candidate must be claimable by multiple S1s"


def test_phonetic_duplicate_ids_collapse():
    c = _phonetic(["smith"], ["smith", "smith"], ["S2-1", "S2-1"])
    assert c[0] == {"S2-1"}, dict(c)


def test_phonetic_zero_candidate_absent():
    c = _phonetic(["zzzqqq"], ["smith"], ["S2-1"])
    assert 0 not in c, dict(c)


def test_phonetic_deterministic():
    a = _phonetic(["smith", "smyth"], ["smith", "smithe"], ["S2-1", "S2-2"])
    b = _phonetic(["smith", "smyth"], ["smith", "smithe"], ["S2-1", "S2-2"])
    assert dict(a) == dict(b)


# ---------------------------------------------------------------------------
# initialism_blocking
# ---------------------------------------------------------------------------
def test_initialism_empty_input():
    _require()
    assert dict(initialism_blocking([], ["IBM"], ["S2-1"])) == {}
    assert dict(initialism_blocking(["IBM"], [], [])) == {}
    assert dict(initialism_blocking([""], ["IBM"], ["S2-1"])) == {}


def test_initialism_acronym_query_matches_full_name():
    _require()
    c = initialism_blocking(["IBM"], ["International Business Machines"], ["S2-1"])
    assert c[0] == {"S2-1"}, dict(c)


def test_initialism_full_name_query_matches_acronym():
    _require()
    c = initialism_blocking(["International Business Machines"], ["IBM"], ["S2-1"])
    assert c[0] == {"S2-1"}, dict(c)


def test_initialism_bucket_multiple_candidates():
    _require()
    c = initialism_blocking(
        ["IBM"],
        ["International Business Machines", "Internet Business Machines"],
        ["S2-1", "S2-2"],
    )
    assert c[0] == {"S2-1", "S2-2"}, dict(c)


def test_initialism_oversized_bucket_dropped():
    # REGRESSION (bucket safety): hot initialism bucket (>30) is dropped.
    _require()
    t = ["Alpha Beta Gamma"] * 40          # all map to initialism "ABG"
    ids = [f"S2-{i}" for i in range(40)]
    dropped = initialism_blocking(["ABG"], t, ids)
    assert 0 not in dropped, f"oversized bucket not dropped: {dict(dropped)}"
    kept = initialism_blocking(["ABG"], t, ids, max_bucket=40)
    assert len(kept[0]) == 40, dict(kept)


def test_initialism_no_match_zero_candidate_absent():
    _require()
    c = initialism_blocking(["Acme Traders"], ["Zeta Wholesale"], ["S2-1"])
    assert 0 not in c, dict(c)


def test_initialism_deterministic():
    _require()
    a = initialism_blocking(["IBM"], ["International Business Machines"], ["S2-1"])
    b = initialism_blocking(["IBM"], ["International Business Machines"], ["S2-1"])
    assert dict(a) == dict(b)


# ---------------------------------------------------------------------------
# address_tfidf_candidates
# ---------------------------------------------------------------------------
def test_address_empty_input():
    _require()
    assert len(address_tfidf_candidates([], ["1 main st"], ["S2-1"])) == 0
    assert len(address_tfidf_candidates(["1 main st"], [], [])) == 0


def test_address_empty_addresses_yield_nothing():
    _require()
    assert len(address_tfidf_candidates([""], ["123 main street"], ["S2-1"])) == 0
    assert len(address_tfidf_candidates(["123 main street"], [""], ["S2-1"])) == 0


def test_address_forward_retrieval():
    _require()
    c = address_tfidf_candidates(
        ["123 main street springfield"],
        ["123 main street springfield illinois", "nowhere lane"],
        ["S2-1", "S2-2"],
        threshold=0.3,
    )
    assert "S2-1" in c[0], dict(c)
    assert "S2-2" not in c[0], dict(c)


def test_address_top_k_cap():
    _require()
    c = address_tfidf_candidates(
        ["123 main street"],
        ["123 main street one", "123 main street two", "123 main street three"],
        ["S2-1", "S2-2", "S2-3"],
        threshold=0.1,
        top_k=2,
    )
    assert len(c[0]) == 2, dict(c)


def test_address_duplicate_ids_collapse():
    _require()
    c = address_tfidf_candidates(
        ["123 main street"], ["123 main street", "123 main street"],
        ["S2-1", "S2-1"], threshold=0.3,
    )
    assert c[0] == {"S2-1"}, dict(c)


def test_address_zero_candidate_absent():
    _require()
    c = address_tfidf_candidates(
        ["nowhere lane"], ["123 main street springfield"], ["S2-1"], threshold=0.3
    )
    assert 0 not in c, dict(c)


def test_address_deterministic():
    _require()
    args = (
        ["123 main street springfield"],
        ["123 main street springfield illinois"],
        ["S2-1"],
    )
    a = address_tfidf_candidates(*args, threshold=0.3)
    b = address_tfidf_candidates(*args, threshold=0.3)
    assert dict(a) == dict(b)


# ---------------------------------------------------------------------------
# minhash_lsh_candidates (datasketch optional -> rapidfuzz fallback path)
# ---------------------------------------------------------------------------
def test_minhash_empty_input():
    assert dict(_minhash([], ["acme"], ["S2-1"])) == {}
    assert dict(_minhash(["acme"], [], [])) == {}


def test_minhash_identical_match():
    c = _minhash(["acme robotics"], ["acme robotics"], ["S2-1"])
    assert "S2-1" in c[0], dict(c)


def test_minhash_fuzzy_match():
    c = _minhash(
        ["acme robotics"], ["acme robotics inc"], ["S2-1"], threshold=0.3
    )
    assert "S2-1" in c[0], dict(c)


def test_minhash_returns_only_target_ids():
    # Guards against the rapidfuzz fallback returning scores instead of IDs
    # (suspected tuple-unpack bug in _rapidfuzz_fallback_candidates).
    ids = ["S2-1", "S2-2", "S2-3"]
    c = _minhash(
        ["acme robotics"], ["acme robotics", "acme robotics inc", "delta foods"],
        ids, threshold=0.3,
    )
    assert c[0], "expected at least one candidate"
    assert c[0] <= set(ids), f"candidates contain non-entity-id values: {c[0]!r}"


def test_minhash_duplicate_ids_collapse():
    c = _minhash(
        ["acme robotics"], ["acme robotics", "acme robotics"], ["S2-1", "S2-1"]
    )
    assert c[0] == {"S2-1"}, dict(c)


def test_minhash_deterministic():
    args = (["acme robotics"], ["acme robotics inc"], ["S2-1"])
    a = _minhash(*args, threshold=0.3)
    b = _minhash(*args, threshold=0.3)
    assert dict(a) == dict(b)


def main():
    print("=" * 60)
    print("BASELINE TESTS — src/blocking.py")
    print(f"import mode: {_IMPORT_MODE}")
    if _DEPS_ABSENT:
        print(f"missing deps (probed): {sorted(_DEPS_ABSENT)}")
    print("=" * 60)
    tests = sorted((k, v) for k, v in globals().items() if k.startswith("test_"))
    failed, blocked = [], []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS    {name}")
        except Exception as exc:
            if str(exc).startswith("BLOCKED"):
                blocked.append((name, exc))
                print(f"  BLOCKED {name}: {exc}")
            else:
                failed.append((name, exc))
                print(f"  FAIL    {name}: {type(exc).__name__}: {exc}")
    print("-" * 60)
    print(f"  {len(tests) - len(failed) - len(blocked)}/{len(tests)} passed, "
          f"{len(failed)} failed, {len(blocked)} blocked")
    for title, rows in (("failed", failed), ("blocked", blocked)):
        if rows:
            print(f"\n  {title}:")
            for name, exc in rows:
                print(f"    - {name}: {exc}")
    sys.exit(1 if failed or blocked else 0)


if __name__ == "__main__":
    main()
