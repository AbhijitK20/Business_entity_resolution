"""Tests for scripts/make_synthetic_data.py — K2: the synthetic generator must
resemble the VERIFIED real distribution (docs/COMPETITIVE_INTEL.md §1):

    89% multi-match · 5.58% singleton · mean 3.46 · max 11
    ~3% blank addresses (gallery only) · 47% shared S1 names
    22.7% India cross-script names
    train US 60 / India 40 · test US 38 / India 47 / France 15

Tolerances are used (not exact equality) so normal variation never fails a
test; allocation-based statistics are checked more tightly because the
generator allocates them deterministically.

Run: python tests/test_synthetic_data.py   (or pytest)
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.make_synthetic_data import (
    MATCH_COUNT_PMF,
    allocate_match_counts,
    generate,
    has_indic_script,
    largest_remainder,
    parse_mix,
    sample_match_counts,
    summarize_split,
    to_devanagari,
    to_telugu,
)
import random


# ------------------------------------------------------ allocation primitives
def test_largest_remainder_exact_and_deterministic():
    out = largest_remainder(600, {"US": 0.6, "India": 0.4})
    assert sum(out.values()) == 600
    assert out["US"] == 360 and out["India"] == 240
    assert largest_remainder(600, {"US": 0.6, "India": 0.4}) == out
    odd = largest_remainder(7, {"a": 0.5, "b": 0.3, "c": 0.2})
    assert sum(odd.values()) == 7


def test_match_pmf_targets_at_scale():
    """The allocation reproduces the real distribution at large n."""
    counts = allocate_match_counts(100_000)
    n = len(counts)
    assert n == 100_000
    multi = sum(1 for c in counts if c >= 2) / n
    single = sum(1 for c in counts if c == 0) / n
    mean = sum(counts) / n
    assert abs(multi - 0.8902) < 0.003, multi       # ~89% multi-match
    assert abs(single - 0.0558) < 0.002, single     # ~5.6% singleton
    assert abs(mean - 3.46) < 0.02, mean            # mean 3.46
    assert max(counts) == 11, max(counts)           # maximum 11 reachable
    assert 10 in counts


def test_sample_match_counts_seeded_deterministic():
    a = sample_match_counts(300, random.Random(42))
    b = sample_match_counts(300, random.Random(42))
    c = sample_match_counts(300, random.Random(7))
    assert a == b, "same seed must give identical match counts"
    assert sorted(a) == sorted(b)
    assert a != c or len(a) < 5                     # shuffled differently


# ------------------------------------------------------------ cross-script K2
def test_indic_renderers_produce_native_script():
    deva = to_devanagari("sanjay textiles")
    telu = to_telugu("sanjay textiles")
    assert deva != "sanjay textiles"
    assert has_indic_script(deva)
    assert has_indic_script(telu)
    assert has_indic_script("राम मार्केटिंग")
    assert not has_indic_script("Sanjay Textiles")
    assert not has_indic_script("")
    # deterministic
    assert to_devanagari("sanjay textiles") == deva
    assert to_telugu("krishna sweets") == to_telugu("krishna sweets")


# ------------------------------------------------------------ full generation
def _gen(n_s1=600, seed=42):
    tmp = Path(tempfile.mkdtemp(prefix="synth_test_"))
    stats = generate(tmp, n_s1, seed)
    return tmp, stats


def test_match_distribution_matches_targets():
    _, stats = _gen(600, seed=42)
    for split in ("train", "test"):
        s = stats[split]
        assert abs(s["multi_match_pct"] - 89.0) < 3.0, (split, s)
        assert abs(s["singleton_pct"] - 5.58) < 2.0, (split, s)
        assert abs(s["mean_matches"] - 3.46) < 0.15, (split, s)
        assert s["max_matches"] <= 11, (split, s)   # real maximum
        assert s["max_matches"] >= 6, (split, s)    # tail present at n=600


def test_blank_addresses_gallery_only():
    tmp, _ = _gen(600, seed=42)
    for split in ("train", "test"):
        d = tmp / "dataset" / split
        for src in ("source1", "source2", "source3"):
            rows = (d / f"{split}_{src}.tsv").read_text().splitlines()[1:]
            blank = sum(1 for r in rows
                        if not r.split("\t")[2].strip())
            rate = blank / max(len(rows), 1)
            if src == "source1":
                assert blank == 0, "S1 addresses must never be blank"
            else:
                assert 0.0 <= rate < 0.07, (split, src, rate)   # ~3% ± tol


def test_name_collisions_at_target_rate():
    _, stats = _gen(600, seed=42)
    for split in ("train", "test"):
        s = stats[split]
        assert abs(s["shared_name_pct"] - 47.0) < 3.0, (split, s)


def test_country_distribution_train_and_test():
    _, stats = _gen(600, seed=42)
    tr = stats["train"]["country_mix"]
    te = stats["test"]["country_mix"]
    assert abs(tr.get("US", 0) - 0.60) < 0.02
    assert abs(tr.get("India", 0) - 0.40) < 0.02
    assert "France" not in tr, "France must not appear in training"
    assert abs(te.get("US", 0) - 0.38) < 0.03
    assert abs(te.get("India", 0) - 0.47) < 0.03
    assert abs(te.get("France", 0) - 0.15) < 0.03


def test_cross_script_names_on_india_gallery_only():
    tmp, _ = _gen(600, seed=42)
    for split in ("train", "test"):
        d = tmp / "dataset" / split
        india = 0
        india_indic = 0
        us_indic = 0
        s1_indic = 0
        for src in ("source1", "source2", "source3"):
            rows = (d / f"{split}_{src}.tsv").read_text().splitlines()[1:]
            for r in rows:
                _, name, _, country = r.split("\t")
                if src == "source1":
                    if has_indic_script(name):
                        s1_indic += 1
                    continue
                if country == "India":
                    india += 1
                    if has_indic_script(name):
                        india_indic += 1
                elif country == "US" and has_indic_script(name):
                    us_indic += 1
        assert s1_indic == 0, "S1 names must be Latin (real: 0 non-ASCII S1)"
        assert us_indic == 0, "cross-script only applies to India"
        if india:
            rate = india_indic / india
            assert abs(rate - 0.227) < 0.06, (split, rate)


def test_deterministic_output_with_fixed_seed(tmp_path=None):
    import filecmp
    a, _ = _gen(200, seed=42)
    b, _ = _gen(200, seed=42)
    c, _ = _gen(200, seed=123)
    same = True
    diff = False
    for rel in ("dataset/train/train_source1.tsv",
                "dataset/train/train_ground_truth.tsv",
                "dataset/test/test_source2.tsv"):
        assert filecmp.cmp(a / rel, b / rel, shallow=False), rel
        if not filecmp.cmp(a / rel, c / rel, shallow=False):
            diff = True
    assert same and diff, "fixed seed must reproduce; other seeds must differ"


def test_integrity_no_duplicate_true_identities():
    """Every gallery record is a true match of AT MOST one S1 (the verified
    real-data invariant), all matched ids exist, distractors never enter GT."""
    tmp, _ = _gen(400, seed=42)
    for split in ("train", "test"):
        d = tmp / "dataset" / split
        s1 = (d / f"{split}_source1.tsv").read_text().splitlines()[1:]
        s2 = (d / f"{split}_source2.tsv").read_text().splitlines()[1:]
        s3 = (d / f"{split}_source3.tsv").read_text().splitlines()[1:]
        gt = (d / f"{split}_ground_truth.tsv").read_text().splitlines()[1:]

        s1_ids = [r.split("\t")[0] for r in s1]
        assert len(s1_ids) == len(set(s1_ids)), "duplicate S1 ids"
        gallery_ids = [r.split("\t")[0] for r in s2 + s3]
        assert len(gallery_ids) == len(set(gallery_ids)), "duplicate gallery ids"
        gallery_set = set(gallery_ids)

        seen_targets = set()
        for row in gt:
            s1_id, targets = row.split("\t")
            ids = [t for t in targets.split(",") if t.strip()]
            assert len(ids) == len(set(ids)), "duplicate ids in a match list"
            for t in ids:
                assert t in gallery_set, f"{t} not in gallery"
                assert t not in seen_targets, \
                    f"{t} claimed by two S1 entities (duplicate identity)"
                seen_targets.add(t)

        # singletons exist and are represented as empty lists
        assert any(not row.split("\t")[1].strip() for row in gt)


def test_summarize_split_fields_present():
    _, stats = _gen(100, seed=3)
    s = stats["train"]
    for key in ("n_s1", "n_gallery", "multi_match_pct", "singleton_pct",
                "mean_matches", "max_matches", "blank_gallery_addr_pct",
                "shared_name_pct", "indic_name_pct_gallery", "total_matches"):
        assert key in s, key
    assert s["n_s1"] == 80


def test_parse_mix():
    assert parse_mix("US=0.6,India=0.4") == {"US": 0.6, "India": 0.4}
    try:
        parse_mix("")
        assert False, "empty mix must raise"
    except ValueError:
        pass


def test_pmf_keys_cover_zero_to_max():
    assert set(MATCH_COUNT_PMF) == set(range(12))
    # source table rounds to 4 decimals (sum = 1.0001); allocation normalizes
    assert abs(sum(MATCH_COUNT_PMF.values()) - 1.0) < 1e-3


if __name__ == "__main__":
    import unittest

    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except unittest.SkipTest as exc:
            print(f"  SKIP  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} synthetic-generator tests passed.")
    sys.exit(1 if failed else 0)
