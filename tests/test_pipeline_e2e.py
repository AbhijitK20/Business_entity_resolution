"""End-to-end pipeline integration test on a tiny synthetic world.

Catches refactor regressions in the run() path (the _run_inference NameError
slipped through because no test exercised the full pipeline).

Run: python tests/test_pipeline_e2e.py
Runs in ~1-2 minutes (fast mode, tiny world, dense disabled to stay offline).
"""

import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
        FAILED += 1


def test_pipeline_end_to_end():
    tmp = Path(tempfile.mkdtemp(prefix="er_e2e_"))
    try:
        world = tmp / "world"
        out = tmp / "out"
        # 1) Tiny synthetic world
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"),
             "--out", str(world), "--n-s1", "120", "--seed", "7"],
            check=True, capture_output=True, timeout=300,
        )
        assert (world / "dataset" / "train").exists(), "world train dir missing"
        assert (world / "dataset" / "test").exists(), "world test dir missing"

        # 2) Full pipeline (fast mode; dense disabled for speed/offline safety)
        proc = subprocess.run(
            [sys.executable, "-m", "src.pipeline",
             "--data-dir", str(world), "--output-dir", str(out),
             "--fast", "--max-candidates", "20", "--no-dense-cap"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=900,
        )
        assert proc.returncode == 0, (
            f"pipeline failed (rc={proc.returncode}):\n"
            f"{proc.stderr[-2000:]}"
        )

        # 3) Outputs exist and are well-formed
        matching = out / "matching_results.tsv"
        candidates = out / "candidate_pairs.tsv"
        assert matching.exists() and candidates.exists(), "output files missing"

        with open(matching) as f:
            rows = list(csv.reader(f, delimiter="\t"))
        assert rows[0] == ["source1_entity_id", "matched_entity_ids"]
        assert len(rows) > 1, "no matching rows"

        with open(candidates) as f:
            crows = list(csv.reader(f, delimiter="\t"))
        assert crows[0] == ["source1_entity_id", "candidate_entity_ids"]
        assert len(crows) == len(rows), "candidate/matching row count mismatch"

        # Matches must be a subset of candidates (competition rule)
        cand_by_id = {r[0]: set(r[1].split(",")) if r[1] else set()
                      for r in crows[1:]}
        for r in rows[1:]:
            matched = set(r[1].split(",")) if r[1] else set()
            assert matched <= cand_by_id[r[0]], (
                f"match not in candidates for {r[0]}"
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("=== pipeline end-to-end integration test ===")
    check("pipeline_end_to_end", test_pipeline_end_to_end)
    print(f"\n{PASSED}/{PASSED + FAILED} e2e tests passed.")
    sys.exit(1 if FAILED else 0)
