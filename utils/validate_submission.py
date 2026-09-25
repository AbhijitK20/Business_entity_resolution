"""Submission validator for Amazon ML Challenge 2026 — Business Entity Resolution.

Stdlib only. Mirrors the rules from the official problem statement.

Usage:
    python3 utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
"""
import argparse
import csv
import sys
from pathlib import Path


def load_test_ids(test_dir: Path):
    """Load valid S1/S2/S3 entity IDs from test source files."""
    ids = {"S1": set(), "S2": set(), "S3": set()}
    for source in ("source1", "source2", "source3"):
        path = test_dir / f"test_{source}.tsv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                eid = (row.get("entity_id") or "").strip()
                if eid.startswith("S1-"):
                    ids["S1"].add(eid)
                elif eid.startswith("S2-"):
                    ids["S2"].add(eid)
                elif eid.startswith("S3-"):
                    ids["S3"].add(eid)
    return ids


def parse_id_list(raw: str):
    """Parse a comma-separated ID list. Returns list of IDs (may be empty)."""
    raw = (raw or "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def validate_file(path: Path, id_column: str, valid_ids: set, label: str, issues: list,
                  require_all_s1: bool = True, s1_ids: set = None):
    """Validate one submission file against all rules."""
    if not path.exists():
        issues.append(f"{label}: file not found at {path}")
        return None

    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        columns = reader.fieldnames
        if columns is None or len(columns) != 2 or "source1_entity_id" not in columns or id_column not in columns:
            issues.append(
                f"{label}: expected exactly 2 tab-separated columns "
                f"[source1_entity_id, {id_column}], got {columns}"
            )
            return None

        for line_no, row in enumerate(reader, start=2):
            s1 = (row.get("source1_entity_id") or "").strip()
            raw = (row.get(id_column) or "").strip()
            if not s1:
                issues.append(f"{label} line {line_no}: empty source1_entity_id")
                continue

            # Duplicate S1 rows
            if s1 in rows:
                issues.append(f"{label} line {line_no}: duplicate source1_entity_id '{s1}'")
                continue

            # S1 must be a valid S1 ID
            if s1_ids and s1 not in s1_ids:
                issues.append(f"{label} line {line_no}: unknown source1_entity_id '{s1}'")

            ids = parse_id_list(raw)

            # Duplicates within a list
            if len(ids) != len(set(ids)):
                dupes = {x for x in ids if ids.count(x) > 1}
                issues.append(f"{label} line {line_no} ({s1}): duplicate IDs in list: {sorted(dupes)}")

            # IDs must be S2/S3 only and must exist in test set
            for eid in ids:
                if eid.startswith("S1-"):
                    issues.append(f"{label} line {line_no} ({s1}): self-match to Source 1 '{eid}'")
                elif not (eid.startswith("S2-") or eid.startswith("S3-")):
                    issues.append(f"{label} line {line_no} ({s1}): invalid ID prefix '{eid}'")
                elif eid not in valid_ids:
                    issues.append(f"{label} line {line_no} ({s1}): ID '{eid}' not in test set")

            rows[s1] = set(ids)

    # Every S1 must appear
    if require_all_s1 and s1_ids is not None:
        missing = s1_ids - set(rows.keys())
        if missing:
            sample = sorted(missing)[:5]
            issues.append(
                f"{label}: {len(missing)} Source 1 entities missing "
                f"(e.g. {', '.join(sample)})"
            )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Validate submission files")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to dataset/test")
    args = parser.parse_args()

    issues = []
    test_dir = Path(args.test_dir)
    ids = load_test_ids(test_dir)
    s1_ids = ids["S1"]
    valid_s2_s3 = ids["S2"] | ids["S3"]

    if not s1_ids:
        issues.append(f"No Source 1 entities found in {test_dir}")
    if not valid_s2_s3:
        issues.append(f"No Source 2/3 entities found in {test_dir}")

    matching = validate_file(
        Path(args.matching), "matched_entity_ids", valid_s2_s3,
        "matching_results.tsv", issues, require_all_s1=True, s1_ids=s1_ids,
    )
    candidate = validate_file(
        Path(args.candidate), "candidate_entity_ids", valid_s2_s3,
        "candidate_pairs.tsv", issues, require_all_s1=True, s1_ids=s1_ids,
    )

    # Final matches must be a subset of candidates
    if matching is not None and candidate is not None:
        for s1, matched in matching.items():
            cand_ids = candidate.get(s1, set())
            not_in_candidates = matched - cand_ids
            if not_in_candidates:
                issues.append(
                    f"matching_results.tsv ({s1}): matched IDs not present in "
                    f"candidate_pairs.tsv: {sorted(not_in_candidates)[:5]}"
                )

    if issues:
        print(f"FAIL — {len(issues)} issue(s) found:\n")
        for i, issue in enumerate(issues, start=1):
            print(f"  {i}. {issue}")
        sys.exit(1)

    print("PASS — submission is valid and safe to submit.")
    sys.exit(0)


if __name__ == "__main__":
    main()
