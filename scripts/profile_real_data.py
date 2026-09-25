"""Profile the REAL dataset — one pass per file, memory-safe.

Computes: row counts, country distribution, blank fields, ground-truth
match distribution (singletons, multi-match), and cross-checks the audited
numbers reported by other teams.

Usage:
    python scripts/profile_real_data.py --data-dir data
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path


def profile_source(path: Path, chunk=500_000):
    """Stream a source TSV; return counts, countries, blanks."""
    rows = 0
    countries = Counter()
    blank_name = blank_addr = 0
    t0 = time.time()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for rec in reader:
            rows += 1
            if len(rec) < 4:
                continue
            name, addr, country = rec[1], rec[2], rec[3]
            countries[country.strip()] += 1
            if not name.strip():
                blank_name += 1
            if not addr.strip():
                blank_addr += 1
    return {
        "rows": rows,
        "countries": dict(countries.most_common()),
        "blank_name": blank_name,
        "blank_addr": blank_addr,
        "seconds": round(time.time() - t0, 1),
    }


def profile_ground_truth(path: Path):
    """Stream the ground truth; return singleton/multi-match distribution."""
    dist = Counter()
    total_pairs = 0
    rows = 0
    t0 = time.time()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for rec in reader:
            if len(rec) < 2:
                continue
            rows += 1
            matched = rec[1].strip()
            if not matched:
                dist[0] += 1
            else:
                ids = [x for x in matched.split(",") if x.strip()]
                dist[len(ids)] += 1
                total_pairs += len(ids)
    singletons = dist.get(0, 0)
    return {
        "rows": rows,
        "total_pairs": total_pairs,
        "singletons": singletons,
        "singleton_pct": round(100 * singletons / rows, 2) if rows else 0,
        "distribution": dict(sorted(dist.items())),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="docs/data_profile.json")
    args = ap.parse_args()

    base = Path(args.data_dir) / "dataset"
    report = {}

    print("=" * 60)
    print("REAL DATASET PROFILE")
    print("=" * 60)

    for split in ("train", "test"):
        for src in ("source1", "source2", "source3"):
            path = base / split / f"{split}_{src}.tsv"
            if not path.exists():
                print(f"  MISSING: {path}")
                continue
            print(f"\nProfiling {split}_{src} ...", flush=True)
            info = profile_source(path)
            report[f"{split}_{src}"] = info
            countries = ", ".join(f"{k}={v:,}" for k, v in info["countries"].items())
            print(f"  rows={info['rows']:,}  [{info['seconds']}s]")
            print(f"  countries: {countries}")
            print(f"  blank name={info['blank_name']:,}  blank addr={info['blank_addr']:,}")

    gt_path = base / "train" / "train_ground_truth.tsv"
    if gt_path.exists():
        print("\nProfiling ground truth ...", flush=True)
        gt = profile_ground_truth(gt_path)
        report["ground_truth"] = gt
        print(f"  rows={gt['rows']:,}  total_pairs={gt['total_pairs']:,}  "
              f"[{gt['seconds']}s]")
        print(f"  singletons={gt['singletons']:,} ({gt['singleton_pct']}%)")
        print(f"  match distribution: {gt['distribution']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Cross-check against the audited numbers (COMPETITIVE_INTEL §1)
    print("\n" + "=" * 60)
    print("CROSS-CHECK vs audited numbers")
    print("=" * 60)
    expected = {
        "train_source1": 2_206_821, "train_source2": 5_034_616,
        "train_source3": 5_285_603, "test_source1": 1_732_544,
        "test_source2": 4_887_273, "test_source3": 5_082_316,
    }
    for key, exp in expected.items():
        got = report.get(key, {}).get("rows")
        status = "✅" if got == exp else f"❌ got {got:,}" if got else "—"
        print(f"  {key:15s} expected {exp:>10,}  {status}")


if __name__ == "__main__":
    main()
