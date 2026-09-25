"""Build a sampled "world" from the real train data for fast, realistic iteration.

Follows SABER's measured 2.5%-world protocol:
  1. Sample a fraction of Source 1 entities (stratified by country).
  2. Include ALL their true matches in the gallery.
  3. Add a proportional sample of distractor records (so candidate density is
     realistic — a thin gallery makes retrieval artificially easy).
  4. Write a normal dataset layout (train_source1/2/3 + ground truth) so the
     entire pipeline runs unchanged.

Usage:
    python scripts/make_sample_world.py --data-dir data --out tests/world_2p5 \
        --fraction 0.025 --seed 42
"""
import argparse
import csv
import random
from pathlib import Path


def read_country_map(source1_path: Path) -> dict:
    """{s1_id: country} from train_source1."""
    out = {}
    with open(source1_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for rec in reader:
            if len(rec) >= 4:
                out[rec[0]] = rec[3].strip()
    return out


def read_ground_truth(gt_path: Path) -> dict:
    """{s1_id: [matched_ids]} (only non-empty)."""
    out = {}
    with open(gt_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for rec in reader:
            if len(rec) < 2:
                continue
            matched = rec[1].strip()
            if matched:
                out[rec[0]] = [x for x in matched.split(",") if x]
    return out


def sample_ids_per_country(country_map: dict, fraction: float, rng: random.Random) -> set:
    """Stratified sample of S1 ids by country."""
    by_country = {}
    for sid, c in country_map.items():
        by_country.setdefault(c, []).append(sid)
    sampled = set()
    for country, ids in by_country.items():
        k = max(1, int(len(ids) * fraction))
        sampled.update(rng.sample(ids, k))
        print(f"  sampled {country}: {k:,} / {len(ids):,}")
    return sampled


def write_rows(path: Path, rows: list, header: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


def filter_source(path: Path, keep_ids: set, out_rows: list, header: list):
    """Keep rows whose entity_id is in keep_ids."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for rec in reader:
            if len(rec) >= 4 and rec[0] in keep_ids:
                out_rows.append(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="tests/world_2p5")
    ap.add_argument("--fraction", type=float, default=0.025,
                    help="fraction of S1 to sample (SABER used 0.025)")
    ap.add_argument("--distractor-fraction", type=float, default=None,
                    help="fraction of distractor records to keep (default = fraction)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = Path(args.data_dir) / "dataset" / "train"
    out = Path(args.out)
    (out / "dataset" / "train").mkdir(parents=True, exist_ok=True)
    d = out / "dataset" / "train"

    print("=" * 60)
    print("BUILDING SAMPLED WORLD")
    print("=" * 60)

    print("\n[1] Reading country map + ground truth...")
    country_map = read_country_map(base / "train_source1.tsv")
    gt = read_ground_truth(base / "train_ground_truth.tsv")
    print(f"  S1 entities: {len(country_map):,} | with matches: {len(gt):,}")

    print(f"\n[2] Sampling {args.fraction:.1%} of S1 per country...")
    sampled_s1 = sample_ids_per_country(country_map, args.fraction, rng)

    # True matches of sampled S1 must all be in the gallery
    needed_matches = set()
    for sid in sampled_s1:
        needed_matches.update(gt.get(sid, []))
    print(f"  S1 sampled: {len(sampled_s1):,} | their true matches: {len(needed_matches):,}")

    # Distractor sampling: keep `distractor_fraction` of records NOT in needed_matches
    df_frac = args.distractor_fraction if args.distractor_fraction is not None else args.fraction

    print(f"\n[3] Building gallery (all true matches + {df_frac:.1%} distractors)...")
    for src in ("source2", "source3"):
        src_path = base / f"train_{src}.tsv"
        kept_rows = []
        kept_matches = 0
        distractors_kept = 0
        total = 0
        with open(src_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for rec in reader:
                if len(rec) < 4:
                    continue
                total += 1
                eid = rec[0]
                if eid in needed_matches:
                    kept_rows.append(rec)
                    kept_matches += 1
                elif rng.random() < df_frac:
                    kept_rows.append(rec)
                    distractors_kept += 1
        write_rows(d / f"train_{src}.tsv", kept_rows,
                   ["entity_id", "business_name", "business_address", "country"])
        print(f"  {src}: total={total:,} | matches kept={kept_matches:,} | "
              f"distractors kept={distractors_kept:,} | gallery={len(kept_rows):,}")

    print("\n[4] Writing sampled S1 + ground truth...")
    s1_rows = []
    filter_source(base / "train_source1.tsv", sampled_s1, s1_rows,
                  ["entity_id", "business_name", "business_address", "country"])
    write_rows(d / "train_source1.tsv", s1_rows,
               ["entity_id", "business_name", "business_address", "country"])

    gt_rows = [[sid, ",".join(gt.get(sid, []))] for sid in sampled_s1]
    write_rows(d / "train_ground_truth.tsv", gt_rows,
               ["source1_entity_id", "matched_entity_ids"])

    n_with = sum(1 for sid in sampled_s1 if gt.get(sid))
    print(f"  S1 rows: {len(s1_rows):,} | with matches: {n_with:,} | "
          f"singletons: {len(s1_rows) - n_with:,}")

    print(f"\nWorld written → {d}")
    print(f"  S1={len(s1_rows):,}  S2={sum(1 for _ in open(d / 'train_source2.tsv')) - 1:,}  "
          f"S3={sum(1 for _ in open(d / 'train_source3.tsv')) - 1:,}")


if __name__ == "__main__":
    main()
