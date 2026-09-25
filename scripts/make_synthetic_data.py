"""Synthetic dataset generator — mirrors the noise patterns described in the problem.

Creates train/test datasets with realistic name & address variations:
  - Abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited, Rd/Road)
  - Punctuation (& vs "and"), typos, word-order transpositions
  - Missing address components, landmark references
  - Singletons (~30%), 1-to-many matches, US/India/France countries

Usage:
    python scripts/make_synthetic_data.py --out tests/fixtures_synth --n-s1 300 --seed 42
"""
import argparse
import random
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------- name pools
US_NAMES = [
    "Acme Robotics", "Delta Foods", "Bright Cafe", "Zen Traders", "Kappa Motors",
    "Summit Logistics", "Blue Wave Analytics", "Ironclad Security", "Nova Interiors",
    "Redwood Legal", "Silverline Media", "Pioneer Chemicals", "Harbor Freight",
    "Lakeside Dental", "Meadow Farm Supply", "Orbit Software", "Peak Performance",
    "Quartz Electronics", "Riverside Auto", "Sterling Finance",
]
IN_NAMES = [
    "Sanjay Textiles", "Gupta Medical Store", "Krishna Sweets", "Sharma Electricals",
    "Lakshmi Jewellers", "Verma Hardware", "Patel Travels", "Bharat Motors",
    "Ganesh Provision", "Deshmukh Constructions", "Iyer Catering", "Reddy Chemicals",
    "Nair Ayurveda", "Joshi Stationers", "Kulkarni Furnitures",
]
FR_NAMES = [
    "Boulangerie Dupont", "Cafe de la Paix", "Garage Martin", "Pizzeria Roma",
    "Librairie Flammarion", "Pharmacie Centrale", "Boucherie Bernard",
    "Fleuriste Marie", "Patisserie Lyon", "Restaurant Provence",
]
US_CITIES = [("San Jose", "CA"), ("Austin", "TX"), ("Reno", "NV"), ("Boise", "ID"),
             ("Seattle", "WA"), ("Denver", "CO"), ("Boston", "MA"), ("Miami", "FL")]
IN_CITIES = ["Bengaluru", "New Delhi", "Mumbai", "Hyderabad", "Chennai", "Pune",
             "Jaipur", "Kochi"]
FR_CITIES = ["Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux", "Nice", "Nantes"]

US_STREETS = ["Market St", "Oak Ave", "Pine St", "Hill Rd", "Lake Dr", "Maple Blvd",
              "Cedar Ln", "Washington St", "Lincoln Ave", "Park Rd"]
IN_STREETS = ["MG Road", "Gandhi Marg", "Nehru Nagar", "Station Road",
              "Main Bazaar", "Ring Road", "Market Road"]
FR_STREETS = ["Rue de la Republique", "Avenue Victor Hugo", "Boulevard Saint-Michel",
              "Rue du Commerce", "Place de la Mairie"]

SUFFIXES = {
    "US": ["Inc", "Corp", "LLC", "Co", "Corporation", "Incorporated", "Ltd", ""],
    "India": ["Pvt Ltd", "Private Limited", "Ltd", "LLP", "", ""],
    "France": ["SARL", "SAS", "SA", ""],
}


# ------------------------------------------------------------- noise helpers
def vary_name(name: str, rng: random.Random, country: str) -> str:
    """Apply one realistic name transformation, then optionally add a suffix."""
    out = name
    pick = rng.random()
    if pick < 0.30 and len(out) > 4:
        i = rng.randrange(1, len(out) - 1)
        out = out[:i] + ("x" if out[i] != "x" else "z") + out[i + 1:]
    elif pick < 0.50:
        toks = out.split()
        if len(toks) >= 2:
            i = rng.randrange(len(toks) - 1)
            toks[i], toks[i + 1] = toks[i + 1], toks[i]
            out = " ".join(toks)
    elif pick < 0.65:
        if " & " in out:
            out = out.replace(" & ", " and ")
        elif "and" in out:
            out = out.replace("and", "&")
    elif pick < 0.80:
        repl = {"Corporation": "Corp", "Incorporated": "Inc", "Limited": "Ltd",
                "Private": "Pvt", "International": "Intl", "Company": "Co"}
        for full, abbr in repl.items():
            if full.lower() in out.lower():
                out = out.replace(full, abbr)
                break
    elif pick < 0.90 and len(out.split()) > 2:
        toks = out.split()
        out = " ".join(toks[:1] + toks[2:])

    suffix = rng.choice(SUFFIXES[country])
    out = out.strip()
    if suffix and rng.random() < 0.7:
        out = f"{out} {suffix}"
    return out


def vary_address(addr: str, city: str, country: str, rng: random.Random) -> str:
    """Apply one realistic address transformation."""
    pick = rng.random()
    if pick < 0.25:
        out = (addr.replace("Street", "St").replace("Avenue", "Ave")
                   .replace("Road", "Rd").replace("Boulevard", "Blvd"))
    elif pick < 0.45:
        out = (addr.replace(" St", " Street").replace(" Ave", " Avenue")
                   .replace(" Rd", " Road"))
    elif pick < 0.65:
        landmarks = ["Near SBI ATM", "Opposite City Hall", "Near Bus Stand",
                     "Behind Post Office", "Near Railway Station"]
        out = f"{rng.choice(landmarks)}, {city}"
    elif pick < 0.80:
        out = city if rng.random() < 0.5 else addr
    else:
        out = addr

    if country == "France":
        out = (unicodedata.normalize("NFKD", out)
               .encode("ascii", "ignore").decode("ascii"))
    return out.strip()


def make_entities(country: str, n: int, rng: random.Random):
    """Generate n (name, city, address, country) dicts for one country."""
    if country == "US":
        pool, cities = US_NAMES, [c for c, _ in US_CITIES]
        streets = US_STREETS
    elif country == "India":
        pool, cities = IN_NAMES, IN_CITIES
        streets = IN_STREETS
    else:
        pool, cities = FR_NAMES, FR_CITIES
        streets = FR_STREETS

    out = []
    for i in range(n):
        base = pool[i % len(pool)]
        if i >= len(pool):
            base = f"{base} {i // len(pool) + 1}"
        city = rng.choice(cities)
        street = rng.choice(streets)
        number = rng.randint(1, 999)
        address = f"{number} {street}, {city}"
        out.append({"name": base, "city": city, "address": address,
                    "country": country})
    return out


def _write_tsv(path: Path, rows, cols):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def generate(out_dir: Path, n_s1: int, seed: int):
    rng = random.Random(seed)
    out_dir = Path(out_dir)

    n_us = int(n_s1 * 0.5)
    n_in = int(n_s1 * 0.35)
    n_fr = n_s1 - n_us - n_in

    entities = []
    for country, count in (("US", n_us), ("India", n_in), ("France", n_fr)):
        entities.extend(make_entities(country, count, rng))
    rng.shuffle(entities)

    split = int(len(entities) * 0.8)
    splits = {"train": entities[:split], "test": entities[split:]}

    for tag, pool in splits.items():
        s1_rows, s2_rows, s3_rows, gt_rows = [], [], [], []
        s1_i = s2_i = s3_i = 0

        for ent in pool:
            country = ent["country"]
            s1_i += 1
            s1_id = f"S1-{100000 + s1_i}"
            suffix = rng.choice(SUFFIXES[country])
            s1_name = f"{ent['name']} {suffix}".strip() if suffix else ent["name"]
            s1_rows.append({
                "entity_id": s1_id,
                "business_name": s1_name,
                "business_address": ent["address"],
                "country": country,
            })

            # ~30% singletons
            if rng.random() < 0.30:
                gt_rows.append({"source1_entity_id": s1_id,
                                "matched_entity_ids": ""})
                continue

            # 1..3 matches across S2/S3
            n_matches = rng.choices([1, 2, 3], weights=[0.55, 0.30, 0.15])[0]
            matches = []
            for _ in range(n_matches):
                src = rng.choice(["S2", "S3"])
                name = vary_name(ent["name"], rng, country)
                addr = vary_address(ent["address"], ent["city"], country, rng)
                if src == "S2":
                    s2_i += 1
                    sid = f"S2-{500000 + s2_i}"
                    s2_rows.append({"entity_id": sid, "business_name": name,
                                    "business_address": addr, "country": country})
                else:
                    s3_i += 1
                    sid = f"S3-{700000 + s3_i}"
                    s3_rows.append({"entity_id": sid, "business_name": name,
                                    "business_address": addr, "country": country})
                matches.append(sid)

            # Look-alike distractor: same name, different address, same country (~25%)
            if rng.random() < 0.25:
                same_country = [e for e in pool if e["country"] == country]
                other = rng.choice(same_country) if same_country else ent
                s2_i += 1
                s2_rows.append({
                    "entity_id": f"S2-{500000 + s2_i}",
                    "business_name": vary_name(ent["name"], rng, country),
                    "business_address": other["address"],
                    "country": country,
                })

            gt_rows.append({"source1_entity_id": s1_id,
                            "matched_entity_ids": ",".join(matches)})
        # Extra unmatched distractor records in the pool
        for ent in pool:
            if rng.random() < 0.35:
                country = ent["country"]
                name = vary_name(ent["name"], rng, country)
                addr = vary_address(ent["address"], ent["city"], country, rng)
                if rng.random() < 0.5:
                    s2_i += 1
                    s2_rows.append({"entity_id": f"S2-{500000 + s2_i}",
                                    "business_name": name, "business_address": addr,
                                    "country": country})
                else:
                    s3_i += 1
                    s3_rows.append({"entity_id": f"S3-{700000 + s3_i}",
                                    "business_name": name, "business_address": addr,
                                    "country": country})

        d = out_dir / "dataset" / tag
        d.mkdir(parents=True, exist_ok=True)
        cols = ["entity_id", "business_name", "business_address", "country"]
        _write_tsv(d / f"{tag}_source1.tsv", s1_rows, cols)
        _write_tsv(d / f"{tag}_source2.tsv", s2_rows, cols)
        _write_tsv(d / f"{tag}_source3.tsv", s3_rows, cols)
        # Synthetic data: write GT for BOTH splits so we can self-evaluate
        _write_tsv(d / f"{tag}_ground_truth.tsv", gt_rows,
                   ["source1_entity_id", "matched_entity_ids"])

        n_matches = sum(len(r["matched_entity_ids"].split(","))
                        for r in gt_rows if r["matched_entity_ids"])
        print(f"  {tag}: S1={len(s1_rows)} S2={len(s2_rows)} S3={len(s3_rows)} "
              f"GT rows={len(gt_rows)} matches={n_matches}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/fixtures_synth")
    ap.add_argument("--n-s1", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Generating synthetic dataset: {args.n_s1} S1 entities, seed={args.seed}")
    generate(Path(args.out), args.n_s1, args.seed)
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
