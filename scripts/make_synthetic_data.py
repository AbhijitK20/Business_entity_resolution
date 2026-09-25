"""Synthetic dataset generator — calibrated to the VERIFIED real distribution.

Real-distribution targets (docs/COMPETITIVE_INTEL.md §1, measured by two
independent teams on the official files):

    match counts per S1 root:  89.0% multi-match (>=2), 5.58% singleton,
                               mean 3.46, maximum 11
    gallery blank addresses:   ~3% of S2/S3 records ONLY (S1 never blank)
    shared S1 names:           47% of S1 entities share a name with another S1
    India cross-script names:  22.7% of India S1-S2 pairs are cross-script
    countries:  train US 60% / India 40%
                test   US 38% / India 47% / France 15%   (France = test-only)

All generation is deterministic given --seed. No external data, APIs, or
geocoding — pure local string transformation.

Usage:
    python scripts/make_synthetic_data.py --out tests/fixtures_synth \
        --n-s1 300 --seed 42
"""
import argparse
import json
import random
import unicodedata
from pathlib import Path

# ------------------------------------------------- real match-count pmf (K2)
# Shares measured on train_ground_truth.tsv (COMPETITIVE_INTEL §1).
# 10-11 combined = 0.03% (571 roots) → split evenly.
MATCH_COUNT_PMF = {
    0: 0.0558,
    1: 0.0540,
    2: 0.1700,
    3: 0.2406,
    4: 0.2194,
    5: 0.1459,
    6: 0.0747,
    7: 0.0290,
    8: 0.0085,
    9: 0.0019,
    10: 0.00015,
    11: 0.00015,
}

TRAIN_COUNTRY_MIX = {"US": 0.60, "India": 0.40}
TEST_COUNTRY_MIX = {"US": 0.38, "India": 0.47, "France": 0.15}

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

COUNTRY_POOLS = {
    "US": (US_NAMES, [c for c, _ in US_CITIES], US_STREETS),
    "India": (IN_NAMES, IN_CITIES, IN_STREETS),
    "France": (FR_NAMES, FR_CITIES, FR_STREETS),
}

# ---------------------------------------------------------------------------
# Cross-script (Indic) rendering — deterministic local transliteration TARGET.
# We generate the *native-script* side of a cross-script pair: the Latin name
# is rendered into Devanagari/Telugu with a fixed letter table (no external
# API, no dataset download). src/normalize.py (Vishwesh) transliterates back.
# ---------------------------------------------------------------------------
DEVANAGARI_CONS = {
    "k": "क", "g": "ग", "c": "च", "j": "ज", "t": "त", "d": "द", "n": "न",
    "p": "प", "b": "ब", "m": "म", "y": "य", "r": "र", "l": "ल", "v": "व",
    "w": "व", "s": "स", "h": "ह", "x": "क्स", "z": "ज़", "q": "क़",
    "f": "फ़",
}
DEVANAGARI_VOWELS = {
    "a": ("अ", "ा"), "i": ("इ", "ि"), "u": ("उ", "ु"),
    "e": ("ए", "े"), "o": ("ओ", "ो"),
}
TELUGU_CONS = {
    "k": "క", "g": "గ", "c": "చ", "j": "జ", "t": "త", "d": "ద", "n": "న",
    "p": "ప", "b": "బ", "m": "మ", "y": "య", "r": "ర", "l": "ల", "v": "వ",
    "w": "వ", "s": "స", "h": "హ", "x": "క్స", "z": "జ్", "q": "క్",
    "f": "ఫ్",
}
TELUGU_VOWELS = {
    "a": ("అ", "ా"), "i": ("ఇ", "ి"), "u": ("ఉ", "ు"),
    "e": ("ఏ", "ే"), "o": ("ఓ", "ో"),
}
INDIC_RANGES = ((0x0900, 0x0D7F),)


def _render_indic(text: str, cons: dict, vowels: dict) -> str:
    """Render Latin text in an Indic script (deterministic, local tables)."""
    out = []
    prev_cons = False
    for ch in text.lower():
        if ch in vowels:
            standalone, matra = vowels[ch]
            out.append(matra if prev_cons else standalone)
            prev_cons = False
        elif ch in cons:
            out.append(cons[ch])
            prev_cons = True
        else:
            out.append(ch)          # spaces, digits, punctuation pass through
            prev_cons = False
    return "".join(out)


def to_devanagari(text: str) -> str:
    return _render_indic(text, DEVANAGARI_CONS, DEVANAGARI_VOWELS)


def to_telugu(text: str) -> str:
    return _render_indic(text, TELUGU_CONS, TELUGU_VOWELS)


def to_indic(text: str, script: str) -> str:
    return to_devanagari(text) if script == "Devanagari" else to_telugu(text)


def has_indic_script(text: str) -> bool:
    return any(any(lo <= ord(ch) <= hi for lo, hi in INDIC_RANGES)
               for ch in (text or ""))


# ------------------------------------------------- deterministic allocation
def largest_remainder(n: int, weights: dict) -> dict:
    """Proportional integer allocation (largest remainder / Hamilton).

    Deterministic: ties broken by descending fractional part then ascending
    key order. Guarantees sum(values) == n.
    """
    total = sum(weights.values())
    keys = sorted(weights.keys())
    quotas = {k: n * weights[k] / total for k in keys}
    floors = {k: int(quotas[k]) for k in keys}
    remainder = n - sum(floors.values())
    order = sorted(keys, key=lambda k: (-(quotas[k] - floors[k]), k))
    for i in range(remainder):
        floors[order[i % len(order)]] += 1
    return floors


def allocate_match_counts(n: int, pmf: dict = None) -> list:
    """Exact proportional allocation of the real match-count distribution."""
    pmf = pmf or MATCH_COUNT_PMF
    counts = largest_remainder(n, pmf)
    out = []
    for k in sorted(counts):
        out.extend([k] * counts[k])
    return out


def sample_match_counts(n: int, rng: random.Random, pmf: dict = None) -> list:
    """Match counts for n S1 entities: real pmf, deterministic given seed."""
    out = allocate_match_counts(n, pmf)
    rng.shuffle(out)
    return out


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


# --------------------------------------------------------------- entity prep
def _assign_names(countries, share_name_rate, rng):
    """Assign S1 display names so that ~share_name_rate of entities share a
    name with at least one other entity (47% in the real data).

    Deterministic count via largest-remainder, then shuffled. Shared names are
    clustered (cluster size >= 2) so every shared name really is duplicated;
    unique names carry a serial and are never reused.
    """
    by_country = {}
    for idx, country in enumerate(countries):
        by_country.setdefault(country, []).append(idx)

    shared_idx = set()
    for country, idxs in by_country.items():
        if len(idxs) < 2:
            continue
        n_share = largest_remainder(len(idxs),
                                    {"share": share_name_rate,
                                     "unique": 1 - share_name_rate})["share"]
        n_share = min(n_share, len(idxs) - 1)   # keep >=1 unique per country
        order = idxs[:]
        rng.shuffle(order)
        shared_idx.update(order[:n_share])

    names = [None] * len(countries)
    serial = {c: 0 for c in by_country}

    # unique names first (shared picks never steal a unique name)
    for idx in range(len(countries)):
        if idx in shared_idx:
            continue
        country = countries[idx]
        pool = COUNTRY_POOLS[country][0]
        serial[country] += 1
        base = pool[(serial[country] - 1) % len(pool)]
        names[idx] = f"{base} {serial[country]}"

    # shared names: cluster shared entities into groups of >= 2 per country;
    # every cluster gets the BARE pool name (no serial) so the name is shared
    for country, idxs in by_country.items():
        members = [i for i in idxs if i in shared_idx]
        if not members:
            continue
        pool = COUNTRY_POOLS[country][0]
        clusters = []
        pos = 0
        while pos < len(members):
            remaining = len(members) - pos
            if remaining == 1:
                if clusters:
                    clusters[-1].append(members[pos])   # fold straggler in
                else:
                    clusters.append([members[pos]])     # lone shared (n=1)
                pos += 1
            else:
                size = 3 if remaining % 2 == 1 else 2
                clusters.append(members[pos:pos + size])
                pos += size
        for ci, cluster in enumerate(clusters):
            label = pool[(ci // 2) % len(pool)]         # bare shared name
            for idx in cluster:
                names[idx] = label
    return names


def _make_address(country, cities, streets, rng, used_addresses):
    """Distinct address per business (shared-name businesses must not share an
    address — that would create a duplicate identity, not a distractor)."""
    for _ in range(50):
        city = rng.choice(cities)
        street = rng.choice(streets)
        number = rng.randint(1, 999)
        address = f"{number} {street}, {city}"
        if address not in used_addresses:
            used_addresses.add(address)
            return address, city
    city = rng.choice(cities)
    address = f"{len(used_addresses) + 1} {rng.choice(streets)}, {city}"
    used_addresses.add(address)
    return address, city


def make_entities(country: str, n: int, rng: random.Random,
                  share_name_rate: float = 0.47):
    """Generate n S1 entities for one country with real-distribution naming."""
    pool, cities, streets = COUNTRY_POOLS[country]
    countries = [country] * n
    names = _assign_names(countries, share_name_rate, rng)
    used_addresses = set()
    out = []
    for i in range(n):
        address, city = _make_address(country, cities, streets, rng,
                                      used_addresses)
        out.append({"name": names[i], "city": city, "address": address,
                    "country": country})
    return out


def _write_tsv(path: Path, rows, cols):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def parse_mix(spec: str) -> dict:
    """Parse 'US=0.6,India=0.4' → {'US': 0.6, 'India': 0.4}."""
    out = {}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        out[k.strip()] = float(v)
    if not out:
        raise ValueError(f"empty country mix: {spec!r}")
    return out


def _make_gallery_row(name, address, country, rng, cross_script_rate,
                      blank_rate):
    """One S2/S3 record: noise + optional cross-script name + optional blank
    address (gallery-only; S1 addresses are never blank)."""
    if country == "India" and rng.random() < cross_script_rate:
        script = rng.choice(["Devanagari", "Telugu"])
        name = to_indic(name, script)
    if address and rng.random() < blank_rate:
        address = ""
    return {"business_name": name, "business_address": address,
            "country": country}


# ---------------------------------------------------------------- generation
def generate(out_dir: Path, n_s1: int, seed: int,
             train_mix: dict = None, test_mix: dict = None,
             blank_rate: float = 0.03,
             share_name_rate: float = 0.47,
             cross_script_rate: float = 0.227,
             same_name_distractor_rate: float = 0.25,
             same_addr_distractor_rate: float = 0.15,
             pool_distractor_rate: float = 0.35,
             train_frac: float = 0.8):
    """Generate train/test datasets calibrated to the real distribution."""
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    train_mix = train_mix or dict(TRAIN_COUNTRY_MIX)
    test_mix = test_mix or dict(TEST_COUNTRY_MIX)

    n_train = int(round(n_s1 * train_frac))
    n_test = n_s1 - n_train

    splits = {"train": (n_train, train_mix), "test": (n_test, test_mix)}

    # global id counters (unique across splits)
    s1_i = s2_i = s3_i = 0
    stats = {}

    for tag, (n_entities, mix) in splits.items():
        if n_entities <= 0:
            continue
        country_counts = largest_remainder(n_entities, mix)
        countries = []
        for c in sorted(country_counts):
            countries.extend([c] * country_counts[c])
        rng.shuffle(countries)

        # match counts follow the real pmf (independent of country)
        match_counts = sample_match_counts(n_entities, rng)

        # build entities per country bucket, preserving the shuffled order
        entities = [None] * n_entities
        for country in sorted(country_counts):
            idxs = [i for i, c in enumerate(countries) if c == country]
            for slot, ent in zip(idxs, make_entities(
                    country, len(idxs), rng, share_name_rate)):
                entities[slot] = ent
        for i, ent in enumerate(entities):
            ent["n_matches"] = match_counts[i]

        # S1 display names: ONE suffix per distinct base name, so entities
        # sharing a base name keep an IDENTICAL business_name in the S1 file
        # (the real "47% of S1 share names" statistic survives suffix noise).
        suffix_cache = {}
        s1_pairs = set()
        for ent in entities:
            base = ent["name"]
            if base not in suffix_cache:
                suffix_cache[base] = rng.choice(SUFFIXES[ent["country"]])
            sfx = suffix_cache[base]
            ent["s1_name"] = f"{base} {sfx}".strip() if sfx else base
            s1_pairs.add((ent["s1_name"], ent["address"]))

        s1_rows, s2_rows, s3_rows, gt_rows = [], [], [], []

        def emit(s2: bool, name: str, address: str, country: str,
                 distractor: bool = False) -> str:
            """Write one gallery record. Distractors are guarded against being
            exact copies of an S1 record (would be a duplicate identity that
            ground truth does not label)."""
            nonlocal s2_i, s3_i
            if distractor and (name, address) in s1_pairs:
                address = f"{rng.randint(1000, 9999)} {address}"
            gal = _make_gallery_row(name, address, country, rng,
                                    cross_script_rate, blank_rate)
            if s2:
                s2_i += 1
                sid = f"S2-{500000 + s2_i}"
                s2_rows.append({"entity_id": sid, **gal})
            else:
                s3_i += 1
                sid = f"S3-{700000 + s3_i}"
                s3_rows.append({"entity_id": sid, **gal})
            return sid

        for ent in entities:
            country = ent["country"]
            s1_i += 1
            s1_id = f"S1-{100000 + s1_i}"
            s1_rows.append({
                "entity_id": s1_id,
                "business_name": ent["s1_name"],
                "business_address": ent["address"],
                "country": country,
            })

            matches = []
            for _ in range(ent["n_matches"]):
                src_s2 = rng.random() < 0.48          # real: 3.69M S2 / 3.94M S3
                name = vary_name(ent["name"], rng, country)
                addr = vary_address(ent["address"], ent["city"], country, rng)
                sid = emit(src_s2, name, addr, country)
                matches.append(sid)

            # hard negative 1: SAME name, DIFFERENT address (look-alike)
            if rng.random() < same_name_distractor_rate:
                name = vary_name(ent["name"], rng, country)
                other_addr = rng.choice(entities)["address"]
                if other_addr == ent["address"]:
                    other_addr = f"{rng.randint(1000, 9999)}, {ent['city']}"
                emit(rng.random() < 0.5, name, other_addr, country,
                     distractor=True)

            # hard negative 2: DIFFERENT name, SAME address (shared premise)
            if rng.random() < same_addr_distractor_rate:
                pool = COUNTRY_POOLS[country][0]
                other_name = rng.choice(pool)
                if other_name == ent["name"]:
                    other_name = rng.choice([p for p in pool
                                             if p != ent["name"]] or pool)
                name = vary_name(other_name, rng, country)
                emit(rng.random() < 0.5, name, ent["address"], country,
                     distractor=True)

            gt_rows.append({"source1_entity_id": s1_id,
                            "matched_entity_ids": ",".join(matches)})

        # background distractors in the gallery (real: 2.68M train distractors)
        for ent in entities:
            if rng.random() < pool_distractor_rate:
                pool = COUNTRY_POOLS[ent["country"]][0]
                name = vary_name(rng.choice(pool), rng, ent["country"])
                addr = vary_address(ent["address"], ent["city"],
                                    ent["country"], rng)
                emit(rng.random() < 0.5, name, addr, ent["country"],
                     distractor=True)

        d = out_dir / "dataset" / tag
        d.mkdir(parents=True, exist_ok=True)
        cols = ["entity_id", "business_name", "business_address", "country"]
        _write_tsv(d / f"{tag}_source1.tsv", s1_rows, cols)
        _write_tsv(d / f"{tag}_source2.tsv", s2_rows, cols)
        _write_tsv(d / f"{tag}_source3.tsv", s3_rows, cols)
        # Synthetic data: write GT for BOTH splits so we can self-evaluate
        _write_tsv(d / f"{tag}_ground_truth.tsv", gt_rows,
                   ["source1_entity_id", "matched_entity_ids"])

        stats[tag] = summarize_split(s1_rows, s2_rows, s3_rows, gt_rows)
        stats[tag]["country_mix"] = _country_mix(s1_rows)
        print(f"  {tag}: " + _format_stats(stats[tag]))

    (out_dir / "synthetic_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    return stats


# ----------------------------------------------------------------- reporting
def _country_mix(rows) -> dict:
    total = len(rows) or 1
    counts = {}
    for r in rows:
        counts[r["country"]] = counts.get(r["country"], 0) + 1
    return {k: round(v / total, 4) for k, v in sorted(counts.items())}


def summarize_split(s1_rows, s2_rows, s3_rows, gt_rows) -> dict:
    """Distribution statistics of one generated split (K2 validation surface)."""
    counts = []
    by_id = {r["source1_entity_id"]: r for r in gt_rows}
    for r in s1_rows:
        raw = by_id.get(r["entity_id"], {}).get("matched_entity_ids", "")
        counts.append(len([x for x in raw.split(",") if x.strip()]) if raw else 0)
    n = len(counts) or 1
    multi = sum(1 for c in counts if c >= 2)
    single = sum(1 for c in counts if c == 0)
    gallery = s2_rows + s3_rows
    blank = sum(1 for r in gallery if not (r.get("business_address") or "").strip())
    indic = sum(1 for r in gallery if has_indic_script(r.get("business_name", "")))
    name_seen = {}
    for r in s1_rows:
        name_seen[r["business_name"]] = name_seen.get(r["business_name"], 0) + 1
    shared = sum(1 for r in s1_rows if name_seen[r["business_name"]] > 1)
    return {
        "n_s1": len(s1_rows),
        "n_gallery": len(gallery),
        "multi_match_pct": round(100.0 * multi / n, 2),
        "singleton_pct": round(100.0 * single / n, 2),
        "mean_matches": round(sum(counts) / n, 4),
        "max_matches": max(counts) if counts else 0,
        "blank_gallery_addr_pct": round(100.0 * blank / max(len(gallery), 1), 2),
        "shared_name_pct": round(100.0 * shared / max(len(s1_rows), 1), 2),
        "indic_name_pct_gallery": round(
            100.0 * indic / max(len(gallery), 1), 2),
        "total_matches": sum(counts),
    }


def _format_stats(s: dict) -> str:
    return (f"S1={s['n_s1']} gallery={s['n_gallery']} "
            f"multi={s['multi_match_pct']}% singleton={s['singleton_pct']}% "
            f"mean={s['mean_matches']} max={s['max_matches']} "
            f"blank_addr={s['blank_gallery_addr_pct']}% "
            f"shared_names={s['shared_name_pct']}% "
            f"indic={s['indic_name_pct_gallery']}% "
            f"countries={s.get('country_mix')}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="tests/fixtures_synth")
    ap.add_argument("--n-s1", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blank-rate", type=float, default=0.03,
                    help="blank address rate on GALLERY side only (real ~3%%)")
    ap.add_argument("--share-name-rate", type=float, default=0.47,
                    help="fraction of S1 sharing a name (real 47%%)")
    ap.add_argument("--cross-script-rate", type=float, default=0.227,
                    help="India gallery rows rendered in Indic script — "
                         "pair-level proxy for the real 22.7%% cross-script "
                         "rate among India S1-S2 pairs")
    ap.add_argument("--same-name-distractor-rate", type=float, default=0.25)
    ap.add_argument("--same-addr-distractor-rate", type=float, default=0.15)
    ap.add_argument("--pool-distractor-rate", type=float, default=0.35)
    ap.add_argument("--train-country-mix", default="US=0.6,India=0.4")
    ap.add_argument("--test-country-mix", default="US=0.38,India=0.47,France=0.15")
    ap.add_argument("--train-frac", type=float, default=0.8)
    args = ap.parse_args()

    print(f"Generating synthetic dataset: {args.n_s1} S1 entities, "
          f"seed={args.seed}")
    stats = generate(
        Path(args.out), args.n_s1, args.seed,
        train_mix=parse_mix(args.train_country_mix),
        test_mix=parse_mix(args.test_country_mix),
        blank_rate=args.blank_rate,
        share_name_rate=args.share_name_rate,
        cross_script_rate=args.cross_script_rate,
        same_name_distractor_rate=args.same_name_distractor_rate,
        same_addr_distractor_rate=args.same_addr_distractor_rate,
        pool_distractor_rate=args.pool_distractor_rate,
        train_frac=args.train_frac,
    )
    print(f"Done -> {args.out} (stats: {args.out}/synthetic_stats.json)")


if __name__ == "__main__":
    main()
