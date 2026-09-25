# 🏗️ IMPLEMENTATION BLUEPRINT — Definitive Technical Plan

**Date:** 25 Sep 2026 · **Authors:** Team (Abhijit · Vishwesh · Karan)
**Sources:** PS PDF + official guidelines + official video + 13 team repos + 26 research repos

> This is the code-level blueprint. [MASTERPLAN.md](../MASTERPLAN.md) is the team overview; this document specifies **what to build and how**.

---

## PART 0 — PDF ALIGNMENT AUDIT (are we on the right track?)

**Verdict: ✅ YES on fundamentals — with 6 critical gaps.**

### What the PDF demands vs what we have

| PS PDF requirement | Our status | Verdict |
|--------------------|-----------|---------|
| Read TSVs with `sep="\t"` | ✅ everywhere | ✅ |
| S1 = reference; find matches in S2/S3 | ✅ pipeline does this | ✅ |
| 0/1/many matches per S1 — never top-1 | ✅ no top-1 logic | ✅ |
| Macro F_0.5 per S1, averaged, singletons included | ✅ `find_best_macro_f05_threshold` + `scripts/evaluate.py` | ✅ |
| Singleton: empty pred = 1.0; any match = 0.0 | ✅ implemented | ✅ |
| Every S1 appears exactly once | ✅ `_generate_output` + validator | ✅ |
| No duplicate IDs in a list | ✅ generation + validator | ✅ |
| Only test-set S2/S3 IDs | ✅ official validator `--check-ids` | ✅ |
| `matching_results.tsv` only scored file | ✅ | ✅ |
| `candidate_pairs.tsv` = exact model-input candidate set | ✅ fixed (was writing matches) | ✅ |
| Matches ⊆ candidates | ✅ enforced | ✅ |
| No external data/APIs/geocoding | ✅ zero network calls in `src/` | ✅ |
| MIT/Apache ≤8B model | ✅ LightGBM/XGBoost/RF (all MIT/BSD) | ✅ |
| Country open-set, France included | ⚠️ never hard-coded, but no script/FR-specific handling | 🟡 |
| Blocking = recall ceiling → invest first | ✅ measured, but no candidate budget curve | 🟡 |
| Validate before submitting | ✅ official validator installed | ✅ |
| Documentation template | ✅ official template in repo root | ✅ |
| Max 5 submissions/day | ⚠️ no submission log | 🟡 |

### The 6 critical gaps (from competitive intel)

1. **Indic transliteration missing** — 22.7% of India S1–S2 pairs are cross-script; we ASCII-fold (destroys them). SABER fixed this: India recall@20 0.835 → 0.992.
2. **No candidate caps** — real test = 1.73M S1; we must cap ~20–30/source/S1 and measure the recall-vs-K curve.
3. **No one-to-one exclusivity** — each S2/S3 matches ≤1 S1 (verified, zero exceptions); global assignment is the top precision lever.
4. **No expected-F0.5 set selection** — plain global threshold underperforms per-entity prefix selection by 0.0013–0.0017 (SABER measured).
5. **No scale engineering** — 24.2M records; our feature loop is row-wise Python; need chunked/vectorized + Parquet.
6. **Blocker similarity + bidirectional ranks discarded** — top teams feed these as model features.

**Conclusion: architecture is right, execution has specific fixable gaps. Proceed.**

---

## PART 1 — TARGET ARCHITECTURE (consensus of all top teams)

```
TSVs (24.2M records)
   │
   ▼
[1] NORMALIZE ONCE → Parquet
    · Indic→Latin transliteration (ITRANS + schwa deletion)
    · NFKD + ligatures (France) + lowercase + & → and
    · Keep RAW alongside normalized (contradiction detection)
    · ~18 min on 10 processes (SABER measured)
   │
   ▼
[2] BLOCKING (recall-first, bidirectional, adaptive-K) → candidate_pairs.tsv
    Legs (union, never intersection):
    A. char 3-gram TF-IDF (region-partitioned)     — forward top-50 + reverse top-5
    B. exact keys: address ≥12 chars, name core+last2addr tokens (drop buckets >30)
    C. [P2] dense encoder (arctic-xs / ml-e5-small) — forward + reverse
    D. rare-token + PIN/ZIP indexes
    Adaptive-K: rank < kmin OR score ≥ top1 − gap, up to kmax
    Target: ≥99% pair recall @ 20–30 candidates/S1/source
   │
   ▼
[3] PAIR FEATURES (~35–45)
    · string similarity (rapidfuzz: ratio, token_set, partial, JW)
    · blocker evidence (cosine, bidirectional ranks, leg flags)
    · competition (n_cand_s1, n_cand_r, rank_in_s1, rank_in_r)
    · structure (digit Jaccard, house-number eq, region overlap)
    · IDF/record (rare-token overlap, name/addr frequency, non-Latin flag)
    · missingness + contradiction
   │
   ▼
[4] MATCHER: LightGBM binary (MIT)
    · negatives sampled FROM the blocking candidates (train = inference distribution)
    · hard negatives: same name/different address; same address/different name
    · OOF training to keep holdout clean
   │
   ▼
[5] DECISION (the metric lives here)
    a. isotonic calibration on a held-out calibration fold
    b. one-to-one exclusivity: highest-p S1 owns each candidate record
    c. expected-F0.5 prefix selection per S1 (incl. empty set)
    d. margin/competition features feed a second-round model
   │
   ▼
[6] OUTPUT: matching_results.tsv + candidate_pairs.tsv → official validator → upload
```

---

## PART 2 — MODULE SPECIFICATIONS (code-level)

### 2.1 `src/normalize.py` — Indic transliteration (GAP #1, highest value)

**Port SABER's exact approach** (proven: India recall@20 0.835 → 0.992):

```python
import re, unicodedata
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

BLOCKS = [(0x0900, sanscript.DEVANAGARI), (0x0980, sanscript.BENGALI),
          (0x0A00, sanscript.GURMUKHI), (0x0A80, sanscript.GUJARATI),
          (0x0B00, sanscript.ORIYA), (0x0B80, sanscript.TAMIL),
          (0x0C00, sanscript.TELUGU), (0x0C80, sanscript.KANNADA),
          (0x0D00, sanscript.MALAYALAM)]

def _script(ch):
    o = ord(ch)
    for b, s in BLOCKS:
        if b <= o < b + 0x80: return s
    return None

_run = re.compile(r"[\u0900-\u0D7F\u200c\u200d]+")

def _translit(m):
    t = m.group(0).replace("\u200c", "").replace("\u200d", "")
    sc = next((_script(c) for c in t if _script(c)), None)
    if sc is None: return t
    out = transliterate(t, sc, sanscript.ITRANS)
    out = re.sub(r"(?<=[^aeiouAEIOU\s])a\b", "", out)     # word-final schwa deletion
    return out.replace("~N", "n").replace(".N", "n").replace("M", "n").replace("JN", "gy")

_LIG = str.maketrans({"œ":"oe","Œ":"OE","æ":"ae","Æ":"AE","ß":"ss","ø":"o",
                      "Ø":"O","ł":"l","Ł":"L","đ":"d","’":"'"})

def norm(s):
    if not s: return ""
    s = _run.sub(_translit, s)
    s = s.translate(_LIG)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9/ ,.-]", " ", s)
    s = re.sub(r"(?<![a-z0-9])[-.,/]+|[-.,/]+(?![a-z0-9])", " ", s)
    return re.sub(r"\s+", " ", s).strip()
```

- Dependency: `indic-transliteration` (pure Python, local, no network → compliant)
- Keep RAW + normalized both (SABER stores both in parquet)
- **Also add**: script flag feature (`is_non_latin`) for the model
- Legal suffixes for keys: private, pvt, limited, ltd, llc, inc, incorporated, corp, corporation, co, company, llp, plc, the, and, pc, pllc, lp, sarl, sas, sa, eurl, sci, gmbh

### 2.2 `src/blocking.py` — hybrid + adaptive-K + bidirectional

**Adaptive-K prune (SABER exact):**
```python
def prune(idx, sc, kmin, kmax, gap):
    """Keep rank < kmin, OR score >= top1 - gap, up to kmax."""
    k = idx.shape[1]; r = np.arange(k)[None, :]
    return (idx >= 0) & (r < kmax) & ((r < kmin) | (sc >= sc[:, :1] - gap))
```

**Defaults:** `vq_kmin=5, vq_kmax=30, vq_gap=0.10` (forward) · `vr_kmin=2, vr_kmax=5, vr_gap=0.05` (reverse) · same for trigram with `tq_gap=0.15, tr_gap=0.10`.

**Key legs:** R-side buckets >30 records dropped; address key = exact normalized address ≥12 chars; name key = core name (legal-stripped) + last two address tokens, core ≥3 chars.

**Region partitioning (multi-membership):**
- Vocabulary = every normalized comma-part appearing in ≥0.05% of that country's S1 records (unsupervised, works for France)
- A record belongs to every region named by any part, its first/last token, first/last two tokens (direct or alias)
- Compare if regions intersect OR either is unknown
- Measured: US 98.68% pair retention @ 15% comparisons; India 99.94% @ 22%

**Why bidirectional matters (SABER measured):** reverse-trigram top-5 alone recovers 0.986 India / 0.984 US pairs. Forward-only union = India 0.9779; with reverse+keys+adaptive-K = 0.9903.

**Dead end to avoid:** one-region-per-record (last address part) lost 4.1% of pairs, 92% of losses in India.

### 2.3 `src/features.py` — the ~41-feature set (SABER's proven list)

**Leg evidence:** `cos_v` (dense), `cos_t` (trigram), `in_vq/in_vr/in_tq/in_tr/in_a/in_n` flags, `rank_vq/rank_vr/rank_tq/rank_tr` (0-based, 99=absent)

**Competition:** `n_cand_s1`, `n_cand_r`, `cos_v_rank_in_s1`, `cos_v_rank_in_r` (1-based)

**String:** `name_ratio`, `name_tset`, `name_partial` (on core names), `core_jw`, `core_equal`, `addr_tset`, `addr_tsort`, `addr_partial`, `name_len_diff`

**Structure:** `num_jacc` (digit-set Jaccard, −1 if empty), `house_eq` (first digit token, −1 if null), `r_addr_empty`, `region_overlap` (−1 unknown)

**IDF/record (per country, unsupervised):** `name_wjacc`, `addr_wjacc`, `name_rare_miss_s1/r`, `addr_rare_miss_s1/r`, `name_idf_match`, `nfreq_s1/r`, `afreq_s1/r`, `r_nonlatin`

**Class separation (SABER measured, India):** matches score name_tset 86.9 vs 54.6; addr_tset 90.3 vs 52.8; cos_v 0.850 vs 0.649; num_jacc 0.642 vs 0.015. These features WORK.

### 2.4 `src/decision.py` — expected-F0.5 set selection (GAP #4)

**SABER's algorithm (measured +0.0013–0.0017 over hard threshold):**

```
1. EXCLUSIVITY: per (country, candidate_id), highest-p S1 owns it (tie-break: s1_id asc),
   only if p >= p_min (default 0.05)
2. expected_truth_i = Σ p of owned children
3. For k = 1..n (sorted by p desc): expected_f(k) = 1.25·Σ_{j≤k} p_j / (0.25·expected_truth_i + k)
4. empty_score = 1 − p_has_match   (baseline proxy: 1 − max_p)
5. Emit top-k prefix if expected_f(k*) > empty_score, else empty set
```

**Never multiply dependent probabilities** (`∏(1−p_j)` is wrong without a singleton classifier; SABER docstring is emphatic).

**One-to-one greedy assignment (resolvers, simpler alternative):**
```python
above = pairs[pairs.score >= threshold].sort_values("score", ascending=False)
claimed, rows = {}, {}
for s1_id, cand_id, score in above.itertuples(index=False):
    if cand_id in claimed: continue          # already assigned to a better S1
    claimed[cand_id] = s1_id
    rows.setdefault(s1_id, []).append(cand_id)
```

**Country-holdout proxy for France:** train US→eval India, train India→eval US; use the gap to set stricter cutoffs where unsupervised.

### 2.5 Scale engineering (GAP #5)

| Rule | Implementation |
|------|----------------|
| Never dense (query × gallery) | chunked sparse matmul (`chunk @ target.T`), `dense_output=False` |
| Transposed sparse mm is 35× slower | use contiguous `(V × b)` block (SABER measured: 1.197s → 0.034s) |
| No row-wise Python feature loops | vectorized `rapidfuzz.process.cpdist(workers=-1)` |
| Chunk sizes | reads 200K rows; features ≤4M pairs/chunk; TSV written streaming |
| Store intermediates | Parquet (not pickle/CSV) |
| Candidate shards | cut at ≤2M pairs, on S1 boundaries |
| Memory ceiling | never materialize >2 GB (SIBAM contract) |
| `del + gc.collect()` after every chunk | mandatory |

### 2.6 Validation (both masterplans + RF audit agree)

- **Splits:** 80% fit / 10% calibration / 10% locked holdout, grouped by S1 identity (each S2/S3 belongs to ≤1 S1 → S1 grouping is leakage-safe; verified zero fan-out)
- Stratify by country + match-count buckets `{0, 1, 2, 3–4, 5+}`
- **Deterministic partition hashing:** `zlib.crc32(f"{seed}:{id}")/2**32` (SABER pattern)
- **Report separately:** candidate-oracle F0.5 ceiling vs achieved F0.5
- **Oracle formula:** `Oracle_i = 1 if t_i==0 else 5·r_i/(4·r_i + t_i)` where `r_i = |C_i ∩ T_i|`
- Bootstrap **business groups** (not pairs) for uncertainty
- Keep the locked holdout untouched until final threshold freeze

### 2.7 Evaluation (exact metric, all teams agree)

```python
def entity_f05(truth, pred):
    if not truth: return float(not pred)
    tp = len(truth & pred); fp = len(pred - truth); fn = len(truth - pred)
    return 5.0 * tp / (5 * tp + 4 * fp + fn)      # equivalent to 1.25PR/(0.25P+R)
```

Our `scripts/evaluate.py` already implements this correctly (verified against the worked example). Add: per-country breakdown, match-count buckets, oracle ceiling, singleton accuracy.

---

## PART 3 — WHAT WE HAVE vs WHAT WE BUILD

| Module | Have | Build |
|--------|------|-------|
| normalize | 10-step + legal suffixes | **+ Indic transliteration** (port SABER) |
| blocking | 7-layer union + top-K caps + chunked sparse | **+ adaptive-K + bidirectional + region partition + key legs** |
| features | 35 (name/addr/country/cross/missingness/contradiction) | **+ blocker evidence (cos/ranks/leg flags) + competition + IDF/record** |
| model | LGB+XGB+RF → OOF meta | keep; sample negatives from blocking candidates |
| decision | global macro-F0.5 threshold | **+ calibration + exclusivity + expected-F0.5 prefix** |
| scale | smoke-tested at 300 rows | **vectorize + Parquet + chunk contract** |
| eval | macro F0.5 + singleton accuracy | **+ oracle ceiling + country/bucket breakdowns** |

---

## PART 4 — EXECUTION ORDER (next 48 hours)

| # | Task | Owner | Blocks |
|---|------|-------|--------|
| 1 | **Indic transliteration** in normalize.py + unit tests | Vishwesh | blocking |
| 2 | **Adaptive-K + bidirectional + key legs** in blocking.py | Vishwesh | features |
| 3 | **Vectorized feature pipeline** + Parquet | Abhijit | model |
| 4 | **Blocker evidence features** (cos/ranks/flags) | Vishwesh + Abhijit | model |
| 5 | **Negatives from blocking candidates** + OOF | Abhijit | decision |
| 6 | **Calibration + exclusivity + expected-F0.5** in decision.py | Abhijit | submission |
| 7 | **Oracle + country/bucket eval** in evaluate.py | Karan | iteration |
| 8 | **Synthetic generator → real distribution** | Karan | testing |
| 9 | **Scale benchmark** (50K roots) | All | full run |
| 10 | **Full train run → test run → validate → upload** | Abhijit | score |

---

## PART 5 — GUARDRAILS (from all sources)

1. Never force a match because a neighbor exists — empty is a valid learned decision.
2. Never top-1; S1 accepts many S2/S3 targets.
3. Never multiply dependent edge probabilities for the empty set.
4. Never tune on the locked holdout; bootstrap business groups.
5. Never hard-code country values; France must flow through.
6. Never drop duplicate-text IDs — they are separate required targets.
7. Never materialize dense matrices or >2 GB intermediates.
8. Never use external data/APIs/geocoding — zero outbound calls.
9. Never claim unmeasured numbers; every score in docs must have an evidence file.
10. `candidate_pairs.tsv` must be produced in the same run as `matching_results.tsv`; matches ⊆ candidates.
