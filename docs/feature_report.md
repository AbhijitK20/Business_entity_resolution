# Feature & Evaluation Report — Karan (K1–K5)

**Stream:** oracle/segmented evaluation · real-distribution synthetic data · feature tests · country holdout · error analysis
**Status:** K1 ✅ K2 ✅ K3 ✅(with A1-blocked families) K4 ✅ K5 ✅ — all on **synthetic data, clearly labelled**
**Real dataset:** unavailable in this environment (`data/` symlink target missing → H4 blocker). Every number below is from the seeded synthetic generator unless stated otherwise. Re-run commands are listed in §7.

---

## 1. Scope & data provenance

| Item | Value |
|---|---|
| Generator | `scripts/make_synthetic_data.py` (seed 42, deterministic) |
| Verified targets | `docs/COMPETITIVE_INTEL.md` §1 (real train/test distribution) |
| Evaluation harness | `scripts/evaluate.py` (oracle, segments, bootstrap, error buckets) |
| Country holdout | `scripts/country_holdout.py` |
| Error-analysis demo | `scripts/error_analysis_demo.py` |
| Tests | `tests/test_evaluate.py` (24) · `tests/test_synthetic_data.py` (14) · `tests/test_features.py` (14 pass + 5 A1-blocked skips) · `tests/test_country_holdout.py` (10) |
| France | Test-only in the real competition — **never fabricated** in training data or metrics |

**Synthetic vs real — coverage of verified targets** (n_s1=600, seed 42):

| Statistic | Real target | Synthetic train | Synthetic test |
|---|---|---|---|
| Multi-match entities | 89.0% | 88.96% | 89.17% |
| Singletons | 5.58% | 5.62% | 5.83% |
| Mean matches | 3.46 | 3.458 | 3.475 |
| Max matches | 11 | 9¹ | 8¹ |
| Blank addresses (gallery only) | ~3% | 2.76% | 5.18%² |
| S1 entities sharing a name | 47% | 46.88% | 46.67% |
| India cross-script names | **22.7% of India S1–S2 pairs** (real, verified) | generator renders ~22.7% of India **gallery rows** in Indic script — a pair-level proxy, India-only, ±6pp test-asserted | same |
| Country mix | US 60/India 40 (train); US 38/India 47/France 15 (test) | exact (largest-remainder) | exact |

¹ At n=600 the 11-match tail (0.026% of entities) is not expected; the **allocator** reproduces `max=11` at n=100k (test: `test_match_pmf_targets_at_scale`), and no split ever exceeds 11.
² Small-split noise; test asserts gallery blank ∈ [0, 7%) and S1 blank = 0.

Integrity invariants (test-enforced): no gallery record is a true match of two S1 entities; all matched ids exist in the gallery; distractors never enter ground truth; fixed seed → byte-identical output.

---

## 2. Feature-engineering methodology (drop-in section for `Documentation_template.md` §4)

**Feature set:** 35 features in `src/features.py`, computed per (S1, candidate) pair after the shared normalization pipeline (Indic transliteration → Unicode fold → lowercase → legal-suffix stripping → abbreviation expansion → punctuation strip).

| Family | Count | What it measures | Semantics notes |
|---|---|---|---|
| Name | 10 | token-sort / partial / WRatio, Jaro-Winkler, token Jaccard, edit ratio, trigram Jaccard, Soundex, Metaphone, length ratio | All ∈ [0,1]; empty names handled via missingness flags, never NaN |
| Address | 6 | token-sort / partial / WRatio, token Jaccard, trigram Jaccard, length ratio | abbreviation-expanded before compare |
| Country | 1 | exact label match | open-set safe: unknown ≠ conflict; 0/1 only |
| Cross | 8 | name⊕address avg/max/min, combined trigram, is_company, phonetic vote, surname-length diff | `avg` is exactly the mean of its parts (test-enforced) |
| Missingness | 6 | field-presence flags | whitespace-only counts as missing |
| Contradiction | 4 | country conflict, house-number conflict, city conflict, sum | **missing ≠ conflict** (test-enforced); count = sum of flags |
| *Pending A1* | 32 | blocker-evidence (12), competition/rank (4), IDF/record (12), structure (4: `num_jacc`/`house_eq`/`r_addr_empty`/`region_overlap`) | **Blocked on A1 (Abhijit)** — tests exist and SKIP with an explicit `BLOCKED on A1` note; they start asserting the moment the features land, incl. the `-1` sentinel semantics, the 0-based/`99`=absent leg ranks, and the `r_addr_empty` flag |

**Design decisions worth defending in the methodology:**

1. **Hard negatives mirror the real distribution.** 47% of S1 entities share their exact name with other records, so name-only similarity is *not* sufficient — the synthetic generator reproduces this, and the feature tests verify behavior *under that regime*, not on toy pairs.
2. **Missing is not contradictory.** Blank country/address on either side never raises a conflict flag — it raises a missingness flag. This prevents the model from learning "blank = mismatch" (real data has ~3% blank gallery addresses).
3. **Open-set country.** France (and any unseen label) is just "different country", never a crash and never a fabricated training example.
4. **Sentinels, not zeros.** `num_jacc`/`house_eq`/`region_overlap` use `-1` for unknown (A1 spec) so "no digits on one side" can never masquerade as "digit sets disjoint = 0.0".
5. **Dud detection over dud deletion.** `detect_dud_features()` in `tests/test_features.py` flags constant / near-constant / all-NaN / duplicate features and *reports* them (duplicates: identical value vectors, first occurrence kept); deletion stays the model owner's call.

**Class-separation evidence (K3):**

- *Controlled pairs* (4 matches / 5 negatives): every core feature separates with the right sign — `name_WRatio` median 0.95 vs 0.50, `addr_WRatio` 0.92 vs 0.38, `contradiction_count` 0 vs 2 (higher on negatives by design).
- *Real-shaped synthetic pairs* (positives from GT + hardest same-country, non-matching distractors):

| Group | Verdict on hard negatives | Why |
|---|---|---|
| Address features (6/6) | ✅ separate (e.g. `addr_jaccard` med 1.00 vs 0.00) | true matches share premises; distractors don't |
| Cross features (`_avg`, `_min`, `combined_trigram`) | ✅ separate | combine both signals |
| Name features (10/10) | ⚠️ medians equal (neg = 1.0000) | **by construction**: hard negatives are same-name businesses (the verified 47% regime) |
| Missingness / contradiction flags | constant in this sample | sample has no blanks & same-country negatives — controlled tests prove they vary correctly when conditions occur |

**Actionable conclusions for the model owner (A1/A2):**
- Name features must be **paired with address/contradiction evidence** (or leg-evidence/IDF features from A1) — never trusted alone; phonetic votes actively anti-separate in the same-name regime.
- `name_phonetic_vote` and `name_metaphone_match` show negative deltas under same-name hard negatives → candidates for interaction terms, not standalone weights.
- Contradiction flags will only earn weight once negatives include cross-country/blank-address cases — ensure A2's negative sampling keeps them, otherwise they are dead columns.

---

## 3. Evaluation harness (K1 — `scripts/evaluate.py`)

Leaderboard-exact macro F_0.5 (per-S1, singletons included) plus:

- **Candidate-oracle ceiling** `Oracle_i = 1 if t_i=0 else 5·r_i/(4·r_i+t_i)` → separates *blocking loss* from *selector loss* (`selector_loss = oracle − achieved`).
- **Segments:** per-country (France only when real test rows exist) × match-count buckets `{0, 1, 2, 3-4, 5+}` with candidate recall, reduction ratio, complete-match coverage.
- **Bootstrap CIs** resampling **business groups (S1 entities)**, never pairs — 1000 reps, 95%, seed 42 (documented in output).
- **Singleton false-merge rate** tracked separately (F_0.5 punishes it hard).

All of the above are covered by `tests/test_evaluate.py` — 24/24, incl. hand-computed oracle values, singleton semantics, bootstrap determinism and CI coverage behavior, and error-case-only worst-entity lists (never padded with correct entities).

---

## 4. Country-holdout stress test (K4 — `scripts/country_holdout.py`)

Protocol: train US → eval India, train India → eval US (+ pooled all-country model as final-model proxy). **Clean country holdout:** each direction trains only on its own country's pairs — the eval country's gallery (and any test-only country such as France) never appears in training, not even as negatives; the JSON records `train_negative_countries` per direction as proof. Two eval regimes: **hard** (top same-country distractors — drives the France margin) and **full** (every gallery record a candidate — leaderboard-like density). Run: `n_s1=600, seed=42, optuna n_trials=10, full density` → `output/k4_country_holdout.json`.

| Direction | in-country P@t | transfer P (hard) | Δ precision | Δ macro F_0.5 | precision-preserving cutoff | margin |
|---|---|---|---|---|---|---|
| US → India | 0.9784 @ t=0.85 | 0.9443 | −0.0340 (3.5%) | −0.0956 | 0.93 | **0.08** |
| India → US | 0.9754 @ t=0.71 | 0.9472 | −0.0282 (2.9%) | −0.0309 | 0.93 | **0.22** |

- Pooled final-model proxy: **t = 0.88**, val macro F_0.5 = 0.973.
- **France cutoff recommendation: `t_france = 0.95`** — **CAPPED**: pooled 0.88 + max margin 0.22 = 1.10 exceeds the 0.95 grid end, so the report shows `uncapped_cutoff: 1.1`, `capped_at_0_95: true` and a `capping_note`. Both directions' precision-preserving margins were reachable in this run (`directions_with_unreachable_margin: []`).
- Full-density transfer precision under a *plain threshold* decision drops to 0.59 (US→India) / 0.36 (India→US) vs ~0.95 in-country: at leaderboard-like density, threshold-only decisions are unusable — the decision layer (exclusivity + expected-F0.5, A3) is mandatory, and margins must be derived under the hard regime (which is what this script does).

**Caveats:** synthetic pairs only. An earlier audit run had training negatives sampled from the eval country's gallery (a protocol leak that biased the transfer gap); the numbers above are from the **clean protocol** (train-country-only negatives), recorded in `output/k4_country_holdout.json`. **Re-run on real data before freezing any cutoff (H4).**

---

## 5. Error analysis (K5 — buckets, worst entities, actions)

Harness: `scripts/error_analysis_demo.py` — trains LightGBM on 80% of S1 entities, scores only the **held-out 20%** (no train/test entity leakage), simulates name-similarity blocking (top-K candidates), runs the repo's real decision layer (`select_sets_expected_f05`), then invokes `evaluate.py --pair-scores --threshold --feature-evidence`.

### Bucket counts (96 held-out entities, seed 42)

| Simulated blocking | macro F_0.5 | oracle F_0.5 | candidate recall (micro) | retrieval | matching | decision-policy | integrity |
|---|---|---|---|---|---|---|---|
| top-15 candidates | 0.4378 (CI [0.368, 0.510]) | 0.5279 | 0.259 | **257** | 31 | 4 | 0 |
| top-60 candidates | 0.6605 (CI [0.607, 0.707]) | 0.9519 | 0.839 | **56** | **164** | 16 | 0 |

Reading: as retrieval improves, errors **shift from blocking to features/model** — exactly the diagnostic the buckets exist for. The demo's simulated blocking is deliberately simple (name-only top-K); real `src/blocking.py` (Vishwesh) should be re-measured the same way, not judged from these numbers.

### Error-bucket → owner → action

| Bucket | Definition | Owner | Action |
|---|---|---|---|
| retrieval (257 / 56) | true match never in candidates | blocking (Vishwesh) | add address/PIN/TF-IDF legs; measure `candidate_recall_micro` + oracle F_0.5 first |
| matching (31 / 164) | retrieved, scored on the wrong side of cutoff | features/model (Abhijit + A1) | land blocker-evidence/IDF features; hard-negative mining keeps same-name cases |
| decision-policy (4 / 16) | score fine, exclusivity/expected-F_0.5 overrode it | decision layer (A3) | calibrate (isotonic) before thresholding; audit `p_min` |
| integrity (0 / 0) | predicted id ∉ candidates | pipeline | **currently clean — keep it an assertion in CI** |

### Worst entities (top-20, feature evidence attached)

`evaluate.py --top-k 20 --feature-evidence` output (in `output/k5_demo/evaluation.json`): dominated by `f05=0.0000` entities with `|T|=3–7` and `retrieved_true=0` — i.e. blocking misses where *no* true match entered the top-K. Sample diagnosis:

```
S1-100173 US f05=0.0000 |T|=7 |C|=15 retrieved_true=0 cat=retrieval_error
    missed=[S2-500324 … S3-700353] pred=[]
    likely cause: blocking never surfaced the true match (blocking miss)
```

Once blocking recall is high (top-60 run), the worst list shifts to `matching_error` entities where predictions contain wrong same-name records (e.g. `S1-100288`: `pred=['S2-500780', 'S2-500800', 'S3-700396']` vs missed `['S2-500537', 'S2-500538']`) — the address/contradiction features are what must break those ties.

Singleton accuracy: 1.0 (top-15) / 0.8 (top-60) on 5 singletons; false-merge rate on single-match entities 1–2 of 3 — small-sample, watch on real data.

---

## 6. Blockers

| Blocker | Blocks | Needed from |
|---|---|---|
| **H4 — real dataset** (`data/` symlink dangling) | real K4/K5 numbers, France evaluation, threshold freeze | teammates / dataset mount |
| **A1 — blocker-evidence, competition, IDF features + sentinels** | 4 feature families (32 features) asserted by `tests/test_features.py` (currently explicit SKIPs) | Abhijit |
| `tests/fixtures/*.tsv` gitignored (`*.tsv`) | `tests/test_smoke.py` locally | generate locally, never commit |

---

## 7. Reproduce (all seed 42)

```bash
# K2 — generator + distribution checks
python scripts/make_synthetic_data.py --out tests/fixtures_synth --n-s1 600 --seed 42
python tests/test_synthetic_data.py

# K3 — feature semantics, class separation, dud scan (5 SKIPs until A1)
python tests/test_features.py

# K1 — harness tests
python tests/test_evaluate.py

# K4 — country holdout (France proxy) + JSON
python scripts/country_holdout.py --n-s1 600 --n-trials 10 --json-out output/k4_country_holdout.json

# K5 — error buckets + top-20 worst entities (sensitivity: 15 vs 60)
python scripts/error_analysis_demo.py --n-s1 600 --out output/k5_demo
python scripts/error_analysis_demo.py --n-s1 600 --top-cands 60 --out output/k5_demo_t60

# everything at once
python -m pytest tests/test_evaluate.py tests/test_synthetic_data.py \
                 tests/test_features.py tests/test_country_holdout.py -q
```

When the real dataset lands: replace the generator with `--data-dir <real>` in K4/K5 (both scripts accept it) and freeze thresholds only after the real country-holdout re-run.
