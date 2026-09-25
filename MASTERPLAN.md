# 🎯 MASTERPLAN — Amazon ML Challenge 2026

## Business Entity Resolution · 72-Hour Hackathon

**Team repo:** https://github.com/AbhijitK20/Business_entity_resolution
**Competition window:** 25–27 September 2026 (72 hours)
**Date created:** 25 Sep 2026

---

## 1. THE PROBLEM (one paragraph)

A business signs up on Amazon Business — we capture its **name** and **address**. The same real-world business is described differently by 3 independent data sources (different vendors, formats, conventions), with **no shared identifier**. Given **Source 1 as the clean reference list**, our job is to find **every matching record in Source 2 and Source 3** for each Source 1 entity. One S1 entity may match **zero, one, or many** records.

```
SOURCE 1 (reference)          SOURCE 2 (noisy)                    SOURCE 3 (noisy)
┌─────────────────────┐      ┌──────────────────────────┐        ┌──────────────────────────┐
│ S1-732914           │      │ S2-118820 Acme Robotics  │  ✓     │ S3-905477 Acme Robotics  │ ✓
│ Acme Robotics Inc.  │─────▶│ S2-540221 Acme Robotix   │  ✗     │ S3-063118 Acme Bakery    │ ✗
│ 500 Market St, SJ   │      │            (look-alike)   │        │         (shares address) │
└─────────────────────┘      └──────────────────────────┘        └──────────────────────────┘
```

---

## 2. DATA (confirmed from official video)

### Fields (per record)
| Column | Example | Notes |
|--------|---------|-------|
| `entity_id` | `S1-732914`, `S2-118820`, `S3-905477` | Prefix encodes the source |
| `business_name` | `Acme Robotics Inc` | Noisy: abbreviations, typos, transpositions |
| `business_address` | `500 Market St, San Jose` | Noisy: abbreviations, landmarks, missing parts |
| `country` | `US`, `India`, `France` | Open set — France appears **only in test** |

### Files
```
dataset/
├── train/
│   ├── train_source1.tsv           # S1 reference records
│   ├── train_source2.tsv           # S2 records
│   ├── train_source3.tsv           # S3 records
│   └── train_ground_truth.tsv      # source1_entity_id → comma-separated matches
└── test/
    ├── test_source1.tsv            # predict matches for EVERY entity here
    ├── test_source2.tsv
    └── test_source3.tsv
```

**⚠️ Read TSVs with `sep="\t"` — addresses and ID lists contain commas.**

### Noise patterns to expect
- **Names:** `Corp`↔`Corporation`, `Pvt`↔`Private`, `Ltd`↔`Limited`, DBA names, `&`↔`and`, word-order swaps, typos, transliterations
- **Addresses:** `Rd`↔`Road`, `St`↔`Street`, missing PIN/state, landmark refs (`Near SBI ATM`), municipal numbering, component reordering
- **Region:** US / India / France patterns differ (suffixes like SARL/SAS, address formats)

---

## 3. EVALUATION — Macro F_0.5 (precision-weighted)

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

- **Macro-averaged per Source 1 entity**, then averaged across ALL entities.
- **Singletons included:** correct empty prediction = **1.0**; any match predicted = **0.0**.
- **A false merge costs ~2× a missed match** → *when in doubt, do not merge.*

### Worked example
| Predicted | Truth | P | R | F_0.5 |
|-----------|-------|---|---|-------|
| [S2-47, S2-93, S3-12] | [S2-47, S3-12] | 2/3 | 1.0 | 0.714 |

---

## 4. WHAT WE SUBMIT

### A. During the hackathon (leaderboard)
`matching_results.tsv` — one row per Source 1 entity:
```tsv
source1_entity_id	matched_entity_ids
S1-732914	S2-118820,S3-905477
S1-889301	S2-397155,S3-651230
S1-205774	
```
Rules: every S1 present · no duplicates in a list · only S2/S3 IDs from the test set · empty = singleton.

### B. Final package (audited)
```
<team>_submission.zip
├── output/
│   ├── matching_results.tsv      # final matches
│   └── candidate_pairs.tsv       # blocking candidates (audited; NOT scored)
├── code/business_entity_resolution/
│   ├── src/                      # runnable pipeline
│   ├── README.md
│   └── requirements.txt
└── Documentation_template.md     # 1–2 page methodology
```

**Every match in `matching_results.tsv` MUST appear in `candidate_pairs.tsv`** (pipeline sanity rule).

---

## 5. OUR ARCHITECTURE

```
                    ┌──────────────────────────────────────────────┐
 Raw TSVs ─────────▶│ 1. NORMALIZE                                 │
                    │    Unicode->ASCII, lowercase, legal suffixes │
                    │    (Inc/Corp/Pvt/SARL/...), abbreviations,   │
                    │    punctuation, collapse whitespace          │
                    └───────────────────┬──────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │ 2. BLOCKING  (recall ceiling!)               │
                    │    7-layer union of candidate generators:    │
                    │    name TF-IDF · token-sorted · phonetic     │
                    │    (Soundex/Metaphone) · initialism ·        │
                    │    address TF-IDF · country partition ·      │
                    │    MinHash LSH                               │
                    │    → candidate_pairs.tsv                     │
                    └───────────────────┬──────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │ 3. PAIR FEATURES (25)                        │
                    │    name (10) · address (6) · country (1) ·   │
                    │    cross (8, incl. phonetic vote, trigrams)  │
                    └───────────────────┬──────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │ 4. TRAIN (leak-free stacking)                │
                    │    positives = ground truth                  │
                    │    negatives = 70% hard + 30% random         │
                    │    base models: LightGBM + XGBoost + RF      │
                    │    → OOF preds → shallow LightGBM meta       │
                    └───────────────────┬──────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │ 5. THRESHOLD (macro F_0.5 on validation)     │
                    │    scan 0.05→0.95 step 0.01                  │
                    └───────────────────┬──────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │ 6. INFERENCE → matching_results.tsv          │
                    │    + candidate_pairs.tsv → validate → upload │
                    └──────────────────────────────────────────────┘
```

### Critical design principles
1. **Blocking sets the recall ceiling** — a true match you never consider can never be recovered. Target ≥ 95% recall on train.
2. **Precision > recall** — tune the threshold for F_0.5, not F_1. When unsure, predict empty.
3. **Singletons matter** — 30%+ of entities likely have no match; predicting them right is as valuable as finding matches.
4. **No external data** — no APIs, no geocoding, no databases. Disqualification offense.
5. **Leak-free stacking** — meta-learner trains ONLY on out-of-fold predictions.

---

## 6. TEAM DIVISION (3 people, parallel streams)

| Stream | Owner | Deliverable | Can start |
|--------|-------|-------------|-----------|
| **A · Blocking** | Member 2 | `src/normalize.py`, `src/blocking.py` + recall report | Immediately |
| **B · Features** | Member 3 | `src/features.py` (25 features) + validation | Immediately |
| **C · Pipeline + Model** | Lead (you) | `src/pipeline.py`, `src/model.py`, submissions | Immediately |
| **D · Evaluation & Docs** | Shared rotation | `scripts/evaluate.py`, methodology doc, package | Hour 12+ |

### Ownership boundaries (avoid merge conflicts!)
- Member 2 → only `src/normalize.py`, `src/blocking.py`, `tests/test_blocking.py`
- Member 3 → only `src/features.py`, `tests/test_features.py`
- Lead → everything else + final integrations
- **Shared interfaces are frozen** (see §7) — change them only by agreement.

---

## 7. FROZEN INTERFACES (do not break)

```python
# normalize.py
normalize_name(name: str) -> str
normalize_address(addr: str) -> str
apply_normalization(df: pd.DataFrame) -> pd.DataFrame   # adds *_clean columns

# blocking.py
tfidf_blocking_candidates(q: list[str], t: list[str], ids: list[str],
                          threshold: float) -> dict[int, set[str]]
phonetic_blocking(...)   -> dict[int, set[str]]
initialism_blocking(...) -> dict[int, set[str]]
address_tfidf_candidates(...) -> dict[int, set[str]]
minhash_lsh_candidates(...) -> dict[int, set[str]]
union_candidates(*dicts) -> dict[int, set[str]]
measure_blocking_quality(cands, ground_truth, total_pairs, s1_ids) -> dict

# features.py
FEATURE_NAMES: list[str]          # EXACTLY 25, fixed order
compute_all_features(name_a, name_b, addr_a, addr_b, country_a, country_b) -> dict

# model.py
train_base_models(X_train, y_train, X_val, y_val, n_trials) -> dict
train_meta_learner(base_results, y_train, y_val, n_trials) -> (meta, oof, val)
find_best_macro_f05_threshold(y_true, y_proba, s1_ids) -> (thresh, score)
```

---

## 8. 72-HOUR TIMELINE

| Hours | Goal | Owner | Checkpoint |
|-------|------|-------|------------|
| **0–2** | Repo + env + dataset loaded | Lead | `git clone` works, venv ready |
| **2–6** | Normalize + block train data | M2 + Lead | Blocking recall ≥ 95% on train |
| **6–10** | Features + first training pairs | M3 + Lead | Feature matrix, no NaN |
| **10–14** | **First end-to-end submission** | Lead | Uploaded to leaderboard |
| **14–24** | Threshold tuning, error analysis | All | Validation F_0.5 ≥ 0.85 |
| **24–36** | Improve blocking recall + features | M2 + M3 | +2–5% score |
| **36–48** | Model tuning, ensemble variants | Lead | Best CV score recorded |
| **48–60** | Freeze model, run test inference | All | Output files generated |
| **60–68** | Validation script passes, docs written | All | `validate_submission.py` PASS |
| **68–72** | Final submission + zip package | Lead | Uploaded + zipped |

---

## 9. KNOWN ISSUES & FINDINGS (living section)

| # | Finding | Status | Owner |
|---|---------|--------|-------|
| 1 | Legal suffix list missed `Pvt`/`Private`/`SARL`/`SAS` — fixed | ✅ Fixed | Lead |
| 2 | `optuna.TPESampler` → `optuna.samplers.TPESampler` (v5 API) — fixed | ✅ Fixed | Lead |
| 3 | Meta-learner was trained on validation data (leakage) — fixed with OOF stacking | ✅ Fixed | Lead |
| 4 | `candidate_pairs.tsv` was writing final matches instead of blocking candidates — fixed | ✅ Fixed | Lead |
| 5 | Singletons had no negatives in training — fixed (singleton hard negatives added) | ✅ Fixed | Lead |
| 6 | **Address TF-IDF blocking generates too many false candidates** (342/484 on test) — needs tuning or gating by name similarity | 🔴 OPEN | M2 |
| 7 | Tiny-data CV guards added (`make_cv_splits`, `_safe_ap`) | ✅ Fixed | Lead |
| 8 | Threshold optimizes to lowest bound (0.10) on synthetic — needs investigation with real data | 🟡 OPEN | Lead |

### Open hypothesis for #6
Address-based blocking is high-recall but low-precision: many businesses share cities/streets.
**Proposed fix:** require *both* a name signal AND an address signal for address-derived candidates,
or raise the address TF-IDF threshold from 0.25 → 0.5.

---

## 10. REPO MAP

```
Amazon ML/
├── src/
│   ├── data_loader.py      # TSV loading, ground-truth parsing
│   ├── normalize.py        # name/address normalization  [M2]
│   ├── blocking.py         # 7-layer candidate generation [M2]
│   ├── features.py         # 25 pairwise features        [M3]
│   ├── training.py         # pairs + hard negatives
│   ├── model.py            # OOF stacking + thresholds
│   └── pipeline.py         # end-to-end orchestrator     [Lead]
├── scripts/
│   ├── make_synthetic_data.py  # generate noisy fixtures for testing
│   └── evaluate.py             # macro F_0.5 evaluator (leaderboard-style)
├── tests/
│   ├── fixtures/           # hand-built from video examples
│   ├── fixtures_synth/     # 300-entity synthetic dataset
│   └── test_smoke.py       # end-to-end smoke test
├── utils/
│   └── validate_submission.py  # format validator (all rules)
├── docs/
│   ├── problem_statement.pdf
│   ├── video_transcript.md
│   ├── IDEAS.md
│   └── ULTIMATE_STRATEGY.md
├── PRD.md · TASK_BREAKDOWN.md · MASTERPLAN.md (this file)
└── requirements.txt
```

---

## 11. COMMANDS CHEATSHEET

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
uv pip install -r requirements.txt

# Generate synthetic data for testing (no real data needed)
python scripts/make_synthetic_data.py --out tests/fixtures_synth --n-s1 300

# Smoke test (fixtures)
python tests/test_smoke.py

# Run pipeline (fast mode for testing)
python -c "from src.pipeline import EntityResolutionPipeline as P; P('tests/fixtures_synth','output',fast_mode=True).run()"

# Run pipeline (production)
python -m src.pipeline --data-dir data --output-dir output

# Evaluate locally (leaderboard-style)
python scripts/evaluate.py --predictions output/matching_results.tsv \
                           --ground-truth data/dataset/train/train_ground_truth.tsv

# Validate before submitting
python utils/validate_submission.py --matching output/matching_results.tsv \
                                    --candidate output/candidate_pairs.tsv \
                                    --test-dir data/dataset/test
```

---

## 12. SUCCESS CRITERIA

| Milestone | Target | Hard requirement |
|-----------|--------|------------------|
| Working submission | Within 14 hours | ✅ |
| Blocking recall (train) | ≥ 95% | ✅ |
| Validation macro F_0.5 | ≥ 0.85 | ⭐ |
| `validate_submission.py` | PASS | ✅ MUST |
| No external data used | 100% compliant | ✅ MUST |
| Methodology doc | 1–2 pages | ✅ MUST |
| ZIP package | Complete structure | ✅ MUST |

---

## 13. RISK REGISTER

| Risk | Impact | Mitigation |
|------|--------|------------|
| Dataset not yet downloaded | Blocker | Lead to fetch from portal first thing |
| Blocking miss on France (unseen in train) | Recall loss | Multilingual-proof normalization; no country hardcoding |
| Threshold tuned on wrong distribution | Score loss | Tune on entity-level macro F_0.5, validate on test-like split |
| Submission format error | **Rejection** | Always run `validate_submission.py` |
| Team merge conflicts | Time loss | Frozen interfaces + file ownership |
| Time runs out | Partial package | Prioritize: submission file → validation → docs |

---

## 14. FIRST ACTIONS (right now)

1. **Lead:** obtain the dataset from the competition portal → `data/dataset/`
2. **M2:** tune address-blocking precision (issue #6) — gate address candidates by name similarity
3. **M3:** verify features on real names/addresses from the dataset; report any NaN or constant columns
4. **All:** run `python tests/test_smoke.py` to confirm local setup works before touching real data

---

*This masterplan is a living document. Update §9 (Known Issues) and §8 (Timeline) as the hackathon progresses.*
