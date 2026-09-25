# 📋 TASK BREAKDOWN — Team Business Entity Resolution

**Team:** Abhijit · Vishwesh · Karan
**Event:** Amazon ML Challenge 2026 · 72-hour hackathon (25–27 Sept 2026)
**Repo:** https://github.com/AbhijitK20/Business_entity_resolution

> **Reading order:** [MASTERPLAN.md](MASTERPLAN.md) → [docs/IMPLEMENTATION_BLUEPRINT.md](docs/IMPLEMENTATION_BLUEPRINT.md) → this file.

---

## 👥 Team Roster & Roles

| Member | Role | Owns | Primary files |
|--------|------|------|---------------|
| **Abhijit** | Lead — pipeline, model, decision, submissions | Integration, vectorized features, model training, decision layer, final package | `src/pipeline.py`, `src/model.py`, `src/training.py`, `src/decision.py`, `src/data_loader.py` |
| **Vishwesh** | Blocking engineer | Normalization + candidate generation, blocking recall | `src/normalize.py`, `src/blocking.py`, `tests/test_blocking.py`, `tests/test_normalize.py` |
| **Karan** | Feature/eval engineer | Evaluation tooling, synthetic data, error analysis, docs | `scripts/evaluate.py`, `scripts/make_synthetic_data.py`, `tests/test_features.py`, `docs/feature_report.md` |

### File ownership rules
- Only the owner edits their files. Others request changes via PR/issues.
- **Frozen interfaces** (MASTERPLAN §7) change only by team agreement.
- Every PR: 1 teammate approval + `python tests/test_smoke.py` passes.

---

## 🚀 Workstreams at a Glance

```
        NOW                    HOUR +8               HOUR +16             HOUR +24              HOUR +36            HOUR +48
        ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
ABHIJIT vectorize+Parquet ██ negatives+OOF ██████ calibration+exclusivity ██████ scale benchmark ████ full run+upload ████ package
VISHWESH Indic translit ████ adaptive-K+bidir █████ region partition ████████ budget curve ████████ tune+freeze ██████ docs
KARAN   oracle+buckets █████ synth real-dist ██████ feature validation ██████ country-holdout ██████ error analysis ████ docs
```

---

## 🔵 VISHWESH — Blocking Stream

### V1 · Indic → Latin transliteration *(highest recall impact)*
**Files:** `src/normalize.py` · **Spec:** BLUEPRINT §2.1
- [ ] Add `indic-transliteration` dependency (`uv pip install indic-transliteration`)
- [ ] Implement the 9-script-block transliteration (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam) with ITRANS + word-final schwa deletion
- [ ] Add ligature map (œ→oe, æ→ae, ß→ss, ø, ł, đ) for France
- [ ] Keep RAW + normalized strings side by side (contradiction detection needs raw)
- [ ] Add `is_non_latin` script flag
- [ ] Unit tests: `राम मार्केटिंग` → `ram marketing`; `Café de la Paix SARL` → `cafe de la paix`; Latin passthrough unchanged

**Evidence to record:** before/after normalization samples from real data; count of records whose normalized form changed.

### V2 · Adaptive-K + bidirectional + key legs
**Files:** `src/blocking.py` · **Spec:** BLUEPRINT §2.2
- [ ] Implement `prune(idx, sc, kmin, kmax, gap)` — `rank < kmin OR score ≥ top1 − gap, up to kmax`
- [ ] Defaults: forward `kmin=5, kmax=30, gap=0.10`; reverse `kmin=2, kmax=5, gap=0.05`
- [ ] Run every leg **both directions** (S1→gallery forward, gallery→S1 reverse)
- [ ] Key legs: address key (exact normalized ≥12 chars), name key (core name + last 2 addr tokens; drop buckets >30)
- [ ] Measure per-leg marginal recall (which legs actually add pairs)

**Evidence to record:** per-leg contribution table; union recall with/without reverse legs.

### V3 · Region partitioning (multi-membership)
**Files:** `src/blocking.py` · **Spec:** BLUEPRINT §2.2
- [ ] Vocabulary: normalized comma-parts appearing in ≥0.05% of that country's S1 records (unsupervised → works for France)
- [ ] Multi-membership: part, first/last token, first/last two tokens
- [ ] Compare if regions intersect OR either unknown (unknown → whole country)
- [ ] **Dead end to avoid:** one-region-per-record (last address part) loses ~4% of pairs, 92% in India

**Evidence to record:** % of pairs retained vs % of comparisons, per country.

### V4 · Candidate budget curve + freeze
**Files:** `src/blocking.py` · **Spec:** BLUEPRINT §2.2
- [ ] Sweep K (candidates/S1/source): 10/20/30/50 + uncapped; measure pair recall at each
- [ ] Pick the smallest K that doesn't sacrifice recall (target ≥99% @ ~20–30)
- [ ] Freeze blocking; write blocking section of methodology doc

---

## 🟢 KARAN — Feature/Eval Stream

### K1 · Oracle ceiling + country/bucket evaluation
**Files:** `scripts/evaluate.py` · **Spec:** BLUEPRINT §2.6–2.7
- [ ] Add candidate-oracle F0.5: `Oracle_i = 1 if t_i==0 else 5·r_i/(4·r_i + t_i)`, `r_i = |C_i ∩ T_i|`
- [ ] Add per-country breakdown (US/India/France-proxy)
- [ ] Add match-count buckets: `{0, 1, 2, 3–4, 5+}`
- [ ] Add complete-match coverage + reduction ratio + singleton false-merge rate
- [ ] Bootstrap **business groups** (not pairs) for confidence intervals

### K2 · Synthetic generator → real distribution
**Files:** `scripts/make_synthetic_data.py` · **Spec:** COMPETITIVE_INTEL §1
- [ ] Match real distribution: **89% multi-match, 5.6% singleton**, mean 3.46, max 11
- [ ] Add cross-script names (Indic script variants) in S2/S3
- [ ] Blank addresses on gallery side only (~3%)
- [ ] Name collisions (47% of S1 share names) + same-address/different-business distractors
- [ ] Country mix: US 60%/India 40% train; test US 38%/India 47%/France 15%

### K3 · Blocker-evidence feature validation
**Files:** `tests/test_features.py` · **Spec:** BLUEPRINT §2.3
- [ ] Unit tests for each new feature family (leg evidence, competition, IDF/record)
- [ ] Verify class separation: matches should score higher than negatives (compare medians)
- [ ] Check `-1` sentinel handling (`num_jacc`, `house_eq`, `region_overlap`)
- [ ] Report dud/constant features

### K4 · Country-holdout stress test (France proxy)
**Files:** `scripts/evaluate.py` or new `scripts/country_holdout.py`
- [ ] Train US → eval India; train India → eval US
- [ ] Report the transfer gap (how much precision drops on unseen country)
- [ ] Recommend France cutoff margin from the gap

### K5 · Error analysis + docs
- [ ] Bucket errors: retrieval vs matching vs decision-policy (BLUEPRINT §2.6)
- [ ] Top-20 worst entities with diagnosis
- [ ] Feature-engineering section of methodology doc

---

## 🟡 ABHIJIT — Pipeline & Model Stream

### A1 · Vectorized features + Parquet
**Files:** `src/features.py`, `src/pipeline.py` · **Spec:** BLUEPRINT §2.5
- [ ] Replace row-wise loop with `rapidfuzz.process.cpdist(workers=-1)` batch computation
- [ ] Chunk pairs ≤4M; `del + gc.collect()` after each chunk
- [ ] Store intermediates as Parquet; stream final TSV writers
- [ ] Never materialize >2 GB; never dense (query × gallery)

### A2 · Negatives from blocking candidates + OOF
**Files:** `src/training.py` · **Spec:** BLUEPRINT §2.4
- [ ] Sample training negatives **from the actual blocking candidates** (train = inference distribution)
- [ ] Hard negatives: same-name/different-address, same-address/different-name, unit differences
- [ ] Sample by S1 root first (avoid many-candidate entities dominating)
- [ ] Keep all known positives; exclude all of an anchor's positives from its negatives
- [ ] OOF training to keep the holdout clean

### A3 · Decision layer: calibration + exclusivity + expected-F0.5
**Files:** `src/decision.py` (new) · **Spec:** BLUEPRINT §2.4
- [ ] Isotonic calibration on a held-out calibration fold
- [ ] One-to-one exclusivity: highest-p S1 owns each candidate (tie-break s1_id asc), `p ≥ 0.05`
- [ ] Expected-F0.5 prefix selection: `expected_f(k) = 1.25·Σp / (0.25·E|T| + k)`; empty score `1 − max_p`
- [ ] Never multiply dependent edge probabilities
- [ ] Emit empty set when it wins

**Evidence:** measured macro F0.5 vs plain-threshold baseline (expect +0.001–0.002).

### A4 · Scale benchmark + full run
- [ ] Smoke test on 2–5K roots → benchmark on ~50K representative roots (all countries, missing-address cases, long records)
- [ ] Measure: ingestion rows/s, blocking time, candidates/S1, features/s, peak RAM, export time
- [ ] Project full-run time from measured rates (no guesses)
- [ ] Run full train → full test inference

### A5 · Validate + upload
- [ ] `python utils/validate_submission.py --matching ... --candidate ... --test-dir ... --check-ids` → PASS
- [ ] Upload `matching_results.tsv` (max 5/day — log every submission with score + file hash in `docs/leaderboard_log.md`)
- [ ] Compare public score vs local holdout → investigate any gap (likely France)

### A6 · Final package
- [ ] Fill `Documentation_template.md` (official template, already in repo root)
- [ ] Build zip: `output/` + `code/business_entity_resolution/{src,README.md,requirements.txt}` + filled template
- [ ] Verify clean-room reproduction (fresh venv, `pip install -r requirements.txt`, run pipeline)
- [ ] `MODEL_LICENSE_AUDIT.md` for every shipped model (LightGBM MIT, XGBoost Apache-2.0, sklearn BSD)

---

## 🔄 Handoff Points

| Handoff | From → To | Trigger |
|---------|-----------|---------|
| H1 | Abhijit → All | Dataset extracted to `data/dataset/{train,test}/` |
| H2 | Vishwesh → Abhijit | Indic transliteration + blocking frozen (recall ≥99% @ budget) |
| H3 | Karan → Abhijit | Oracle evaluator ready → training can be measured properly |
| H4 | Abhijit → Karan | First end-to-end run → error analysis begins |
| H5 | Karan → Vishwesh | Error analysis shows blocking misses → blocking push |
| H6 | All → Abhijit | Final freeze → package assembly |

---

## 📡 Communication Protocol

### Async updates (every ~6 hours, group chat)
```
✅ Done: <task> — <evidence/result>
🔨 Doing: <task> — <ETA>
🚧 Blocked: <task> — <what I need>
```

### Branching
```
main                  ← always working, always valid
 ├── feat/blocking-*  ← Vishwesh
 ├── feat/features-*  ← Karan
 └── feat/model-*     ← Abhijit
```
PR → 1 approval → merge. No direct pushes to `main` after hour 14 (except Abhijit for submissions).

### Escalation rule
If blocked > 30 minutes: post in chat immediately. Time is the scarcest resource.

---

## ✅ Definition of Done (per task)

A task is DONE only when:
1. Code committed with a clear message
2. Tests pass or manual evidence attached
3. **Measurable result recorded** (recall %, F_0.5, candidate reduction, timing — whatever the task promises)
4. Owner marked it `✅ Done` in the update protocol

**"Works on my machine" is not done. "Evidence attached" is done.**

---

## 🎯 Team Goals (priority order)

1. **Working submission on leaderboard** — within 14 hours of data landing
2. **Blocking ≥99% pair recall @ ≤30 candidates/S1/source** — the ceiling
3. **Validation macro F_0.5 ≥ 0.85** — stretch: ≥0.90
4. **Oracle ceiling vs achieved reported separately** — no hiding losses
5. **Complete package + methodology doc + validator PASS** — by hour 68
6. **Top 100 (PPI interviews)** — target 🏆

---

## 📌 Current Status Board

| Task | Owner | Status | Evidence |
|------|-------|--------|----------|
| **V1** Indic transliteration | Vishwesh | ⬜ Ready (spec written) | — |
| **V2** Adaptive-K + bidirectional + keys | Vishwesh | ⬜ Ready | adaptive-K spec in BLUEPRINT §2.2 |
| **V3** Region partitioning | Vishwesh | ⬜ Ready | — |
| **V4** Candidate budget curve | Vishwesh | ⬜ Blocked on V2/V3 | — |
| **K1** Oracle + bucket evaluator | Karan | ⬜ Ready | oracle formula in BLUEPRINT §2.6 |
| **K2** Synthetic → real distribution | Karan | ⬜ Ready | real numbers in COMPETITIVE_INTEL §1 |
| **K3** Feature validation | Karan | ⬜ Blocked on A1 | — |
| **K4** Country-holdout test | Karan | ⬜ Blocked on H4 | — |
| **K5** Error analysis + docs | Karan | ⬜ Blocked on H4 | — |
| **A1** Vectorized features + Parquet | Abhijit | ⬜ Ready | — |
| **A2** Negatives from blocking + OOF | Abhijit | ⬜ Blocked on A1 | — |
| **A3** Decision layer (calibrate+excl+expected-F0.5) | Abhijit | ⬜ Ready (spec written) | — |
| **A4** Scale benchmark + full run | Abhijit | ⬜ Blocked on dataset | — |
| **A5** Validate + upload | Abhijit | ⬜ Blocked on A4 | official validator ✅ |
| **A6** Final package | Abhijit | ⬜ Blocked on A5 | official template ✅ |

### Completed foundation (before this board)
| Item | Status | Evidence |
|------|--------|----------|
| Repo + venv + dependencies | ✅ | `44525e1`…`deb07de` pushed |
| 35 pairwise features (incl. missingness + contradiction) | ✅ | `tests/test_smoke.py` PASS (35 features) |
| Scale-safe blocking (chunked sparse + top-K caps) | ✅ | measured 99.2% recall @ 17 cand/S1 on synthetic |
| Leak-free OOF stacking | ✅ | meta OOF AP 0.9849 / val AP 0.9648 (synthetic) |
| Macro F_0.5 threshold + evaluator | ✅ | verified against official worked example |
| Official validator integrated | ✅ | PASS on synthetic outputs |
| Synthetic generator + smoke test | ✅ | `tests/fixtures_synth` (300 entities) |
| Competitive intel + implementation blueprint | ✅ | `docs/COMPETITIVE_INTEL.md`, `docs/IMPLEMENTATION_BLUEPRINT.md` |
