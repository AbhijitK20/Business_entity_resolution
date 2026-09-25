# 📍 PROJECT STATE — Living Handoff Document

**Last updated:** 25 Sep 2026 (afternoon) · **Updated by:** Abhijit
**Purpose:** One-page snapshot of where we are, what's verified, and what's next. Update at the end of every work session.

---

## 1. Where We Are

| Phase | State |
|-------|-------|
| Problem understood (PDF + video + guidelines) | ✅ Done |
| Competitive intel (13 team repos + 26 research repos) | ✅ Done |
| Code foundation (modules, tests, validator, synthetic data) | ✅ Done |
| Implementation blueprint (code-level specs) | ✅ Done |
| **Dataset** | ⏳ Downloading — not yet extracted to `data/dataset/` |
| Real-data work (scale benchmark, training, submission) | 🔴 Blocked on dataset |

---

## 2. What Is Built and Verified (evidence attached)

| Component | File | Verified by | Result |
|-----------|------|-------------|--------|
| Normalization (10-step + legal suffixes) | `src/normalize.py` | `tests/test_smoke.py` | ✅ `Pvt`→stripped, SARL/SAS handled |
| 35 pairwise features (name/addr/country/cross/missingness/contradiction) | `src/features.py` | `tests/test_smoke.py` | ✅ 35 features, no NaN |
| Scale-safe blocking (chunked sparse + top-K caps) | `src/blocking.py` | synthetic benchmark | ✅ 99.2% recall @ 17 cand/S1 (vs 100% @ uncapped 4910) |
| Training pairs + hard negatives + singleton negatives | `src/training.py` | smoke test | ✅ 19 pairs from 6-entity fixture |
| Leak-free OOF stacking (LGB+XGB+RF → meta) | `src/model.py` | synthetic run | ✅ meta OOF AP 0.9849 / val AP 0.9648 |
| Macro F_0.5 threshold | `src/model.py` | official worked example | ✅ matches 0.714 exactly |
| Pipeline orchestrator | `src/pipeline.py` | synthetic run | ✅ end-to-end, both TSVs written |
| Evaluator (leaderboard-style) | `scripts/evaluate.py` | hand calculation | ✅ 0.9259 matches manual |
| Official validator | `utils/validate_submission.py` | synthetic outputs | ✅ PASS |
| Synthetic data generator | `scripts/make_synthetic_data.py` | 300-entity run | ✅ 240 train / 60 test |
| Smoke test | `tests/test_smoke.py` | every commit | ✅ PASS |

---

## 3. What We Know About the Real Data (verified by 2 other teams)

| Fact | Value | Source |
|------|-------|--------|
| Train S1/S2/S3 | 2.21M / 5.03M / 5.29M | RF_AMAZON_2026 audit + resolvers |
| Test S1/S2/S3 | 1.73M / 4.89M / 5.08M | same |
| France in test | 259,452 S1 (~15%) | same |
| Ground truth | 7,638,365 positive pairs | same |
| Singletons | 5.58% (123,247) | same |
| Multi-match roots | 89.0% (mean 3.46, max 11) | same |
| Every S2/S3 matches ≤1 S1 | verified, zero exceptions | same |
| True matches with same normalized name | only ~20% | labeled sample |
| India cross-script pairs | 22.7% | labeled sample |
| Shared S1 names | 47% | SABER |
| Blank addresses | ~3% of S2/S3 only | resolvers |
| All-empty baseline | 0.0558 | RF audit |
| Test cross-product | 17.27 trillion pairs | RF audit |

Full detail: [COMPETITIVE_INTEL.md](COMPETITIVE_INTEL.md).

---

## 4. What We Know We Must Fix (6 critical gaps)

| # | Gap | Fix spec | Owner | Task |
|---|-----|----------|-------|------|
| 1 | No Indic transliteration (22.7% India pairs cross-script) | BLUEPRINT §2.1 | Vishwesh | V1 |
| 2 | No candidate caps / budget curve | BLUEPRINT §2.2 | Vishwesh | V4 |
| 3 | No one-to-one exclusivity | BLUEPRINT §2.4 | Abhijit | A3 |
| 4 | No expected-F0.5 set selection | BLUEPRINT §2.4 | Abhijit | A3 |
| 5 | No scale engineering (row-wise loops) | BLUEPRINT §2.5 | Abhijit | A1 |
| 6 | Blocker similarity + ranks discarded | BLUEPRINT §2.3 | Vishwesh+Abhijit | V2/A1 |

---

## 5. Next Actions (in order)

### The moment the dataset extracts to `data/dataset/`
1. **Abhijit:** profile the real files (`wc -l`, null counts, country distribution) → `docs/data_profile.md`
2. **Abhijit:** 50K-root scale benchmark (blocking time, candidates/S1, feature throughput, peak RAM)
3. **All:** project full-run time from measured rates; decide candidate budget

### Parallel (no dataset needed)
4. **Vishwesh:** V1 Indic transliteration + tests
5. **Vishwesh:** V2 adaptive-K + bidirectional + key legs
6. **Karan:** K1 oracle/bucket evaluator
7. **Karan:** K2 synthetic generator → real distribution
8. **Abhijit:** A1 vectorized features + Parquet
9. **Abhijit:** A3 decision layer (calibration + exclusivity + expected-F0.5)

### Then
10. Train → validate → upload (max 5 submissions/day, log every one)
11. Package + methodology doc + clean-room reproduction

---

## 6. Risks We Are Watching

| Risk | Mitigation | Status |
|------|-----------|--------|
| Dataset arrives late | All non-data tasks ready to start now | 🟡 |
| Full run doesn't finish in time | Scale benchmark first; sample if needed | 🟡 |
| France generalization | Country-holdout proxy (K4) + language-agnostic features | 🟡 |
| Submission format error | Official validator `--check-ids` before every upload | ✅ |
| Merge conflicts under time pressure | File ownership + PR-only workflow | ✅ CONTRIBUTING.md |
| Someone pushes broken code to main | **Branch protection enabled** | ✅ |

---

## 7. Key Decisions Log (so we don't relitigate)

| # | Decision | Why |
|---|----------|-----|
| D1 | LightGBM-family matcher (not deep learning first) | All top teams; fast, MIT, interpretable |
| D2 | Leak-free OOF stacking for meta-learner | Prevents the 0.97→0.24 val/test gap we measured |
| D3 | Macro F_0.5 optimized per S1 entity, not pair-level | Matches leaderboard exactly |
| D4 | Country never a hard filter (open-set) | Official requirement; France unseen |
| D5 | `candidate_pairs.tsv` = exact model input, same run as matches | Official requirement + audit |
| D6 | Dense encoder only if lexical blocking ceiling insufficient | SABER: lexical+keys already reach 99%+ |
| D7 | No external data/APIs — zero outbound calls | Disqualification rule |
| D8 | PR-only workflow on protected `main` | Team coordination under time pressure |

---

## 8. Open Questions

| # | Question | Who decides | When |
|---|----------|-------------|------|
| Q1 | Candidate budget K per source (10/20/30/50)? | Vishwesh (from recall curve) | After benchmark |
| Q2 | Dense retrieval leg needed? | Abhijit + Vishwesh (from recall gap) | After full blocking |
| Q3 | France cutoff margin? | Karan (from country-holdout gap) | After K4 |
| Q4 | When to freeze the holdout? | Abhijit | Before final threshold |
