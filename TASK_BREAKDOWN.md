# 📋 TASK BREAKDOWN — Team Business Entity Resolution

**Team:** Abhijit · Vishwesh · Karan
**Event:** Amazon ML Challenge 2026 · 72-hour hackathon (25–27 Sept 2026)
**Repo:** https://github.com/AbhijitK20/Business_entity_resolution

> Read [MASTERPLAN.md](MASTERPLAN.md) first — it's the source of truth for the problem, data, architecture, and frozen interfaces.

---

## 👥 Team Roster & Roles

| Member | Role | Owns | Primary files |
|--------|------|------|---------------|
| **Abhijit** | Lead / Pipeline & Model | Integration, ensemble training, threshold optimization, submissions, final package | `src/pipeline.py`, `src/model.py`, `src/training.py`, `src/data_loader.py` |
| **Vishwesh** | Blocking Engineer | Normalization + candidate generation, blocking recall | `src/normalize.py`, `src/blocking.py`, `tests/test_blocking.py` |
| **Karan** | Feature Engineer | Pairwise features, feature validation, evaluation tooling | `src/features.py`, `tests/test_features.py`, `scripts/evaluate.py` |

**Shared responsibilities (rotate):** documentation, error analysis, packaging, leaderboard submissions.

### File ownership rules
- Only the owner edits their files. Others request changes via PR/issues.
- **Frozen interfaces** (MASTERPLAN §7) may only change by team agreement.
- Every PR: 1 teammate approval + `python tests/test_smoke.py` passes.

---

## 🚀 Workstreams at a Glance

```
        HOUR 0─6            HOUR 6─14           HOUR 14─36          HOUR 36─60          HOUR 60─72
        ─────────────────────────────────────────────────────────────────────────────────────────
ABHIJIT ██ setup+data ████ pipeline run ████ first submit ████████ model tuning ████ package+upload
VISHWESH██ normalize  ████ block+recall ████ tune addr ██████████ recall push  ████ freeze+docs
KARAN   ██ features   ████ validate    ████ eval+errors██████████ feature+   ████ docs
                                                                  ablation
```

---

## 🔵 VISHWESH — Blocking Stream

### V1 · Normalization hardening *(Hours 0–4)*
**Files:** `src/normalize.py`
- [ ] Verify the 10-step pipeline on real dataset samples (once data lands)
- [ ] Audit legal suffix list: US (`Inc/Corp/LLC/Co/Incorporated`), India (`Pvt/Private/Ltd/LLP`), France (`SARL/SAS/SA/EURL`), Germany (`GmbH/AG/KG`)
- [ ] Verify abbreviation expansion (`&`→`and`, `Intl`→`international`, street abbreviations)
- [ ] Confirm transliteration handling (accented French names → ASCII)
- [ ] Add unit tests: `tests/test_normalize.py` with edge cases:
  - `"Acme Corp."` → `"acme"` · `"Sanjay Textiles Pvt Ltd"` → `"sanjay textiles"` · `"Café de la Paix SARL"` → `"cafe de la paix"` · empty/NaN → `""`

**Done when:** `pytest tests/test_normalize.py` passes and manual samples from the real dataset look correct.

### V2 · Address blocking precision fix *(Hours 4–14)* 🔴 CRITICAL
**Files:** `src/blocking.py`
**Problem:** Address TF-IDF generates too many false candidates (342/484 on synthetic test). Many businesses share cities/streets — address alone is not strong enough evidence.
- [ ] Add a **name-gating rule**: an address-derived candidate is kept only if the pair also has name similarity above a floor (e.g., token Jaccard ≥ 0.15 or WRatio ≥ 55)
- [ ] Alternatively/additionally raise address TF-IDF threshold from 0.25 → 0.45–0.55 (experiment)
- [ ] Measure candidate count reduction + recall change on train ground truth
- [ ] Target: candidate_pairs reduced ≥ 30% with **zero recall loss**

**Done when:** `measure_blocking_quality` shows pair_recall ≥ 0.95 and candidates reduced ≥ 30% vs current.

### V3 · Blocking recall push *(Hours 14–36)*
**Files:** `src/blocking.py`
- [ ] Run full blocking quality report on train data; list every missed true match
- [ ] For each missed match, identify why (typo? transliteration? landmark-only address? dropped token?) and add a bridging layer
- [ ] Tune per-layer thresholds (name TF-IDF 0.25, LSH 0.3, etc.) against ground truth
- [ ] Keep a blocking report: `docs/blocking_report.md` — per-layer recall/contribution table
- [ ] Add `tests/test_blocking.py` covering all 5 candidate generators + union + quality measurement

**Done when:** blocking pair_recall ≥ 0.97 on train with documented per-layer contribution.

### V4 · Freeze + docs *(Hours 60–72)*
- [ ] Freeze `normalize.py` / `blocking.py` (no more edits)
- [ ] Write blocking section of the methodology document (what layers, thresholds, recall achieved)

---

## 🟢 KARAN — Features Stream

### K1 · Feature validation on real data *(Hours 0–6)*
**Files:** `src/features.py`, `tests/test_features.py`
- [ ] Once data lands: compute all 25 features for sample pairs
- [ ] Check for constant columns, NaNs, inf, or broken ranges on real names/addresses
- [ ] Verify each feature discriminates: true-match pairs should score higher than negative pairs
- [ ] Report: feature distribution table + any dud features

**Done when:** feature report delivered (`docs/feature_report.md`), zero NaNs on real data.

### K2 · Feature unit tests *(Hours 6–12)*
**Files:** `tests/test_features.py`
- [ ] Test each feature function with known inputs:
  - `compute_name_features("acme robotics", "acme robotics")` → all name features = 1.0
  - identical addresses → addr features = 1.0
  - different countries → `same_country` = 0.0
  - empty strings → no crash, features = 0.0
- [ ] Test `compute_all_features` returns exactly 25 keys matching `FEATURE_NAMES`

**Done when:** `pytest tests/test_features.py` passes.

### K3 · Error analysis + evaluation tooling *(Hours 12–36)*
**Files:** `scripts/evaluate.py`, `docs/error_analysis.md`
- [ ] After Abhijit's first submission: compute macro F_0.5 locally
- [ ] Segment errors: false merges vs misses; by country (US/India/France); by singleton vs multi-match
- [ ] Recommend the #1 fix per error category (features? threshold? blocking?)
- [ ] Extend `evaluate.py` if needed: per-country breakdown, confusion examples

**Done when:** error analysis doc lists top 10 worst entities with diagnosis.

### K4 · Feature improvement *(Hours 36–60)*
**Files:** `src/features.py`
- [ ] Propose and test up to 5 new features based on error analysis. Candidate ideas:
  - token-level overlap on **surnames only** (for business names ending in distinctive words)
  - **number/city consistency** features (street number match, city match)
  - **country-specific** address token matches (PIN code digits for India, ZIP for US)
  - **acronym ↔ expansion** feature (IBM ↔ International Business Machines)
- [ ] Run ablation: add one feature at a time, measure val macro F_0.5 delta
- [ ] Only keep features with positive delta; keep `FEATURE_NAMES` in sync (frozen interface!)

**Done when:** ablation table in `docs/feature_report.md` with per-feature delta.

### K5 · Docs *(Hours 60–72)*
- [ ] Write feature engineering section of the methodology document
- [ ] Include: feature list, rationale, importance rankings (SHAP if time)

---

## 🟡 ABHIJIT — Pipeline & Model Stream

### A1 · Repo + env + data intake *(Hours 0–2)*
**Files:** repo root
- [x] Repo created, structure in place, all modules built
- [x] venv + dependencies installed
- [ ] **Download dataset from competition portal** → `data/dataset/{train,test}/`
- [ ] Run `python -c` profile: record counts S1/S2/S3, field completeness, country distribution, match distribution
- [ ] Save profile output to `docs/data_profile.md`

**Done when:** all 7 TSVs present and profiled.

### A2 · First end-to-end pipeline run *(Hours 2–10)*
**Files:** `src/pipeline.py`, `src/model.py`
- [x] Pipeline built and smoke-tested on synthetic fixtures
- [ ] Run full pipeline on real train data (normal mode, 25 Optuna trials)
- [ ] Inspect: blocking metrics, training pair counts, model AP scores, threshold choice
- [ ] Sanity-check `matching_results.tsv` + `candidate_pairs.tsv` on test
- [ ] `validate_submission.py` must PASS

**Done when:** two output files generated on real data and validated.

### A3 · First leaderboard submission *(Hours 10–14)*
- [ ] Upload `matching_results.tsv` to portal
- [ ] Record the score + run number in `docs/leaderboard_log.md`
- [ ] This is our baseline — every improvement is measured against it

**Done when:** score visible on leaderboard.

### A4 · Model tuning round *(Hours 14–36)*
**Files:** `src/model.py`
- [ ] Increase Optuna trials (25 → 50) for base models if time allows
- [ ] Try meta-learner variants: shallow LGB (current) vs logistic regression vs weighted average
- [ ] Calibrate probabilities (isotonic) before thresholding — check if it helps macro F_0.5
- [ ] Compare ensemble vs single LightGBM (ensemble may not always win)
- [ ] Log every experiment in `docs/experiments.md` (config → val macro F_0.5)

**Done when:** best config identified with a documented val score.

### A5 · Threshold re-optimization *(Hours 36–60)*
- [ ] After every change, re-run `find_best_macro_f05_threshold`
- [ ] Check threshold stability (is it at a boundary like 0.10? that signals a problem)
- [ ] Verify singleton behavior: how many test entities predicted empty? Compare with expected singleton rate (~30%)
- [ ] Consider **per-country thresholds** if error analysis shows different precision regimes

**Done when:** threshold chosen with a stable, non-boundary optimum.

### A6 · Final package + submission *(Hours 60–72)*
- [ ] Freeze all code (tag the commit)
- [ ] Run the pipeline one final time on the real test set
- [ ] `validate_submission.py` PASS
- [ ] Build the zip:
  ```
  team_submission.zip
  ├── output/{matching_results.tsv, candidate_pairs.tsv}
  ├── code/business_entity_resolution/{src/, README.md, requirements.txt}
  └── Documentation_template.md
  ```
- [ ] Verify the zip is runnable from scratch (fresh venv, `pip install -r requirements.txt`, run pipeline)
- [ ] Final leaderboard upload + package submission

**Done when:** package submitted and leaderboard score recorded.

---

## 🔄 Handoff Points (who waits on whom)

| Handoff | From → To | Trigger |
|---------|-----------|---------|
| H1 | Abhijit → All | Dataset downloaded — everyone starts real-data work |
| H2 | Vishwesh → Abhijit | Blocking frozen (recall ≥ 0.95) — pipeline can run |
| H3 | Karan → Abhijit | Features validated — training can proceed |
| H4 | Abhijit → Karan | First submission → error analysis begins |
| H5 | Karan → Vishwesh | Error analysis shows blocking misses → blocking push |
| H6 | Everyone → Abhijit | Final freeze — package assembly |

---

## 📡 Communication Protocol

### Async updates (every ~6 hours, in group chat)
```
✅ Done: <task> — <evidence/result>
🔨 Doing: <task> — <ETA>
🚧 Blocked: <task> — <what I need>
```

### Daily sync (every 12 hours, 15 min voice)
1. Blocking recall status (Vishwesh)
2. Feature/error analysis status (Karan)
3. Score status + next experiment (Abhijit)
4. Re-assign anything blocked

### Branching
```
main                  ← always working, always valid (releases)
 ├── feat/blocking-*  ← Vishwesh
 ├── feat/features-*  ← Karan
 └── feat/model-*     ← Abhijit
```
PR → 1 approval → merge. Never push directly to `main` after hour 14 (except Abhijit for submissions).

### Escalation rule
If blocked > 30 minutes: post in chat immediately, don't grind silently. Time is the scarcest resource.

---

## ✅ Definition of Done (per task)

A task is DONE only when:
1. Code committed with a clear message
2. Tests pass (`pytest tests/` for that module) or manual evidence attached
3. Measurable result recorded (recall %, F_0.5, candidate reduction — whatever the task promises)
4. Owner marked it `✅ Done` in the update protocol

**"Works on my machine" is not done. "Evidence attached" is done.**

---

## 🎯 Team Goals (priority order)

1. **Working submission on leaderboard** — by hour 14
2. **Validation F_0.5 ≥ 0.85** — by hour 36
3. **Blocking recall ≥ 0.97** — by hour 48
4. **Complete package + methodology doc** — by hour 68
5. **Top 50 (PPI interviews)** — stretch goal 🏆

---

## 📌 Current Status Board (update as you go)

| Task | Owner | Status | Evidence |
|------|-------|--------|----------|
| V1 Normalization hardening | Vishwesh | ⬜ Not started | — |
| V2 Address blocking precision | Vishwesh | 🔴 Critical, not started | — |
| V3 Blocking recall push | Vishwesh | ⬜ Not started | — |
| K1 Feature validation (real data) | Karan | ⬜ Blocked on data | — |
| K2 Feature unit tests | Karan | ⬜ Not started | — |
| K3 Error analysis tooling | Karan | ⬜ Not started | — |
| K4 Feature improvement | Karan | ⬜ Not started | — |
| A1 Data intake + profile | Abhijit | ⏳ Waiting for dataset | — |
| A2 First pipeline run (real) | Abhijit | ⬜ Blocked on A1 | smoke test ✅ |
| A3 First submission | Abhijit | ⬜ Blocked on A2 | — |
| A4 Model tuning | Abhijit | ⬜ Not started | synthetic meta AP 0.965 ✅ |
| A5 Threshold re-optimization | Abhijit | ⬜ Not started | — |
| A6 Final package | Abhijit | ⬜ Not started | — |
