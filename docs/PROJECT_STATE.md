# 📍 PROJECT STATE — Living Handoff Document

**Last updated:** 26 Sep 2026 (early hours) · **Updated by:** Abhijit + agent session
**Purpose:** One-page snapshot of where we are, what's verified, and what's next. Update at the end of every work session.

---

## 1. Where We Are

| Phase | State |
|-------|-------|
| Problem understood (PDF + video + guidelines) | ✅ Done |
| Competitive intel (13 team repos + 26 research repos) | ✅ Done |
| Code foundation (modules, tests, validator, synthetic data) | ✅ Done |
| **Dataset** | ✅ Extracted, profiled, SSD copy (`local_data/`), `data` symlink |
| Real-data work (blocking benchmark, end-to-end runs) | ✅ First full pipeline run on the 2.5% world |
| GPU dense stack (torch cu130 + sentence-transformers + faiss) | ✅ RTX 4050 working |
| Encoder fine-tuning | ⏳ Script ready, not yet run |

---

## 2. Measured Results (real sampled world — 55,169 S1 / 443,927 gallery / 191,103 gt pairs)

### Blocking (the recall ceiling)
| Configuration | Recall | Cands/S1 | Notes |
|---|---|---|---|
| 7-leg lexical union | 0.9413 | 108.4 | after phonetic/initialism/minhash caps (50.5M → 1.2M) |
| + dense e5 top-30 added | 0.9622 | 127.5 | dense adds +2.1pp |
| Raw first-K cap @30 | **0.2907** | 30 | ← the failure mode that was fixed |
| Scored cap (dense rerank) @30 | **0.8481** | 30 | `cap_candidates_scored`, 11.8s |
| Scored cap @50 | **0.8917** | 50 | pipeline default |
| Scored cap @100 | **0.9316** | 100 | ≈ union ceiling |

### Dense legs (zero-shot, license-safe)
| Model | recall@20 | @30 | @50 | Encode time |
|---|---|---|---|---|
| snowflake-arctic-embed-xs (Apache-2.0) | 0.7385 | 0.7581 | 0.7821 | 83s GPU |
| **intfloat/multilingual-e5-small (MIT)** | **0.7804** | **0.8005** | **0.8228** | 161s GPU |

### End-to-end (held-out test split — 11,033 S1, honest)
| Run | Test macro F0.5 | Singleton acc | Notes |
|---|---|---|---|
| v1: 35 lexical features | 0.8033 | 48.6% | first honest end-to-end |
| v2: + `name_dense_cosine` feature | **0.8122** | 50.6% | dense feature as model input |
| v3: entity-level split | 0.7990 | 46.2% | partition sensitivity |
| v4: honest validation + calibration | (see `output/world_2p5_split_v4`) | | calibrator fit on full-candidate distribution |
| Honest val (v4, full candidates + decision) | 0.8498 | | replaces the misleading 0.99 pair-level val |

**Official validator: PASS** on all runs (`utils/validate_submission.py --check-ids`).

---

## 3. What Is Built and Verified (evidence attached)

| Component | File | Verified by | Result |
|-----------|------|-------------|--------|
| Normalization (10-step + legal suffixes + Indic transliteration) | `src/normalize.py` | `tests/test_smoke.py` | ✅ |
| 36 pairwise features (+ `name_dense_cosine`) | `src/features.py` | `tests/test_smoke.py` | ✅ 36 features, no NaN |
| Scale-safe blocking (7 legs, caps, adaptive-K) | `src/blocking.py` | `benchmarks/blocking_world_2p5_capped.log` | ✅ union 0.9413 @108/S1 |
| **Scored top-K cap** (fuzzy/dense/hybrid) | `src/blocking.py` | `tests/test_scored_cap.py` (5/5) + `benchmarks/scored_cap_world_2p5.json` | ✅ 0.29 → 0.85 @30/S1 |
| **GPU dense blocking leg** | `src/dense_blocking.py` | `tests/test_dense_blocking.py` (6/6) + `benchmarks/dense_world_2p5.json` | ✅ e5 best zero-shot |
| Training pairs + scale-safe hard negatives | `src/training.py` | `tests/test_training_negatives.py` (5/5) | ✅ 2K S1 × 100K gallery < 60s |
| Entity-level train/val split (no leakage) | `src/training.py` | same test file | ✅ no entity overlap |
| Leak-free OOF stacking (LGB+XGB+RF → meta) | `src/model.py` | synthetic + real runs | ✅ |
| Decision layer (calibration + exclusivity + expected-F0.5) | `src/decision.py` | `tests/test_decision.py` (12/12) | ✅ |
| **Honest validation** (full candidates + decision on held-out entities) | `src/pipeline.py` | v4 run log | ✅ 0.8498 |
| Pipeline orchestrator (GPU dense + scored cap + dense feature) | `src/pipeline.py` | `tests/test_pipeline_e2e.py` (1/1) | ✅ e2e |
| Official validator | `utils/validate_submission.py` | every run | ✅ PASS |
| Fine-tune script (e5-small, MNRL) | `scripts/finetune_encoder.py` | smoke pending | ⏳ |

---

## 4. Critical Facts About the Real Data (verified by 2 other teams)

| Fact | Value |
|------|-------|
| Train S1/S2/S3 | 2.21M / 5.03M / 5.29M |
| Test S1/S2/S3 | 1.73M / 4.89M / 5.08M |
| France in test | 259,452 S1 (~15%) |
| Ground truth | 7,638,365 positive pairs |
| Singletons | 5.58% (123,247) |
| Multi-match roots | 89.0% (mean 3.46, max 11) |
| Every S2/S3 matches ≤1 S1 | verified, zero exceptions → exclusivity valid |
| True matches with same normalized name | only ~20% → semantic features matter |

---

## 5. Next Actions (in order)

1. **Fine-tune e5-small** (`scripts/finetune_encoder.py`, GPU) — SABER fine-tuned
   arctic to 0.9964 recall@20 vs our 0.8005 zero-shot. Biggest single lever.
   Benchmark recall@20/30 after; if better, re-run pipeline with `--dense-model`.
2. **Budget experiment @100/S1** — ceiling 0.9316 vs 0.8917 @50; costs ~2× features.
3. **Singleton handling** (50.6% correct) — decision-layer/feature work.
4. **Full-scale test run** (1.73M S1 / 10M gallery) — needs sharded embedding
   encode+search (embeddings for 10M records = 15GB fp32, > disk/RAM).
5. **Submit** (max 5/day, log every one) once full-scale pipeline is proven.

---

## 6. Risks We Are Watching

| Risk | Mitigation | Status |
|------|-----------|--------|
| Full-scale embeddings don't fit (10M × 384d) | Shard encode + search; fp16; IVF | 🟡 design needed |
| Disk space (3GB free on root) | Prune old runs; embeddings cache is reusable | 🟡 |
| GPU 6GB VRAM | fp16 + chunked search (already done) | ✅ |
| France generalization | Country-holdout (K4) + language-agnostic features | 🟡 |
| Submission format | Official validator before every upload | ✅ |
| Merge conflicts | PR-only, file ownership | ✅ CONTRIBUTING.md |

---

## 7. Key Decisions Log (so we don't relitigate)

| # | Decision | Why |
|---|----------|-----|
| D1 | LightGBM-family matcher | All top teams; fast, MIT, interpretable |
| D2 | Leak-free OOF stacking for meta-learner | Prevents val/test gap |
| D3 | Macro F_0.5 per S1 entity | Matches leaderboard exactly |
| D4 | Country never a hard filter | Official requirement; France unseen |
| D5 | `candidate_pairs.tsv` = exact model input | Official requirement + audit |
| D6 | Dense encoder added (e5-small) after all | Lexical ceiling 0.9413 < target; dense +2.1pp union, +0.9pp e2e |
| D7 | No external data/APIs | Disqualification rule |
| D8 | PR-only workflow on protected `main` | Team coordination |
| D9 | Scored top-K cap (not raw first-K) | Raw cap collapsed recall 0.94 → 0.29 |
| D10 | Dense cosine as model feature 36 | ~80% of true matches have different names |
| D11 | Calibrate + validate on full-candidate distribution | Pair-level val 0.99 vs honest 0.85 misleads decisions |
| D12 | Entity-level train/val split | Pair-level split leaks entities |
| D13 | Hard negatives from blocking candidates | Scale-safe (was ~6B comparisons) + matches inference distribution |

---

## 8. Open Questions

| # | Question | Who decides | When |
|---|----------|-------------|------|
| Q1 | Fine-tuned encoder worth the time? | Abhijit (from recall delta) | After fine-tune |
| Q2 | Final budget: 50 or 100/S1? | Abhijit (from F0.5 vs cost) | After budget run |
| Q3 | Singleton threshold strategy? | Abhijit + Karan | Before freeze |
| Q4 | Full-scale sharding design | Abhijit | Before full run |
