# 🔍 GAP ANALYSIS — Teammate Masterplans vs Our Implementation

**Date:** 25 Sep 2026 · **Author:** Abhijit · **Status:** Action required

> Deep-dive comparison of the two teammate masterplans (`team_masterplan_v.md`, `team_masterplan_v2.md`) against what we have actually built, audited against the official problem statement + guidelines.

---

## 0. NEW OFFICIAL FACTS DISCOVERED

While auditing, we fetched the **official starter scaffold** (`Sugandh-vI/Amazon-ML-challenge`) — these are now in `official/` and copied into our repo:

| Fact | Source | Impact on us |
|------|--------|-------------|
| **Challenge window: 25 Sep 12:00 AM – 27 Sep 11:59 PM IST** | guidelines PDF | ~2.5 days left at time of writing |
| **Max 5 submissions/day** (15 total) | guidelines PDF | Don't waste submissions — validate locally first |
| **Test set is ~1.7M entities** | official validator docstring | 🔴 **Scale changes architecture — see §3** |
| **Top 100 teams** advance (not top 50) | guidelines PDF | PPI interviews follow for top cohort |
| **Official validator** exists (`utils/validate_submission.py`) | scaffold repo | ✅ Now installed in our repo |
| **Official Documentation template** exists | scaffold repo | ✅ Now in repo root as `Documentation_template.md` |
| **`--check-ids` is optional**; matches ⊆ candidates is a WARNING, not a failure | official validator | Our stricter local checks are fine to keep |
| **Shortlisting uses BOTH public and private leaderboards** | guidelines PDF | No overfitting to public LB |
| **Version history of all submissions must be kept** | guidelines PDF | Log every submission (score + file hash) |

See `docs/guidelines_and_key_instructions.pdf` and `docs/student_resource_readme.md`.

---

## 1. WHO ARE THE MASTERPLANS FROM?

| File | Author (inferred) | Character |
|------|-------------------|-----------|
| `docs/team_masterplan_v2.md` | Merged draft (A+B) | 19 sections; methodology-first; explicit OOF/leakage rigor |
| `docs/team_masterplan_v.md` | "V" (Vishwesh) | 60 sections; organizer-alignment audit; P0–P3 tagging; strongest on guardrails |

Both are **higher quality than our current MASTERPLAN.md in methodology discipline.** We should absorb them, not replace them.

---

## 2. WHAT WE ALREADY HAVE (confirmed good)

| Our component | Status | Masterplan verdict |
|---------------|--------|--------------------|
| 10-step name/address normalization | ✅ Built | Aligned; needs multi-view (RAW retention) — see G1 |
| 7-layer blocking union | ✅ Built | Aligned on union-never-intersect; needs top-K caps (G2) + provenance (G3) |
| 25 pairwise features | ✅ Built | Missing 3 families: missingness (G4), contradiction (G5), entity-level (G6) |
| Hard negative mining (70/30) | ✅ Built | Needs OOF isolation for iterative rounds (G7) |
| Base models + shallow meta (OOF, leak-free) | ✅ Built | Aligned with v2 §10–11; keep |
| Macro F_0.5 threshold search | ✅ Built | Aligned; add entity-level decision layer (G6) |
| Synthetic data generator + evaluator + smoke tests | ✅ Built | Both plans recommend exactly this; extend with adversarial cases (G8) |
| Official validator integration | ✅ Now | Was missing; now uses official script |

---

## 3. 🔴 CRITICAL: SCALE (~1.7M TEST ENTITIES)

The official validator states the full test set is **~1.7M entities**. Our current code has three scale blockers:

| Blocker | Where | Why it breaks at 1.7M |
|---------|-------|----------------------|
| Dense TF-IDF cosine matrix | `blocking.tfidf_blocking_candidates` | `cosine_similarity(q, t)` materializes (n_q × n_t) — OOM |
| Row-wise Python feature loop | `features.compute_features_batch` | ~1 ms/pair × millions of pairs = hours |
| Uncapped candidate union | `blocking.union_candidates` | No top-K per channel → candidate explosion |

**Fixes (P0):**
1. **Top-K capping per channel** (e.g. K=50) before union — controls candidate count regardless of corpus size.
2. **Sparse/chunked similarity** — never materialize dense (n_q × n_t); process in query chunks and keep top-K.
3. **Vectorized feature computation** — use `rapidfuzz.process.cdist` batch APIs; chunk pairs.
4. **Sampling for training** — tune/validate on stratified samples; never compute features for the full cross-product.

**Verify before building:** confirm actual file sizes when the dataset lands (`wc -l`, `du -h`). If train is small (≤50K), we keep the simple path for training and only harden inference for test scale.

---

## 4. HIGH-VALUE GAPS (prioritized, actionable)

### 🔴 P0 — Direct score impact

#### G1. Multi-view normalization (V §5)
**Gap:** We produce ONE normalized string per field and discard intermediates.
**Why it matters:** Contradiction detection needs RAW vs normalized; address parsing failures (landmark addresses) need a raw fallback.
**Fix:** Keep `business_name_raw`, `business_name_light`, `business_name_aggressive`, and address equivalents. Similarity features computed on ≥2 views; the gap between views becomes informative.

#### G2. Top-K per channel + candidate budget (v2 §8, V §12)
**Gap:** We take ALL pairs above a threshold — no per-entity budget.
**Why:** Candidate explosion at scale; also redundant candidates dilute precision features.
**Fix:** Per channel, keep top-K (K≈50) per S1 by that channel's score. Union. Cap final per-entity budget (≈100–200) chosen from a recall-vs-budget curve. Add a **difficult-record fallback**: any S1 with zero candidates gets a relaxed pass so genuine matches aren't lost to blocker miss.

#### G3. Retrieval provenance features (v2 §9)
**Gap:** After union we lose which channel found each candidate.
**Why:** A candidate found by 3 independent channels is stronger evidence than one found by the weakest channel. Both plans make this a first-class feature.
**Fix:** Track `channels_found` per (s1, cand) pair → features: `n_channels`, `found_by_name_tfidf`, `found_by_phonetic`, `found_by_address`, `found_by_minhash`, `found_by_initialism`. Store during blocking; consume in features.

#### G4. Missingness features (V §13)
**Gap:** Models can learn "less data ⇒ no-match" as a shortcut; similarity on a missing field silently collapses to 0.
**Fix:** Add explicit present/absent flags: `name_a_present`, `name_b_present`, `addr_a_present`, `addr_b_present`, `both_names_present`, `both_addrs_present`. (6 features; missingness is organizer-acknowledged — abbreviated addresses, landmarks.)

#### G5. Contradiction features (V §14, v2 §9.1)
**Gap:** We have no explicit negative-evidence features. The organizer's own example — *same address, different business* — is exactly what a similarity-only matcher over-trusts.
**Fix:** Add contradiction flags computed only when both sides present: `country_conflict`, `addr_number_conflict` (digit-run mismatch), `city_conflict`, `addr_length_conflict` (one side <40% of the other with no containment). Aggregate: `contradiction_count`.

#### G6. Entity-level decision layer (V §22–26, v2 §12)
**Gap:** We threshold each pair independently. The leaderboard scores **entity-level decisions** (0/1/many), and singletons get 1.0 only if we predict empty.
**Fix:** After pairwise scoring, compute per-S1: `candidate_count`, `top_score`, `second_score`, `score_gap`. Apply decision policy:
- zero candidates → empty (already done)
- top_score < floor → empty (no-match engine)
- else accept all candidates above threshold (never force top-1)
- flag `ambiguous` when top two scores are within margin (diagnostic + optional Tier-3)
**AND** expose `candidate_count`, `score_margin`, `n_above_threshold` as *features* at training time — but computed **within-fold only** (leakage risk per V §13).

#### G7. OOF-isolated hard-negative mining (v2 §11.1)
**Gap:** We mine negatives once, from the whole training pool, before splitting — mild leakage risk and no iteration.
**Fix:** Iterative loop entirely inside the training split: K-fold → mine high-scoring OOF false positives → add to training → retrain → repeat until plateau. Validation holdout untouched by mining.

#### G8. Adversarial guardrail suite (V §36–37)
**Gap:** No adversarial fixtures.
**Fix:** Build from the organizer's own examples: (a) same name + different address; (b) **same address + different business** (highest priority); (c) abbreviation/typo stacks; (d) missing-field pairs; (e) multi-transform stacks. Treat as **guardrails, not objective** — never override real held-out F_0.5.

### 🟡 P1 — High value

#### G9. Country verification protocol (V §2.1a)
**Gap:** We use country as a blocking-adjacent feature (`same_country`) without measuring its actual signal in THIS dataset.
**Fix:** Before relying on it, run an experiment on train ground truth: for matched vs non-matched candidate pairs, what fraction agree/conflict/miss on country? Log as `country_active: bool`. Keep the feature only if it discriminates. **Never** hard-filter/hard-code regardless.

#### G10. Connected-component validation split (v2 §7.2)
**Gap:** We split at S1-entity level only. If one S2/S3 record matches multiple S1s, it can straddle train/val.
**Fix:** Build the ground-truth graph (nodes = records, edges = matches), compute connected components, assign whole components to train or val. Also run the **source fan-out check** (§3 of both plans): does any S2/S3 ID appear in >1 S1's match list?

#### G11. Multi-view-aware features + rare-token/IDF features (v2 §9, V §7)
**Gap:** No IDF-weighted features. Generic tokens ("services", "group") inflate similarity.
**Fix:** Compute token document frequency once over the combined corpus; add `rare_token_overlap`, `rarest_shared_token_idf`. Data-gated: keep only if ablation shows positive ΔF_0.5.

#### G12. Calibration test (v2 §12, V §21)
**Gap:** Untested. Probability calibration (isotonic/Platt) may change where the optimal F_0.5 threshold sits.
**Fix:** Ablate: raw vs isotonic-calibrated probabilities → compare macro F_0.5 at each one's own optimal threshold. Keep only if it helps.

### 🟢 P2 — Experimental (only with ablation evidence)

| Gap | Source | Trigger condition |
|-----|--------|-------------------|
| Ranking model (LambdaMART) vs binary classification | v2 §10 | Only if binary underperforms on entity metric |
| Dense multilingual embeddings (E5/BGE) + FAISS | both | Only if lexical blocking recall < target on real data |
| Cross-encoder reranker (top-K, ambiguous only) | both | Only if ambiguous population is material |
| Cross-source consistency features | V §29 | Only as evidence feature, ablated |
| Graph/transitive reasoning | V §30, v2 | Only for ambiguous residual, its own ablation |
| Pseudo-labeling | V §34 | Strict guardrails; likely skip given time |

---

## 5. ALIGNMENT AUDIT vs OFFICIAL PROBLEM STATEMENT

Checked both masterplans + our implementation against the PS PDF and official validator:

| Official requirement | Our status | Masterplans |
|----------------------|-----------|-------------|
| Macro F_0.5 metric, per-S1 then averaged | ✅ implemented (`find_best_macro_f05_threshold`, `scripts/evaluate.py`) | ✅ both correct |
| Singletons: empty = 1.0, any match = 0.0 | ✅ implemented | ✅ both emphasize |
| 0/1/many matches — never force top-1 | ✅ no top-1 anywhere | ✅ both emphasize |
| TSV with `sep="\t"` | ✅ everywhere | ✅ both warn |
| Every S1 appears in output | ✅ enforced in `_generate_output` | ✅ both flag as P0 |
| No duplicates in ID lists | ✅ validator + generation logic | ✅ both |
| Only S2/S3 IDs from test set | ✅ official validator `--check-ids` | ✅ both |
| `candidate_pairs.tsv` = final pre-model candidate set (superset of matches) | ✅ fixed earlier (was writing matches) | ✅ both explicit; V §2.2 adds "same run, never regenerated" — we comply |
| No external data/APIs/geocoding | ✅ no network calls in `src/` | ✅ both; V §47 adds a network-disabled test we should run |
| MIT/Apache-2.0, ≤8B params | ✅ LightGBM/XGBoost/RF, tiny params | ✅ both; v2 §16 asks for `MODEL_LICENSE_AUDIT.md` — **we should add** |
| Country open-set (France in test) | ⚠️ used as `same_country` feature — **not hard-coded** but not verified (G9) | ✅ both; V §2.1a protocol is the fix |
| Blocking = recall ceiling | ✅ measured (`measure_blocking_quality`) | ✅ both emphasize |
| Validation script before submitting | ✅ official script now in `utils/` | ✅ both |
| Documentation template | ✅ official template now in repo root | ✅ both mention |
| Max 5 submissions/day | ⚠️ not tracked yet | v2 §0 knew it; **add submission log** |

**Alignment verdict:** our implementation is fundamentally aligned. The masterplans' edge is **measurement discipline** (P0 tags, ablation gates, guardrails) and the **gaps listed above** — not a different architecture.

---

## 6. THE MASTERPLANS' BEST IDEAS WE SHOULD ADOPT (ranked)

1. **OOF-isolated hard-negative mining loop** — rigorous, cheap, real gains (G7)
2. **Contradiction + missingness features** — direct precision wins under F_0.5 (G4, G5)
3. **Entity-level decision layer** — where the metric actually lives (G6)
4. **Retrieval provenance features** — free signal we're discarding (G3)
5. **Top-K capping + difficult-record fallback** — scale survival (G2)
6. **Country verification protocol** — removes an unvalidated assumption (G9)
7. **Adversarial guardrail suite from organizer's own examples** — same-address-different-business (G8)
8. **MODEL_LICENSE_AUDIT.md + submission log** — compliance and version history (guidelines requirement)
9. **Multi-view normalization with RAW retention** — enables contradictions (G1)
10. **Connected-component split** — protects the validation number we steer by (G10)

---

## 7. WHAT WE EXPLICITLY DECLINE (and why)

| Idea | Why we decline for now |
|------|------------------------|
| Full 60-section build order as-is | Time budget: 72h hackathon, ~2.5 days left; must prioritize |
| Neural reranker, ColBERT, graph reasoning | No ablation evidence yet; high complexity; both plans mark P3 |
| LLM in scored pipeline | Prohibited (external API) + license constraints; both plans agree |
| Separate S1→S2 / S1→S3 models | Test with source feature first; split models only if evidence |
| Pseudo-labeling | Risk >> reward at this stage |

---

## 8. EXECUTION PLAN FROM THIS ANALYSIS

| # | Action | Owner | When |
|---|--------|-------|------|
| 1 | Add missingness + contradiction features (G4, G5) | Karan | Immediately |
| 2 | Add retrieval provenance tracking (G3) | Vishwesh | Immediately |
| 3 | Top-K capping per channel + fallback (G2) | Vishwesh | Immediately |
| 4 | Entity-level decision layer + features (G6) | Abhijit | Immediately |
| 5 | Country verification script (G9) | Abhijit | After data lands |
| 6 | Fan-out + connected-component split check (G10) | Abhijit | After data lands |
| 7 | Adversarial fixtures (G8) | Karan | Next |
| 8 | OOF hard-negative loop (G7) | Abhijit | After first submission |
| 9 | `MODEL_LICENSE_AUDIT.md` + submission log | Vishwesh | Before final package |
| 10 | Multi-view normalization (G1) | Vishwesh | After first submission |

---

*This gap analysis is the bridge between the teammate masterplans and our codebase. Every G-item becomes a task in TASK_BREAKDOWN.md.*
