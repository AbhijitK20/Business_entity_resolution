# Amazon ML Challenge 2026 — Business Entity Resolution
## Master Plan — Final Merged & Hardened Version

> Provenance: this document merges two independently drafted masterplans (referred to below as **Draft A**, 147 sections, and **Draft B**, 110 sections), verifies every external claim against the actual competition repo, and adds four methodological fixes that neither draft originally contained (OOF-isolated hard negatives, fan-out/connected-component leakage detection, explicit `[P0]–[P3]` prioritization, and corrected country framing). Where this document says "fix," it means the gap existed in both drafts and is being closed here — not that one draft already solved it.

---

# 0. Verified Ground Truth

Confirmed directly against `Sugandh-vI/Amazon-ML-challenge` (the official starter scaffold — data + validator only, **no solution code**; treat it as the source of truth for format and rules, not as prior art):

```text
student_resource/
├── dataset/
│   ├── train/  train_source1.tsv, train_source2.tsv, train_source3.tsv,
│   │           train_ground_truth.tsv
│   └── test/   test_source1.tsv, test_source2.tsv, test_source3.tsv
└── utils/      validate_submission.py, Documentation_template.md
```

```text
Max 5 leaderboard submissions/day, 3 days total.
Public leaderboard = subset of test set (live). Private = remainder, revealed
after the challenge ends. Predict the FULL test set regardless of split.
A false merge costs roughly 2× what a miss costs (precision-weighted F0.5).
Final package = output/ (both TSVs) + code/business_entity_resolution/
(src/, README.md, requirements.txt) + filled Documentation_template.md.
```

Two other repo names appeared in earlier drafts (`Ankittian/amazon_2026_toolkit`, `SaikatxAlpha/AWS-ML-Business-Entity-Resolution-Challenge`) — **unverified**. Check them the same way (fetch, read the README, confirm they contain actual solution code and not just the scaffold) before treating anything from them as real prior art.

Always run before submitting:
```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

---

# 1. The Two Problems (read this before anything else)

This is not binary classification. It's two coupled optimization problems, each with its own metric, and conflating them is the single most common way teams underperform:

| | Problem A — Candidate Generation | Problem B — Matching & Decision |
|---|---|---|
| Objective | Maximize **candidate recall** | Maximize **macro F0.5** |
| Failure mode if ignored | True match never enters candidate_pairs.tsv → unrecoverable, no matter how good the model is | False merges destroy singleton scores; a false positive costs ~2× a miss |
| What "done" looks like | Recall@K ≥ ~98–99% at an acceptable candidate budget | High precision on accepted matches, correct abstention on true singletons |

`candidate_pairs.tsv` is **not** a debug artifact — it's graded as the actual candidate universe fed to the model at inference time. Everything downstream is capped by what's in it.

---

# 2. Problem Definition & Data Contract

Three independent sources, no shared IDs. Source 1 is the deduplicated reference set; find its matches (zero, one, or many) in Source 2 and/or Source 3.

Each record: `entity_id`, `business_name`, `business_address`, `country`.

```text
Name noise:     typos, abbreviations, legal-suffix variants, punctuation,
                word-order changes, transliteration, DBA/trade names,
                missing/extra words
Address noise:  abbreviations, reordered components, missing components,
                missing postcode, landmarks, municipal numbering, partial
                addresses
```

**Country — corrected framing.** The official guidance requires treating `country` as an **open-set string field**: training covers US + India only, test adds France (unseen). This means two things simultaneously, and it's important not to collapse them into one:

1. **Never hard-code** the country set (`{US, India}` or `{US, India, France}`) anywhere in the pipeline — no country-specific branching logic that would break on a fourth country.
2. **Do not therefore discard country as a signal.** Country mismatch/agreement is still a legitimate categorical feature (§8) — the constraint is about *generalization*, not about *down-weighting the field*. A model that ignores country entirely gives up real evidence; a model that hard-codes the three known values will silently misbehave on anything else. The correct approach is: country as an arbitrary categorical feature, encoded in a way that degrades gracefully to an unseen value (e.g., target/frequency encoding with an explicit "unseen" fallback, or simply exact-match/mismatch as a binary feature rather than one-hot on fixed categories).

---

# 3. Evaluation Metric

```text
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Computed **per Source-1 entity**, then **macro-averaged**. A correct empty prediction on a true singleton scores 1.0; one wrong candidate added scores 0.0.

```text
Primary metric (optimize this directly, always):
    macro F0.5, entity-level

Secondary metrics (report on every experiment):
    candidate recall, candidate reduction ratio, pairwise precision/recall/
    PR-AUC, entity-level precision/recall, singleton accuracy, multi-match
    recall, false-merge rate, calibration error (ECE), Recall@{5,10,20,50,100},
    runtime, memory
```

Never optimize a proxy (pairwise F1, AUC, accuracy) and assume it transfers to entity-level macro F0.5 — validate the actual metric at every stage.

---

# 4. Target Architecture

```text
SOURCE 1 RECORD
      │
      ▼
MULTI-VIEW NORMALIZATION (raw / basic / alnum / tokens / legal-stripped —
      never destroy the raw field)
      │
      ├── Exact/Token Blocking ────┐
      ├── Char + Word TF-IDF ──────┤
      ├── Dense Embeddings (E5/BGE)┼──► CANDIDATE UNION ──► DEDUP + PROVENANCE
      └── Optional ColBERT ────────┘         (retain which channel(s) found
                                               each candidate — it's a feature)
                                        │
                                        ▼
                          RICH PAIR FEATURE ENGINE
              (name / address / cross-field / country / source /
               missingness / embedding / rare-token / retrieval-provenance /
               candidate-difficulty features — §8)
                                        │
                          ┌─────────────┴─────────────┐
                          ▼                            ▼
                     LightGBM                   Neural Reranker
                  (baseline matcher)         (cross-encoder, top-K only,
                                               only if it beats LightGBM
                                               on macro F0.5)
                          │                            │
                          └─────────────┬──────────────┘
                                        ▼
                              ENSEMBLE (weights tuned on
                              validation, never by hand)
                                        ▼
                          PROBABILITY CALIBRATION
                          (Platt / isotonic / temperature)
                                        ▼
                          ENTITY-LEVEL DECISION LAYER  (§9 — its own
                          problem, not a threshold slapped on scores)
                            ┌───────────┴───────────┐
                            ▼                        ▼
                         MATCH(ES)                ABSTAIN
                            └───────────┬────────────┘
                                        ▼
                    matching_results.tsv + candidate_pairs.tsv
```

**Mental model** — five machines, each independently testable:
`FIND CANDIDATES → UNDERSTAND PAIR → SCORE → DECIDE → VERIFY OUTPUT`.
The ML model is one component of five; most of the leaderboard gap comes from Machine 4 (decision policy) and Machine 5 (validation discipline), not from model sophistication.

---

# 5. Priority-Tagged Build Order

Tags: **`[P0]`** = must be right before anything else matters (garbage in → garbage out downstream, or a wrong choice here silently invalidates every later measurement). **`[P1]`** = high-leverage, do next. **`[P2]`** = valuable, but only after P0/P1 are validated. **`[P3]`** = speculative/advanced — implement only if experiments on the earlier phases specifically justify it.

```text
[P0] Dataset forensics — record counts, missingness, match distribution,
     duplicate analysis, country distribution (§6)
[P0] Reproducible data layer — TSV → Parquet, DuckDB/Polars, schema validation
[P0] Validation split with leakage protection (§7 — entity-level split
     AND fan-out/connected-component check)
[P0] Multi-view normalization (name + address)
[P0] Exact + token-blocking candidate generator, with candidate-recall
     measurement — establishes the recall ceiling before anything else
[P1] Char + word TF-IDF retrieval, Recall@K report
[P1] Pairwise feature engine (§8)
[P1] LightGBM baseline matcher + validation metrics
[P1] F0.5-aware decision optimization (search thresholds against macro
     F0.5, not 0.5, not pairwise F1)
[P2] Dense multilingual embeddings (E5/BGE) + FAISS retrieval
[P2] Hard-negative mining loop — OOF-isolated (§7.2)
[P2] Calibration (Platt/isotonic/temperature)
[P2] Entity-level decision layer refinement — singleton detector, margin
     rule, contradiction rule (§9)
[P3] Neural reranker (cross-encoder, top-K candidates only)
[P3] Late interaction / ColBERT — only if P2 dense retrieval clearly helps
     and GPU budget allows
[P3] Probabilistic linkage (Splink/Dedupe) as baseline or feature source
[P3] Ensemble of complementary models — only if it beats the single
     strongest model on validation
[P3] Graph-based / cross-source triangulation reasoning
```

Do not start a `[P1]` item before its `[P0]` prerequisites are measured and understood. Do not start `[P3]` work speculatively "because it sounds advanced" — every `[P2]`/`[P3]` item must answer: *what problem does it solve, what's the evidence it will help on this dataset specifically, and what's the compute cost?*

---

# 6. Phase 0 — Dataset Forensics `[P0]`

Before any modeling, measure:

```text
Record counts:      N(S1), N(S2), N(S3)
Missingness:        % missing/empty/unique for business_name, business_address,
                     country, per source
Match distribution: % singleton, % with 1/2/3/>3 matches; avg/median/max
                     matches per S1
Source split:       S1→S2 only, S1→S3 only, S1→S2+S3
Country dist:       full breakdown per source (do NOT assume it matches the
                     "US, India, +France in test" summary exactly — verify)
Duplicates:         duplicate normalized names/addresses/name+address within
                     each source — determine whether the same business
                     legitimately appears multiple times in S2/S3 (this
                     affects whether one-to-many decisions are even possible)
```

Deliverables: `reports/dataset_profile.json`, `reports/match_distribution.csv`, `reports/missingness.csv`, `reports/duplicate_analysis.csv`, `docs/DATASET_FORENSICS.md`.

---

# 7. Validation Strategy `[P0]` — with Leakage Protection

## 7.1 Entity-level split (baseline requirement)

Split at the **Source-1 entity level**, never by candidate pair. All candidates associated with a validation-set S1 entity must stay entirely in validation. Stratify by country, match-count, and singleton/non-singleton where the dataset is large enough to support it.

## 7.2 Fan-out / connected-component leakage check — **fix, not present in either draft**

Standard entity-level splitting has a real gap: if a single Source-2 or Source-3 record is the true match for **multiple** Source-1 entities (a legitimate possibility — nothing in the problem forbids fan-in from S1's perspective, and S1 itself can fan out to multiple S2/S3 records), then that S2/S3 record can end up **appearing in both the train and validation candidate pools simultaneously**, via different S1 entities on either side of the split. The model then implicitly "sees" validation-side identity information during training through a shared record, even though no single S1 entity crosses the split.

**Fix:** before splitting, build a graph where nodes are records (S1 ∪ S2 ∪ S3) and edges are ground-truth match links. Compute **connected components**. Split at the **connected-component level**, not the raw S1-entity level — i.e., treat every component as an atomic unit that goes entirely into train or entirely into validation. This guarantees no S2/S3 record straddles the split. In practice most components will be small (one S1 + its few matches), but auditing for larger components is essential — a large component would indicate a genuinely ambiguous or heavily-reused record worth inspecting directly.

```text
1. Build graph: nodes = all record IDs appearing in train_ground_truth.tsv
   (S1 side and matched S2/3 side), edges = ground-truth match pairs
2. Find connected components (union-find or graph library)
3. Report component-size distribution — flag any component above a small
   size (e.g. >5) for manual inspection before deciding how to split it
4. Assign whole components to train or validation, stratifying by the
   component's dominant country / match-count profile
5. Verify post-split: no S2/S3 ID present in both train and validation
   candidate pools
```

## 7.3 Full inference simulation

For every validation S1 entity: hide ground truth → run the **complete** candidate-generation pipeline (not a shortcut) → generate candidates → score → decide → compare to hidden ground truth → compute macro F0.5. Never generate candidates using ground truth; never let ground truth influence normalization, TF-IDF vocabulary fitting, or embedding tuning for the validation split.

## 7.4 Unseen-country simulation

Train on US+India only; validate as if evaluating an unseen country to approximate the France gap. Leave-one-country-out experiments (train on A, test on B) give a direct generalization estimate, since France itself can't be tuned against.

---

# 8. Candidate Generation — Multi-Channel Union

Never depend on a single blocking rule; take the **union** across channels, never the intersection.

```text
A. Exact normalized retrieval   name/address exact + legal-stripped variants
                                 (country + normalized_name, country + rare
                                 token, etc. — do not assume postal codes exist)
B. Token blocking                rare tokens >> common tokens; learn token
                                 frequency from the data itself rather than a
                                 fixed stoplist
C. Character n-gram TF-IDF       2–6 grams; strongest classical channel for
                                 typos, transliteration, abbreviations,
                                 punctuation variance; separate indexes for
                                 name / address / name+address
D. Word-level TF-IDF             handles word reordering, missing words
E. Fuzzy retrieval (RapidFuzz)   feature generator / narrow retrieval only —
                                 not the whole matching system on its own
F. Address-specific retrieval    house number, street token, postal token,
                                 city token — inferred from supplied data
                                 ONLY, never external geocoding
G. Dense multilingual embeddings E5-small/base/large, BGE-M3; benchmark
                                 empirically, don't assume the largest wins
H. Optional: ColBERT / late      token-level interaction; only if G proves
   interaction (`[P3]`)          valuable and the candidate budget/compute
                                 justifies it
```

Deduplicate the union and retain **retrieval provenance** per candidate (which channel(s) found it) — a candidate independently found by multiple channels is stronger evidence and should be a feature, not thrown away. Also compute **mutual retrieval** (does S2 retrieve S1 back under the same channel?) as a consensus signal.

Apply a **candidate budget** (e.g., top-50 per channel → union → cheap filter → 50–200 final per S1) determined experimentally, not assumed.

Measure **Candidate Recall@K** (K = 5/10/20/50/100) per channel and for the union, split by singleton/one-match/multi-match and by country. Recall must be validated as high *before* investing further downstream — a downstream model can never fix a candidate that was never retrieved.

---

# 9. Pairwise Feature Engineering `[P1]`

```text
Name features         exact/normalized-exact, Levenshtein, Jaro-Winkler,
                       token_sort/set_ratio, WRatio, partial_ratio, char
                       TF-IDF cosine, word TF-IDF cosine, token Jaccard/
                       overlap, length ratio/diff, common/rare token count,
                       prefix/suffix similarity, first/last token match

Address features       same similarity family + numeric/house-number overlap,
                       postal-token match, city/street-token overlap

Cross-field features   name_sim × address_sim, max/min/mean/weighted-mean;
                       explicitly encode (strong name + weak address) vs
                       (weak name + strong address) vs (strong+strong) —
                       identical name + very different address may indicate
                       a different branch, not the same entity (§9.1 below)

Country features       exact match / mismatch / unknown, encoded to
                       generalize to unseen values (§2) — never a fixed
                       one-hot over {US, India, France}

Source features         candidate_source (S2 vs S3), source_pair indicator
                       — S2 and S3 may have different noise profiles

Missingness features   explicit present/missing indicator for every field
                       on both sides; similarity on a missing field must
                       NOT silently collapse to a raw zero — the model
                       needs to distinguish "known mismatch" from "unknown"

Embedding features      name/address/combined cosine similarity, per model
                       benchmarked (don't include every model by default —
                       ablate)

Rare-token/IDF          sum/max/mean IDF of shared tokens, rarest shared
                       token — "xyzengineering" is far more discriminative
                       than "shop" or "starbucks"

Retrieval features      per-channel rank, per-channel similarity, count of
                       channels that independently retrieved this candidate,
                       mutual-retrieval flag

Candidate difficulty    score_margin (top vs second-best for this S1),
                       rank, number of similarly-scored competing candidates
```

## 9.1 Contradiction features — treated explicitly

High name similarity does not always mean "same entity." Build features that let the model learn *negative* evidence directly, rather than hoping it infers this from raw similarity scores alone:

```text
name_strong + address_strong        → strong positive evidence
name_strong + address_contradictory → possible different branch (not
                                       automatically a match)
name_weak + address_strong          → possible DBA/trade-name variant
name_weak + address_contradictory   → strong negative evidence
```

The ground truth in training defines what counts as "same entity" for branch cases — the model should learn this pattern rather than being told a fixed rule.

---

# 10. Modeling — Baseline Ladder `[P1]`

```text
B0  Exact normalized name
B1  Exact name + address
B2  RapidFuzz features
B3  RapidFuzz + TF-IDF retrieval
B4  TF-IDF + LightGBM
B5  + dense retrieval features
B6  + hard-negative mining (OOF-isolated, §11)
B7  + neural reranker (top-K only, `[P3]`)
B8  + calibrated ensemble (`[P3]`)
B9  + entity-level decision optimization (§9 fully applied)
```

Every rung reports macro F0.5, candidate recall, runtime, memory — advance only if it wins on the entity-level metric, not a proxy.

Benchmark matcher families: LightGBM, XGBoost, CatBoost, logistic regression, random forest. Also test a **ranking** formulation (LambdaMART / pairwise or listwise loss) against plain binary classification — ranking candidates per S1 may align better with the actual task ("which of these should be accepted") than independent pair classification. LightGBM is the default starting matcher: MIT-licensed, and tree models handle the heterogeneous feature set (similarity scores, ranks, binary flags, counts, categoricals) well without heavy tuning.

---

# 11. Hard-Negative Mining — OOF-Isolated `[P2]`

Random negatives ("Shree Ganesh Medicals" vs "Toyota Motor Corporation") teach the model almost nothing. Hard negatives — same/similar name at a different address, same address with a different business, high fuzzy score but known non-match, dense-retrieval false positives — are far more valuable training signal.

## 11.1 The leakage risk in naive iterative mining — fix, not present in either draft

The naive loop ("train → score a large pool → take high-scoring false positives → add to training set → retrain → repeat") has a subtle problem if it's run against the same validation holdout used for model selection: each iteration lets the model implicitly "see" validation-adjacent signal through what gets promoted into the training set, and the holdout stops being a clean estimate of generalization by the time you're ready to pick a final threshold.

**Fix — mine exclusively from out-of-fold (OOF) predictions, never from the holdout:**

```text
1. K-fold the TRAINING portion only (never touch the validation holdout)
   — fold boundaries must also respect the connected-component grouping
     from §7.2, so a record's train/OOF fold assignment is consistent
     with its component
2. For each fold k: train on folds != k, score fold k's candidate pool
   out-of-fold
3. Collect high-scoring false positives from OOF predictions across all
   folds — these are candidates the model was NOT trained on when scored,
   so they're genuinely informative hard negatives, not artifacts of
   memorization
4. Add OOF-derived hard negatives to the training set, retrain on the
   full training portion
5. Repeat until OOF hard-negative yield plateaus
6. Evaluate the final model ONCE against the held-out validation set
   (§7), which has never been touched by any mining iteration
```

This keeps the validation holdout genuinely clean for final threshold selection and model comparison — the number that goes into your decision-making at the end (§14 threshold search) hasn't been indirectly optimized against.

---

# 12. Calibration & Entity-Level Decision Policy `[P2]` — treated as its own problem

This is where many otherwise-good pipelines lose points, because it's tempting to treat "get a good score" and "make a good decision" as the same problem. They aren't — a well-ranked candidate list and a well-chosen decision policy are two different achievements, and the metric (macro F0.5, entity-level, precision-weighted) is scored on the *decision*, not the ranking.

1. **Calibrate** raw scores (Platt scaling / isotonic / temperature scaling) on a held-out calibration set, separate from both training and the final validation holdout if data volume allows.
2. **Never use a universal 0.5 threshold.** Search a fine-grained range against actual entity-level macro F0.5 (not pairwise F1, not accuracy). Because the metric is precision-weighted, the optimal threshold is very likely well above 0.5.
3. **Singleton detection as a first-class subproblem.** A dedicated `P(any match | S1 entity)` head, separate from per-candidate scoring, can decide "return `[]` immediately" before ever ranking candidates — this directly targets how singletons are scored (1.0 for correct empty, 0.0 for any wrong addition) rather than relying on every individual candidate score happening to fall below threshold.
4. **Balance the three cases explicitly**, since they're scored identically by the metric but need different handling:
   - *Zero matches (singleton):* be conservative — a false positive here is maximally costly.
   - *Single match:* accept on strong absolute score AND sufficient margin over the next-best candidate.
   - *Multiple matches:* don't force one-to-one; accept each candidate independently against threshold + contradiction check, since the ground truth explicitly allows one-to-many.
5. **Score-gap / margin analysis:** `score_margin = top_score − second_score`. Large margin + high score → decisive, act on it. Small margin between top candidates → ambiguous, worth extra scrutiny (this is exactly where a neural reranker, if used, should be spent — see §5's `[P3]` ambiguity-aware reranking) rather than spending expensive compute uniformly on every candidate.
6. **Contradiction check** (§9.1) applied at decision time, not just as a training feature — a candidate with a strong name match but a strongly contradictory address should require a materially higher bar to accept.

---

# 13. Common Failure Modes — Explicit Guardrails

```text
✗ best_candidate = candidates.iloc[0]; return best_candidate
  → forces a match on every S1 entity; destroys singleton scoring.

✗ matches = candidates[score > 0.5]
  → unvalidated universal threshold; near-certainly wrong for a
    precision-weighted metric.

✗ Using only business_name, or only address, or only embeddings, or only
  fuzzy matching, in isolation
  → each is individually blind to a different noise category (shared
    names, incomplete/shared addresses, semantic-but-wrong matches,
    transliteration/long-range noise respectively). Combine them.

✗ Hard-coding country ∈ {US, India} or {US, India, France}
  → violates the open-set requirement (§2); will misbehave on the
    private leaderboard if it contains anything unexpected.

✗ Treating candidate_pairs.tsv as a debug artifact
  → it's graded as the real pre-inference candidate universe.

✗ Mining hard negatives from the same holdout used for final validation
  → contaminates the number you use to pick your final threshold (§11.1).

✗ Splitting only at the S1-entity level with no fan-out check
  → a shared S2/S3 record can straddle train/validation, leaking
    identity signal across the split (§7.2).

✗ Treating "candidate ranking quality" and "final decision quality" as
  the same problem
  → macro F0.5 is scored on the decision, not the ranking; a great
    ranker with a naive threshold still loses (§12).
```

---

# 14. Error Taxonomy — Separate Failure Types

For every missed true match, classify:

```text
True match NOT in candidate_pairs.tsv     → retrieval/candidate-generation
                                             failure (fix: §8, add/tune a
                                             channel, raise budget)
True match IN candidates, model rejects   → matching/scoring failure
                                             (fix: §9 features, §10 model,
                                             §11 hard negatives)
Model score good, threshold rejects it    → decision-policy failure
                                             (fix: §12)
```

Track proportions from real validation runs (illustrative structure only — actual numbers must come from your own error analysis): e.g. "40% retrieval failure / 30% matching failure / 20% decision-policy failure / 10% genuine data ambiguity." This tells you where to actually spend engineering time — don't guess.

Further slice by: typo, abbreviation, transliteration, word reorder, missing field, wrong-branch ambiguity, common vs rare business name, country mismatch, singleton vs multi-match.

---

# 15. Research Questions to Answer Before Finalizing Any Component

```text
Q1  Is LightGBM actually the best pairwise matcher vs XGBoost/CatBoost/
    logistic regression/neural cross-encoder/probabilistic linkage/
    ranking formulations — on THIS dataset, measured by macro F0.5?
Q2  Is dense retrieval actually better than character TF-IDF? Measure
    Recall@K, don't assume.
Q3  Is E5 better than BGE-M3 here specifically?
Q4  Does multilingual embedding retrieval help with the ACTUAL noise
    present in this data, not generic benchmark scores?
Q5  Does ColBERT improve candidate recall enough to justify its added
    complexity and compute cost?
Q6  Would a cross-encoder outperform LightGBM on macro F0.5 (not
    pairwise F1 — the two can disagree)?
Q7  Would a ranking formulation outperform binary classification?
Q8  Would separate S1→S2 and S1→S3 models outperform one universal
    model with a source feature?
Q9  Would a dedicated singleton detector (§12.3) measurably improve
    macro F0.5 over threshold-only decisions?
Q10 Would entity-level graph/connected-component reasoning at
    INFERENCE time (beyond its use for leakage-safe splitting, §7.2)
    improve results — test only after a strong baseline exists.
Q11 Would hard-negative contrastive learning improve retrieval quality
    specifically, not just matching quality?
Q12 Would pseudo-labeling improve a neural model — test cautiously,
    high risk of label contamination if the confidence threshold isn't
    very strict.
Q13 Would synthetic noise augmentation improve robustness, validated on
    real error-taxonomy slices (§14) rather than assumed?
Q14 Would an ensemble actually beat the single strongest model on
    validation — never assume yes; check error overlap (§15.1) first.
```

## 15.1 Ensemble diversity check (only relevant if Q14 is being tested)

Before ensembling, compare error overlap between candidate models: if Model A's mistakes are frequently corrected by Model B, ensembling adds value. If both models make identical mistakes, ensembling adds little beyond noise — don't ensemble reflexively.

## 15.2 Final Decision Framework

| Component | Option A | Option B | Option C | Winner (by validation) |
|---|---|---|---|---|
| Normalization | Rules | Learned | Hybrid | TBD |
| Retrieval | TF-IDF | E5/BGE | ColBERT | TBD |
| Pair matcher | LightGBM | Cross-encoder | Ranking model | TBD |
| Probabilistic | Splink | None | Hybrid | TBD |
| Neural training | None | Contrastive | Cross-encoder fine-tune | TBD |
| Decision | Threshold | Margin | Learned entity model | TBD |
| Ensemble | No | Weighted | Rank fusion | TBD |

Fill "winner" strictly from experiment results — never from intuition, novelty, or which one "sounds more advanced."

---

# 16. License & Compliance `[P0]` before any final submission

```text
Final model:      MIT or Apache 2.0
Parameter count:  ≤ 8 billion
External data:    forbidden — no geocoding, business registries, directories,
                   or web lookups of any kind for resolving actual entities.
                   (Researching algorithms/papers/libraries online is fine —
                   the restriction is on business-identity data, not
                   general research.)
```

Maintain `MODEL_LICENSE_AUDIT.md` per shipped model/library:
```text
name · repository · model card · software license · model license ·
parameter count · version · used in final pipeline? · reason
```

Known-MIT candidates (verify at time of use — licenses change): `intfloat/multilingual-e5-*`, `BAAI/bge-m3`, LightGBM. `lightonai/GTE-ModernColBERT-v1` is listed Apache-2.0 on its current HF model card as of this drafting — re-check before shipping. Individually verify any Jina embedding variant (some are CC-BY-NC and incompatible with the final-model requirement) — fine for `[P3]` research, not for the final pipeline.

---

# 17. Repository Structure & Execution

```text
amazon-entity-resolution/
├── README.md · MASTER_PLAN.md · MODEL_LICENSE_AUDIT.md · requirements.txt
├── data/{raw,processed,parquet,splits}/
├── notebooks/  01_dataset_forensics · 02_normalization · 03_exact_blocking ·
│               04_tfidf_retrieval · 05_dense_retrieval · 06_feature_engineering ·
│               07_lightgbm · 08_probabilistic_er · 09_neural_reranker ·
│               10_hard_negatives · 11_ensemble · 12_calibration ·
│               13_f05_optimization · 14_error_analysis
├── src/business_entity_resolution/
│   ├── normalization/ blocking/ retrieval/ candidates/ features/
│   ├── models/ neural/ calibration/ decision/ evaluation/ ensemble/
│   └── inference/ submission/
├── configs/  base.yaml · retrieval.yaml · features.yaml · models.yaml ·
│             decision.yaml
├── experiments/  (MLflow or a structured registry — every run stores:
│                 experiment_id, git_commit, dataset_version, candidate
│                 generator config, candidate recall, feature set, model,
│                 hyperparameters, threshold, macro F0.5, precision, recall,
│                 singleton accuracy, runtime, memory)
├── models/ embeddings/ reports/ research/ tests/
├── output/{matching_results.tsv, candidate_pairs.tsv}
└── scripts/  prepare_data.py · generate_candidates.py · train_matcher.py ·
              mine_hard_negatives.py · optimize_threshold.py ·
              run_inference.py · validate_submission.py
```

Production entry point — final inference must not depend on manual notebook execution:
```bash
python -m business_entity_resolution.run \
    --data-dir dataset --output-dir output --config configs/final.yaml
```

---

# 18. Golden Rules

```text
1.  Candidate generation optimizes RECALL. Matching optimizes PRECISION.
    They are different problems with different metrics — don't conflate them.
2.  Never use a 0.5 threshold without validating it against entity-level
    macro F0.5.
3.  Always UNION retrieval channels, never intersect.
4.  candidate_pairs.tsv is the real, final, pre-inference candidate set,
    not a debug file — it caps everything downstream.
5.  Country is an open-set categorical feature: use it, but never hard-code
    its value set (§2).
6.  No external business data, ever — explicit disqualification risk.
7.  Select a model because it improves validation macro F0.5 — never
    because it's newer, bigger, a Transformer, or "sounds more advanced."
8.  Separate retrieval failures from matching failures from decision-policy
    failures (§14) — each needs a different fix.
9.  Treat singleton prediction as a first-class subproblem (§12.3), not
    an afterthought of thresholding.
10. Mine hard negatives out-of-fold (§11.1) — never contaminate the
    validation holdout you'll use for final threshold selection.
11. Split validation at the connected-component level (§7.2), not just
    the S1-entity level — check for fan-out leakage explicitly.
12. Verify any external claim (a repo, a license, a benchmark number)
    before relying on it. "Plausible" is not "confirmed."
```

---

# 19. Definition of Done

```text
[ ] Dataset fully audited (§6)
[ ] Validation split is entity-level AND connected-component-safe (§7.1–7.2)
[ ] Full pipeline simulated end-to-end on validation, no ground-truth leakage
    into normalization/blocking/vocabulary/embedding tuning (§7.3)
[ ] Unseen-country generalization tested via leave-one-country-out (§7.4)
[ ] Candidate recall measured per channel and for the union (§8)
[ ] Pairwise feature matrix built, including explicit contradiction
    features (§9, §9.1)
[ ] LightGBM baseline trained; every subsequent addition beats it on
    validation macro F0.5 or is dropped (§10)
[ ] Hard-negative mining run OOF-isolated to convergence (§11)
[ ] Calibration applied; threshold optimized against entity-level macro
    F0.5, not 0.5, not pairwise F1 (§12)
[ ] Singleton detection, margin analysis, and contradiction checks applied
    at decision time, not just as training features (§12)
[ ] Error taxonomy completed — retrieval vs matching vs decision-policy
    failures quantified, not guessed (§14)
[ ] Research questions (§15) answered from actual experiments, decision
    table filled from results, not intuition
[ ] MODEL_LICENSE_AUDIT.md complete for every shipped component (§16)
[ ] Full pipeline runs end-to-end via one command, no notebook dependency
[ ] validate_submission.py passes on final output
[ ] Reproducibility package assembled: output/, code/, filled
    Documentation_template.md
```
