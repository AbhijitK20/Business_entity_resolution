# 🕵️ COMPETITIVE INTELLIGENCE — Other Teams' Repos

**Date:** 25 Sep 2026 · **Sources:** 9 team repos cloned in `peoples prototype/` (gitignored)
**Purpose:** Extract verified facts + winning ideas. **Nothing here is copied code — it's intelligence.**

---

## 1. THE BIGGEST FIND: REAL DATASET AUDIT

Team `AcID3r/RF_AMAZON_2026` published a **full dataset audit** (`docs/analysis/dataset_profile.json` + `implementation.md`) — measured, with SHA-256 hashes. Team `amazon-ml-challenge-the-resolvers` independently confirms the counts.

### Actual scale (VERIFIED by two independent teams)

| File | Rows | US | India | France | Blank addresses |
|------|------:|-----:|-------:|-------:|----------------:|
| `train_source1.tsv` | **2,206,821** | 1,323,633 | 883,188 | 0 | 0 |
| `train_source2.tsv` | **5,034,616** | 3,016,817 | 2,017,799 | 0 | 168,967 (3.4%) |
| `train_source3.tsv` | **5,285,603** | 3,170,056 | 2,115,547 | 0 | 175,916 (3.3%) |
| `test_source1.tsv` | **1,732,544** | 663,106 | 809,986 | **259,452 (15%)** | 0 |
| `test_source2.tsv` | **4,887,273** | 1,871,330 | 2,312,565 | 703,378 | 129,408 (2.7%) |
| `test_source3.tsv` | **5,082,316** | 1,945,701 | 2,405,000 | 731,615 | 136,098 (2.7%) |
| `train_ground_truth.tsv` | 2,206,821 | — | — | — | — |

**Total: 24,229,173 source records · 2.52 GB TSVs.** Test gallery = 9,969,589 S2/S3 records.

### Ground truth distribution (MEASURED)

| Matches per S1 root | Roots | Share |
|--------------------:|------:|------:|
| 0 (singleton) | 123,247 | **5.58%** |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.06% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7 | 63,968 | 2.90% |
| 8 | 18,680 | 0.85% |
| 9 | 4,205 | 0.19% |
| 10–11 | 571 | 0.03% |

- **7,638,365 positive pairs** (3.69M → S2, 3.94M → S3)
- **89.0% of roots have MULTIPLE matches** — multi-match is the dominant case, not the exception
- Mean matches/root = 3.46 (3.67 among non-singletons)
- **All-empty baseline = 0.0558** (equals singleton fraction)

### 🔴 THE GAME-CHANGER FACT (both teams verified independently)

> **Every S2/S3 record matches AT MOST ONE S1 entity.** Across all 7,638,365 matched-ID occurrences: zero exceptions. No target is shared between roots.

**Why this matters enormously:** it means we can enforce a **global one-to-one assignment** as a final precision lever — if two S1 entities both claim the same S2 record, the lower-scored claim can be dropped. The resolvers team ranks this their **#1 highest-leverage move**. Our pipeline does NOT do this yet.

### Noise measured on a labeled sample (17,315 roots / 59,547 pairs)

| Pair slice | Same normalized name | Same normalized address | Both identical |
|------------|---------------------:|------------------------:|---------------:|
| US, S1–S2 | 19.5% | 11.7% | 2.1% |
| US, S1–S3 | 19.2% | 4.3% | 0.0% |
| India, S1–S2 | 10.1% | 9.6% | 0.9% |
| India, S1–S3 | 11.6% | 4.3% | 0.0% |

**Implications:**
- **~80% of true matches have DIFFERENT normalized names** — fuzzy/phonetic/embedding retrieval is mandatory
- **India S1–S2: 22.7% Indic-script mismatch in names** — cross-script handling is mandatory
- 0 non-ASCII training S1 names, but **764,608 non-ASCII training S2 names** — script diversity lives in the gallery
- Real examples: Telugu-script name vs Latin; `Quick Stock Private Limited` vs `Mirabelozeph`; `Legacy Keystone Gas LLC` vs `legacykeystonegas.com`

### Integrity facts
- No duplicate record IDs, no malformed rows, no blank names/countries
- Ground truth covers all training S1 exactly once
- **No labeled positive crosses countries** (supports country-partitioned blocking)
- 2,681,854 training gallery records are distractors absent from all truth lists (must be included in validation)
- Exact-text duplicates in S2/S3 exist under different IDs (25,873 train S2, 18,860 train S3) — **keep every ID**, don't dedupe text

---

## 2. SCALE MATH (why our current code must change)

| Quantity | Value |
|----------|-------|
| Full test cross-product | **17.27 trillion pairs** |
| Country-restricted cross-product | 6.72 trillion pairs |
| Target candidate budget per S1 | 40–100 (planning) → **69M–173M scored pairs** |
| 1024-dim FP16 embeddings for test gallery | **19.0 GiB** (raw vectors alone) |
| 64 float32 features × 69–173M pairs | **16.5–41.3 GiB** (raw values alone) |
| Feature TSVs on disk | several GB |

**Rules that follow:**
1. Never materialize dense (query × gallery) matrices → chunked sparse ops (`sparse_dot_topn` recommended by RF team)
2. Never per-row Python loops for features → vectorized/`process.cdist`
3. Store intermediates as Parquet, stream the final TSV writers
4. Cap candidates per S1 per source before scoring
5. Compute candidate-oracle ceiling after every filter

---

## 3. WHAT OTHER TEAMS ARE BUILDING

| Team | Approach | Notable detail |
|------|----------|----------------|
| **RF_AMAZON_2026** | Multi-channel lexical + multilingual embeddings (Qwen3-0.6B / BGE-M3) + LightGBM + Qwen3 reranker + calibration; 80/10/10 grouped splits; cloud multi-GPU | Most rigorous; full audit; licenses audited; entity-group splits; candidate oracle math |
| **the-resolvers** | 4-pass blocking (name tokens, phonetic, address/digits, MinHash) + LightGBM on 8 feature families + **global one-to-one assignment** | Ranked one-to-one assignment as #1 precision lever; explicitly skipping deep learning |
| **SIBAM890** | `Blocker` class: two-pass chunked inverted indexes (exact name, rare token, rare addr token, PIN) with 2 GB memory ceiling | Best example of memory-disciplined streaming code; PIN-code index for India |
| **Business-Entity-Resolution…** | Agent-spec style; evaluation.py with exact F_0.5 + worked-example unit test | Good test discipline; read-only dataset rule |
| Others (AmazonML_Challenge_2026, TensorTrek-ML, etc.) | Mostly scaffolding/READMEs | Not informative |

---

## 4. IDEAS WE SHOULD ADOPT (ranked by leverage)

### 🔴 P0 — likely score-critical

1. **Global one-to-one assignment** (resolvers #1): after thresholding, sort all accepted (S1, cand) pairs by score desc; when a cand is claimed, drop it from other S1s. Verified-safe by the no-shared-target fact.
2. **Candidate caps per S1 per source** (40–100 target): controls compute AND improves precision; measure recall-ceiling-vs-cap curve before freezing.
3. **Cross-script handling**: Unicode NFKC + script detection; keep Indic/Arabic text (don't ASCII-fold away); add script-mismatch feature; lean on address/numeric when names are cross-script.
4. **Entity-grouped splits**: group S1 + its targets; 80/10/10 fit/calibration/locked-holdout; stratify by country + match-count buckets `{0,1,2,3-4,5+}`.
5. **Candidate oracle metric**: `Oracle_i = 1 if t_i==0 else 5*r_i/(4*r_i + t_i)`; report after every filter. This is the blocking ceiling in F_0.5 units.
6. **Exact entity-level scorer** with the 4 empty/nonempty cases + worked-example unit test (we have this — keep).

### 🟡 P1 — high value

7. **Rare-token + PIN-code indexes**: token frequency from data; PIN = `\b\d{5,6}\b` (India), ZIP = 5 digits (US). High-precision blocking signals.
8. **Mutual-nearest-neighbor feature**: is this candidate S1's top pick AND is S1 the candidate's top pick.
9. **Address-missing explicit feature** (we have this now via missingness features) — candidate-side blanks are ~3%.
10. **Blocking provenance features** (which passes surfaced the candidate + block cardinality) — resolvers use these as model features.
11. **Calibration** (isotonic/Platt) before thresholding + assignment.
12. **Country-holdout stress test** (train US→eval India and vice versa) as a France-generalization proxy.
13. **Sample negatives from actual retrieval candidates** — "every other candidate in that S1's blocked set" (resolvers) — this makes train match inference distribution.

### 🟢 P2 — only with ablation evidence

14. Multilingual dense retrieval: `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0) or `BAAI/bge-m3` (MIT); reranker `Qwen/Qwen3-Reranker-0.6B` (Apache-2.0). All ≤0.6B — license-safe. Only if lexical ceiling is insufficient.
15. Supervised contrastive fine-tune of dual encoder (RF plan §8.4).
16. Distillation to smaller model.

---

## 5. THINGS WE GOT WRONG / MUST FIX

| Our assumption | Reality | Fix |
|----------------|---------|-----|
| ~30% singletons | **5.6%** singletons; 89% multi-match | Fix synthetic generator; adjust expectations; don't over-tune for singletons |
| "One-to-many is occasional" | Multi-match is the **dominant** case | Never top-1; assignment logic must preserve multi-match |
| English/Latin text only | **764K non-ASCII S2 names**, 22.7% India cross-script | Script-aware normalization; keep Unicode; add script features |
| Feature computation row-wise OK | 69–173M scored pairs | Vectorize; chunk; Parquet |
| No fan-out → S1-level split fine | Confirmed no shared targets | S1-level split OK, but still group by identity (targets belong to one root) |
| Synthetic scale (300) representative | Real scale is 7000× larger | Scale-test with ≥50K roots before full run |
| candidate_pairs uncapped | Budget 40–100 per root | Add top-K caps + oracle curve |

---

## 6. WHAT WE DO BETTER / KEEP

- ✅ Leak-free OOF stacking already built (most teams haven't done this)
- ✅ Official validator already wired in
- ✅ Synthetic generator + evaluator + smoke tests (adapt to real distribution)
- ✅ 35 features incl. missingness + contradiction (comparable to the best teams)
- ✅ Two-view normalization idea already in gap analysis (extend to multi-view + script-aware)

---

## 7. IMMEDIATE ACTION LIST (updated with SABER learnings)

| # | Action | Owner | Priority |
|---|--------|-------|----------|
| 1 | **Indic → Latin transliteration** in normalization (ITRANS + schwa deletion) — fixes 22.7% India cross-script pairs | Vishwesh | 🔴 P0 |
| 2 | **Global one-to-one exclusivity** in decision (each S2/S3 → best S1 only; margin feature) | Abhijit | 🔴 P0 |
| 3 | **Candidate caps**: top-K per source per S1 (start K=20), measure recall-ceiling-vs-K curve | Vishwesh | 🔴 P0 |
| 4 | **Expected-F0.5 set selection per S1** (replaces plain threshold; includes singleton probability) | Abhijit | 🔴 P0 |
| 5 | **Vectorize features** (chunked, no row loops); store Parquet intermediates | Abhijit | 🔴 P0 |
| 6 | **Reuse blocker similarity + bidirectional ranks as features** | Vishwesh + Abhijit | 🔴 P0 |
| 7 | **Fix synthetic generator** to real distribution (89% multi, 5.6% singleton, cross-script, blank gallery addresses) | Karan | 🟡 P1 |
| 8 | **Entity-grouped split + candidate-oracle F0.5** in evaluator | Abhijit | 🟡 P1 |
| 9 | **Calibration** (isotonic) before decision | Abhijit | 🟡 P1 |
| 10 | **Country-holdout stress test** (US↔India) to set France margins | Karan | 🟡 P1 |
| 11 | **Rare-token + PIN/ZIP indexes** | Vishwesh | 🟡 P1 |
| 12 | **Dense retrieval leg** with license-safe encoder (arctic-embed-xs / multilingual-e5-small), fine-tuned contrastively with sorted-window hard negatives | Abhijit | 🟢 P2 (only if lexical ceiling insufficient) |
| 13 | **Scale benchmark** on 50K roots before full run | All | 🔴 P0 |

---

## 8. SABER (yogeshwars-cys) — MEASURED SOTA-LEVEL RESULTS

The most advanced public implementation. Their measured numbers (2.5% held-out world: 55K queries vs 258K pool, 191K true pairs):

| Recall@20, dense leg | US | India |
|----------------------|----:|------:|
| Stock MiniLM-L6 | 0.948 | 0.835 |
| Best zero-shot (arctic-embed-xs) | 0.990 | 0.929 |
| Best zero-shot (multilingual-e5-small) | 0.988 | **0.946** |
| **Fine-tuned arctic-embed-xs** | **0.9964** | **0.9924** |
| Char-trigram TF-IDF (reference) | 0.996 | 0.975 |

**Full hybrid blocker (stock encoder): US 0.9978 @ 24.8 cand/S1 · India 0.9903 @ 29.1 cand/S1**

### Their key techniques (all measured, all worth stealing)

1. **Indic → Latin transliteration**: ITRANS mapping + word-final schwa deletion. `राम मार्केटिंग` → `ram marketing`. Covers Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam. **This is the fix for the 22.7% India cross-script problem.**
2. **Hybrid blocker legs**: fine-tuned encoder (binary sign codes + INT8 rescore) + char-trigram TF-IDF (chunked sparse×dense) + exact keys; **every leg runs in both directions**; region-partitioned trigram; adaptive K merge.
3. **Contrastive fine-tuning**: symmetric InfoNCE (scale 30), Matryoshka 192/full-dim, anchor = `name, address`, positive resampled each epoch, **hard negatives from (country, name)-sorted contiguous windows** (same name, different address — the hard case since **47% of S1 names are shared**).
4. **Throughput reality check**: 6-layer 384-d models encode **~5,800 records/s in fp16 on an RTX 3050 (6 GB laptop GPU)** → 24M records ≈ 70 min. Normalization: 24.2M records ≈ 18 min on 10 cores. **Our RTX 4050 can do this.**
5. **Selector design** (their matching stage):
   - LightGBM on ~30 features: leg evidence (cosines, bidirectional ranks, leg flags), string similarity, structure (house number, number-token Jaccard, region overlap, source), competition context (candidate counts, within-S1/within-record ranks)
   - **Exclusivity**: keep only the highest-probability S1 per candidate record; margin `p(best) − p(second)` as a feature
   - **Expected-F0.5 set selection per S1**: sort candidates by calibrated probability; for each cutoff k=0..n estimate expected F0.5 of predicting top-k (Monte-Carlo or DP); pick argmax. **Optimizes the metric directly instead of one global threshold.**
   - **Singleton probability**: `∏(1−p_j)` blended with a separate singleton classifier on S1-level features
   - **Group consistency**: mean similarity to S1's other high-prob candidates (S2↔S3 agreement), reciprocal-best-match flag, per-S1 S2/S3 count priors
   - Country NOT used as a feature; France gets stricter cutoffs (no supervision)
6. **Validation discipline**: exact macro F0.5 on held-out S1s against **full-density** candidates; report **oracle ceiling vs achieved**; breakdowns per country + match-count bucket; country-holdout transfer (US↔India) to set France's cutoff margin.

### Model choices that are license-safe and laptop-runnable

| Model | HF id | License | Params | Dim | Notes |
|-------|-------|---------|-------:|----:|-------|
| arctic-embed-xs | `Snowflake/snowflake-arctic-embed-xs` | Apache-2.0 | 22.6M | 384 | Best US zero-shot; fastest; fine-tunes to 0.996/0.992 |
| multilingual-e5-small | `intfloat/multilingual-e5-small` | MIT | 117.7M | 384 | Best India zero-shot (0.946) |
| e5-small-v2 | `intfloat/e5-small-v2` | MIT | 33.4M | 384 | Strong all-round |
| granite-30m | `ibm-granite/granite-embedding-30m-english` | Apache-2.0 | 30.3M | 384 | Small alternative |

**Do not use**: `bge-m3` (568M — too slow: >10h on 6GB GPU), `gte-multilingual-base` (crashed under transformers 5.x), Jina variants (some CC-BY-NC).

---

## 9. vaibhav05-cloud — SIMPLE CLEAN BASELINE (reference for our v1)

Minimal 5-file pipeline, good as a sanity reference:

1. **Blocking**: country partition (dynamic values, open-set safe) → char_wb 3–5 gram TF-IDF → `NearestNeighbors(cosine)` **top-20 per source per S1**, `min_sim=0.15` → `candidate_pairs.tsv`
2. **Features (13)**: blocking_similarity (reused!), name Levenshtein/token-sort/partial/Jaccard, addr Levenshtein/token-sort/Jaccard, **addr_digit_overlap**, country_match, name/addr length diffs
3. **Model**: LightGBM
4. Same feature function at train and predict time (no drift)

**Takeaway:** this is roughly what our v1 should look like at minimum — but with candidate caps (top-20/source), digit-overlap, and the blocker similarity reused as a feature (which we don't currently do).

---

## 10. CONVERGED BEST PRACTICE (what ALL top teams agree on)

| # | Consensus | Our status |
|---|-----------|------------|
| 1 | Blocking must reach ≥99% pair recall at 20–30 cand/S1 | ⚠️ We have no caps; recall measured but not at budget |
| 2 | Cross-script handling (Indic transliteration) is mandatory | ❌ Missing — we ASCII-fold, which destroys Indic |
| 3 | LightGBM on 15–30 pair features is the matcher | ✅ We have 35 features + ensemble |
| 4 | **Entity-level set decision, not per-pair threshold** | ⚠️ We have macro threshold; no per-entity expected-F0.5 selection |
| 5 | **One-to-one exclusivity** (each S2/S3 → ≤1 S1) | ❌ Missing — big precision lever |
| 6 | Calibrate probabilities before deciding | ❌ Missing |
| 7 | Country open-set, never a hard filter | ✅ We never hard-filter (but do use same_country feature — fine) |
| 8 | Grouped entity splits + country-holdout proxy for France | ⚠️ We split by S1 only (fine given no fan-out, but no country-holdout test) |
| 9 | Report oracle ceiling vs achieved separately | ⚠️ We measure blocking recall; no oracle F0.5 |
| 10 | Chunked/streamed processing; Parquet intermediates | ❌ Our feature loop is row-wise Python |
| 11 | Blocker similarity reused as a model feature | ❌ We discard blocking scores |
| 12 | Bidirectional retrieval (S1→gallery and gallery→S1) | ❌ One-directional |

---

## 11. REPO INVENTORY (`peoples prototype/`, gitignored)

| Repo | Value | Contents |
|------|-------|----------|
| `SABER` (yogeshwars-cys) | ⭐⭐⭐⭐⭐ | Measured encoder screen + fine-tune results + hybrid blocker + selector design + real code |
| `RF_AMAZON_2026` | ⭐⭐⭐⭐⭐ | Full dataset audit + 14-section implementation plan |
| `amazon-ml-challenge-the-resolvers` | ⭐⭐⭐⭐ | Verified facts + one-to-one assignment + PIN-code blocking |
| `Business-Entity-Resolution-Amazon-Ml-Challenge-2026` | ⭐⭐⭐ | Chunked Blocker class + F_0.5 evaluator |
| `vaibhav-solution` | ⭐⭐⭐ | Clean minimal 5-file baseline (13 features, top-20 blocking) |
| `SIBAM890/Business-Entity-Resolution-…` | ⭐⭐⭐ | Memory-disciplined two-pass indexing |
| Others (10 repos) | ⭐ | Scaffolding only |

