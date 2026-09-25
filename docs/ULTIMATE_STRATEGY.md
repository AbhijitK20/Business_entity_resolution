# Ultimate Entity Resolution Strategy — 26 Repos Synthesized

## The Golden Rule: Quality > Quantity

From 26 repos, the biggest wins come from:
1. **Perfect blocking** (recall ceiling)
2. **Rich features** (20+ per pair)
3. **Hard negatives** (not more data)
4. **Ensemble stacking** (combine models)
5. **Precision-heavy threshold** (F_0.5 = 2× precision weight)

---

## BLOCKING STRATEGY (7-Layer Union)

From: BlockingPy, StringMatcher, goldenmatch, canonmap, UBS-ER, LeadDeDupe, armory

| Layer | Strategy | What it Catches | Implementation |
|-------|----------|-----------------|----------------|
| 1 | **Normalized name** (strip legal suffixes) | "Corp" vs "Corporation" | `re.sub(suffixes, '', name.lower())` |
| 2 | **Token-sorted TF-IDF** | Word transpositions | Sort tokens → TF-IDF cosine > 0.3 |
| 3 | **Soundex/Metaphone** | Transliterations | `jellyfish.soundex()` → exact match bucket |
| 4 | **Initialism** | "IBM" ↔ "International Business Machines" | First letters joined uppercase |
| 5 | **TF-IDF on addresses** | Address variants | TF-IDF char 3-4 grams, cosine > 0.3 |
| 6 | **Country partition** | Reduces search space | Only same-country records compared |
| 7 | **MinHash LSH** | Catch-all fuzzy | Character 3-grams, threshold 0.4, 128 permutations |

**Key insight from canonmap:** Initialism matching is BIDIRECTIONAL — "IBM" (2-6 chars, alpha only) is treated AS an initialism, so it matches "International Business Machines" and vice versa.

**Key insight from StringMatcher:** Lower LSH threshold (0.4) = more candidates = higher recall. Post-filter with scoring threshold instead.

---

## TEXT NORMALIZATION (10-Step Pipeline)

From: due-diligence-agents, business-record-matcher, UBS-ER, LeadDeDupe, goldenmatch

```python
def normalize_name(name):
    if pd.isna(name): return ""
    name = str(name)
    
    # Step 1: Unicode normalization (handle accented chars)
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    
    # Step 2: Lowercase
    name = name.lower()
    
    # Step 3: Remove parenthesized text
    name = re.sub(r"\([^)]*\)", " ", name)
    
    # Step 4: Strip legal suffixes (18 patterns, iterative up to 3x)
    suffixes = [
        r"\bincorporated\b", r"\binc\.?\b", r"\bcorporation\b",
        r"\bcorp\.?\b", r"\bcompany\b", r"\bco\.?\b", r"\blimited\b",
        r"\bltd\.?\b", r"\bllc\b", r"\bllp\b", r"\blp\b", r"\bplc\b",
        r"\bgmbh\b", r"\bag\b", r"\bsa\b", r"\bgroup\b", r"\bholdings\b",
        r"\benterprises?\b", r"\bassociates?\b", r"\bpartners?\b",
        r"\bthe\b", r"\band\b",
    ]
    for _ in range(3):  # Iterative to handle stacked suffixes
        for suffix in suffixes:
            name = re.sub(suffix, " ", name)
    
    # Step 5: Expand abbreviations
    abbrevs = {
        r"&": "and", r"\bintl\b": "international", r"\bmfg\b": "manufacturing",
        r"\bsvcs?\b": "services", r"\btechs?\b": "technology",
        r"\bmgmt\b": "management", r"\bgrp\b": "group",
        r"\bassocs?\b": "associates", r"\bsys\b": "systems",
    }
    for pattern, replacement in abbrevs.items():
        name = re.sub(pattern, replacement, name)
    
    # Step 6: Replace punctuation with spaces
    for ch in "&'/,.":
        name = name.replace(ch, " ")
    
    # Step 7: Strip non-alphanumeric
    name = re.sub(r"[^\w\s]", " ", name)
    
    # Step 8: Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    
    # Step 9: Remove digit-only words (for names)
    name = re.sub(r"\w*\d\w*", "", name)
    
    return name

def normalize_address(addr):
    if pd.isna(addr): return ""
    addr = str(addr).lower()
    
    # Expand street abbreviations
    street_abbrevs = {
        r"\bstreet\b": "st", r"\bavenue\b": "ave", r"\bboulevard\b": "blvd",
        r"\bdrive\b": "dr", r"\broad\b": "rd", r"\blane\b": "ln",
        r"\bcourt\b": "ct", r"\bparkway\b": "pkwy", r"\bplace\b": "pl",
        r"\bnorth\b": "n", r"\bsouth\b": "s", r"\beast\b": "e", r"\bwest\b": "w",
    }
    for pattern, replacement in street_abbrevs.items():
        addr = re.sub(pattern, replacement, addr)
    
    addr = re.sub(r"[^\w\s]", " ", addr)
    addr = re.sub(r"\s+", " ", addr).strip()
    return addr
```

---

## PAIRWISE FEATURES (25 Features)

From: ted-entity-resolution, entity-deduplication, name-matching, StringMatcher, UBS-ER, canonmap

### Name Features (10)
| # | Feature | Source | What it Captures |
|---|---------|--------|------------------|
| 1 | `name_token_sort_ratio` | RapidFuzz | Order-invariant comparison |
| 2 | `name_partial_ratio` | RapidFuzz | Best substring match |
| 3 | `name_WRatio` | RapidFuzz | Weighted multi-algorithm |
| 4 | `name_jaro_winkler` | jellyfish | Prefix-weighted edit distance |
| 5 | `name_jaccard` | Manual | Token-level set overlap |
| 6 | `name_edit_ratio` | Manual | 1 - (edit_dist / max_len) |
| 7 | `name_trigram_jaccard` | Manual | Character 3-gram overlap |
| 8 | `name_soundex_match` | jellyfish | Phonetic match (binary) |
| 9 | `name_metaphone_match` | jellyfish | Phonetic match (binary) |
| 10 | `name_length_ratio` | Manual | min/max length ratio |

### Address Features (6)
| # | Feature | Source | What it Captures |
|---|---------|--------|------------------|
| 11 | `addr_token_sort_ratio` | RapidFuzz | Order-invariant comparison |
| 12 | `addr_partial_ratio` | RapidFuzz | Best substring match |
| 13 | `addr_WRatio` | RapidFuzz | Weighted multi-algorithm |
| 14 | `addr_jaccard` | Manual | Token-level set overlap |
| 15 | `addr_trigram_jaccard` | Manual | Character 3-gram overlap |
| 16 | `addr_length_ratio` | Manual | min/max length ratio |

### Country Feature (1)
| # | Feature | Source | What it Captures |
|---|---------|--------|------------------|
| 17 | `same_country` | Binary | Must match for valid pair |

### Cross Features (8)
| # | Feature | Source | What it Captures |
|---|---------|--------|------------------|
| 18 | `name_addr_WRatio_avg` | Combined | Average similarity |
| 19 | `name_addr_WRatio_max` | Combined | Best of both fields |
| 20 | `name_addr_WRatio_min` | Combined | Worst of both fields |
| 21 | `name_addr_jaccard_avg` | Combined | Average token overlap |
| 22 | `is_company` | UBS-ER | Binary: has legal suffix |
| 23 | `name_phonetic_vote` | UBS-ER | 2/3 phonetic encodings match |
| 24 | `surname_length_diff` | Manual | Surname length difference |
| 25 | `combined_trigram` | Manual | Char 3-grams on concatenated name+addr |

**Key insight from UBS-ER:** Phonetic voting — if 2 of 3 encodings (Soundex, Metaphone, NYSIIS) match, that's strong evidence even when spelling differs.

**Key insight from StringMatcher:** `partial_token_sort_ratio` and `partial_token_set_ratio` are valuable additional features that most pipelines miss.

---

## MODEL ARCHITECTURE (Dual-Layer Stacking)

From: MetaBoost, imbalance-benchmark, entity-deduplication

```
Layer 0: Base Models (3)
┌─────────────────────────────────────────────────────────┐
│  LightGBM          XGBoost           CatBoost/RF        │
│  (histogram)       (histogram)       (diversity)        │
│                                                                 │
│  class_weight      scale_pos_weight  class_weight        │
│  = balanced        = neg/pos         = balanced          │
└──────────┬──────────────┬──────────────┬────────────────┘
           │              │              │
           ▼              ▼              ▼
    OOF predictions (3-fold StratifiedKFold)
    Each model predicts on held-out fold
    → 3 columns of probabilities

Layer 1: Meta-Learner
┌─────────────────────────────────────────────────────────┐
│  Input: [lgb_oof, xgb_oof, rf_oof]                      │
│                                                                 │
│  LightGBM (shallow)  OR  Ridge Regression                 │
│  max_depth: 2-5            alpha: 0.01-100               │
│  num_leaves: 4-16                                          │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
    Final probability (calibrated)

Threshold: F_0.5 optimized on validation set
```

**Key insight from MetaBoost:** The meta-learner should be DELIBERATELY SHALLOW (depth 2-5, leaves 4-16) to avoid overfitting on the small meta-feature space.

**Key insight from imbalance-benchmark:** SMOTE hurts when class weighting is already used — they're redundant. Use ONE mechanism (class_weight="balanced").

---

## HARD NEGATIVE MINING (The Secret Weapon)

From: name-matching, ted-entity-resolution, LLM4ER, adaptive-HNM, negminer, sentence-transformers

### Strategy: 70/30 Hard/Random Split

```python
def generate_hard_negatives(s1_entities, s2_s3_pool, ground_truth, n_neg=2):
    negatives = []
    for s1_id, s1_record in s1_entities.items():
        positives = set(ground_truth[s1_id])
        
        # 70% HARD: same country, similar name, NOT in positives
        same_country = s2_s3_pool[s2_s3_pool['country'] == s1_record['country']]
        sims = []
        for _, cand in same_country.iterrows():
            if cand['entity_id'] not in positives:
                sim = fuzz.WRatio(s1_record['name_clean'], cand['name_clean'])
                sims.append((cand['entity_id'], sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        hard_count = int(n_neg * 0.7)
        for cand_id, _ in sims[:hard_count]:
            negatives.append({'s1': s1_id, 's2': cand_id, 'label': 0})
        
        # 30% RANDOM: different country or very different name
        random_count = n_neg - hard_count
        diff_country = s2_s3_pool[s2_s3_pool['country'] != s1_record['country']]
        if len(diff_country) >= random_count:
            sampled = diff_country.sample(random_count)
            for _, cand in sampled.iterrows():
                negatives.append({'s1': s1_id, 's2': cand['entity_id'], 'label': 0})
    
    return negatives
```

**Key insight from LLM4ER:** Hard negatives are MORE important than more positive data. The paper shows that standard benchmarks with easy negatives overestimate performance by 10-20%.

**Key insight from adaptive-HNM:** Curriculum-based training — start with easy negatives (random), gradually increase to hard negatives as the model learns.

---

## THRESHOLD OPTIMIZATION FOR F_0.5

From: ted-entity-resolution, imbalance-benchmark

```python
def find_best_f05_threshold(y_true, y_proba):
    """
    F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    Weights precision 2x over recall.
    """
    best_thresh, best_f05 = 0.5, 0
    for thresh in np.arange(0.1, 0.95, 0.01):
        preds = (y_proba >= thresh).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f05 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0
        if f05 > best_f05:
            best_f05, best_thresh = f05, thresh
    return best_thresh, best_f05
```

**Key insight from imbalance-benchmark:** Scan in 0.01 steps, not 0.05. The difference between 0.63 and 0.67 can be significant for F_0.5.

**Key insight from due-diligence-agents:** Length-based thresholds — shorter names need HIGHER thresholds (95% for 6-8 chars) to avoid false positives.

---

## SINGLETON HANDLING (Free Points)

From: problem statement, name-matching, sandx-er

- Every S1 entity MUST appear in output
- If no candidates → empty match list
- If no candidates score above threshold → empty match list
- Correctly predicting "no match" = 1.0 score
- False merges on singletons = 0.0 score

**Conservative threshold is key.** Better to miss a true match (recall penalty) than to create a false merge (precision penalty with 2× weight).

---

## PIPELINE ARCHITECTURE

```
Input TSVs
    │
    ▼
[1] Normalize all names/addresses (10-step pipeline)
    │
    ▼
[2] Build blocking index (7-layer union)
    │  ├── Normalized name → TF-IDF
    │  ├── Token-sorted TF-IDF
    │  ├── Soundex/Metaphone bucket
    │  ├── Initialism matching
    │  ├── Address TF-IDF
    │  ├── Country partition
    │  └── MinHash LSH (threshold 0.4)
    │
    ▼
[3] Generate candidate pairs (union of all blocking)
    │
    ▼
[4] Compute 25 pairwise features per candidate
    │  ├── 10 name features
    │  ├── 6 address features
    │  ├── 1 country feature
    │  └── 8 cross features
    │
    ▼
[5] Train: 80/20 split
    │  ├── Positive pairs from ground truth
    │  ├── Hard negatives (70%) + random negatives (30%)
    │  └── Feature matrix + labels
    │
    ▼
[6] Model: Dual-layer stacking
    │  ├── Layer 0: LightGBM + XGBoost + RandomForest
    │  ├── OOF predictions from 3-fold CV
    │  └── Layer 1: Shallow LightGBM meta-learner
    │
    ▼
[7] Threshold: F_0.5 optimization on validation
    │  └── Scan 0.1 to 0.95 in 0.01 steps
    │
    ▼
[8] Inference: Score all test candidates
    │  ├── Apply threshold → match/no-match
    │  ├── Handle singletons (empty list)
    │  └── Deduplicate within each match list
    │
    ▼
[9] Output: matching_results.tsv + candidate_pairs.tsv
    │
    ▼
[10] Validate: python3 utils/validate_submission.py
```

---

## EXPECTED PERFORMANCE GAINS

| Technique | Expected F_0.5 Gain | Confidence |
|-----------|---------------------|------------|
| 7-layer blocking (vs 3-layer) | +5-10% recall | High |
| 25 features (vs 8) | +3-5% F_0.5 | High |
| Hard negatives (vs random) | +5-8% F_0.5 | High |
| Ensemble stacking (vs single LGB) | +2-3% F_0.5 | Medium |
| F_0.5 threshold optimization | +1-2% F_0.5 | High |
| 10-step normalization (vs basic) | +2-3% F_0.5 | Medium |
| **Total estimated improvement** | **+18-31%** | |

---

## LIBRARIES NEEDED

```
# Already installed
rapidfuzz==3.14.5
torch==2.13.0+cpu (need CUDA version)
transformers==5.14.1
numpy==2.3.5

# Need to install
lightgbm
scikit-learn
pandas
jellyfish
optuna
shap
sentence-transformers
faiss-cpu
xgboost
catboost  (optional, for diversity in ensemble)
```
