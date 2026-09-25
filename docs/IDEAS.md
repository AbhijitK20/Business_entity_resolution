# Entity Resolution Strategy — Synthesized from 15 Repos

## Executive Summary

After analyzing 15 GitHub repos, here are the best ideas for building a high-F_0.5 entity resolution pipeline for the Amazon ML Challenge.

---

## Key Insight: Blocking = Recall Ceiling

> "Blocking recall is a hard ceiling on fusion recall: any true match that no blocking key puts in a shared block can never be matched downstream, at any threshold, by any model."

**Investment priority: Blocking > Features > Model**

---

## Phase 1: Multi-Signal Blocking (Union Strategy)

From: LeadDeDupe, goldenmatch, BlockingPy, armory, sandx-er

### Blocking Keys to Implement

| # | Key | Source | What it catches |
|---|-----|--------|----------------|
| 1 | **Normalized name** (strip legal suffixes, lowercase, collapse whitespace) | armory, goldenmatch | Abbreviations: "Corp" vs "Corporation" |
| 2 | **Token sort** (sorted name tokens) | goldenmatch | Word transpositions: "ABC Corp" ↔ "Corp ABC" |
| 3 | **Soundex/Metaphone on name** | LeadDeDupe, goldenmatch | Transliteration variants |
| 4 | **Address prefix** (first 4-5 chars) | armory | Address typos |
| 5 | **Country partition** | Problem statement | Reduces search space, enforces country match |
| 6 | **TF-IDF cosine > 0.3 on names** | name-matching | Semantic similarity |
| 7 | **TF-IDF cosine > 0.3 on addresses** | name-matching | Address similarity |
| 8 | **Dense blocking** (sentence embeddings + FAISS ANN) | BlockingPy, serf, entity-embed | Catches semantic variants |

### Blocking Evaluation (from armory)

Measure before proceeding to matching:
```
reduction_ratio = 1 - (candidate_pairs / total_pairs)
pair_recall = matches_retained / labeled_matches  # THE CEILING
```

Target: **pair_recall >= 0.95** with **reduction_ratio >= 0.99**

---

## Phase 2: Text Normalization

From: LeadDeDupe, name-matching, goldenmatch

### Name Cleaning Pipeline
```python
def normalize_name(name):
    name = name.lower()
    name = re.sub(r'[^\w\s]', '', name)          # Remove punctuation
    name = re.sub(r'\b(inc|corp|corporation|llc|ltd|limited|pvt|private|co|company|group|holdings|gmbh|plc|sa|sons)\b', '', name)  # Remove legal suffixes
    name = re.sub(r'\s+', ' ', name).strip()      # Collapse whitespace
    return name
```

### Address Cleaning Pipeline
```python
def normalize_address(addr):
    addr = addr.lower()
    abbrevs = {'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
               'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place',
               'cir': 'circle', 'pkwy': 'parkway', 'ste': 'suite'}
    for abbr, full in abbrevs.items():
        addr = re.sub(rf'\b{abbr}\b', full, addr)
    addr = re.sub(r'[^\w\s]', '', addr)
    addr = re.sub(r'\s+', ' ', addr).strip()
    return addr
```

---

## Phase 3: Pairwise Feature Engineering

From: ted-entity-resolution (17 features), entity-deduplication (19 features), name-matching (8 features)

### Feature Set (25+ features)

#### Name Features (7)
| # | Feature | Implementation | Source |
|---|---------|---------------|--------|
| 1 | `name_token_sort_ratio` | `rapidfuzz.fuzz.token_sort_ratio / 100` | name-matching |
| 2 | `name_partial_ratio` | `rapidfuzz.fuzz.partial_ratio / 100` | name-matching |
| 3 | `name_WRatio` | `rapidfuzz.fuzz.WRatio / 100` | ted-entity-resolution |
| 4 | `name_jaro_winkler` | `rapidfuzz.distance.JaroWinkler.similarity` | entity-deduplication |
| 5 | `name_jaccard` | Token-level Jaccard | ted-entity-resolution |
| 6 | `name_tfidf_cosine` | TF-IDF bigram cosine similarity | name-matching |
| 7 | `name_edit_ratio` | `1 - edit_dist / max(len_a, len_b)` | name-matching |

#### Address Features (5)
| # | Feature | Implementation |
|---|---------|---------------|
| 8 | `addr_token_sort_ratio` | `rapidfuzz.fuzz.token_sort_ratio / 100` |
| 9 | `addr_partial_ratio` | `rapidfuzz.fuzz.partial_ratio / 100` |
| 10 | `addr_WRatio` | `rapidfuzz.fuzz.WRatio / 100` |
| 11 | `addr_jaccard` | Token-level Jaccard |
| 12 | `addr_tfidf_cosine` | TF-IDF cosine similarity |

#### Country Feature (1)
| # | Feature | Implementation |
|---|---------|---------------|
| 13 | `same_country` | Binary: 1.0 if both present and equal, else 0.0 |

#### Cross Features (3)
| # | Feature | Implementation |
|---|---------|---------------|
| 14 | `name_length_ratio` | `min(len_a, len_b) / max(len_a, len_b)` |
| 15 | `addr_length_ratio` | `min(len_a, len_b) / max(len_a, len_b)` |
| 16 | `name_addr_complement` | `(name_score + addr_score) / 2` |

#### Missing Data Handling (from ted-entity-resolution)
- Missing-missing = 0.0 (NOT 1.0) — prevents missing data from being evidence of match
- Missing-present = 0.0

---

## Phase 4: Training Strategy

From: ted-entity-resolution, entity-deduplication, name-matching

### Positive Pair Generation
1. Use ground truth from training data
2. Augment with programmatic variants:
   - Typo injection (single-char swaps, keyboard proximity)
   - Legal suffix variations (Corp ↔ Corporation)
   - Word transpositions
   - Abbreviation expansion/contraction

### Negative Pair Generation (Hard Negatives)
From: name-matching, ted-entity-resolution
1. **70% hard negatives**: Same country, similar name but different entity
2. **30% random negatives**: Different country or very different name
3. Negative ratio: 1:1 to 1:3 (positive:negative)

### LightGBM Configuration
From: ted-entity-resolution (conservative), name-matching
```python
params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "class_weight": "balanced",
    "num_leaves": 16-63,         # Conservative
    "max_depth": 3-8,            # Shallow
    "learning_rate": 0.01-0.08,  # Slow
    "n_estimators": 200-500,
    "min_child_samples": 80-200, # Prevents overfitting
    "subsample": 0.7-1.0,
    "colsample_bytree": 0.7-1.0,
    "reg_lambda": 1.0-10.0,
    "force_col_wise": True,
}
```

### Threshold Optimization for F_0.5
From: ted-entity-resolution
```python
def find_best_f05_threshold(y_true, proba):
    best_thresh, best_f05 = 0.5, 0
    for thresh in np.arange(0.1, 0.95, 0.05):
        preds = (proba >= thresh).astype(int)
        p = precision_score(y_true, preds)
        r = recall_score(y_true, preds)
        f05 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0
        if f05 > best_f05:
            best_f05, best_thresh = f05, thresh
    return best_thresh
```

**Key:** F_0.5 rewards precision 2× over recall. Threshold should be HIGHER than for F_1.

---

## Phase 5: Post-Processing

From: name-matching, sandx-er, goldenmatch

### Transitive Closure with Caution
- **Connected Components** (Union-Find): Fast, may over-merge
- **Correlation Clustering** (Kwik-Cluster): Slower, corrects transitivity errors

**Recommendation:** Use Connected Components first, then check for suspicious merges.

### Singleton Handling
- If no candidate scores above threshold → empty match list (worth 1.0)
- Correctly predicting "no match" is FULL credit

### Validation
From: ted-entity-resolution
- 80/20 stratified split
- Bootstrap confidence intervals (500 iterations)
- SHAP feature importance for debugging

---

## Architecture Decision: What to Build

### Option A: LightGBM Pipeline (Recommended)
- Blocking → Feature Engineering → LightGBM → Threshold → Output
- **Pros:** Fast, interpretable, strong baseline, CPU-only
- **Cons:** Needs good features

### Option B: Embedding + LightGBM Hybrid
- Dense blocking (sentence embeddings) + LightGBM matching
- **Pros:** Better blocking recall, semantic understanding
- **Cons:** Slower, needs GPU for embeddings

### Option C: Pure Neural (Not Recommended)
- Sentence-transformers for everything
- **Pros:** End-to-end
- **Cons:** May not beat LightGBM on structured features, slower

**Recommendation: Option A with embedding-based blocking as enhancement**

---

## Key Libraries to Install

```
lightgbm          # Classifier
rapidfuzz         # String similarity (Rust-backed, fast)
scikit-learn      # TF-IDF, evaluation metrics
pandas            # Data manipulation
numpy             # Numerical ops
jellyfish         # Soundex/Metaphone
sentence-transformers  # Embeddings (optional)
faiss-cpu         # ANN search (optional)
optuna            # Hyperparameter tuning
shap              # Feature importance
```

---

## Critical Success Factors

1. **Blocking recall >= 0.95** — measure this FIRST
2. **Hard negative mining** — don't just use random negatives
3. **F_0.5 threshold optimization** — tune directly, not via F_1
4. **Singleton accuracy** — correctly predicting "no match" = 1.0
5. **Country as open set** — don't hard-code US/India/France
6. **Every S1 entity must appear** — missing = rejection
7. **No external data** — only provided training data

---

## Benchmark Comparison (from repos)

| Approach | F1 on standard datasets |
|----------|------------------------|
| LightGBM + 17 features (ted) | 0.9755 |
| MLP + 19 features (entity-dedup) | 0.973 |
| LSH + Jaccard (sandx-er, Febrl4) | 0.977 |
| SNM + Jaccard (sandx-er, DBLP-ACM) | 0.927 |
| GoldenMatch auto-config (DBLP-ACM) | 0.972 |

**Target for our pipeline: F_0.5 >= 0.85 on validation** (F_0.5 is harder than F_1 due to precision emphasis)
