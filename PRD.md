# Amazon ML Challenge 2026 — Business Entity Resolution

## Team PRD (Product Requirements Document)

---

## 1. Problem Statement

**Goal:** Determine which business records, arriving from **three independent sources** with **no shared identifier**, describe the **same real-world business**.

**Real-world context:** When businesses sign up on Amazon Business, we capture their **name** and **address**. The same business may sign up multiple times across different sources, creating duplicate records. We need to link these duplicates.

---

## 2. Data Structure (FROM VIDEO — CRITICAL!)

### Fields Per Record
| Field | Description | Example |
|-------|-------------|---------|
| `entity_id` | Unique ID with source prefix | `S1-732914`, `S2-118820`, `S3-905477` |
| `business_name` | Business name (may have variations) | `Acme Robotics Inc` |
| `business_address` | Business address (may have variations) | `500 Market St, San Jose` |

**NOTE:** The video shows ONLY name and address. Country is mentioned in the PDF but may or may not appear in the actual data.

### Example Records (from video frame 15)

**SOURCE 1 (Reference — deduplicated):**
```
S1-732914 · Acme Robotics Inc        · 500 Market St, San Jose
S1-889301 · Delta Foods              · 8 Oak Ave, Austin
S1-410562 · Bright Cafe LLC          · 22 Pine St, Reno
S1-205774 · Zen Traders              · 4 Hill Rd, Boise
```

**SOURCE 2:**
```
S2-118820 · Acme Robotics Inc        · 500 Market Street, San Jose    ← MATCH
S2-540221 · Acme Robotix             · 12 Elm Rd, San Jose            ← LOOK-ALIKE (different biz)
S2-397155 · Delta Foods Co           · 8 Oak Avenue, Austin           ← MATCH
S2-663049 · Bright Cafe              · 22 Pine Street, Reno           ← MATCH
```

**SOURCE 3:**
```
S3-905477 · Acme Robotics            · Nr City Hall, San Jose         ← MATCH
S3-063118 · Acme Bakery              · 500 Market St, San Jose        ← LOOK-ALIKE (different biz, same address!)
S3-651230 · Delta Foods Ltd          · Oak Ave, Austin                ← MATCH
S3-472088 · Kappa Motors             · 90 Lake Dr, Fargo              ← NO MATCH
```

### Key Observation from Video
- Same address can belong to DIFFERENT businesses (Acme Robotics vs Acme Bakery at same address)
- Name variations: `Inc` vs `Incorporated`, `Foods` vs `Foods Co` vs `Foods Ltd`
- Address variations: `St` vs `Street`, `Ave` vs `Avenue`, `Nr City Hall` (landmark-based)

---

## 3. Files Provided

### Training Set
| File | Description |
|------|-------------|
| `train_source1.tsv` | Source 1 records (reference) |
| `train_source2.tsv` | Source 2 records |
| `train_source3.tsv` | Source 3 records |
| `train_ground_truth.tsv` | Ground truth labels |

### Test Set
| File | Description |
|------|-------------|
| `test_source1.tsv` | Source 1 records (predict matches for EVERY entity) |
| `test_source2.tsv` | Source 2 records |
| `test_source3.tsv` | Source 3 records |

### Ground Truth Format
```tsv
source1_entity_id    matched_entity_ids
S1-732914           S2-118820,S3-905477
S1-889301           S2-397155,S3-651230
S1-410562           S2-663049
S1-205774           
```
- One row per Source 1 entity
- Comma-separated list of matching S2/S3 IDs
- Empty = singleton (no matches)

---

## 4. What We Must Produce

### Output Files
| File | Description | Scored? |
|------|-------------|---------|
| `matching_results.tsv` | Final matches | **YES** (leaderboard) |
| `candidate_pairs.tsv` | Blocking candidates | No (audited) |

### matching_results.tsv Format
```tsv
source1_entity_id    matched_entity_ids
S1-732914           S2-118820,S3-905477
S1-889301           S2-397155,S3-651230
S1-410562           S2-663049
S1-205774           
```

### candidate_pairs.tsv Format
```tsv
source1_entity_id    candidate_entity_ids
S1-732914           S2-118820,S2-540221,S3-905477,S3-063118
S1-889301           S2-397155,S3-651230
S1-410562           S2-663049
S1-205774           S3-472088
```

### Validation
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

---

## 5. Evaluation Metric

**F_0.5 Score (precision-weighted):**

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

- **Macro-averaged** per Source 1 entity
- **Singletons included** — predicting empty for a singleton = 1.0, predicting any match = 0.0
- **False merges cost 2× more than misses** — when unsure, DO NOT MERGE

---

## 6. Critical Rules

| Rule | Consequence |
|------|-------------|
| Every S1 entity must appear in output | Rejection |
| No duplicate IDs in match lists | Rejection |
| Only S2/S3 IDs from test set | Rejection |
| No external data lookup | **Disqualification** |
| Model must be MIT/Apache 2.0, ≤8B params | Checked in review |

---

## 7. Key Tips (from video)

1. **"Blocking sets your recall ceiling"** — invest there first
2. **"A false merge costs twice as much as a miss"** — when unsure, don't merge
3. **"Singletons earn full 1.0 if predicted correctly"** — don't force matches
4. **"Account for region-specific patterns"** — India/US/France address formats differ
5. **"No external data"** — only use provided training data

---

## 8. Submission Package Structure

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

---

## 9. Timeline (72 Hours)

| Phase | Time | Focus |
|-------|------|-------|
| **Phase 1** | Hours 0-6 | Data exploration, simple baseline, first submission |
| **Phase 2** | Hours 6-24 | Improve blocking, add features, iterate |
| **Phase 3** | Hours 24-48 | Advanced models, threshold optimization |
| **Phase 4** | Hours 48-72 | Polish, documentation, final submission |

---

## 10. Success Criteria

| Metric | Target |
|--------|--------|
| First submission | Within 6 hours |
| Blocking recall | ≥ 95% |
| F_0.5 on validation | ≥ 0.85 |
| Code quality | Modular, documented |
| Documentation | Complete methodology write-up |
