# Task Breakdown — 3-Person Team

## Team Structure

| Role | Person | Focus Area |
|------|--------|------------|
| **Lead / Pipeline Owner** | You | Architecture, integration, model training, submission |
| **Blocking Engineer** | Member 2 | Blocking strategies, candidate generation |
| **Feature Engineer** | Member 3 | Normalization, feature engineering, evaluation |

---

## Parallel Work Streams

### Stream A: Blocking (Member 2) — CAN START IMMEDIATELY
No data dependency. Build blocking module using example data from video.

**Task A1: Text Normalization** (Hours 0-4)
- [ ] Name cleaning (lowercase, strip legal suffixes, expand abbreviations)
- [ ] Address cleaning (expand street abbreviations, normalize)
- [ ] Handle missing values
- [ ] Unit tests with examples from video

**Task A2: Blocking Strategies** (Hours 4-12)
- [ ] TF-IDF blocking on names (char n-grams, cosine similarity)
- [ ] Token-sorted blocking (catches word transpositions)
- [ ] Phonetic blocking (Soundex, Metaphone)
- [ ] Initialism blocking ("IBM" ↔ "International Business Machines")
- [ ] Address-based blocking
- [ ] Union of all strategies

**Task A3: Blocking Evaluation** (Hours 12-16)
- [ ] Measure blocking recall on training ground truth
- [ ] Tune blocking thresholds
- [ ] Generate candidate_pairs.tsv format
- [ ] Validate with validate_submission.py

**Deliverables:**
- `src/normalize.py`
- `src/blocking.py`
- Blocking evaluation report

---

### Stream B: Features (Member 3) — CAN START IMMEDIATELY
No data dependency. Build feature module using example pairs.

**Task B1: Pairwise Features** (Hours 0-8)
- [ ] Name features (10):
  - token_sort_ratio, partial_ratio, WRatio
  - Jaro-Winkler, Jaccard, edit_ratio
  - Trigram Jaccard, Soundex match, Metaphone match
  - Length ratio
- [ ] Address features (6):
  - token_sort_ratio, partial_ratio, WRatio
  - Jaccard, Trigram Jaccard, Length ratio
- [ ] Country feature (1): same_country
- [ ] Cross features (8):
  - Name-addr averages/max/min
  - Is_company, phonetic_vote
  - Surname_length_diff, combined_trigram

**Task B2: Feature Validation** (Hours 8-12)
- [ ] Test features on example pairs from video
- [ ] Check feature distributions
- [ ] Remove redundant features
- [ ] Document feature importance

**Task B3: Feature Pipeline** (Hours 12-16)
- [ ] Batch feature computation
- [ ] Handle missing data in features
- [ ] Feature scaling (if needed)
- [ ] Feature selection (if needed)

**Deliverables:**
- `src/features.py`
- Feature validation report

---

### Stream C: Pipeline + Model (You) — COORDINATE OTHERS
Depends on Stream A and B outputs.

**Task C1: Project Setup** (Hours 0-2)
- [ ] Create GitHub repo
- [ ] Set up project structure
- [ ] Create requirements.txt
- [ ] Add team members as collaborators
- [ ] Create shared documentation

**Task C2: Data Loading** (Hours 2-4)
- [ ] Load all TSV files
- [ ] Profile data (scale, distribution, completeness)
- [ ] Parse ground truth
- [ ] Understand matching patterns

**Task C3: Pipeline Integration** (Hours 4-8)
- [ ] Integrate blocking module
- [ ] Integrate feature module
- [ ] Build training data construction
- [ ] Test end-to-end pipeline

**Task C4: Model Training** (Hours 8-16)
- [ ] Train LightGBM (primary model)
- [ ] Optuna hyperparameter tuning
- [ ] Cross-validation
- [ ] Evaluate on validation set

**Task C5: Threshold Optimization** (Hours 16-20)
- [ ] Scan F_0.5 thresholds (0.1 to 0.95)
- [ ] Select optimal threshold
- [ ] Bootstrap confidence intervals

**Task C6: Inference & Submission** (Hours 20-24)
- [ ] Run on test set
- [ ] Generate matching_results.tsv
- [ ] Generate candidate_pairs.tsv
- [ ] Validate submission
- [ ] Upload to leaderboard

**Deliverables:**
- `src/pipeline.py`
- `src/model.py`
- `src/training.py`
- First leaderboard submission

---

## Task Dependencies

```
C1 (Setup) ──┬── A1 (Normalization) ──┬── A2 (Blocking) ──┬── A3 (Eval)
              │                        │                    │
              └── B1 (Features) ───────┴── B2 (Validation) ─┴── B3 (Pipeline)
                                          │
                                          ▼
                                    C3 (Integration) ── C4 (Model) ── C5 (Threshold) ── C6 (Submit)
```

**Critical Path:** C1 → A1 → A2 → C3 → C4 → C5 → C6

**Parallel Opportunities:**
- A1 and B1 can run in parallel (no dependency)
- A2 and B2 can run in parallel
- C2 can run while A1/B1 are in progress

---

## Communication Protocol

### Daily Standup (Every 12 hours)
- What did you complete?
- What are you working on next?
- Any blockers?

### Shared Artifacts
- `src/` — All source code
- `docs/` — Documentation, notes, findings
- `output/` — Submission files
- `models/` — Saved models

### Code Review
- All PRs require 1 approval before merge
- Run tests before merging
- Document any design decisions

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Blocking recall too low | Add more blocking signals, lower thresholds |
| Too many false positives | Increase classification threshold, add features |
| Model overfits | Use cross-validation, conservative params |
| Time runs out | Prioritize blocking + simple model first |
| Team member blocked | Rotate tasks, pair programming |

---

## Quick Reference Commands

```bash
# Setup environment
source venv/bin/activate
uv pip install -r requirements.txt

# Run pipeline
python -m src.pipeline

# Validate submission
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test

# Run tests
pytest tests/
```
