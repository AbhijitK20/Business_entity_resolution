# Amazon ML Challenge 2026 — Business Entity Resolution

## Team Members
- [Your Name] — Lead / Pipeline Owner
- [Member 2] — Blocking Engineer
- [Member 3] — Feature Engineer

## Problem
Determine which business records from 3 independent sources describe the same real-world business, using only **name** and **address** fields.

## Evaluation
- **Metric:** F_0.5 (precision-weighted)
- **Formula:** `F_0.5 = (1.25 × P × R) / (0.25 × P + R)`
- **Key:** False merges cost 2× more than misses

## Project Structure
```
├── src/
│   ├── data_loader.py      # Load TSV files
│   ├── normalize.py        # Text normalization
│   ├── blocking.py         # Blocking strategies
│   ├── features.py         # Pairwise features
│   ├── training.py         # Training data construction
│   ├── model.py            # Model training
│   └── pipeline.py         # Main orchestrator
├── tests/
├── output/
├── models/
├── docs/
├── PRD.md                  # Product requirements
├── TASK_BREAKDOWN.md       # Team task assignments
└── requirements.txt
```

## Quick Start
```bash
# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python -m src.pipeline

# Validate submission
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Timeline
- **Hours 0-6:** Data exploration, simple baseline
- **Hours 6-24:** Improve blocking, add features
- **Hours 24-48:** Advanced models, optimization
- **Hours 48-72:** Polish, documentation, final submission
