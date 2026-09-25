# Amazon ML Challenge 2026 — Business Entity Resolution

**Team:** Abhijit · Vishwesh · Karan
**Team repo:** https://github.com/AbhijitK20/Business_entity_resolution
**Hackathon:** 25–27 September 2026 (72 hours)

> **📖 Start here:** [MASTERPLAN.md](MASTERPLAN.md) — problem, data, architecture, team division, timeline, and current status.

---

## Team & Roles

| Member | Role | Owns |
|--------|------|------|
| **Abhijit** | Lead — pipeline, model, submissions | `src/pipeline.py`, `src/model.py`, `src/training.py`, `src/data_loader.py` |
| **Vishwesh** | Blocking engineer | `src/normalize.py`, `src/blocking.py` |
| **Karan Sasane** | Feature & evaluation engineer | `src/features.py`, `scripts/evaluate.py`, `docs/feature_report.md` |

➡️ Detailed assignments: [TASK_BREAKDOWN.md](TASK_BREAKDOWN.md)

### Contributions — Karan Sasane (feature & evaluation stream)

Delivered in the `feat/features-eval-karan` branch; details in [docs/feature_report.md](docs/feature_report.md).

- **Evaluation tooling** (`scripts/evaluate.py`): candidate-oracle F_0.5 ceiling, per-country and match-count-bucket segments, complete-match coverage, reduction ratio, singleton false-merge rate, business-group (S1) bootstrap confidence intervals, structured error buckets (retrieval / matching / decision-policy / integrity), and top-K worst-entity diagnosis with feature evidence
- **Synthetic-data distribution** (`scripts/make_synthetic_data.py`): calibrated to the verified real distribution — match-count PMF (89% multi-match, mean 3.46), 47% shared S1 names, ~3% gallery-only blank addresses, Indic cross-script name variants, same-name/same-address distractors, train/test country mixes
- **Feature validation** (`tests/test_features.py`): semantic tests per feature family, class-separation checks (match vs negative medians), `-1` sentinel handling and blocked-family tests, dud/constant/duplicate feature detection
- **Country holdout** (`scripts/country_holdout.py`): train-US↔eval-India transfer stress test (hard and full-density regimes) with precision-preserving margins and a France cutoff recommendation (France is never presented as a supervised result — test-only in the data)
- **Error analysis** (`scripts/error_analysis_demo.py`): held-out-entity error bucketing with pair-score evidence, retrieval-vs-matching-vs-decision attribution, and top-20 worst-entity diagnostics
- **Tests**: `tests/test_evaluate.py`, `tests/test_synthetic_data.py`, `tests/test_country_holdout.py`, `tests/test_features.py` — `pytest -q` → 74 passed, 5 skipped (skips = feature families pending A1, explicitly marked `BLOCKED on A1`)
- **Documentation**: [docs/feature_report.md](docs/feature_report.md) — feature-engineering methodology, evaluation harness, country-holdout and error-analysis findings (synthetic-data labelled; real-data re-runs pending dataset availability)

---

## Quick Links

| Document | Purpose |
|----------|---------|
| [MASTERPLAN.md](MASTERPLAN.md) | Team overview: problem, verified data, roles, timeline |
| [docs/IMPLEMENTATION_BLUEPRINT.md](docs/IMPLEMENTATION_BLUEPRINT.md) | **Code-level spec** — exact algorithms, module specs, execution order |
| [TASK_BREAKDOWN.md](TASK_BREAKDOWN.md) | Task ownership + live status board |
| [docs/COMPETITIVE_INTEL.md](docs/COMPETITIVE_INTEL.md) | What 13 other teams found (verified dataset audit, SABER results) |
| [docs/GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md) | Teammate masterplans vs our implementation |
| [PRD.md](PRD.md) | Product requirements (problem, data, output format) |
| [docs/video_transcript.md](docs/video_transcript.md) | Official problem walkthrough transcript |
| [docs/ULTIMATE_STRATEGY.md](docs/ULTIMATE_STRATEGY.md) | Strategy synthesized from 26 research repos |

## Quick Start

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
uv pip install -r requirements.txt

# Verify everything works (no real dataset needed — uses synthetic fixtures)
python tests/test_smoke.py

# Generate a larger synthetic dataset for pipeline testing
python scripts/make_synthetic_data.py --out tests/fixtures_synth --n-s1 300

# Run the pipeline (fast mode)
python -c "from src.pipeline import EntityResolutionPipeline as P; P('tests/fixtures_synth','output',fast_mode=True).run()"

# Evaluate (leaderboard-style macro F_0.5)
python scripts/evaluate.py --predictions output/matching_results.tsv \
                           --ground-truth tests/fixtures_synth/dataset/test/test_ground_truth.tsv

# Validate submission format before uploading
python utils/validate_submission.py --matching output/matching_results.tsv \
                                    --candidate output/candidate_pairs.tsv \
                                    --test-dir tests/fixtures_synth/dataset/test
```

## Pipeline Overview

```
TSVs → normalize → 7-layer blocking → 35 pairwise features
     → leak-free stacking (LGB + XGB + RF → meta) → macro-F_0.5 threshold
     → matching_results.tsv + candidate_pairs.tsv
```

## Evaluation

Macro F_0.5, precision-weighted — a false merge costs ~2× a missed match.
**When in doubt, do not merge.**

## Rules

- ⚠️ **No external data**: no APIs, no geocoding, no databases — disqualification offense.
- Model must be MIT/Apache-2.0 licensed, ≤ 8B parameters.
- Submission must pass `utils/validate_submission.py`.
