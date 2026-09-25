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
| **Karan** | Feature engineer | `src/features.py`, `scripts/evaluate.py` |

➡️ Detailed assignments: [TASK_BREAKDOWN.md](TASK_BREAKDOWN.md)

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
TSVs → normalize → 7-layer blocking → 25 pairwise features
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
