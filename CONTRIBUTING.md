# 🤝 CONTRIBUTING — Team Workflow Rules

**Repo:** https://github.com/AbhijitK20/Business_entity_resolution
**Team:** Abhijit · Vishwesh · Karan

> ⚠️ **`main` is protected. Nobody pushes directly to `main` — including the lead.**

---

## 1. The Rule

```
main  ← protected. Only merges via Pull Request with review.
 │
 ├── feat/blocking-*      ← Vishwesh
 ├── feat/features-*      ← Karan
 ├── feat/model-*         ← Abhijit
 └── docs/*               ← anyone
```

**Every change goes through a branch + PR.** No exceptions after hour 14 of the hackathon.

---

## 2. Daily Workflow

### Start your work
```bash
git checkout main
git pull                                    # get latest
git checkout -b feat/<your-area>/<topic>    # e.g. feat/blocking/indic-transliteration
```

### While working
```bash
git add <files>
git commit -m "feat(blocking): add Indic transliteration

- port ITRANS transliteration with schwa deletion
- covers 9 script blocks
- unit tests: राम मार्केटिंग → ram marketing"
```

**Commit message convention:** `type(area): short description`
- `feat` new capability · `fix` bug · `docs` documentation · `test` tests · `refactor` cleanup · `perf` speed

### Before pushing
```bash
python tests/test_smoke.py                  # must PASS
python tests/test_<your_module>.py          # your module's tests
```

### Push + open PR
```bash
git push -u origin feat/<your-area>/<topic>
gh pr create --fill --base main
```

### PR rules
| Rule | Detail |
|------|--------|
| **Title** | Same convention as commits: `feat(blocking): ...` |
| **Description** | What changed · evidence (measured numbers) · what to verify |
| **Reviewers** | At least 1 teammate must approve |
| **Tests** | `tests/test_smoke.py` must pass in the PR description |
| **Evidence** | Any performance claim needs a measured number attached |
| **Merge** | Squash-merge into `main` after approval |
| **Delete branch** | After merge, delete the branch |

### After merge
```bash
git checkout main && git pull
git branch -d feat/<your-area>/<topic>
```

---

## 3. File Ownership (avoid conflicts)

| Member | Owns (only they edit) |
|--------|----------------------|
| **Vishwesh** | `src/normalize.py`, `src/blocking.py`, `tests/test_normalize.py`, `tests/test_blocking.py` |
| **Karan** | `src/features.py`, `scripts/evaluate.py`, `scripts/make_synthetic_data.py`, `tests/test_features.py`, `docs/feature_report.md` |
| **Abhijit** | `src/pipeline.py`, `src/model.py`, `src/training.py`, `src/decision.py`, `src/data_loader.py`, `utils/`, package assembly |

**Frozen interfaces** (MASTERPLAN §7) change only by team agreement — announce in chat first.

**Shared docs** (`MASTERPLAN.md`, `TASK_BREAKDOWN.md`, `docs/*.md`): anyone can PR, but keep edits scoped.

---

## 4. What NOT to Commit

`.gitignore` already blocks these — never force-add them:
- `data/` (dataset — large, and we must not redistribute it)
- `research/`, `peoples prototype/`, `official/` (cloned reference repos)
- `venv/`, `__pycache__/`, `*.pkl`, `output/`, `*.zip`, `.env`

---

## 5. Emergency Rule

If `main` is broken and blocking everyone:
1. Post in chat immediately: what's broken, what you need
2. **Do not** push a fix directly to `main` — open a PR and tag the other two for instant review
3. If truly urgent (e.g. submission deadline), the lead may temporarily lift protection, merge, and re-enable — this must be announced in chat and re-enabled immediately

---

## 6. Submission-Day Exception

Only `output/matching_results.tsv` uploads happen on the competition portal — that's not a git push. The repo stays PR-only the whole time.
