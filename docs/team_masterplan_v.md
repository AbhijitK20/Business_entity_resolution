# MASTERPLAN.md (Revision 2)
## Business Entity Resolution — Amazon ML Challenge 2026
### Organizer-Aligned, Precision-First, Dataset-Driven Architecture, Optimized for Macro F0.5

---

## 0. HOW TO READ THIS DOCUMENT

This is a build plan, not a wish list. Every component is tagged:

| Tag | Meaning | Rule |
|---|---|---|
| **[P0 — MANDATORY]** | Required for a valid, competitive submission, and confirmed by either the official PDF or the organizer video | Build first, no exceptions |
| **[P1 — HIGH VALUE]** | Very likely to help based on the problem structure, but not explicitly confirmed by the organizers | Build after P0 is stable and measured |
| **[P2 — EXPERIMENTAL / DATA-DEPENDENT]** | May or may not help; unconfirmed by organizers; must earn its place with an ablation on *this* dataset | Build only after profiling gives a concrete reason, and remove if it doesn't move held-out F0.5 |
| **[VERIFY]** | Depends on a fact not yet confirmed by the official PDF, the official video, or your own data profiling | Do not architect around this until confirmed |

**The rule that overrides all enthusiasm:** if a component cannot show a measured, held-out ΔF0.5 improvement on *this* dataset, it does not ship. This revision exists specifically to strip out any place where the previous version stated something as fact when it was actually an assumption, and to re-center the whole document on the two fields the organizers explicitly told you to rely on: **business name and business address.**

---

## OFFICIAL VIDEO ALIGNMENT (read this before anything else)

The organizer video (Amazon ML Challenge 2026 problem walkthrough) confirms the following as **facts**, not design choices. Every section below has been audited against this list; anywhere the two conflicted, the video/PDF wins and the section was corrected.

1. **Three independent sources, no shared identifier.** S1 is the clean, deduplicated reference list. S2 and S3 are noisy fragments reconciled against it.
2. **"We have deliberately limited the data to names and addresses."** The organizers explicitly frame **business_name + business_address** as the fields the reconciliation is built on. `country` exists in the file schema (confirmed by the official PDF), but the video does not name it as a reconciliation field. This document treats country as **conditional, verification-gated evidence**, never as a foundational identity signal on par with name/address — see the new Section 2.1a.
3. **0, 1, or many matches per S1 entity.** Never assume top-1.
4. **Blocking uses a cheap key built from both name and address**, and records can land in the same block through a similar name *or* a shared address — this is the organizer's own conceptual description of blocking, and it is the reason name-blocking and address-blocking are both treated as the **foundational** blocking families in Section 10, with everything else framed explicitly as an extension of that idea.
5. **"Blocking favors recall."** The organizers explicitly say a block is *allowed* to pull in lookalikes — "a business with a similar name at a different address, and a different business that happens to share an address" — and that **the matching model removes those in the next step.** This is now stated as the core separation of concerns throughout the document: blocking is recall-oriented and is not penalized for imprecision; precision is entirely the matcher's and the decision engine's job.
6. **Two outputs.** `candidate_pairs.tsv` (blocking's output, unscored, used only to audit blocking quality) and `matching_results.tsv` (the only scored file). The video is explicit that candidate pairs are a **blocking-quality audit artifact**, not itself an optimization target — Section 5 (candidate_pairs semantics) and the new optimization-hierarchy diagram in Section 11 make this explicit.
7. **Ground truth is one row per S1 entity**, comma-separated match ids, empty when no match. Files are tab-separated.
8. **Macro F0.5**, precision weighted twice recall. "When in doubt, it is safer not to merge."
9. **Singletons matter as much as real matches.** Correctly predicting empty = 1.0 for that entity; any predicted match on a true singleton = 0.0. The video states this as a direct recommendation, not a side note.
10. **"Your blocking strategy sets the ceiling on the recall you can achieve. Invest in it first, because you cannot match a record you never consider."** This is now the single most heavily emphasized measurement in the document (Section 11).
11. **"Pay attention to region-specific patterns in both names and addresses."** Reframed throughout as a data-driven discovery task (Section 3, Section 10), never as hard-coded geographic rules, and never sourced from anything outside the provided data.
12. **"This is a pure machine learning challenge, so external databases, APIs, and lookups are strictly prohibited. Use only the provided data."** Stated as an absolute, structural constraint — zero outbound network calls anywhere in the scored pipeline (Section 33, Section 47, Section 60).

**What the video does NOT confirm, and what this revision therefore stops asserting as fact:** that country is a reliable or intended matching signal; that any specific technique (embeddings, graphs, transformers) is expected or rewarded; that any specific false-positive pattern (e.g. generic-token collisions) is actually present at meaningful volume in this dataset — all such claims are now phrased as **hypotheses to validate against the real data**, not conclusions.

---

## 1. EXECUTIVE SUMMARY

**The task.** Three data sources describe overlapping business entities with no shared keys. Source 1 (S1) is a clean, deduplicated reference set. Sources 2 and 3 (S2, S3) are noisy — inconsistent naming, partial addresses, transliteration, typos, landmark references. For every S1 entity, you must return the set of S2/S3 records that refer to the same real business: zero, one, or many. The organizers deliberately scoped this challenge to **name and address** as the fields carrying the reconciliation signal.

**Why this is hard.**
- There is no identifier to join on. Matching is purely evidence-based, primarily on name and address similarity.
- Real business names collide on purpose ("Corp", "Services", "International", "Ltd" are shared by unrelated companies) — **this is a hypothesis about the data, to be confirmed by profiling in Section 3, not an assumed fact**, though it is a well-documented general failure mode in business-name matching.
- Addresses are semi-structured free text with missing components, landmark references, and formatting drift across sources — this one *is* explicitly confirmed by the organizer video ("one vendor may write Acme Robotics Inc., another abbreviates the address, and a third references a nearby landmark").
- The relationship is not one-to-one: an S1 entity can have 0, 1, or N true matches — explicitly confirmed. A system that assumes top-1 matching is structurally wrong on multi-match and zero-match entities regardless of similarity quality.
- Whether `country` carries usable signal at all is **unconfirmed** — see Section 2.1a. The architecture must not depend on it.
- All predicted matches must exist inside your own `candidate_pairs.tsv`. Blocking is a hard recall ceiling: a match your blocker never generates as a candidate can never be recovered later, no matter how good your model is — this is the organizers' own stated framing, not an inference.

**Why naive fuzzy matching fails.** A system that computes one similarity score and thresholds it has no principled mechanism for "no match," no mechanism for legitimate multi-match, and (per the organizer's own worked example) no way to distinguish "similar name, correct address" from "similar name, wrong address" from "different business, shared address" — three cases the video explicitly describes as landing in the *same block* by design.

**Why blocking is necessary, not optional, and why it is allowed to be imprecise.** The video's own conceptual pipeline is: cheap name+address key → block → candidate pairs (recall-favoring, lookalikes included on purpose) → matching model removes the lookalikes → final matches. Blocking is optimized for **recall and computational feasibility**, never for candidate precision — a blocker that returns a "wrong" candidate is not a blocking bug, it is blocking working as intended, and rejecting that candidate is explicitly the matcher's job, not the blocker's.

**Why F0.5 changes the strategy.** F0.5 weights precision 2× recall, computed **per S1 entity, then macro-averaged**:

```
F0.5 = 1.25 · P · R / (0.25 · P + R)
```

Two consequences, both organizer-confirmed:
1. A correctly-predicted singleton scores a full 1.0; a false merge on a true singleton scores 0.0, with no partial credit. **Precision on singletons is worth exactly as much as precision on any real match** — the no-match engine (Section 25) is core to the score, not a side feature, and the organizers say so directly.
2. "When in doubt, it is safer not to merge" is the organizers' own stated decision heuristic. This must be operationalized empirically (a validated threshold, Section 24), not turned into blanket over-conservatism that also destroys recall.

**What makes this architecture competition-grade**: explicit, organizer-endorsed candidate-recall measurement before any model is trained (Sections 3, 11); a strict separation between recall-oriented blocking and precision-oriented matching, matching the organizers' own mental model exactly (Section 10); an entity-level decision layer explicitly separate from the pairwise classifier (Section 22); a dataset-first methodology that refuses to assume which advanced techniques (if any) this specific dataset needs (Section 40, Section 52); an adversarial suite used as a guardrail, not a target that could distort the real objective (Section 36-37); and a submission validator that runs before every leaderboard upload (Section 50).

### 1.1 Architecture at a Glance

```
S1 entity (test) ─────────────────────────────────────────────┐
                                                                │
   ┌────────────────────────────────────────────────────────┐ │
   │ STAGE 0  Ingestion & Validation      [P0]                │◄┘
   │ STAGE 1  Multi-view Normalization    [P0]  (name+address) │
   │ STAGE 2  Name+Address Blocking       [P0]  → candidates  │
   │ STAGE 3  Candidate Recall Audit      [P0]  (offline)     │
   │ STAGE 4  Pairwise Feature Engine     [P0]  (country cond.)│
   │ STAGE 5  GBDT Matcher + Calibration  [P0/P1]              │
   │ STAGE 6  Entity-Level Decision       [P0]  → NONE/1/N    │
   │ STAGE 7  Difficulty Routing          [P1]  (Tier-2/3)    │
   │ STAGE 8  Cross-source consistency    [P2]  (evidence only)│
   │ STAGE 9  Adversarial + Ablation Lab  [P0]  (guardrail)   │
   │ STAGE 10 Submission Validator        [P0]                │
   └────────────────────────────────────────────────────────┘
                                                                │
                                          matching_results.tsv ─┘  (scored)
                                          candidate_pairs.tsv       (audit only)
```

The philosophy, in the organizers' own terms: **blocking favors recall and is allowed to be wrong; the matching model removes the wrong candidates; the entity-level decision converts pairwise scores into the final 0/1/many set that F0.5 actually judges.**

---

## 2. COMPLETE PROBLEM CONTRACT

Transcribed and reconciled from the official PDF and the official organizer video — the two authoritative sources. Anything not confirmed by either is marked `[VERIFY]`.

### 2.1 Inputs
- `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`
- `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv` (no ground truth)
- Columns per source file: `entity_id` (prefixed `S1-`/`S2-`/`S3-`), `business_name`, `business_address`, `country`.
- Files are **tab-separated**; commas legitimately occur inside addresses and inside comma-joined id lists. Always `sep="\t"`.
- `business_name` and `business_address` are the fields the organizer video explicitly identifies as what reconciliation is built on ("the only fields we can rely on are the business name and address... we have deliberately limited the data to names and addresses").
- `country` is present in the schema per the official PDF (US/India in training, France additionally in test, unseen in training). **Its status as a genuine matching signal for this task is not confirmed by the organizer video and must be treated as conditional — see 2.1a.**

### 2.1a FEATURE CONTRACT VERIFICATION `[P0 — do this before building any country-dependent logic]`

The official PDF lists `country` as a column. The official video, describing the intended reconciliation logic, names only name and address. These are not necessarily in conflict — country may simply be contextual metadata, or it may be weak auxiliary evidence — but the architecture must not assume which, and must not be built as if `business_name + business_address + country` are three equally-weighted identity fields.

**Required verification steps, in order, before country is used predictively anywhere in the pipeline:**
1. Re-read the official PDF's exact wording on `country` (already done: it is described as a field, with an explicit instruction to treat it as an *open set* and never hard-filter or hard-code against `{US, India}`). This confirms country is *present and legitimate to read*, but says nothing about how predictive it should be treated as.
2. Treat the organizer video as the authoritative statement of *intended* reconciliation logic: name + address. Do not silently expand this to include country as a foundational signal.
3. Empirically test, on the training ground truth (Section 3): for matched pairs, what fraction have identical / conflicting / missing country values on each side? For non-matched pairs (near-miss candidates from blocking), what's the same breakdown? If country reliably agrees on true matches and reliably disagrees or is a genuine discriminator on hard negatives, it is safe to use as **auxiliary evidence**. If it's mostly missing, mostly agreeing even on non-matches (e.g. because almost everything in a subset is the same country), or unreliable, it contributes little and adds unnecessary surface area.
4. **Country is never a hard filter or blocking key, regardless of what step 3 shows** — this is an explicit rule from the original PDF (open-set country, France unseen in training) and is *reinforced*, not contradicted, by the video's silence on country as a reconciliation field.
5. If step 3 shows measurable, validated signal: keep country strictly as one auxiliary feature family (Section 9), ablation-gated like every other P1/P2 component (Section 41), never load-bearing on its own.
6. If step 3 shows no measurable signal, or signal that doesn't survive holdout validation: remove country-based predictive features entirely and keep country only as passthrough metadata for debugging/explainability, not as a model input.

This verification step must be re-run and logged as a first-class experiment (Section 42) — its outcome determines how much of Sections 9, 13, and 14's country-related content is actually active in the shipped pipeline.

### 2.2 Outputs
Two TSVs, both under `output/`:

**`matching_results.tsv`** (the only file scored — confirmed by both PDF and video):
| Column | Description |
|---|---|
| `source1_entity_id` | S1 entity_id |
| `matched_entity_ids` | comma-separated S2/S3 ids, empty for no match |

Hard constraints: exactly one row per test S1 entity; no duplicate `source1_entity_id` rows; no duplicate ids within one entity's list; ids must reference only S2/S3 records that exist in the **test** files; no S1 ids anywhere in the matched list.

**`candidate_pairs.tsv`** (explicitly **not** scored — an audit artifact for blocking quality, per both PDF and video):
| Column | Description |
|---|---|
| `source1_entity_id` | S1 entity_id |
| `candidate_entity_ids` | comma-separated candidate S2/S3 ids considered by the matcher |

Semantic point, stated by the organizers directly ("the candidate set your blocking stage produced, before your final matching model narrowed it down... this file is not scored, but is used to audit the quality of your blocking"): `candidate_pairs.tsv` is the exact set the matching model scored at inference — the **last** stage of your candidate generation, not an early, unfiltered blocking dump. It must be produced in the same run as `matching_results.tsv`, never regenerated afterward from blocking alone, or the "final matches ⊆ candidates" invariant can silently break.

**Do not treat candidate-pair quality as a leaderboard optimization target in its own right.** It is diagnostic, not scored. See Section 11's optimization hierarchy.

### 2.3 Ground truth format
`train_ground_truth.tsv`: `source1_entity_id`, `matched_entity_ids` (comma-separated S2/S3 ids; empty string = zero matches). Confirmed identical in structure to what you submit.

### 2.4 Zero / single / multi-match semantics
- **Zero match (singleton)**: correctly predicting empty scores 1.0 for that entity; predicting anything scores 0.0. Organizer-stated, not inferred.
- **Single match**: standard case.
- **Multi-match**: build every entity-level function to natively return a *set*, never a single label — organizer-confirmed ("a source 1 entity may match many records, exactly one, or none at all").

### 2.5 Candidate-pair invariant
Every final predicted match must exist inside `candidate_pairs.tsv`. Enforced by construction (the decision engine only ever selects from already-retrieved candidates) and re-verified by the validator (Section 50).

### 2.6 Additional official constraints
- **No external data lookups of any kind** — organizer-stated directly and unambiguously: "external databases, APIs, and lookups are strictly prohibited. Use only the provided data." This includes entity-resolution APIs, government registries, geocoding services, mapping services, OpenStreetMap, Wikidata, business directories, search engines, and any third-party company-resolution service. Enforce structurally: zero outbound network calls anywhere in `src/`.
- **Final model license/size constraint** `[from official PDF]`: MIT/Apache-2.0, ≤8B parameters, for any model in the scored pipeline.
- **Validate before submitting** using the provided `utils/validate_submission.py` — organizer-stated directly ("run the provided validation script before submitting, so a simple formatting error does not cost you a submission").
- **Public vs private leaderboard**: same full test-set predictions used for both; only the internal scoring split differs.

`[VERIFY]`: exact max file size / row count limits; whether the validator also checks `candidate_pairs.tsv` id-existence against S1; ordering requirements for multi-match lists (none specified — keep deterministic regardless).

---

## 3. DATA DISCOVERY PHASE `[P0]`

**Nothing in this document should be treated as a confirmed dataset characteristic until this stage produces the number.** Every "hypothesized" claim elsewhere in this document is resolved here, one way or the other, before it's allowed to drive an architecture decision.

**Run and log, separately for S1, S2, S3, and separately for S1→S2 vs S1→S3 ground truth:**

| Metric | Feeds decision in |
|---|---|
| Row counts, unique/duplicate `entity_id` counts | Section 4 |
| Duplicate `business_name` / `business_address` counts (exact and lightly normalized) | Section 7, Section 12 |
| Missingness per field | Section 15, Section 35 |
| **Country agreement/conflict/missingness rate specifically on matched vs non-matched candidate pairs** | Section 2.1a — this is the single most important new measurement in this revision |
| Country frequency table; confirm France's presence and volume in test only | Section 2.1a, Section 44 |
| Name/address length distributions (percentiles) | Section 6/8, Section 27 |
| Token frequency table across name fields (IDF basis) | Section 7, Section 10 — **whether generic-token collision is actually a problem in this data is measured here, not assumed** |
| Character/Unicode/script distribution | Section 5, region-pattern discovery |
| Numeric-token patterns in addresses | Section 8 |
| Ground-truth zero/single/multi-match rate, match-count distribution | Section 22, Section 26 |
| Ground-truth match rate by S2 vs S3 | Section 10 |
| **Whether a single S2 or S3 record appears as a valid match for more than one S1 entity** | Section 18 (validation-graph correction, see below) |
| Region-specific patterns: abbreviation conventions, address ordering, postal-code formats (if any), locality conventions, script mixing, by apparent source/country slice | Section 10 — must be *discovered*, never hard-coded from outside knowledge |

**New required check (from the video's emphasis on region-specific patterns, and the validation-graph correction):**
- **Source-record fan-out check**: does any single S2-/S3- id appear in the matched list of more than one S1 entity in ground truth? If yes, the assumption that grouping by S1 alone prevents leakage is incomplete — see Section 19's correction. Measure this explicitly; do not assume the answer.
- **Naive blocker sanity check**: before building the real retrieval index, manually check whether a simple name-token or address-token overlap would even retrieve the true match for a sample of ground-truth pairs. This is the cheapest possible early warning that a planned blocking strategy is viable before investing engineering time in it.

**Deliverable**: `reports/data_profile.md` with every table above, populated with real numbers, plus concrete example rows (a clean match, a hard match, a country-conflicting match if any exist). Every claim elsewhere in this document that currently reads as a hypothesis must be resolved here into either "confirmed — see data_profile.md §X" or "not supported by this dataset — component downgraded/removed."

---

## 4. DATA QUALITY AND INGESTION `[P0]`

Unchanged in substance from the confirmed-safe baseline: typed severity levels (`INFO`/`WARNING`/`ERROR`/`QUARANTINE`), and the same absolute rule — **bad records reduce evidence quality, never coverage.** Every S1 test id must reach the final output even with all fields empty; that entity becomes a (correct, defensible) no-match prediction rather than a dropped row.

Checks: malformed TSV, missing columns, duplicate ids, null/empty ids, invalid UTF-8, zero-width/control characters, oversized strings (Section 48), delimiter corruption, whitespace corruption.

---

## 5. MULTI-VIEW NORMALIZATION `[P0 core views on name+address; P1/P2 exotic views, data-gated]`

Apply to **business_name and business_address** as the primary, organizer-confirmed fields. Apply the same view machinery to `country` only if Section 2.1a confirms it's worth modeling at all — do not build a parallel normalization stack for a field that may end up unused.

| View | Built [P-level] | Used for |
|---|---|---|
| `RAW` | P0 | contradiction detection, audit trail, explainability |
| `LIGHT_NORMALIZED` | P0 | most similarity features |
| `AGGRESSIVE_NORMALIZED` | P0 | exact-match blocking, high-recall retrieval |
| `TOKENS` | P0 | Jaccard/overlap features, rare-token blocking |
| `CHARACTER_NGRAMS` | P0 | typo/word-order robustness, TF-IDF retrieval |
| `NUMERIC_TOKENS` | P1 | address matching (postal/house-number-like digit runs) |
| `PARSED_COMPONENTS` (address) | P1 | address feature engine (Section 8) — always paired with `RAW`, since landmark-style addresses ("Near SBI ATM," explicitly named by the organizers) will routinely fail structured parsing |
| `PHONETIC` | P2 `[data-gated]` | only if Section 3 shows measurable typo severity that lexical methods miss |
| `TRANSLITERATED` | P2 `[data-gated]` | only if Section 3 shows measurable mixed-script name volume — do not build speculatively |

Cover in normalization, with unit tests: lowercasing, Unicode NFKC, punctuation standardization (`&` vs `and`), whitespace collapsing, repeated-character collapsing, legal-suffix normalization (built from the *actual* token-frequency table in Section 3, not a generic guess list), abbreviation expansion (`Rd`↔`Road`), diacritic handling (kept as a *separate* view, never overwriting the original — matters for French test data), zero-width stripping, homoglyph awareness (flag, don't silently rewrite).

**Never discard raw representations.** Aggressive normalization is always paired with a raw/light view downstream so contradictions remain detectable (Section 14).

---

## 6. BUSINESS NAME MATCHING `[P0]`

Unchanged in substance — name remains a foundational, organizer-confirmed field.

**Cheap, always computed**: exact match (aggressive/light normalized), normalized Levenshtein, token Jaccard, token overlap ratio, length ratio, token count difference, prefix/suffix similarity.

**Moderate cost, computed on blocked candidates**: Jaro-Winkler, token-set/token-sort similarity (the organizer's own "Acme Robotics Inc." example is exactly a legal-suffix + minor variation case these catch), character n-gram similarity, substring containment (DBA/trade names), acronym compatibility.

**Rarity-weighted (Section 7)**: rare-token overlap — computed for the matcher's feature vector, never inside blocking's exact-match tier.

Compute key features on more than one normalization view; the *gap* between views is itself informative.

---

## 7. GENERIC TOKEN AND RARE TOKEN HANDLING `[P1 — hypothesis, confirm with Section 3 before treating as load-bearing]`

**This section is a hypothesis, not a confirmed dataset fact.** Generic business terms (`hotel`, `restaurant`, `services`, `group`, `ltd`) are a well-documented general failure mode in business-name matching, but whether *this* dataset actually exhibits meaningful generic-token collision must be measured in Section 3's token-frequency table before this component is treated as essential.

If confirmed: compute document frequency per token across S1∪S2∪S3, IDF-weight, define `rare_token_overlap`, `distinctive_token_count`, `generic_token_ratio` exactly as before. Do not remove generic tokens from the text — downweight in evidence, never delete.

If Section 3 shows generic-token collision is rare or low-impact in this specific dataset: keep the plain Jaccard/overlap features from Section 6, skip the added IDF machinery, and note the empirical finding in the methodology document rather than building unused infrastructure.

---

## 8. ADDRESS RESOLUTION ENGINE `[P0 similarity features; P1 structured parsing]`

**Address is a major, organizer-confirmed identity signal** — explicitly named alongside business name as the two fields the entire reconciliation is built on, and explicitly named as a way records land in the same block ("a shared address"). Treat it with equal engineering weight to name, not as a secondary field.

Always retain `raw_normalized_address` regardless of parser success — landmark-style addresses ("Near SBI ATM") are explicitly named by the organizers as expected, not exceptional, and will routinely defeat structured parsing.

Best-effort structured extraction (house/building number, street, locality, city, state, postal code, unit, landmark phrase), grouped by rough format family discovered from Section 3's profiling, never by hard-coded per-country templates.

Features regardless of parse success: whole-address edit similarity, token similarity, character n-gram similarity, numeric-token similarity (catches house-number/postal-code agreement even when parsing fails to label it).

Features when structured extraction succeeds on both sides: house-number match, postal-code match, city/locality match, street similarity, state match, and **component conflict count** as a first-class contradiction feature (Section 14).

**Critical, organizer-emphasized caveat**: shared address must **never** automatically imply the same business. The organizer video's own worked example includes "a different business that happens to share an address" as an expected, deliberately-retrieved blocking lookalike. Multiple legitimate businesses can occupy the same building, office complex, mall, or business park, or the same street address with different units. This must be a named case in the adversarial suite (Section 36), not an assumption baked into the address-similarity features.

---

## 9. COUNTRY HANDLING `[P2 — conditional on Section 2.1a; never foundational]`

**This section is entirely gated by Section 2.1a's verification.** Do not read this section as license to build country into the core architecture.

If Section 2.1a confirms measurable, holdout-validated signal: exact match on normalized country string, missingness indicators, normalized-string similarity, and an explicit `country_conflict` flag when both sides are present and non-matching — computed exactly like any other auxiliary evidence family, fed into the matcher and into contradiction scoring (Section 14) as one input among many, never as a standalone rule.

If Section 2.1a does not confirm measurable signal: country plays no predictive role. It may still be logged for debugging/explainability and displayed in error-analysis tooling, but it is not part of the feature vector.

**In either case, absolutely and unconditionally**: country is never a hard filter, never a blocking key, and never hard-coded against `{US, India}` or any fixed list — this is required by the official PDF regardless of what Section 2.1a's predictive-value finding turns out to be, because France appears only in test.

---

## 10. BLOCKING / RETRIEVAL ARCHITECTURE `[P0]`

**This section now mirrors the organizer's own conceptual pipeline directly, with the advanced techniques framed explicitly as extensions of it, not replacements.**

```
SOURCE 1
   ↓
NAME + ADDRESS BASED BLOCKING   ← organizer's own description: "a cheap key
   ↓                              built from both the name and the address"
CANDIDATE PAIRS                 ← allowed to include lookalikes, by design
   ↓
MATCHING MODEL                  ← removes the lookalikes
   ↓
FINAL MATCH SET
```

**Foundational blocking families (map directly to the organizer's description):**
- **Name-based blocking**: exact `AGGRESSIVE_NORMALIZED` name match, rare-token inverted index (Section 7, if confirmed useful), character n-gram / TF-IDF retrieval for typo tolerance.
- **Address-based blocking**: address-token inverted index, numeric-token (postal/house-number) inverted index, address character n-gram retrieval.

**Extensions, framed explicitly as elaborations of the same name/address concept, not separate ideas** `[P1, add only as candidate-recall measurement in Section 11 shows a need]`: BM25-style lexical retrieval over combined name+address text (general recall safety net); prefix blocking; phonetic blocking `[P2, data-gated]`; transliteration-aware blocking `[P2, data-gated]`; a difficult-record fallback pass for any entity that produced zero candidates from every blocker above, so no S1 entity silently gets an empty candidate set from blocker miss rather than genuine zero-match.

**The organizer's own worked example must be explicitly satisfied by this design**: for an S1 entity "Acme Robotics," the candidate set legitimately includes (a) the correct match, (b) "Acme Robotics" at a different, wrong address, and (c) a different business that happens to share the correct address. **All three belong in the candidate set.** Rejecting (b) and (c) is the matcher's job (Section 20-22), not the blocker's. A blocking design that tries to be clever enough to exclude (b)/(c) upfront is solving the wrong problem and will cost recall on genuine matches that happen to look like (b) or (c) in the process.

Every candidate carries blocker provenance (which blocker(s) retrieved it) — feeds the matcher (Section 13) and the recall audit (Section 11).

The final, budgeted, deduplicated union of all blockers is written to `candidate_pairs.tsv` — the exact set the matcher scores, per Section 2.2's semantics.

---

## 11. CANDIDATE RECALL `[P0 — organizer-confirmed as the single highest-priority measurement]`

Directly quoting the organizers: *"Your blocking strategy sets the ceiling on the recall you can achieve... you cannot match a record you never consider."* This is not this document's opinion — it's the organizers' own stated priority, and it is treated here as the first number measured after the baseline pipeline exists, before any time is spent tuning the matcher.

**The optimization hierarchy, made explicit (this replaces any implication that candidate-pair quality is itself a target):**

```
              MACRO F0.5                          ← the only leaderboard-scored objective
                  ↑
        FINAL MATCH DECISIONS  (matching_results.tsv)
                  ↑
       MATCHER + ENTITY-LEVEL DECISION
                  ↑
             CANDIDATE QUALITY                    ← matcher's job to filter
                  ↑
             CANDIDATE RECALL                     ← the hard ceiling; measured, not assumed
                  ↑
                BLOCKING                          ← optimized for recall + feasibility,
                                                       NOT for candidate precision
```

`candidate_pairs.tsv` sits at the bottom of this chain as a diagnostic artifact for auditing blocking — it is never itself the thing being optimized, and a blocker that maximizes candidate-set "cleanliness" at the cost of recall is optimizing the wrong end of this hierarchy.

**Measure, before building the matcher**: overall candidate recall on training ground truth; per-blocker marginal recall; recall by source (S2 vs S3); recall by corruption severity (name/address edit-distance quartiles on matched pairs); recall by missingness; recall by name length; and, once Section 2.1a resolves country's role, recall broken out across country values including any internally-simulated unseen-country slice as a rehearsal for the real France test behavior.

**Candidate budget selection**: plot recall vs. candidates-per-entity and choose the smallest budget that doesn't measurably sacrifice recall — never an arbitrary round number.

**Exit criterion**: candidate recall on held-out data is a headline number in every experiment log entry (Section 42), reported alongside F0.5 — a matcher change that improves F0.5 but silently coincides with a retrieval-stage recall regression is a regression, not an improvement, and this is exactly how that gets caught.

---

## 12. CANDIDATE EXPLOSION `[P0]`

Unchanged in substance. Track p50/p90/p95/p99/max candidates-per-entity, per blocker and overall. Safeguards: top-K per blocker, frequency caps on over-common index terms, rare-token-prioritized ranking within a blocker, global per-entity budget chosen from Section 11's recall-vs-budget curve. The difficult-record fallback (Section 10) is exempt from aggressive capping (it only fires on the small zero-candidate residual) but still budget-bounded.

---

## 13. PAIRWISE FEATURE ENGINE `[P0, country family conditional per 2.1a]`

| Family | Examples | Status | Failure mode |
|---|---|---|---|
| NAME | Section 6 | P0 | inflated by generic tokens if rarity-weighting isn't confirmed useful and applied |
| ADDRESS | Section 8 | P0 | silently neutral when address missing on either side — must pair with a missingness flag |
| COUNTRY | Section 9 | **P2, conditional on 2.1a** | must degrade gracefully on unseen country strings if used at all |
| NUMERIC | postal/house-number digit runs | P1 | numeric coincidence can look like strong evidence when it's actually common |
| MISSINGNESS | per-field present/absent flags | P0 | naive models can learn "less data ⇒ no-match" as a shortcut even when available fields agree strongly |
| RARITY | Section 7 | **P1, conditional on Section 3 confirming generic-token collision is real here** | needs corpus-wide, not per-batch, IDF |
| CONTRADICTION | Section 14 | P0 | the most protective family for precision; never drop without strong justification |
| RETRIEVAL | blocker provenance, blocker count | P0 | provenance from a weak blocker shouldn't be weighted like provenance from an exact-match blocker |
| ENTITY-LEVEL | Section 22-23 | P0 | leakage risk if computed with cross-fold information — must stay strictly within-fold |
| CROSS-SOURCE | Section 29 | **P2** | risk of invalid transitive inference |

Every feature carries a docstring: meaning, expected signal, known failure mode — enforced via a lightweight feature-registry convention.

---

## 14. CONTRADICTION FEATURES `[P0 for name/address-derived contradictions; P2 for country_conflict, per 2.1a]`

Explicit negative-evidence features, computed only when both sides have a confident, non-missing value (missing ≠ contradiction):
- `postal_conflict`, `house_number_conflict`, `city_conflict`, `street_conflict` — **P0**, derived from the organizer-confirmed address field.
- `country_conflict` — **P2**, included only if Section 2.1a confirms country carries validated signal; if not, this feature is omitted rather than computed-but-ignored, to avoid adding noise to the feature vector.
- Aggregate `contradiction_count`, `contradiction_on_confident_fields_only`.

**Why this matters more than a plain similarity-only system**: the organizer's own example — "a different business that happens to share an address" surviving into the candidate set — is precisely the case a name/address-similarity-only matcher would over-trust. A confident address-component disagreement should suppress a match far more than the same name score would suppress it in the absence of contradiction — validate this behavior directly against the shared-address adversarial case (Section 36).

---

## 15. INFORMATION QUALITY ENGINE `[P1]`

Unchanged: `name_quality`, `address_quality` (and `country_quality` only if 2.1a keeps country active) from emptiness, short-string, generic-string, punctuation/number-only, and suspicious-Unicode detection. Feeds the Difficulty Engine (Section 27) as a routing signal, never a deletion trigger.

---

## 16. EVIDENCE DIVERSITY `[P1]`

Unchanged in principle, adjusted for the confirmed two-field-primary structure: name-family and address-family are the two genuinely independent evidence sources this challenge is built around (country, if active at all, is a third but weaker one pending 2.1a). Define `independent_evidence_count` across whichever families are actually active in the shipped feature set, and require it as an explicit input to high-confidence decisions in Section 22 — a match built on name similarity alone, with address either missing or non-corroborating, should require a materially higher name-similarity bar than one with address agreement, given address is organizer-confirmed as co-equal identity evidence.

---

## 17. TRAINING PAIR GENERATION `[P0]`

Unchanged: positive pairs from ground truth; random negatives for baseline separation; structured negatives targeting the organizer's own named failure modes directly — same name / different address, same address / different name (this is literally the organizer's worked lookalike example), same city / different business, similar name / conflicting confident address component; hard negatives mined iteratively (Section 18). All negative generation stays fold-local (Section 19).

---

## 18. HARD-NEGATIVE MINING `[P1, corrected methodology]`

```
FULL TRAINING DATA
   ↓
INNER TRAIN / OUT-OF-FOLD (OOF) SPLIT     ← never the final holdout
   ↓
TRAIN on inner-train, PREDICT on OOF
   ↓
MINE hard negatives from high-scoring OOF false positives
   ↓
CLASSIFY error (Section 38 taxonomy)
   ↓
ADD to inner-train only
   ↓
RETRAIN
   ↓
OUTER VALIDATION (still not the final holdout)
   ↓
[repeat as needed]
   ↓
FINAL HOLDOUT — touched exactly once, at the very end, for the final reported number
```

**Correction from the previous revision**: hard negatives must be mined only from an OOF split that is distinct from whatever will later serve as the final holdout. Never mine from, and then train on, examples drawn from the untouched final holdout — that holdout exists specifically to give one trustworthy, unbiased final estimate, and it must remain completely unseen by any training or mining step until the single final confirmation pass.

---

## 19. VALIDATION DESIGN `[P0, corrected for source-record graph structure]`

**Entity-aware splitting by S1 remains the default**, but this revision adds a required check the previous version omitted: **verify the actual relationship graph before assuming S1-grouping alone prevents leakage.**

Specifically, using the fan-out check from Section 3: if any single S2/S3 record legitimately matches more than one S1 entity in ground truth, then splitting purely by `source1_entity_id` could still place the *same* S2/S3 record in both a train fold and a validation fold (attached to different S1 entities), which is a milder but real form of leakage — the model could learn to recognize that specific S2/S3 record rather than learning generalizable matching logic.

**Required process:**
1. Measure S2/S3 record fan-out from ground truth (Section 3).
2. If fan-out is negligible (each S2/S3 record matches at most one S1 entity, or nearly so): S1-grouped `GroupKFold` splitting is sufficient, as originally planned.
3. If fan-out is non-negligible: extend grouping to a connected-components basis — group S1 entities together in the same fold if they share any S2/S3 record, so no S2/S3 record crosses the train/validation boundary either.

**Splits**: train (fitting + hard-negative mining), validation (threshold tuning, frequent iteration), holdout (rarely touched, final trustworthy estimate, never used for tuning). Corruption-stratified validation slices, repeated seeds, and threshold-stability checks across seeds remain as before. Report candidate recall alongside F0.5 on every split to catch retrieval-stage regressions hiding inside apparent matcher improvements.

---

## 20. MODEL SELECTION `[P0]`

Unchanged conclusion, now explicitly framed against the organizers' own stated neutrality on technique (Section 33): the video and PDF impose no requirement for deep learning, and a classical GBDT is a fully valid, competition-appropriate primary matcher.

| Model | Strengths here | Weaknesses here |
|---|---|---|
| Logistic Regression | fast, interpretable, good sanity baseline | can't capture interactions without manual terms |
| Random Forest | handles interactions/missingness reasonably | typically underperforms boosting on this kind of tabular data |
| XGBoost / LightGBM / CatBoost | native missing-value handling (missingness is pervasive and organizer-acknowledged in this data — abbreviated addresses, landmark references, partial fields), fast, strong tabular performance, MIT/Apache-licensed, trivially under the 8B-parameter constraint | needs the feature engine (Sections 6-16) to do the representational work, since it doesn't learn text representations itself |

**Default, to confirm experimentally**: a gradient-boosted tree (LightGBM or CatBoost) as the primary matcher; logistic regression retained permanently as a fast sanity-check baseline, never discarded.

---

## 21. MODEL CALIBRATION `[P1]`

Unchanged: Platt scaling or isotonic regression, plotted and validated, kept only if ablation shows a measurable difference in the entity-level decision engine's behavior — since the decision engine (Section 22) uses both relative (score-gap) and absolute (threshold) signals, calibration is worth testing but not assumed necessary.

---

## 22. ENTITY-LEVEL DECISION ENGINE `[P0 — the highest-leverage design decision, now stated with explicit pair-level vs entity-level separation]`

**Two distinct problems, kept explicitly separate:**

**Pair-level** (what the matcher answers): `P(match | S1 entity, one candidate)` — a single probability for one candidate pair.

**Entity-level** (the actual scored task): for each S1 entity, produce a **set**:
```
S1_001 → {}
S1_002 → {S2_018}
S1_003 → {S2_034, S3_117}
```

A naive `score > threshold → include` rule applied independently per pair is *not* entity-level reasoning — it ignores everything the rest of that entity's candidate set implies. The decision engine must look at the whole per-entity score distribution.

**Per-entity signals, computed after pairwise scoring:** `top_score`, `second_score`, `third_score`, `score_gap = top_score - second_score`, `candidate_count`, `num_above_threshold`, `evidence_diversity` for the top candidate(s) (Section 16), `contradiction_flags` for the top candidate(s) (Section 14).

**Decision categories:**
- **NO MATCH**: `top_score` below a validated floor, OR top score is only moderate with active contradictions and low evidence diversity, OR zero candidates were retrieved at all (Section 25).
- **SINGLE MATCH**: exactly one candidate clears the threshold with acceptable evidence, or the top candidate clears with a large score gap over the second even at a moderate absolute score.
- **MULTIPLE MATCH**: more than one candidate independently clears the threshold with acceptable evidence — never force top-1 (Section 26).
- **AMBIGUOUS** (internal only): tightly clustered near-threshold scores with no clear gap — routed to Tier 2/3 (Section 28).

Keep this as a small, validated decision layer (a lightweight second model or a short, fully-justified rule set) — not a hand-grown cascade of nested if/else rules.

---

## 23. SCORE GAP `[P0]`

The organizer's underlying intuition ("when in doubt, it is safer not to merge") plus the two structurally different cases below drive this section:

**Case A** (confident, likely unique match): `0.97 / 0.43 / 0.21` — a clear leader, safe to treat as a confident single match.

**Case B** (genuinely ambiguous): `0.97 / 0.95 / 0.91` — same top score as Case A, but no clear leader; this needs more evidence (Tier 2/3, Section 28) before deciding, not an automatic accept of the top candidate.

**Important correction**: a high score gap does **not** by itself prove a match — a large gap between two *uniformly low* scores (e.g., `0.31 / 0.05 / 0.02`) is still evidence for **no match**, not for confidently selecting the 0.31 candidate. Score gap is one input among several (absolute score, evidence diversity, contradictions), never a standalone rule. All specific thresholds and interactions here must be tuned and confirmed on validation/holdout data (Section 24), not asserted from the two illustrative cases above.

---

## 24. THRESHOLD OPTIMIZATION `[P0]`

Unchanged in method, restated with the organizer's own heuristic as the guiding principle rather than an assumption: optimize directly for macro F0.5 via a threshold sweep on validation, confirmed stable across seeds/folds, confirmed on the (rarely touched) holdout before locking. No-match impact and multi-match impact plotted separately, since a threshold change can improve one while hurting the other. "Safer not to merge when in doubt" is implemented as *outcome* of this empirical optimization (F0.5's own math already encodes the 2:1 precision weighting) — not as a manually imposed extra conservatism layered on top of the F0.5-optimal threshold, which would just be sub-optimizing against the actual metric.

---

## 25. NO-MATCH ENGINE `[P0 — organizer-confirmed as equally important as matching]`

Directly organizer-stated: *"identifying the businesses with no match is just as important as finding the ones that do match."* This is not a hypothesis to validate — it's a direct instruction from the challenge organizers, and it is why this section remains one of the highest-priority components in the document.

Detect "no plausible candidate" as a genuine class using: absolute top score (validated floor), evidence diversity (a single-family match with no corroboration should lean toward no-match), active contradictions on the best candidate, score-distribution shape (uniformly low scores across all candidates is itself no-match evidence, distinct from one candidate meaningfully ahead of a low pack — see Section 23's correction), candidate provenance quality (candidates from only the weakest blockers warrant lower confidence), and the automatic no-match outcome for any entity with zero retrieved candidates.

**Do not over-correct into blanket conservatism.** "Safer not to merge when in doubt" is encoded by F0.5's own math (precision weighted 2×) and by this engine's design — it is not license to set an arbitrarily high, unvalidated threshold that trades away real recall for a false sense of safety. The threshold is still empirically tuned (Section 24).

---

## 26. MULTI-MATCH ENGINE `[P0]`

Unchanged: every candidate independently clearing the validated threshold is eligible, regardless of whether another candidate also clears it. Evidence-consistency conflicts across a selected multi-match set are surfaced to error analysis (Section 38) rather than silently resolved in v1. Duplicate-candidate suppression must not assume one candidate per entity. Output ordering is deterministic (sorted by id) though not itself scored.

---

## 27. DIFFICULTY ENGINE `[P1]`

Unchanged: `EASY`/`NORMAL`/`DIFFICULT` classification from missingness, generic-name ratio (if confirmed relevant, Section 7), short-name flag, candidate count, low top score, small score gap, contradictions, corruption indicators, and — if measurable — script/language uncertainty. Purely a routing signal for Section 28, never a hidden input to the decision engine's actual accept/reject logic.

---

## 28. THREE-TIER COMPUTATION `[P1]`

Unchanged: Tier 1 (all entities — normalization, blocking, full feature set, matcher scoring), Tier 2 (all entities — the standard entity-level decision engine), Tier 3 (only `DIFFICULT`/`AMBIGUOUS` entities — wider retrieval, cross-source consistency, and any ablation-justified P2 component). Expensive computation only where it can change a decision, applied only to the necessarily small residual population.

---

## 29. CROSS-SOURCE CONSISTENCY `[P2]`

Unchanged, kept strictly evidence-only: mutual S1↔S2 and S1↔S3 corroboration is legitimate supporting evidence only when S1's own evidence for *both* is independently non-trivial. Never chain S2-resembles-S3 into an S1-S3 conclusion without S1-S3 evidence of its own. Feed as one additional feature into Tier 2/3 reasoning, never as a standalone rule, and only build it if profiling/error-analysis shows a real gap it would close.

---

## 30. GRAPH REASONING `[P2 — experimental, tie-breaking role only, unconfirmed by organizers]`

Unchanged in structure from the prior revision: problem it solves (structural inconsistency detection in the `AMBIGUOUS` residual), risks (invalid transitivity, added complexity, harder to explain to judges), test method (Tier-3, ambiguous-only, ablated specifically on that subpopulation), keep condition (positive, seed-stable, holdout-confirmed ΔF0.5 restricted to that subpopulation), and when not to use it (if the ambiguous population is small or already resolved by cheaper evidence-diversity/contradiction features). Explicitly not required or implied by the organizer material — build only if the ambiguous-population size and error analysis (Section 39) justify the investment.

---

## 31. ENTITY CLUSTERING `[P2 — likely not the right formulation, unconfirmed]`

Unchanged conclusion: this task is S1-anchored (find matches *for* each S1 entity), not a general S2-S3 dedup/cluster problem — full transitive clustering is likely unnecessary unless Section 3 profiling reveals systematic S2-S3 duplication an S1-anchored approach handles poorly. If pursued, all the same transitivity, validation, and ablation cautions as Section 29 apply, more acutely.

---

## 32. EMBEDDING RETRIEVAL `[P2 — unconfirmed, high verification bar]`

Unchanged trade-off analysis (semantic recall benefit vs. added runtime, added false-positive risk under a precision-heavy metric, reduced interpretability for judge defense, and the license/parameter/offline constraints from Section 2.6). Explicitly not implied or required by the organizer material, which names no specific technique. Keep only with a clear, holdout-confirmed ΔF0.5 or Δcandidate-recall beyond what the confirmed lexical/structural blockers in Section 10 already deliver — a high bar given the organizer's own blocking description is itself lexical (a "cheap key" from name+address), not embedding-based.

---

## 33. LLM / ADVANCED MODEL USAGE `[P2, strictly offline-development role — never inside the scored pipeline]`

The organizer video and PDF impose **no requirement** for deep learning, LLMs, embeddings, or graph neural networks — this must not be read into the challenge. A classical GBDT pipeline is a complete, valid, competition-appropriate solution if experiments support it (Section 20).

If any advanced technique is used at all, it is restricted to offline, human-in-the-loop development aid (error-analysis pattern-spotting, ambiguous-case explanation for a developer, feature-idea brainstorming) — never as the deterministic matcher inside the scored pipeline, both because of the explicit external-data/API prohibition (an API-based LLM is categorically excluded, Section 2.6) and because of the license/parameter-count constraint on any model that does run inside the pipeline. Any such offline aid still requires the resulting idea to go through the same feature-registry, ablation, and validation process as anything else in this document — an LLM suggesting a feature is not the same as that feature being validated.

---

## 34. SELF-TRAINING / PSEUDO-LABELING `[P2 — experimental, high risk, unconfirmed]`

Unchanged guardrails: high-confidence-only pseudo-labels, independent evidence corroboration required, strict protection of the final holdout (never touched by pseudo-labeled data, consistent with Section 18's corrected methodology), explicit rollback path, and mandatory re-running of the adversarial suites (Sections 36-37) after every pseudo-labeling round to catch uncontrolled error amplification.

---

## 35. EDGE CASE MASTERPLAN `[P0 — build the harness early, populate continuously with real dataset examples once available]`

Same structured template as before (`Scenario | Why difficult | How it can fool the system | Detection | Expected behavior | Fallback | Test strategy | Metric impact`), populated with **actual example record pairs from this dataset once Section 3 profiling is complete**, not only generic illustrative names.

Minimum required coverage, organized around the two confirmed core fields: missingness (empty name/address/both); name pathology (short/generic name, typo, deletion/insertion/swap, reorder, abbreviation, acronym); script/encoding (Unicode variation, zero-width chars, mixed scripts, transliteration, homoglyphs — populate only the ones Section 3 confirms occur); **address pathology, weighted heavily given address's organizer-confirmed importance**: abbreviation, landmark-only reference, missing components, municipal-numbering format variation, component reordering; confusable structure (same name/different address — the organizer's own example; same address/different business — also the organizer's own example; shared building/office-complex/mall address with legitimately different businesses; postal collision; house-number/unit variation); decision-boundary cases (true zero-match, single-match, multi-match, tied candidates, high-score false positive, low-score true positive); systemic (source-specific corruption pattern per Section 3, candidate explosion, blocking failure, malformed input, nondeterminism, distribution shift on any test-only country value, missing feature from upstream parse failure).

---

## 36. ADVERSARIAL FALSE-POSITIVE TESTING `[P0 — guardrail, not the objective]`

**Reframed from the previous revision**: this suite is a **release guardrail**, not the thing being optimized. The true optimization target is always macro F0.5 on the real, held-out data (Section 19, 24). A synthetic test that starts to conflict with genuinely F0.5-optimal behavior on real data is a sign the *test* needs revising, not that the model should be distorted to satisfy it.

Cases, directly reflecting the organizer's own named lookalike patterns: same name + different address (organizer's own example); **same address + different business** (organizer's own example — this is the highest-priority case in this suite given it comes directly from the challenge walkthrough); shared building/office-complex/mall address with different legitimate businesses; same postal code + similar-but-not-identical name; same building + different unit; similar acronym + otherwise-unrelated entity; high name similarity + a confident, independent address contradiction.

**Explicit correction from the previous revision**: "same name + different country" must **not** automatically imply no-match as a hard rule — if Section 2.1a finds country to be unreliable or largely unused, a country mismatch in a test case should not by itself be treated as decisive; the test case (and the system) should rely on whatever evidence families are actually confirmed useful. Every case in this suite should be reviewed against Section 2.1a's finding before being treated as a hard pass/fail gate.

**Target, not absolute mandate**: the system should pass all cases here that reflect genuinely correct behavior, but a failing case is a prompt to investigate — via real held-out F0.5 impact — not an automatic requirement to patch the model into passing a synthetic fixture at the expense of real performance. Track pass rate as a diagnostic in every experiment log entry (Section 42), re-run after every change (this operationalizes Section 40's self-attacking loop), but never let synthetic-suite pass rate override a holdout F0.5 result.

---

## 37. ADVERSARIAL FALSE-NEGATIVE TESTING `[P0 — guardrail, not the objective]`

Same reframing as Section 36 applies. Cases: heavy typo, abbreviation, token reorder, missing address, missing country (should have no effect if country isn't a confirmed feature), transliteration (only if confirmed relevant), Unicode corruption, multiple simultaneous transformations stacked on one pair — the realistic worst case, since noisy real-world data rarely has just one corruption type at a time. A regression here is treated with the same severity as a Section 36 regression — over-correcting toward conservatism to pass Section 36 without also protecting against loss of genuine recall here would itself hurt F0.5, given F0.5 still rewards recall, just less than precision.

---

## 38. ERROR TAXONOMY `[P0]`

Unchanged, extended as a living document as real errors are observed:

```
BLOCKING_FAILURE          NAME_AMBIGUITY           COUNTRY_CONFLICT (only if country active)
GENERIC_NAME              SHORT_NAME               MISSING_DATA
UNICODE_ERROR             TRANSLITERATION          ABBREVIATION
NORMALIZATION_FAILURE     ADDRESS_AMBIGUITY        SHARED_ADDRESS_FALSE_MERGE
MODEL_ERROR               THRESHOLD_ERROR          NO_MATCH_ERROR
MULTI_MATCH_ERROR         DATA_QUALITY_ERROR       CANDIDATE_EXPLOSION
DISTRIBUTION_SHIFT        IMPLEMENTATION_ERROR
```

`SHARED_ADDRESS_FALSE_MERGE` added explicitly given the organizer's own emphasis on this specific failure mode.

---

## 39. ERROR PRIORITIZATION `[P1]`

Unchanged: rank logged error categories by `(F0.5 impact × precision-weighted damage) / fix difficulty`, work top-down, using real held-out error counts, not synthetic-suite counts alone.

---

## 40. SELF-ATTACKING OPTIMIZATION LOOP `[P0 — core methodology]`

Unchanged loop (`BASELINE → MEASURE → ATTACK → FIND FAILURE → ROOT CAUSE → HYPOTHESIS → ONE CHANGE → ABLATION → KEEP/REMOVE → NO-REGRESSION TEST → ATTACK AGAIN`), with "ATTACK" now explicitly meaning both the adversarial guardrail suites (Sections 36-37) *and* direct held-out F0.5 measurement — never the guardrail suite alone, per the Section 36-37 reframing.

---

## 41. ABLATION FRAMEWORK `[P0]`

Same cumulative structure, reordered to reflect what's now confirmed vs. conditional:

```
baseline (name+address normalization, name+address blocking, core name/address
          features, contradiction features, GBDT, validated threshold)
+ rare/generic-token handling        [only if Section 3 confirms relevance]
+ address structured parser
+ char-ngram / phonetic retrieval    [phonetic only if data-justified]
+ country features                   [only if Section 2.1a confirms signal]
+ transliteration handling           [only if data-justified]
+ score gap / entity-level decision refinements
+ cross-source consistency
+ (only if reached) graph / embeddings / calibration / pseudo-labeling
```

Every step measured on held-out data, seed-repeated: ΔF0.5, Δprecision, Δrecall, Δcandidate recall, Δruntime, Δmemory. A component with positive validation ΔF0.5 but flat/negative holdout ΔF0.5 does not ship.

---

## 42. EXPERIMENT TRACKING `[P0]`

Unchanged schema, with `country_active: bool` and `country_verification_result` added as explicit logged fields given Section 2.1a's central role in this revision:

```
experiment_id, git_commit, dataset_version, split, seed,
normalization_version, blockers, candidate_budget, features,
country_active, country_verification_result,
model, hyperparameters, threshold, calibration,
F0.5, precision, recall, candidate_recall, runtime, memory,
error_categories, adversarial_pass_rate
```

---

## 43. PERFORMANCE ENGINEERING `[P1]`

Unchanged: benchmark each stage (loading, normalization, index construction, retrieval, feature computation, model inference, Tier-3 reasoning, output generation) separately; track CPU/RAM/runtime/candidate counts/batch size; apply caching, precomputation, batching, vectorization, indexing, and parallelism only once a bottleneck is actually measured, not preemptively.

---

## 44. DISTRIBUTION SHIFT `[P1]`

Unchanged, with country-related shift now explicitly conditional: compare train vs test name/address length, missingness, token frequency, script distribution, and — once end-to-end — candidate-count and score distributions. Confirm France's presence/volume in test (this one **is** confirmed by the official PDF, unlike most country-related claims), and, if Section 2.1a keeps country active, specifically verify unseen-country behavior doesn't degrade the pipeline (via the internally-simulated unseen-country validation slice from Section 19/44).

---

## 45. SCORE DRIFT `[P1]`

Unchanged: monitor validation vs. test-time score distributions, top-score shape, score-gap distribution, candidate-count distribution as an early, pre-leaderboard warning sign of threshold instability.

---

## 46. REPRODUCIBILITY `[P0]`

Unchanged: pin Python version, exact package versions, all random seeds, full configuration (including the `country_active` flag from Section 2.1a), model/feature version identifiers. A teammate with only the repo and raw data should reproduce the exact submitted output end to end — also an explicit final-package requirement (runnable `code/` folder, `README.md`, `requirements.txt`).

---

## 47. SECURITY / ROBUSTNESS `[P1]`

Unchanged: path traversal, malicious/malformed ids, invalid encodings, oversized strings (Section 48), regex denial-of-service, memory exhaustion, candidate explosion. Add explicitly: **zero outbound network calls anywhere in the pipeline**, tested by running the full pipeline in a network-disabled environment as a release check — the strongest possible enforcement of the organizer's external-data prohibition (Section 2.6).

---

## 48. RESOURCE GUARDS `[P0]`

Unchanged: `MAX_STRING_LENGTH`, `MAX_TOKENS`, `MAX_CANDIDATES_PER_ENTITY`, `MAX_BATCH_SIZE`, `MAX_WORKERS`, `MEMORY_TARGET`, all configuration-driven, all logged when triggered, all degrading gracefully rather than raising uncaught exceptions that could drop an entity from output.

---

## 49. FAILURE RECOVERY `[P0]`

Unchanged: any Tier-3/experimental (P2) component must fail gracefully back to the Tier-1/2 standard-path result, tested by deliberately disabling each P1/P2 component and confirming a full, valid submission still results.

---

## 50. SUBMISSION VALIDATOR `[P0 — run before every leaderboard upload]`

Unchanged, wrapping/extending the provided `utils/validate_submission.py`: full checks on `matching_results.tsv` (coverage, no duplicates, valid test-set ids, no S1 self-matches, correct empty/multi representation) and `candidate_pairs.tsv` (valid ids, no illegal pairs, the matches-are-a-subset-of-candidates invariant treated as a P0 bug if violated), plus global determinism and exact schema conformance. Run as the literal last pipeline step before every upload — organizer-stated directly as a requirement, not a suggestion.

---

## 51. REPOSITORY STRUCTURE `[P0]`

Unchanged structure (`src/{ingestion,normalization,blocking,features,model,decision,evaluation,errors,validator}`, `configs/`, `experiments/`, `reports/`, `tests/{unit,edge_cases,adversarial}`, `README.md`, `requirements.txt`), with `reports/data_profile.md` and the Section 2.1a country-verification finding treated as first-class, checked-in artifacts, not throwaway exploration.

---

## 52. IMPLEMENTATION ROADMAP `[P0 — reordered so data reconnaissance is the very next real action, before any further architecture work]`

| Stage | Deliverable | Exit criterion |
|---|---|---|
| 1 — Contract lock | Section 2 + 2.1a written and confirmed against official PDF/video | team agreement on every ambiguous point, especially country's status |
| 2 — Data reconnaissance | `reports/data_profile.md` fully populated (Section 3), including the country-agreement check, the source-record fan-out check, and a naive-blocker viability check | every hypothesis elsewhere in this document is resolved to confirmed/rejected here, before any further build step |
| 3 — Baseline | ingestion + light normalization + one name+address blocker + simple similarity threshold | valid (if weak) submission, validator passes |
| 4 — Retrieval | full name+address blocking suite (Section 10), extensions added only if Stage 2 justifies them | candidate recall near its ceiling for the chosen budget |
| 5 — Features | full feature engine, country family included only if Section 2.1a confirmed it | every feature's sign matches its documented expectation on spot-check |
| 6 — Matcher | GBDT with corrected entity/graph-aware splitting (Section 19-20) | measurable, seed-stable improvement over baseline |
| 7 — Decision engine | Sections 22-26, separated from the classifier | zero-match and multi-match correctly handled on constructed fixtures |
| 8 — Hard negatives | Section 18's corrected OOF loop, run at least once | measurable precision gain, holdout still untouched |
| 9 — Edge-case + guardrail lab | Sections 35-37 populated with real dataset examples | guardrail pass rate high and any failures individually investigated against real F0.5 impact |
| 10 — Advanced experiments | any P1/P2 component, one at a time, through Section 41's ablation | keep only components with positive, stable, holdout-confirmed ΔF0.5 |
| 11 — Optimization | Section 24 threshold finalization, Section 43 performance pass | thresholds locked from holdout, runtime within any stated limits |
| 12 — Submission | Section 50 validator, Section 51 packaging, methodology doc | validator `PASS`, package matches required structure |

**The next real action after this document is finalized is Stage 2 — data reconnaissance — not further architecture work.**

---

## 53. EXPERIMENT PRIORITY MATRIX `[P1]`

| Experiment | Priority | Keep condition |
|---|---|---|
| Name+address multi-blocking | P0 | always — foundational, organizer-confirmed |
| Contradiction features (address-derived) | P0 | always |
| Entity-level decision engine | P0 | always — correct F0.5 semantics depend on it |
| No-match engine | P0 | always — organizer-confirmed equal importance to matching |
| **Country verification (Section 2.1a)** | **P0** | run once, early; result gates all downstream country work |
| Rare/generic-token handling | P1 | keep only if Section 3 confirms generic-token collision is real here |
| Hard-negative mining | P1 | keep if ΔF0.5 positive and no-regression suite passes, holdout untouched |
| Address structured parsing | P1 | keep if paired with raw fallback and ablation shows gain |
| Country predictive features | P2 | keep only if Section 2.1a confirms holdout-validated signal |
| Cross-source consistency | P2 | keep only as an evidence feature, ablated |
| Calibration | P1 | keep only if ablation shows a difference |
| Phonetic / transliteration blocking | P2 | build only if Section 3 shows real volume justifying it |
| Embeddings | P2 | keep only with clear, stable, holdout ΔF0.5 beyond confirmed lexical blockers |
| Graph reasoning | P2 | keep only for the ambiguous subpopulation, its own ablation |
| Pseudo-labeling | P2 | keep only with strict guardrails and holdout-confirmed gain |

---

## 54. COMPETITION STRATEGY

Unchanged framing, restated without overclaiming: a basic solution stops at single-field fuzzy matching with an arbitrary threshold and no principled zero/multi-match handling. A stronger solution adds multi-blocking and a GBDT matcher but still often treats the pairwise score as the final decision. This architecture's differentiation is in organizer-confirmed measurement discipline: candidate recall tracked as the explicit ceiling it is stated to be; blocking and matching kept in the exact separation the organizers themselves describe; a decision layer that natively handles 0/1/many; and — new in this revision — a refusal to build unconfirmed assumptions (country as core evidence, any specific "advanced" technique) into the architecture before the data or the organizers actually support it. This does not guarantee a result; it guarantees that every claim in the shipped system is either organizer-confirmed or dataset-confirmed, not assumed.

---

## 55. JUDGE DEFENSE

Updated/added answers reflecting this revision:

- **Why blocking, and why is it allowed to retrieve wrong candidates?** Because that's the organizers' own explicit design: blocking favors recall, including deliberate lookalikes, and the matcher is responsible for removing them — verified directly from the challenge walkthrough, not an assumption.
- **Why treat country as conditional rather than a core feature?** Because the organizer video explicitly scopes the challenge to name and address, and the PDF requires open-set, non-hard-coded country handling; we verified empirically (Section 2.1a) whether country adds validated signal before deciding how much weight it gets, rather than assuming a schema column is automatically a strong feature.
- **How do you know a "shared address" isn't automatically treated as a match?** It's a named adversarial case (Section 36) taken directly from the organizer's own worked example, and contradiction features (Section 14) plus evidence-diversity requirements (Section 16) prevent address agreement alone from being sufficient.
- **How do you know an improvement is real, not just a synthetic-test artifact?** The adversarial suites are explicitly guardrails, not the optimization target (Sections 36-37); every real decision is validated against held-out macro F0.5 (Section 19, 41), with the adversarial pass rate reported as a diagnostic alongside it, never in place of it.
- All other answers from the previous revision (why F0.5, why not top-1, how hard negatives/leakage are handled, why GBDT, why not an LLM, what the biggest weakness is) carry over, updated for the corrected hard-negative/OOF methodology (Section 18) and the corrected validation-graph check (Section 19).

---

## 56. DEMO PLAN

Unchanged sequence (easy match, noisy match, address corruption, ambiguous candidates, no-match, multi-match, adversarial false positive, adversarial false negative), with the false-positive demo specifically built around the organizer's own "same address, different business" example, and the ambiguous-candidates demo specifically using a Case-B-style (Section 23) tight score cluster rather than a clear Case-A leader, to show the entity-level decision engine actually distinguishing the two.

---

## 57. FINAL SCORECARD

- [ ] Data reconnaissance complete, including country-verification and source-fan-out checks (Sections 2.1a, 3)
- [ ] Country's predictive role explicitly resolved (active-and-validated, or excluded) before matcher training
- [ ] Multi-view normalization on name+address implemented and unit-tested
- [ ] Name+address blocking suite implemented with provenance tracking, extensions added only where justified
- [ ] Candidate recall measured by subgroup; optimization hierarchy (Section 11) respected — blocking never tuned for candidate precision
- [ ] Full feature taxonomy implemented, country family conditional and clearly flagged
- [ ] GBDT matcher trained with corrected, graph-aware entity splitting
- [ ] F0.5-optimized threshold, stable across seeds, confirmed on holdout
- [ ] No-match engine validated on the singleton-heavy subpopulation — organizer-confirmed priority
- [ ] Multi-match engine validated on the multi-match subpopulation
- [ ] Hard-negative mining uses the corrected OOF methodology; final holdout never touched during mining
- [ ] Edge-case fixture suite built from real dataset examples
- [ ] Adversarial suites treated as guardrails; failures investigated against real holdout F0.5, not patched blindly
- [ ] Shared-address-false-merge case specifically covered and correctly rejected
- [ ] Distribution shift checked, France/unseen-country behavior specifically verified if country is active
- [ ] Runtime/memory benchmarked
- [ ] Every P1/P2 component justified by a logged, holdout-confirmed ablation
- [ ] Zero outbound network calls verified in a network-disabled test run
- [ ] Submission validator passes cleanly
- [ ] Full pipeline reproducible from a clean checkout
- [ ] Final submission zip matches the required structure exactly

---

## 58. FINAL ARCHITECTURE

```
                              SOURCE 1
                                    │
                                    ▼
                            INGESTION + QC
                                    │
                                    ▼
                     MULTI-VIEW NORMALIZATION
                                    │
                                    ▼
                     NAME + ADDRESS BLOCKING
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
                  NAME           ADDRESS         ADVANCED
                 BLOCKS           BLOCKS          BLOCKS
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                             CANDIDATE SET
                                    │
                            candidate recall
                             (hard ceiling,
                          measured, not assumed)
                                    │
                                    ▼
                            FEATURE ENGINE
                        (name + address core;
                         country conditional)
                                    │
                                    ▼
                             GBDT MATCHER
                                    │
                                    ▼
                           ENTITY DECISION
                        (pair-level score →
                         entity-level set)
                                    │
                       ┌────────────┼────────────┐
                       ▼            ▼             ▼
                   NO MATCH      SINGLE        MULTIPLE
                       │            │             │
                       └────────────┼─────────────┘
                                    ▼
                        matching_results.tsv  (scored)
                                    +
                        candidate_pairs.tsv    (audit only)
                                    │
                                    ▼
                             VALIDATOR
```

Tier-3 advanced reasoning exists only for a small, measured `AMBIGUOUS` residual and only where ablation proves it useful — it is not drawn in this diagram's main path because it is explicitly not part of the standard path for the large majority of entities.

---

## 59. FINAL RECOMMENDATION

**FINAL BUILD PATH:**

1. Lock the problem contract, including the country-verification protocol (Sections 2, 2.1a).
2. **Run data reconnaissance now — this is the next real action, not more architecture.** Populate `reports/data_profile.md` fully, including the country-agreement check and the source-record fan-out check.
3. Resolve country's role one way or the other before writing any country-dependent feature code.
4. Build ingestion/QC and name+address normalization, unit-tested.
5. Build the name+address blocking suite; measure candidate recall by subgroup before any model code exists.
6. Build the full feature engine (name/address/contradiction/missingness core; country only if confirmed).
7. Train the GBDT matcher with corrected, graph-aware entity-level validation.
8. Build the entity-level decision engine as a genuinely separate layer; optimize the threshold directly for F0.5.
9. Run at least one corrected (OOF-based) hard-negative mining loop.
10. Build the edge-case and adversarial guardrail suites using real dataset examples; treat failures as investigation prompts against real F0.5, not automatic patches.
11. Only now consider any P1/P2 addition, one at a time, each gated by ablation on holdout data.
12. Run the submission validator, verify zero network calls, package per the required structure, confirm reproducibility.

---

## 60. ABSOLUTE RULES

1. Never claim a guaranteed result.
2. Never assume added complexity is automatically an improvement.
3. Never optimize for anything other than macro F0.5 as the final arbiter.
4. Never let candidate recall go unmeasured — it is the organizer-confirmed hard ceiling.
5. Never assume one-to-one (top-1) matching anywhere in the pipeline.
6. Never ignore zero-match entities — organizer-confirmed as equally important as real matches.
7. Never ignore legitimate multiple matches.
8. **Never treat `country` as a core identity signal equal to name/address until Section 2.1a's verification confirms it — and never hard-filter or hard-code it regardless of that outcome.**
9. Never let validation information leak into training, including during hard-negative mining.
10. **Never mine hard negatives from, or otherwise touch, the final holdout before the single final confirmation pass.**
11. Never add graph/LLM/embedding complexity without a documented, holdout-confirmed ablation result.
12. Never grow a long, ad hoc, manually-tuned rule cascade in place of a validated model or a small, justified rule set.
13. Never trust a single validation split for a final decision.
14. Never ignore runtime or memory.
15. Never treat submission-format correctness as an afterthought — validate before every upload.
16. Never generate `candidate_pairs.tsv` separately from, or after, the actual prediction run that produced `matching_results.tsv`.
17. Never discard raw (unnormalized) representations.
18. Never silently drop a problematic record from the output — every test S1 entity must appear.
19. Never ship an advanced feature without an ablation strategy attached to it.
20. Never leave a major pipeline stage without a measurable exit criterion.
21. Never make a final prediction that can't be explained through structured evidence.
22. Never break reproducibility — pin everything, log the country-active flag alongside every other config value.
23. **Never let a blocker try to be precise — blocking is recall-and-feasibility-oriented by explicit organizer design; rejecting lookalikes is the matcher's job.**
24. **Never treat `candidate_pairs.tsv` as an optimization target in its own right — it is an unscored audit artifact.**
25. **Never assume "same address" implies "same business," or that "same name, different address" implies "different business" — both are named organizer lookalike examples the matcher must resolve, not shortcuts to bake into blocking or features.**
26. **Never let a synthetic adversarial test override a genuine, holdout-confirmed F0.5 result — the suites are guardrails, not the objective.**
27. **Never use any external database, API, geocoding service, mapping service, or third-party lookup anywhere in the scored pipeline — verify with a network-disabled test run.**
28. **Never assert a dataset characteristic (generic-token collision, country reliability, transliteration volume, script mixing) as fact before Section 3's profiling confirms it — state it as a hypothesis until then.**
29. Never design something impractical to actually implement in the available time.
30. **Never lose sight of the actual next action: after this document, the next real step is data reconnaissance, not more architecture.**
