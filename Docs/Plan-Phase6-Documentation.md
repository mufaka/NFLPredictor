# Phase 6: Documentation & Diagnostics Implementation Plan

This document defines the phased implementation plan for the Documentation & Diagnostics phase of the NFL Predictor project, based on [Spec-Phase6-Documentation.md](./Spec-Phase6-Documentation.md). Each phase builds on the previous one and contains checkbox-tracked work items. Requirement IDs (`DD-*`) reference the corresponding entries in the specification.

This is a single-developer learning project. Phases are sized for one person to complete in sittings of an hour or two. Phase 6 is shorter on code than Phase 4 or Phase 5 but longer on prose authoring; the plan splits each deliverable into its own internal phase so commits stay readable. One backward edit to Phase 4 (per-epoch loss-curve emission) and its downstream ripple into Phase 5 are explicit internal phases of their own.

---

## Progress Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Scaffolding and Dependencies | Complete |
| 2 | Phase 4 Backward Edit — Per-Epoch Loss-Curve Emission | Complete |
| 3 | Phase 5 Ripple — Fixture Regen and Test Re-Pin | Complete |
| 4 | Diagnostics Helper Module | Complete |
| 5 | Pipeline Walkthrough Notebook + Export (Deliverable 2) | Complete |
| 6 | Reading-the-Outputs Guide (Deliverable 1) | Pending |
| 7 | Training-Dynamics Doc + Companion Notebook (Deliverable 4) | Pending |
| 8 | Wrap-Up — CLAUDE.md, Idea.md Status, Final Review | Pending |

---

## Current State

- Phases 1–5 are implemented. All entry points (`python -m nflpredictor.{databuild,features,splits,train,evaluate}`) run and pass their fixture-based tests.
- On the dev (CPU-only) machine, the Phase 4 real-data run has not yet been executed — `Data/processed/predictions/` may be empty. Phase 5's `test_evaluate_pipeline_run.py` already skips gracefully when this is the case. Phase 6 inherits the same posture: code, fixtures, and tests land in this phase; the real-data run is owned by the CUDA machine.
- The Phase 6 specification is final at `Docs/Spec-Phase6-Documentation.md`. The two open questions raised in Idea.md ("loss-curve schema" and "notebook execution state") were resolved in-spec: sidecar parquet for loss curves; executed notebooks with cell outputs committed.
- No `src/nflpredictor/diagnostics/` module exists yet.
- No `notebooks/` directory exists yet.
- Two new optional dependencies: `jupyter` and `nbconvert` — declared under a new `[project.optional-dependencies] docs` group in `pyproject.toml`. The default install path used by CI (`pip install -e .[dev]`) does not pull these.

---

## Guiding Principles

1. **Work inside the virtual environment.** All `pip`, `python`, `pytest`, and `jupyter` commands run with `.venv/` activated.
2. **Tests accompany every code phase.** Phases 2, 3, and 4 land code; their tests land in the same phase. Prose phases (5, 6, 7) ship completeness checks alongside the prose.
3. **Phase-aligned commits.** One commit per internal phase. The commit message names the phase (e.g., "Phase 6 (plan phase 2): per-epoch loss-curve emission + train tests").
4. **Single source of truth for the walkthrough.** `notebooks/phase6_walkthrough.ipynb` is canonical. `Docs/Phase6-Walkthrough.md` is generated via `jupyter nbconvert` and never edited by hand.
5. **Documentation does not chase byte-determinism.** The committed cell outputs and markdown exports are the source of truth for offline readers; their exact bytes after re-export are not asserted. Only the underlying parquet has a byte-determinism contract.
6. **Backward edits are minimal and explicit.** Plan-phase 2 touches Phase 4; plan-phase 3 absorbs the ripple into Phase 5; nothing in Phases 1–3 of the project is touched.
7. **The specification is source of truth.** When the plan and the spec disagree, fix the plan or fix the spec — do not silently improvise.

---

## Proposed Repository Layout (additions)

Phase 6 adds the following files to the existing repo. Files outside this list are untouched (Phases 1–3's packages stay read-only; Phase 4 and Phase 5 get the specific touches enumerated in plan-phases 2 and 3 below).

```
NFLPredictor/
├── Data/
│   └── processed/
│       └── training_loss_curves.parquet              # NEW — sidecar artifact for Phase 4 loss curves
├── Docs/
│   ├── Phase6-ReadingTheOutputs.md                   # NEW — Deliverable 1
│   ├── Phase6-Walkthrough.md                         # NEW — Deliverable 2 (generated, do not hand-edit)
│   └── Phase6-TrainingDynamics.md                    # NEW — Deliverable 4
├── notebooks/                                        # NEW dir
│   ├── phase6_walkthrough.ipynb                      # NEW — Deliverable 2 (executed, with outputs)
│   └── phase6_training_dynamics.ipynb                # NEW — Deliverable 4 companion (executed, with outputs)
├── src/
│   └── nflpredictor/
│       └── diagnostics/                              # NEW package
│           ├── __init__.py
│           ├── trace.py                              # per-game trace helpers (raw → predictions → plot coords)
│           ├── encoding.py                           # reconstruct B-flat / B-pos vector for one game
│           └── loss_curves.py                        # load + filter training_loss_curves.parquet for plotting
├── tests/
│   ├── test_diagnostics.py                           # NEW — unit tests for the helpers
│   ├── test_loss_curves.py                           # NEW — schema, sort, determinism, integrity (DD-LC, DD-TEST-01..06)
│   └── test_phase6_docs.py                           # NEW — regex completeness check + walkthrough export presence
├── pyproject.toml                                    # MODIFIED — add [project.optional-dependencies] docs
├── CLAUDE.md                                         # MODIFIED — new "Phase 6" section
└── Docs/Idea.md                                      # MODIFIED — Project Status table row for Phase 6
```

Files modified in pre-existing Phase 4 code (plan-phase 2 of this plan):

```
src/nflpredictor/train/                               # MODIFIED — capture + parquet write
Docs/Spec-Phase4-BaselineLadder.md                    # MODIFIED — amendment block (TR-LC-*)
Docs/Plan-Phase4-BaselineLadder.md                    # MODIFIED — amendment plan-phase
tests/test_train_integration.py                      # MODIFIED — assertions on new artifact + manifest field
tests/test_train_determinism.py                       # MODIFIED — byte-equality on new parquet
tests/fixtures/train/                                 # REGENERATED
```

Files modified by the Phase 5 ripple (plan-phase 3 of this plan):

```
tests/fixtures/evaluate/                              # REGENERATED (mirrors new train fixture)
tests/test_evaluate_integration.py                    # MODIFIED — refreshed SHA pins
tests/test_evaluate_determinism.py                    # MODIFIED — refreshed SHA pins
tests/test_evaluate_cross_phase.py                    # MODIFIED — refreshed SHA pins
```

---

## Phase 1: Scaffolding and Dependencies

Sets up the new directories, the empty module skeleton, and the optional-dependency group. No behavior changes; commit-able after one short sitting.

### 1.1 Dependency Declaration

- [ ] Add a new `[project.optional-dependencies] docs` block to `pyproject.toml` with `jupyter` and `nbconvert` (no upper pin in v1; pins can be added if the export pipeline turns out to be sensitive). Keep the existing `dev` group untouched.
- [ ] Install the new group locally with `pip install -e .[docs]` so subsequent phases have the binaries.

### 1.2 Directory Skeleton

- [ ] Create `notebooks/` at the repo root.
- [ ] Create `src/nflpredictor/diagnostics/` with an empty `__init__.py`.
- [ ] Create three empty stub modules: `src/nflpredictor/diagnostics/trace.py`, `encoding.py`, `loss_curves.py`. Each carries a one-line docstring describing its scope; no functions yet.
- [ ] Verify the package is importable: `python -c "from nflpredictor.diagnostics import trace, encoding, loss_curves"`.

### 1.3 .gitignore Posture

- [ ] Confirm `.ipynb_checkpoints/` is ignored. Add it to `.gitignore` if not already present.
- [ ] Confirm Jupyter doesn't drop kernel-metadata files outside the notebooks themselves (`*.nbconvert.*`, `.virtual_documents/`); add the patterns if needed.

### 1.4 Tests Bootstrap

- [ ] Create empty `tests/test_diagnostics.py`, `tests/test_loss_curves.py`, `tests/test_phase6_docs.py`. Each has a single placeholder test (`def test_placeholder(): pass`) so `pytest -q` continues to collect cleanly.

### 1.5 Verification

- [ ] `pytest -q` from `.venv` passes (existing tests + the three new placeholders).
- [ ] `jupyter --version` and `jupyter nbconvert --version` succeed.

**Commit**: "Phase 6 (plan phase 1): scaffolding — docs extras, diagnostics module skeleton, notebooks dir"

---

## Phase 2: Phase 4 Backward Edit — Per-Epoch Loss-Curve Emission

The only backward edit Phase 6 introduces. Touches `src/nflpredictor/train/` and Phase 4's spec, plan, fixtures, and tests. Implements DD-LC-* and DD-BWD-01..06.

### 2.1 Training-Loop Instrumentation

- [ ] Identify the training-loop module in `src/nflpredictor/train/` and find the per-epoch boundary for the learned rungs. Trivial rungs (rung 0, rung 1) are closed-form; they take no instrumentation.
- [ ] Add a per-epoch capture: at end of each epoch, append a record `(combination_id, fold, epoch, train_loss, val_loss, val_mae)` to an in-memory list scoped to that combination × fold. Trivial rungs contribute zero records (DD-LC-01).
- [ ] Compute `train_loss` as the mean of per-batch L1 loss across the epoch's training batches (DD-LC-06). Compute `val_loss` analogously on the validation batches.
- [ ] Compute `val_mae` per DD-LC-06: per-side-averaged validation MAE using Phase 5's headline formula `mean(|pred_home − true_home| + |pred_away − true_away|) / 2`.
- [ ] Confirm the capture path makes no RNG draws, no extra optimizer steps, and no additional gradient computation. DD-LC-08 requires capture to be optimizer-neutral.

### 2.2 Parquet Writer

- [ ] After all combinations × folds finish, concatenate the captured records into a pandas DataFrame, sort by `(combination_id, fold, epoch)` (DD-LC-02), enforce the dtypes from Spec §4.1, and write to `Data/processed/training_loss_curves.parquet`.
- [ ] Use the same compression and pyarrow encoding settings the existing prediction parquets use, so the determinism contract (DD-NF-01) carries over without surprise.
- [ ] Compute SHA-256 of the new parquet and add `training_loss_curves.parquet: <hex>` to the `output_sha256` dict in `training_manifest.json` (DD-LC-03).
- [ ] Bump `training_version` from `"v1"` to `"v2"` (DD-LC-04).

### 2.3 Train Fixture Regeneration

- [ ] Run `python -m tests.fixtures.train._regenerate` to refresh the train fixture's manifest, prediction parquets, and new loss-curve parquet. Inspect the diff to confirm only the expected files changed.
- [ ] Verify the regenerated `training_manifest.json` shows the bumped `training_version` and the new `output_sha256` key.

### 2.4 Train Tests

- [ ] `tests/test_loss_curves.py`:
  - [ ] Schema test (DD-TEST-03): columns are exactly `[combination_id, fold, epoch, train_loss, val_loss, val_mae]` in that order with the dtypes from Spec §4.1.
  - [ ] Sort-order test (DD-TEST-04): rows are sorted lexicographically by `(combination_id, fold, epoch)`.
  - [ ] Integrity test (DD-TEST-02): for each combination × fold, the final-epoch `val_mae` row matches `training_summaries.<combo>.val_mae` (S1) or the corresponding fold entry (S3) to ~1e-12 relative tolerance.
- [ ] Extend `tests/test_train_determinism.py` (DD-TEST-01): run training twice from the fixture inputs, assert `training_loss_curves.parquet` is byte-identical across runs.
- [ ] Extend `tests/test_train_integration.py` (DD-TEST-05 / DD-BWD-05): assert (a) the parquet exists at the expected path, (b) the manifest's `output_sha256["training_loss_curves.parquet"]` equals `sha256(...)` of the on-disk file, (c) `training_version` is `"v2"` (or whatever the bumped value is — the test reads the fixture's recorded value rather than hardcoding).
- [ ] Add a capture-is-read-only test (DD-TEST-06 / DD-LC-08): control the capture via a test-only flag, run with and without, assert the prediction parquets are byte-identical between the two runs. If the capture is implemented in a way that makes a control flag awkward, document the alternative verification approach (e.g., re-run from the same fixture against a recorded pre-amendment prediction SHA).

### 2.5 Phase 4 Doc Amendments (DD-BWD-03 / DD-BWD-04)

- [ ] `Docs/Spec-Phase4-BaselineLadder.md`: add a "TR-LC: Per-Epoch Loss Curves" subsection under §3 (Functional Requirements) cross-referencing DD-LC-01..08. Add a one-line entry to §12 (Future Considerations) noting that this amendment has landed.
- [ ] `Docs/Plan-Phase4-BaselineLadder.md`: append an amendment plan-phase summarizing the implementation steps (point to this plan as canonical).

### 2.6 Verification

- [ ] `pytest -q tests/test_loss_curves.py tests/test_train_integration.py tests/test_train_determinism.py` passes.
- [ ] Full `pytest -q` passes (no regressions in Phase 1/2/3 tests).

**Commit**: "Phase 6 (plan phase 2): per-epoch loss-curve emission + train tests + Phase 4 amendments"

---

## Phase 3: Phase 5 Ripple — Fixture Regen and Test Re-Pin

Absorbs the Phase 4 manifest-SHA rotation into Phase 5's downstream pins. Pure follow-through; no semantic changes. Implements DD-BWD-07..09.

### 3.1 Evaluate Fixture Regeneration

- [x] Run `python -m tests.fixtures.evaluate._regenerate` to refresh the evaluate fixture. The regen consumes the updated train fixture from plan-phase 2, so the evaluate fixture's source-hash pins on Phase 4's outputs rotate naturally.
- [x] Inspect the diff. Expect: `tests/fixtures/evaluate/` manifest values change; metric values do not (Phase 5's outputs are semantically unchanged per DD-BWD-09). _Observed: only `raw_phase4/training_manifest.json` (new loss-curve SHA + `training_version: v2` + bumped `training_config_sha256`), `raw_phase3/splits_manifest.json` (timestamp + git_commit only), and `expected/evaluation_manifest.json` (the rotated `training_manifest.json` input pin) changed. Every prediction parquet SHA in the eval manifest's input list is unchanged; `expected/metrics_headline.json` and every breakdown parquet in `expected/breakdowns/` are byte-identical pre/post regen._

### 3.2 Test SHA Refresh

- [x] `tests/test_evaluate_integration.py`, `tests/test_evaluate_determinism.py`, `tests/test_evaluate_cross_phase.py`: any hardcoded SHA constants — if present — that pin Phase 4's manifest get refreshed to match the regenerated fixture. If pins are loaded from the fixture's manifest at test time, no code change is needed here. _Confirmed: `grep -lr '[a-f0-9]\{32\}' tests/test_evaluate_*.py` returns nothing; every eval test reads its pins from the fixture manifest at runtime, so the fixture-only regen is sufficient._

### 3.3 Verification

- [x] `pytest -q tests/test_evaluate_*.py` passes.
- [x] `pytest -q tests/test_evaluate_cross_phase.py` passes — confirms Phase 5's recomputed val MAE still equals Phase 4's `training_summaries` values (EV-TEST-08).
- [x] Full `pytest -q` passes. _568 passed, 6 skipped — same baseline as post-plan-phase-2._

**Commit**: "Phase 6 (plan phase 3): Phase 5 fixture regen + test SHA refresh"

---

## Phase 4: Diagnostics Helper Module

Implements the testable helpers the walkthrough notebook and training-dynamics notebook will call. Per DD-INT-01, heavy logic lives in the package, not in the notebooks. Implements DD-INT-01 / DD-INT-04 / DD-WT-06.

### 4.1 Per-Game Trace Helpers (`src/nflpredictor/diagnostics/trace.py`)

- [x] `load_raw_game(game_id: str) -> pandas.Series` — reads the single row from `Data/raw/box_scores_2024.csv` matching `GameId == game_id`. Raises if not found.
- [x] `resolve_starters(game_id: str) -> pandas.DataFrame` — returns one row per starter slot for the given game, joining the raw `_ID` (PFR-style or blank) to its resolved `madden_id` via `Data/processed/player_id_mapping.csv`, plus the `note` column showing the resolution provenance.
- [x] `lookup_split_membership(game_id: str) -> dict` — returns a dict like `{"S1": "val", "S3_folds": [12]}` indicating which split slices contain this game. _Implemented as `{"S1": str|None, "S3_folds": list[int], "S3_test": bool}`; the list contains the `k` values (6–14) of every fold whose val slice contains the game, and `S3_test` flags membership in S3's held-out test slice._
- [x] `lookup_predictions(game_id: str) -> pandas.DataFrame` — returns one row per `(combination_id, slice)` that emitted a prediction for this game, with `pred_home`, `pred_away`, `true_home`, `true_away`, `residual_home`, `residual_away`. _Returns an empty (correctly-typed) frame when `Data/processed/predictions/` is absent — covers the dev-machine "no Phase 4 run yet" case gracefully._

### 4.2 Encoded-Vector Reconstruction (`src/nflpredictor/diagnostics/encoding.py`)

- [x] `encode_one_game_flat(game_id: str) -> pandas.Series` — returns the row from `features_flat_2024.parquet` for this game.
- [x] `encode_one_game_pos(game_id: str) -> pandas.Series` — returns the row from `features_pos_2024.parquet`.
- [x] `explain_categorical(column: str, raw_value: str) -> dict` — for a categorical column, returns `{"raw": raw_value, "vocab_key": ..., "integer_code": ..., "embedding_table": ...}` so the notebook can show the categorical → integer-code → embedding chain. _Implementation also reports `vocab_size` and `routing` (`"embedding"` for >8 entries, `"one_hot"` for ≤8) so the notebook can name the routing decision explicitly. The `integer_code` includes Phase 4's `NULL_BUMP=1` so it matches what the encoder consumes (TR-CAT-07)._

### 4.3 Loss-Curve Loader (`src/nflpredictor/diagnostics/loss_curves.py`)

- [x] `load_loss_curves() -> pandas.DataFrame` — reads `Data/processed/training_loss_curves.parquet`. Verifies the SHA against `training_manifest.json` (raises on mismatch, following the same hash-pinning discipline as Phase 5). _Raises `LossCurvesIntegrityError` on SHA divergence; a `verify_sha=False` escape hatch is provided for hand-crafted fixtures._
- [x] `filter_curves(df, combination_id: str | None = None, fold: int | None = None) -> pandas.DataFrame` — small filtering helper for the plotting notebook.

### 4.4 Tests (`tests/test_diagnostics.py`)

- [x] Unit tests cover each helper against the existing train + evaluate fixtures. Pick one fixture game ID and assert each helper returns the expected shape and a couple of representative values. _Test GameId is `202411280dal` — it lives in S1 val and in S3 fold k=12 val, so it appears in all 12 prediction parquets._
- [x] A negative test asserts that `load_raw_game` on an unknown `GameId` raises a clear error.
- [x] A hash-mismatch test asserts that `load_loss_curves` raises when the parquet on disk has been tampered with (mutate a copy in a tmp dir, point the loader at it via a patchable constant or pass-through arg).

### 4.5 Verification

- [x] `pytest -q tests/test_diagnostics.py` passes. _22 tests pass._
- [x] Full `pytest -q` passes. _589 passed, 6 skipped — up from 568 (+22 new tests, −1 placeholder)._

**Commit**: "Phase 6 (plan phase 4): diagnostics helper module + unit tests"

---

## Phase 5: Pipeline Walkthrough Notebook + Export (Deliverable 2)

Authors `notebooks/phase6_walkthrough.ipynb` and produces `Docs/Phase6-Walkthrough.md`. Implements DD-WT-01..06.

### 5.1 Notebook Authoring

- [x] Create `notebooks/phase6_walkthrough.ipynb`. Pick the walkthrough game (e.g., `202409050kan` — Week 1, KC home opener) and declare it as a constant in the first code cell. _Picked `202411280dal` (Dallas Cowboys Thanksgiving 2024, Week 13) — same game the diagnostics tests use; it lives in S1.val and S3 fold k=12 val so the trace exercises all 12 prediction parquets._
- [x] Structure the notebook into the six top-level sections required by DD-WT-02. Each section opens with a markdown cell explaining what's about to be shown and why; the following code cells call into `nflpredictor.diagnostics.*` and render dataframes / plots.
  - Section 1: Game selection rationale.
  - Section 2: Raw box-score row + Madden join. Show all 44 starter slots with `_ID` → `madden_id` provenance.
  - Section 3: Feature encoding. Show B-flat and B-pos vectors; expand at least one categorical column to walk the raw → integer-code → embedding-lookup chain.
  - Section 4: Split assignment. Show S1 bucket and S3 fold membership.
  - Section 5: Predictions per learned combination. Side-by-side pred vs. actual.
  - Section 6: Locate the game on each Phase 5 plot. Print its row in each breakdown parquet.
- [x] Run all cells in order from a clean kernel. Confirm every cell completes without error and produces expected output. _Executed via `jupyter nbconvert --to notebook --execute --inplace` against the eval fixture; staged composite `processed_dir` from `tests/fixtures/evaluate/` so the CPU-only dev box can render real outputs in every section. Section 6 prints breakdown rows; the actual PNGs are produced by the real-data Phase 5 run on the CUDA machine and the section documents that out-of-fixture path explicitly._
- [x] Commit the notebook in its executed state (cell outputs included) per DD-WT-03.

### 5.2 Markdown Export

- [x] Run `jupyter nbconvert --to markdown notebooks/phase6_walkthrough.ipynb --output ../Docs/Phase6-Walkthrough.md` from the `notebooks/` directory (or the equivalent absolute-path invocation from the repo root) per DD-WT-04 / DD-INT-02. _Exported 22366 bytes; six `## N. ...` section headers present in order._
- [x] Open the generated `Docs/Phase6-Walkthrough.md` and confirm it renders cleanly — no broken image refs, no orphaned `<div>` tags, no escaped HTML where prose should be.
- [x] Commit both files (`.ipynb` and `.md`) in the same commit.

### 5.3 Tests

- [x] `tests/test_phase6_docs.py`: add `test_walkthrough_export_exists` (DD-TEST-07) — asserts `Docs/Phase6-Walkthrough.md` exists and is non-empty (e.g., `len(content) > 500` to catch trivial empties without being brittle to prose changes).
- [x] `tests/test_phase6_docs.py`: add `test_walkthrough_has_required_sections` — regex-scans for the six section headers required by DD-WT-02 (e.g., "Phase 1 — raw box-score row", "Phase 2 — feature encoding", etc.). The exact header strings are committed to the notebook and asserted by the test.

### 5.4 Verification

- [x] `pytest -q tests/test_phase6_docs.py` passes. _2 tests pass._
- [x] Full `pytest -q` passes. _590 passed, 6 skipped — up from 589 (+2 new docs tests, −1 placeholder)._

**Commit**: "Phase 6 (plan phase 5): pipeline walkthrough notebook + markdown export"

---

## Phase 6: Reading-the-Outputs Guide (Deliverable 1)

Authors `Docs/Phase6-ReadingTheOutputs.md`. Implements DD-RG-01..06.

### 6.1 Prose Authoring

- [ ] Open `Docs/Phase6-ReadingTheOutputs.md` and write the "How to use this document" section (DD-RG-01). State that each Phase 5 artifact gets one structured entry, that the goal is interpretation not improvement, and that Phase 7 is where actions get taken.
- [ ] Write one entry per artifact in the DD-RG-02 list (10 entries). Each entry uses the four sub-section structure required by DD-RG-03:
  - **What it shows** — mechanical description.
  - **What good looks like** — calibration intuition.
  - **Red flags** — concrete anti-patterns.
  - **Action to consider** — what to investigate in Phase 7.
- [ ] Embed at least one representative PNG per plot-type entry (DD-RG-04). PNGs reference `Data/processed/evaluation/plots/...` by relative path; the guide does not copy them.
- [ ] Add the "How to read across combinations" section (DD-RG-05) — frame delta comparisons (rung 2 flat vs. rung 2 pos, rung 2 vs. rung 3) as the primary lens.
- [ ] Add the "Vocabulary" appendix (DD-RG-06) mapping Phase 5 terms back to Spec-Phase5 definitions.

### 6.2 Tests

- [ ] `tests/test_phase6_docs.py`: add `test_reading_outputs_has_all_entries` (DD-TEST-08) — regex-scans for the 10 required artifact section headers. The exact header strings the guide commits to are the source of truth; the test reads them from a list constant in the test file.

### 6.3 Verification

- [ ] `pytest -q tests/test_phase6_docs.py` passes.
- [ ] Render the markdown locally (IDE preview or `grip`) and confirm: no broken PNG refs, no broken internal links, every embedded image displays.

**Commit**: "Phase 6 (plan phase 6): reading-the-outputs guide"

---

## Phase 7: Training-Dynamics Doc + Companion Notebook (Deliverable 4)

Authors `Docs/Phase6-TrainingDynamics.md` and `notebooks/phase6_training_dynamics.ipynb`. Implements DD-TD-01..03.

### 7.1 Prose Authoring (`Docs/Phase6-TrainingDynamics.md`)

- [ ] Write the doc covering the six topics required by DD-TD-01 in order:
  1. The training loop — epoch, batch, sample order (per-epoch shuffle, deterministic seed).
  2. The loss function — MAE/L1, why this choice.
  3. The optimizer and learning rate — Adam, fixed LR, no schedule.
  4. How "done" is decided today — `max_epochs`, no early stopping, risks of each direction.
  5. Reading the train↔val gap — the four canonical patterns and their interpretations.
  6. Which knob to reach for — table mapping pattern → `training_config.yaml` field.
- [ ] Add the "What is *not* covered in Phase 6" note (DD-TD-03) listing the Phase 7 items (attribution, ablation, per-team error analysis).

### 7.2 Companion Notebook

- [ ] Create `notebooks/phase6_training_dynamics.ipynb`.
- [ ] First code cell calls `nflpredictor.diagnostics.loss_curves.load_loss_curves()` to load the parquet.
- [ ] Subsequent cells produce one annotated subplot per learned combination (DD-TD-02). For S3 combinations, show all 9 folds on the same subplot (faint lines + bold mean) or as a small multiple — author's call; document the choice in a markdown cell.
- [ ] Annotate at least one combination's subplot with prose callouts pointing at notable inflections (e.g., "val loss flattens at epoch N — likely the underfit knob to reach for is `hidden_dim`").
- [ ] Run all cells from a clean kernel. Commit the notebook with executed outputs per DD-WT-03's spirit (and DD-NF-03 — PNG outputs only).

### 7.3 Tests

- [ ] `tests/test_phase6_docs.py`: add `test_training_dynamics_has_required_topics` — regex-scans `Docs/Phase6-TrainingDynamics.md` for the six topic headers required by DD-TD-01.
- [ ] `tests/test_phase6_docs.py`: add `test_training_dynamics_notebook_exists` — asserts the notebook file is present and non-empty (no content assertion; cell outputs are non-deterministic per DD-NF-04).

### 7.4 Verification

- [ ] `pytest -q tests/test_phase6_docs.py` passes.
- [ ] Render the markdown locally and confirm structure.
- [ ] Open the notebook in JupyterLab or VS Code and confirm cell outputs display.

**Commit**: "Phase 6 (plan phase 7): training-dynamics doc + companion notebook"

---

## Phase 8: Wrap-Up — CLAUDE.md, Idea.md Status, Final Review

Closes the phase: updates the repo-level docs, marks Idea.md's status row complete, runs the full test suite, and does a final review pass.

### 8.1 CLAUDE.md

- [ ] Add a Phase 6 section to `CLAUDE.md` modeled on the existing Phase 1–5 sections. Include:
  - The new artifacts: `Data/processed/training_loss_curves.parquet`, the three `Phase6-*.md` files, the two notebooks.
  - The regen commands for each notebook (`jupyter nbconvert --to markdown ...`).
  - The note that `Docs/Phase6-Walkthrough.md` is generated — do not hand-edit.
  - The location of `src/nflpredictor/diagnostics/` and what each submodule does.
  - The new `docs` optional-dependency group and how to install it.

### 8.2 Idea.md Status

- [ ] Update `Docs/Idea.md`'s Project Status table row for Phase 6 from `Exploratory` to `Implementation complete (YYYY-MM-DD)`.

### 8.3 Full-Suite Verification

- [ ] `pytest -q` from `.venv` passes end-to-end.
- [ ] `python -m nflpredictor.databuild && python -m nflpredictor.features && python -m nflpredictor.splits` still produce byte-identical outputs against the existing manifests (smoke check that Phase 6 hasn't accidentally rippled into Phases 1–3 contrary to DD-BWD-10).

### 8.4 Manual Review

- [ ] Open `Docs/Phase6-ReadingTheOutputs.md`, `Docs/Phase6-Walkthrough.md`, and `Docs/Phase6-TrainingDynamics.md` in a markdown previewer. Confirm:
  - All embedded PNGs resolve.
  - All internal links resolve.
  - No leftover TODO markers, no obvious placeholder text.
- [ ] Open both notebooks in JupyterLab or VS Code. Confirm cell outputs display and the narratives flow.

### 8.5 Commit

**Commit**: "Phase 6 (plan phase 8): CLAUDE.md + Idea.md status + final review"

---

## After Phase 8

When Phase 8 is complete:

- All four Phase 6 deliverables (reading-outputs guide, pipeline walkthrough, loss-curve instrumentation, training-dynamics doc) are committed and tested.
- Phase 4 and Phase 5 fixtures are regenerated to absorb the per-epoch logging amendment; their tests pass against the new SHAs.
- `Docs/Idea.md` Project Status table shows Phase 6 as implementation-complete.
- `CLAUDE.md` documents the new artifacts and regen commands.

At that point, the user can:

1. Read the three Phase 6 docs end-to-end and confirm they build the intuition the phase set out to deliver.
2. Run the real-data Phase 4 + Phase 5 pipeline on the CUDA machine, which will now emit the loss-curve parquet alongside the existing prediction parquets and metric outputs.
3. Re-execute the two Phase 6 notebooks against the real-data outputs (rather than fixtures) to refresh the committed cell outputs with real numbers. The regen commands in `CLAUDE.md` are the documented entry point.
4. Begin Phase 7 (Error Analysis & Iteration) with the diagnostic vocabulary and reading-the-outputs framing in place.

---

## References

- [Spec-Phase6-Documentation.md](./Spec-Phase6-Documentation.md) — the specification this plan implements.
- [Idea.md](./Idea.md) — Phase 6 section under "Documentation & Diagnostics."
- [Spec-Phase4-BaselineLadder.md](./Spec-Phase4-BaselineLadder.md) — the training contract Phase 6 amends in plan-phase 2.
- [Spec-Phase5-Evaluation.md](./Spec-Phase5-Evaluation.md) — the source of every artifact the reading-outputs guide explains.
- [CLAUDE.md](../CLAUDE.md) — repository-level notes; updated in plan-phase 8.
