# Spec: Phase 6 — Documentation & Diagnostics

## 1. Introduction

### 1.1 Purpose

Phase 6 produces the documentation and supporting instrumentation that turn the Phase 1–5 artifacts into a story a single developer can read end-to-end. Its goal is intuition, not better metrics. Four artifacts ship:

1. A "reading the outputs" guide that explains what each Phase 5 plot and breakdown parquet shows, what a good version looks like, and what a red flag suggests.
2. A pipeline walkthrough — a Jupyter notebook plus a Markdown export of that notebook — that traces one concrete game from raw CSV row through every phase to its final residual.
3. Per-epoch loss-curve instrumentation, the one backward edit Phase 6 introduces: Phase 4 gains a sidecar parquet logging `(epoch, train_loss, val_loss, val_mae)` per learned combination × fold so Deliverable 4 has artifacts to point at.
4. A training-dynamics doc that explains epochs, batching, shuffling, learning rate, the meaning of train↔val divergence, and which knob in `training_config.yaml` to reach for in each scenario.

Phase 6 explicitly does *not* execute any iteration cycles, propose feature changes, or compute attribution. Those activities belong to Phase 7 (Error Analysis & Iteration). Phase 6 is the prerequisite that makes Phase 7 actionable for a developer who currently has no intuition for what Phase 5 produces.

### 1.2 Scope

In scope:

- Authoring four documentation artifacts (`Docs/Phase6-ReadingTheOutputs.md`, `Docs/Phase6-Walkthrough.md`, `notebooks/phase6_walkthrough.ipynb`, `Docs/Phase6-TrainingDynamics.md`) plus a small companion notebook for the training-dynamics doc.
- A backward edit to Phase 4 that emits `Data/processed/training_loss_curves.parquet` and registers its SHA in `training_manifest.json`'s `output_sha256` map.
- All downstream test, fixture, and SHA-pin regenerations required by the Phase 4 amendment (Phase 5 source-hash re-pin, fixture regen).

Out of scope:

- **Worst-N narrative analysis, slice-driven hypothesis logs, feature attribution (Captum / Integrated Gradients), ablation experiments.** Phase 7.
- **Any modification to Phase 1, 2, or 3 outputs or schemas.** Phase 6 follows a "minimal backward edits" policy; per-game traces and intermediate-state inspection are reconstructed inside the walkthrough notebook from existing artifacts.
- **Adding new metrics, breakdowns, or plot types to Phase 5.** Phase 6 only documents what Phase 5 already emits.
- **Early-stopping or LR-schedule changes to Phase 4.** Phase 6 instruments the existing training loop; it does not change its behavior.
- **A real-data Phase 4 + Phase 5 run on the CUDA machine.** Phase 6 lands the code, fixture, and test updates; the user runs the full pipeline on their own schedule. Phase 6's CI surface is the fixture-based tests.

### 1.3 Definitions and Acronyms

| Term | Definition |
|------|------------|
| Combination | A `(rung, feature_shape, strategy)` triple — same as Phase 4. Only *learned* rungs (rung 2 + rung 3) contribute loss-curve rows. |
| Fold | For S1, `fold = 0` (single fold). For S3, the integer `k` value in `{6, …, 14}`. Identical to Phase 4's enumeration. |
| Epoch | One full pass of the trainer over the train slice for a given combination × fold. 0-indexed. |
| Reading-guide artifact | A Phase 5 artifact (parquet, JSON, or PNG) for which the guide must produce a structured "what it shows / good / red flag / action" entry. |
| Walkthrough game | The single game (e.g. `202409050kan`) the walkthrough notebook traces end-to-end. The choice is fixed in v1 to keep the notebook deterministic across re-executions. |
| Loss-curve parquet | `Data/processed/training_loss_curves.parquet` — the new sidecar artifact introduced by Phase 6. |
| Notebook companion | A `.md` export of a `.ipynb` produced by `jupyter nbconvert --to markdown`. Committed alongside the notebook so GitHub previews and offline readers see the rendered narrative. |

### 1.4 Design Principles

- **Documentation is the deliverable.** The prose docs and the walkthrough notebook are first-class outputs, not byproducts. Their structure is specified; their exact prose is left to the author.
- **Single source of truth for the walkthrough.** The notebook is canonical. The Markdown export at `Docs/Phase6-Walkthrough.md` is generated from it via `jupyter nbconvert --to markdown` and is never edited by hand. The export is committed alongside the notebook so GitHub previews and offline readers work.
- **Executed notebooks ship with cell outputs.** Both the walkthrough notebook and the training-dynamics companion notebook are committed in their executed state with cell outputs visible. Re-execution is the source of any update.
- **Minimal backward edits.** The only earlier-phase change is Phase 4's loss-curve emission. Per-game traces, encoded vectors, and split-bucket lookups are reconstructed inside Phase 6 from artifacts that already exist on disk.
- **Determinism (parquet content).** `training_loss_curves.parquet` is byte-deterministic per device under the same Phase 4 contract as the prediction parquets — same pinned PyTorch wheel + same resolved device.
- **No determinism contract on PNG-in-markdown or on notebook cell outputs.** The reading-outputs guide embeds Phase 5 PNGs by reference; Phase 6 does not re-render them. Notebook execution metadata (cell IDs, timing) is non-deterministic and not asserted.
- **Specification governs structure, not prose.** When the spec says "the guide covers each of the five breakdown parquets," that is enforceable by a regex test. The exact wording of the entries is the author's call.

### 1.5 Relationship to Phases 1–5

Phase 6 consumes the same processed artifacts Phase 5 does, plus the new loss-curve parquet it introduces:

```
Data/processed/{features_flat_2024.parquet, features_pos_2024.parquet, feature_vocab.json}     (Phase 2)
Data/processed/splits_2024.json                                                                (Phase 3)
Data/processed/predictions/*.parquet                                                           (Phase 4)
Data/processed/training_manifest.json (extended via output_sha256)                             (Phase 4 + Phase 6)
Data/processed/training_loss_curves.parquet                                                    (Phase 6, new)
Data/processed/evaluation/{metrics_headline.json, breakdowns/*, plots/*, evaluation_manifest.json}   (Phase 5)
```

Phase 6 produces:

```
Docs/Phase6-ReadingTheOutputs.md
Docs/Phase6-Walkthrough.md                  ← generated from the notebook; do not hand-edit
Docs/Phase6-TrainingDynamics.md
notebooks/phase6_walkthrough.ipynb          ← committed with executed cell outputs
notebooks/phase6_training_dynamics.ipynb    ← committed with executed cell outputs
Data/processed/training_loss_curves.parquet ← new sidecar artifact
```

## 2. Technology Additions

- `jupyter` and `nbconvert` — for authoring and exporting the walkthrough and training-dynamics notebooks. Both are added to `pyproject.toml` under a new `docs` optional-dependency group so CI and lightweight installs that only need `pytest` are unaffected.
- `jupyter nbconvert --to markdown` is the export command. No additional rendering layer.

No new modeling libraries (no Captum, no SHAP, no plotly). Those are deferred to Phase 7.

## 3. Functional Requirements

### 3.1 Input Contract

| ID | Requirement |
|----|-------------|
| DD-IN-01 | The Phase 6 documentation regeneration (notebook re-runs, markdown re-exports) shall refuse to run if any Phase 2 tracked output, Phase 3 output, Phase 4 prediction parquet, Phase 4 loss-curve parquet, Phase 4 manifest, or Phase 5 manifest on disk does not match its upstream-manifest SHA. The same hash-pinning discipline applies that Phase 5 already enforces (EV-IN-06 / EV-IN-07 / EV-IN-08). |
| DD-IN-02 | The Phase 4 amendment (loss-curve emission) reads no new inputs; it only adds a sidecar output and an entry in `output_sha256`. |

### 3.2 Deliverable 1 — Reading the Outputs Guide (DD-RG)

`Docs/Phase6-ReadingTheOutputs.md`. Prose markdown with embedded PNGs from a real Phase 5 run.

| ID | Requirement |
|----|-------------|
| DD-RG-01 | The guide shall contain a top-level "How to use this document" section explaining that each Phase 5 artifact gets one structured entry and that the goal is interpretation, not metric improvement. |
| DD-RG-02 | The guide shall contain one entry per artifact in the following set (10 artifact types): (1) `metrics_headline.json`, (2) `breakdowns/by_team.parquet`, (3) `breakdowns/by_week.parquet`, (4) `breakdowns/by_home_away.parquet`, (5) `breakdowns/by_surface.parquet`, (6) `breakdowns/by_roof.parquet`, (7) `plots/<combination_id>__<slice>__scatter.png`, (8) `plots/<combination_id>__<slice>__residuals.png`, (9) `plots/<combination_id>__<slice>__by_week.png`, (10) `plots/ladder_summary__{val,test,pooled}.png`. |
| DD-RG-03 | Each entry shall have four labeled sub-sections: **What it shows** (mechanical description), **What good looks like** (calibration intuition), **Red flags** (concrete anti-patterns), **Action to consider** (what to investigate in Phase 7 when a red flag is spotted). |
| DD-RG-04 | Each plot-type entry shall embed at least one PNG from a real Phase 5 run as a representative example. PNGs live under `Data/processed/evaluation/plots/` and are referenced by relative path; the guide does not copy or re-render them. |
| DD-RG-05 | The guide shall include a "How to read across combinations" section explaining that delta comparisons (e.g., rung 2 flat vs. rung 2 pos) are the primary lens for "did this earn its keep" and that absolute metric values are secondary. |
| DD-RG-06 | The guide shall include a short "Vocabulary" appendix mapping Phase 5 terms (`combination_id`, `slice`, `fold`, `pooled`, `headline metric`) back to their Spec-Phase5 definitions so the doc is self-contained for a first-time reader. |

### 3.3 Deliverable 2 — Pipeline Walkthrough (DD-WT)

`notebooks/phase6_walkthrough.ipynb` and `Docs/Phase6-Walkthrough.md`.

| ID | Requirement |
|----|-------------|
| DD-WT-01 | The notebook shall trace one specific game from raw inputs through every phase to its final residual. The chosen game is committed to a constant near the top of the notebook for reproducibility; changing it is a content edit, not a contract change. |
| DD-WT-02 | The notebook shall be organized into six top-level sections in order: (1) Game selection and rationale; (2) Phase 1 — raw box-score row + Madden join, showing the 44 starter slots and how each `_ID` was resolved (deterministic, fuzzy, override, or unmatched); (3) Phase 2 — feature encoding, showing the B-flat and B-pos vectors for this game with the categorical → integer-code → embedding-lookup chain made explicit; (4) Phase 3 — split assignment (S1 bucket and S3 fold membership); (5) Phase 4 — predictions per learned combination that include this game, with prediction vs. actual side-by-side; (6) Phase 5 — locate the game on each relevant scatter, residual, and by-week plot, and print its row in each breakdown parquet. |
| DD-WT-03 | The notebook shall be committed with executed cell outputs (rendered tables, plots, printed values). Re-execution is the source of any update. |
| DD-WT-04 | `Docs/Phase6-Walkthrough.md` shall be the markdown export of the notebook generated by `jupyter nbconvert --to markdown notebooks/phase6_walkthrough.ipynb --output ../Docs/Phase6-Walkthrough.md`. It shall not be edited by hand. |
| DD-WT-05 | A regen command shall be documented in `CLAUDE.md`: a single command that re-executes the notebook and re-exports the markdown, so future re-runs are mechanical. |
| DD-WT-06 | Heavy reconstruction logic (joining `madden_id` back to raw `_ID` provenance, regenerating an encoded vector for one game, locating a game on a plot) shall live in a Python helper module (DD-INT-01), not in the notebook. The notebook calls helpers; helpers are testable. |

### 3.4 Deliverable 3 — Per-Epoch Loss-Curve Instrumentation (DD-LC)

The backward edit to Phase 4.

| ID | Requirement |
|----|-------------|
| DD-LC-01 | The training loop in `src/nflpredictor/train/` shall, for every learned combination × fold, capture `(epoch, train_loss, val_loss, val_mae)` per epoch. Trivial rungs (rung 0 mean, rung 1 team_mean) are closed-form and contribute zero rows. |
| DD-LC-02 | The collected rows shall be written to `Data/processed/training_loss_curves.parquet` at the end of the training build, sorted lexicographically by `(combination_id, fold, epoch)` for byte determinism. |
| DD-LC-03 | `training_manifest.json`'s existing `output_sha256` object shall gain a new key `training_loss_curves.parquet` whose value is the SHA-256 hex digest of the new file. No new top-level manifest keys are added. |
| DD-LC-04 | `training_version` in `training_manifest.json` shall be bumped (e.g., `"v1"` → `"v2"`) to record the schema change. The exact string is at Phase 4's discretion; tests assert *that* it changed, not its value. |
| DD-LC-05 | Per-device byte equality (same contract as Phase 4's prediction parquets, TR-NF-01) shall hold for the loss-curve parquet: identical inputs + identical pinned PyTorch wheel + identical resolved device → identical bytes. CPU↔CUDA byte equality is not asserted. |
| DD-LC-06 | `train_loss` is the per-epoch mean of the L1 loss across all training batches for that epoch. `val_loss` is the per-epoch mean of the L1 loss across all validation batches. `val_mae` is per-side-averaged validation MAE under the same formula as Phase 5's headline (`mean(|pred_home − true_home| + |pred_away − true_away|) / 2`). This makes `val_mae` numerically comparable to the manifest's existing final-epoch val MAE. |
| DD-LC-07 | Phase 4 shall keep emitting its existing `training_summaries` block. The final-epoch `val_mae` row for each combination × fold shall equal the corresponding entry in `training_summaries` to ~1e-12 relative tolerance. A test enforces this tautological invariant. |
| DD-LC-08 | The loss-curve capture shall be read-only with respect to optimization. Sample order, RNG draws, optimizer state, and convergence trajectory shall be byte-identical to a hypothetical run with capture disabled. (Verified by re-running the train fixture with and without capture and comparing the resulting prediction parquets.) |

### 3.5 Deliverable 4 — Training Dynamics Doc (DD-TD)

`Docs/Phase6-TrainingDynamics.md` plus `notebooks/phase6_training_dynamics.ipynb`.

| ID | Requirement |
|----|-------------|
| DD-TD-01 | The prose doc shall cover, in order, the following six topics: (1) **The training loop in this project** — what an epoch is, what a batch is, batch size, sample order (per-epoch shuffle with the deterministic seed); (2) **The loss function** — MAE (L1), and why this choice (interpretable; robust to outliers; matches the headline metric); (3) **The optimizer and learning rate** — Adam, fixed LR, no schedule in v1; (4) **How "done" is decided today** — fixed `max_epochs`, no early stopping; the risks of each direction; (5) **Reading the train↔val gap** — four canonical patterns (both decreasing → still learning; train low / val high → overfit; both flat → optimization stuck; train bottomed / val high → underfit) and their interpretations; (6) **Which knob to reach for** — a table mapping each diagnostic pattern to the relevant `training_config.yaml` field (`max_epochs`, `hidden_dim`, `embedding_dim`, `batch_size`, `lr`, `dropout`). |
| DD-TD-02 | The companion notebook shall load `training_loss_curves.parquet` and produce one annotated subplot per learned combination, plotting train and val loss together on each subplot. The notebook is committed with executed cell outputs. |
| DD-TD-03 | The doc shall explicitly state what is *not* covered in Phase 6: feature attribution, ablation results, per-team error analysis. Those are Phase 7 deliverables. |

## 4. Data Model

### 4.1 `Data/processed/training_loss_curves.parquet` (new)

| Column | Type | Description |
|--------|------|-------------|
| `combination_id` | string | Same identifier as Phase 4's prediction parquet stem (e.g., `rung2_linear__flat__S1`). Only learned-rung combinations appear. |
| `fold` | int32 | `0` for S1 (single fold). For S3, the integer `k` value in `{6, …, 14}`. |
| `epoch` | int32 | 0-indexed epoch number within the combination × fold. |
| `train_loss` | float64 | Per-epoch mean MAE over the training batches for this combination × fold. |
| `val_loss` | float64 | Per-epoch mean MAE over the validation batches. |
| `val_mae` | float64 | Per-side-averaged validation MAE (same formula as Phase 5's headline). |

Rows are sorted lexicographically by `(combination_id, fold, epoch)`. The parquet is written with the same compression and encoding options Phase 4 uses for its prediction parquets so the determinism contract is identical.

For v1 with 2 learned rungs × 2 shapes × 2 strategies = 8 learned combinations; S1 contributes 4 combinations × 1 fold = 4 (combination, fold) pairs; S3 contributes 4 combinations × 9 folds = 36 pairs; total = 40 (combination, fold) pairs × `max_epochs` epochs each. At a representative `max_epochs = 50`, that's 2,000 rows. Trivially small.

### 4.2 `training_manifest.json` extensions

Two changes:

- `output_sha256` gains a new key `training_loss_curves.parquet` whose value is the SHA-256 hex digest of the new file (DD-LC-03).
- `training_version` is bumped one version step to record the schema change (DD-LC-04).

No other manifest-level changes. The existing `training_summaries`, `phase2_source_sha256`, `phase3_source_sha256`, device/torch fields, and git-commit fields remain identical in shape and semantics.

### 4.3 Documentation file layout

```
Docs/
  Spec-Phase6-Documentation.md         ← this document
  Plan-Phase6-Documentation.md         ← implementation plan (drafted next)
  Phase6-ReadingTheOutputs.md          ← Deliverable 1
  Phase6-Walkthrough.md                ← Deliverable 2 markdown export (do not hand-edit)
  Phase6-TrainingDynamics.md           ← Deliverable 4 prose

notebooks/
  phase6_walkthrough.ipynb             ← Deliverable 2 notebook (executed, committed with outputs)
  phase6_training_dynamics.ipynb       ← Deliverable 4 companion notebook (executed, committed with outputs)
```

The `notebooks/` directory is new. The `phase6_*` prefix keeps Phase 7's future additions easy to namespace.

## 5. Backward Edits to Earlier Phases

This section enumerates exactly what Phase 6 changes in already-implemented phases. It exists to answer the question "did the renumbering or the new Phase 6 introduce deltas in upstream contracts?"

**Renumbering alone introduced zero behavioral deltas** — every renumbered reference was forward-looking text (e.g. "this is a Phase 6 concern"), not a behavior contract.

**Phase 6's per-epoch logging deliverable (DD-LC) is the only source of deltas.** They are enumerated below. Because both Phase 4 and Phase 5 currently list "real-data run pending on the CUDA machine," no committed real-data artifact is invalidated; the deltas absorb at the code/test/fixture level only.

### 5.1 Phase 4 — `training_version` bump and loss-curve emission

| ID | Change |
|----|--------|
| DD-BWD-01 | `src/nflpredictor/train/` adds per-epoch loss capture and a new parquet writer call. The training loop's external behavior (sample order, optimizer state, convergence trajectory) shall not change (DD-LC-08). |
| DD-BWD-02 | `training_manifest.json` gains the new `output_sha256` entry (DD-LC-03) and a `training_version` bump (DD-LC-04). |
| DD-BWD-03 | `Docs/Spec-Phase4-BaselineLadder.md` shall gain an amendment block — new requirements under TR-LC-* cross-referencing DD-LC-* in this spec, plus an entry in §12 Future Considerations marking this amendment as landed. The existing TR-MAN-* requirements describing `output_sha256` are unchanged in content; the new key is covered by the existing dict-shape contract. |
| DD-BWD-04 | `Docs/Plan-Phase4-BaselineLadder.md` shall gain an amendment phase enumerating the implementation steps for the new logging. |
| DD-BWD-05 | `tests/test_train_integration.py` and `tests/test_train_determinism.py` shall be extended to assert: (a) the new parquet exists, (b) its rows are sorted, (c) two successive runs produce byte-identical parquets, (d) the manifest's `output_sha256["training_loss_curves.parquet"]` matches the on-disk SHA, (e) the manifest's `training_version` differs from the value present before the amendment, (f) the final-epoch `val_mae` row matches the corresponding `training_summaries` value (DD-LC-07), (g) per DD-LC-08, the prediction parquets are byte-identical to a no-capture baseline. |
| DD-BWD-06 | The train fixture under `tests/fixtures/train/` shall be regenerated (`python -m tests.fixtures.train._regenerate`) and the new artifact + updated manifest committed. |

### 5.2 Phase 5 — source-hash re-pinning and fixture regeneration

Changes (all consequences of Phase 4's manifest SHA rotating):

| ID | Change |
|----|--------|
| DD-BWD-07 | Phase 5's input-contract tests (`tests/test_evaluate_integration.py`, `tests/test_evaluate_determinism.py`, `tests/test_evaluate_cross_phase.py`) will see Phase 4's manifest SHA change. The evaluation fixture under `tests/fixtures/evaluate/` shall be regenerated (`python -m tests.fixtures.evaluate._regenerate`) to match the new train fixture. |
| DD-BWD-08 | `Docs/Spec-Phase5-Evaluation.md` and `Docs/Plan-Phase5-Evaluation.md` shall not require content changes. Their hash-pinning text (EV-IN-06/07/08) is contract-level and remains correct; only the test-time SHA values rotate. |
| DD-BWD-09 | Phase 5's outputs (`metrics_headline.json`, breakdowns, plots) shall not change semantically. Their byte-determinism contract (EV-NF-01, EV-NF-02) is preserved. |

### 5.3 Phases 1–3 — confirmed unaffected

| ID | Change |
|----|--------|
| DD-BWD-10 | Phase 1, Phase 2, and Phase 3 outputs, schemas, specs, plans, fixtures, and tests shall not change. The "minimal backward edits" policy holds: Phase 6's per-game audit needs are reconstructed inside the walkthrough notebook by reading existing artifacts. |

## 6. Integration / Endpoint / Tooling Design

| ID | Requirement |
|----|-------------|
| DD-INT-01 | A new module `src/nflpredictor/diagnostics/` shall hold helper functions the walkthrough notebook (and the training-dynamics notebook) call to load processed artifacts and reconstruct per-game traces (madden-id provenance, encoded vector for one game, split-bucket lookup, prediction-row extraction, plot-coordinate lookup). Keeping the heavy lifting in a package, not in the notebook, makes the notebook small and the helpers testable. |
| DD-INT-02 | The notebook export pipeline is a single command per notebook: `jupyter nbconvert --to markdown notebooks/phase6_walkthrough.ipynb --output ../Docs/Phase6-Walkthrough.md`. The command is documented in `CLAUDE.md`. |
| DD-INT-03 | No CI hook re-runs the notebooks. Notebook re-execution is a developer act, performed when underlying artifacts change. The committed cell outputs are the source of truth for offline readers. |
| DD-INT-04 | The `diagnostics` module shall have at least one unit-test file (`tests/test_diagnostics.py`) covering the per-game trace helpers against the existing train + evaluate fixtures. |

## 7. Changes to Existing Requirements

- **Phase 4** (see §5.1 above). New requirements added under TR-LC-* in Spec-Phase4 referencing DD-LC-*. Plan-Phase4 gains an amendment phase.
- **Phase 5** (see §5.2 above). No spec or plan content changes; fixture and test SHAs rotate.
- **Phases 1–3** (see §5.3 above). No changes.
- **`pyproject.toml`** gains a `docs` optional-dependency group with `jupyter` and `nbconvert`.
- **`CLAUDE.md`** gains a Phase 6 section describing: the new artifacts, the regen commands for both notebooks, the location of the new `diagnostics/` module, and a note that the markdown walkthrough is generated (do not hand-edit).
- **`Docs/Idea.md`**: the Phase 6 section is already in place from the renumbering commit; no further changes from this spec.

## 8. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| DD-NF-01 | `training_loss_curves.parquet` is byte-deterministic per device under the same contract as Phase 4's prediction parquets (TR-NF-01 in Spec-Phase4). |
| DD-NF-02 | The markdown files (`Phase6-ReadingTheOutputs.md`, `Phase6-Walkthrough.md`, `Phase6-TrainingDynamics.md`) and both notebooks shall be committed with LF line endings, matching the repo-wide convention enforced by `.gitattributes`. |
| DD-NF-03 | Notebook cell outputs that embed images (matplotlib plots in the training-dynamics notebook) shall use PNG output. SVG is excluded to keep notebook file size predictable and diffs readable. |
| DD-NF-04 | No determinism contract is asserted on notebook cell outputs themselves. `jupyter nbconvert` and the IPython kernel introduce non-determinism in metadata (cell execution IDs, run timing, kernelspec). The determinism contracts in Phase 6 are limited to the parquet (DD-NF-01) and to the upstream input-hash pins. |
| DD-NF-05 | The `Docs/Phase6-Walkthrough.md` export shall be regeneratable via the documented command (DD-INT-02). The committed copy is *not* asserted byte-identical to a fresh re-export, for the same reasons as DD-NF-04. |

## 9. UI Requirements

Phase 6 has no UI in the conventional sense. The two consumer surfaces are:

- **Markdown rendering** (GitHub web, IDE preview, terminal `cat`). The three `.md` files shall render with no broken image references and no broken internal links.
- **Jupyter notebook rendering** (GitHub web preview, JupyterLab, VS Code's notebook UI). Committed cell outputs shall display the way they rendered at execution time; no client-side execution required.

## 10. Testing Requirements

| ID | Requirement |
|----|-------------|
| DD-TEST-01 | A determinism test (extension of `tests/test_train_determinism.py`) shall run the training build twice against the fixture and assert byte equality of `training_loss_curves.parquet`. |
| DD-TEST-02 | An integrity test shall assert the final-epoch `val_mae` row in `training_loss_curves.parquet` matches the corresponding `training_summaries` value to ~1e-12 relative tolerance per combination × fold (DD-LC-07). |
| DD-TEST-03 | A schema test shall assert the parquet columns are exactly `[combination_id, fold, epoch, train_loss, val_loss, val_mae]` in that order with the dtypes from §4.1. |
| DD-TEST-04 | A sort-order test shall assert the parquet rows are sorted lexicographically by `(combination_id, fold, epoch)`. |
| DD-TEST-05 | A manifest test shall assert `training_manifest.json["output_sha256"]["training_loss_curves.parquet"]` matches `sha256(training_loss_curves.parquet)` and that `training_version` differs from the value present before the amendment (captured in the fixture regen). |
| DD-TEST-06 | A capture-is-read-only test (DD-LC-08) shall confirm that the prediction parquets emitted with capture enabled are byte-identical to a control run with capture disabled (controlled by a test-only flag if needed). |
| DD-TEST-07 | A notebook-export integrity test shall assert that `Docs/Phase6-Walkthrough.md` exists and is non-empty. Its precise content is not asserted (re-export changes byte-level details like cell-execution timestamps). |
| DD-TEST-08 | A documentation-completeness test (lightweight; pure regex) shall scan `Docs/Phase6-ReadingTheOutputs.md` for the 10 required artifact entries (DD-RG-02) by section header presence. |
| DD-TEST-09 | Unit tests for the `diagnostics` helpers (DD-INT-04) covering the per-game-trace utilities against the existing train + evaluate fixtures. |

No test is added that re-executes the walkthrough or training-dynamics notebooks. Notebook execution is a developer act, not a CI gate.

## 11. Security Considerations

No new attack surface. Phase 6 reads existing on-disk artifacts and produces markdown files and a single parquet sidecar. The walkthrough notebook does not load remote URLs, does not execute shell commands beyond the documented `nbconvert` invocation, and does not write outside `Docs/` and `Data/processed/`.

## 12. Future Considerations

Deferred to Phase 7 (Error Analysis & Iteration) or beyond:

- **Worst-N narrative dump.** A separate notebook or script that picks the top-N highest-residual games on the validation slice and produces a structured tagging template. Phase 7.
- **Slice-driven hypothesis log.** A markdown template for recording "I saw X in the by_surface breakdown; hypothesis: feature Y might help" entries with before/after metric deltas. Phase 7.
- **Feature attribution.** Integrated Gradients or SHAP against rung 2/3 predictions. Phase 7.
- **Comparative dashboards.** A multi-run viewer that loads two `training_manifest.json` snapshots and diffs them. Out of scope until iteration cycles begin.
- **Early stopping or LR scheduling in Phase 4.** Phase 6 surfaces the diagnostic vocabulary; whether to *act* on it by adding early stopping is a future Phase 4 amendment, taken up after Phase 7's iteration cycles demand it.
- **Parameterizing the walkthrough game.** v1 fixes a single game for reproducibility. If a future iteration wants per-game audit on demand, the helpers in `diagnostics/` already support arbitrary game IDs; only the notebook wiring would change.

## 13. References

- [Docs/Idea.md](Idea.md) — Phase 6 section under "Documentation & Diagnostics."
- [Docs/Spec-Phase4-BaselineLadder.md](Spec-Phase4-BaselineLadder.md) — the training contract Phase 6 amends (DD-LC, DD-BWD).
- [Docs/Spec-Phase5-Evaluation.md](Spec-Phase5-Evaluation.md) — the source of every artifact the reading-outputs guide explains.
- [Docs/Plan-Phase6-Documentation.md](Plan-Phase6-Documentation.md) — implementation plan (drafted next).
