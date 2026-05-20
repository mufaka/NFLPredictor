# Multi-Year Migration Plan

This document is the cross-phase roadmap for moving the NFL Predictor pipeline from a single 2024 season to all six seasons **2020–2025**. It sits above the per-phase `Plan-Phase*.md` documents: it states the decisions that apply project-wide, the order the phases must change in, and the high-level work each phase needs. Detailed work items still belong in (revised) per-phase plans/specs.

This is a single-developer learning project. The migration is large but mechanical in most phases; the only phase with a genuine new design decision is Phase 3 (Splits).

---

## Why this migration

The original Madden source (`maddennfl24fullplayerratings.csv`) only ever covered 2024. New per-season files now exist for **box scores and Madden ratings, 2020 through 2025** — 12 raw files. The single-season pipeline cannot consume them. Going multi-year roughly 6×'s the training data (≈1,626 games vs 272) and unlocks season-level generalization testing.

---

## Locked decisions

These were decided before this plan was written and are not revisited per-phase:

1. **Scope: full multi-year.** All six seasons 2020–2025 are ingested. The pipeline is no longer single-season.
2. **Player key: keep the synthetic `madden_id`.** The build keeps minting `YYYY-NNNNN` IDs (`ids.py`), now with the season as the vintage prefix (`2020-00001 … 2025-NNNNN`). The new file's own `madden_id` column (a non-unique `NAME_POSGROUP` string) is **not** the key and is dropped on load.
3. **Output layout: combined files.** Phase 1 emits one `madden_all.csv` / `box_scores_all.csv` / `player_id_mapping.csv`, each carrying an explicit `season` column. Downstream phases likewise emit single combined artifacts. The `_2024` filename suffix becomes `_all` project-wide.

---

## Source data summary

| | Old | New |
|---|---|---|
| Madden files | 1 (2024 only) | 6 — `madden_{2020..2025}.csv` |
| Box-score files | 1 (2024 only) | 6 — `box_scores_{2020..2025}.csv` |
| Madden schema | 69 cols, Title Case, stray spaces | 56 cols, lowercase, no spaces |
| Madden team field | nicknames (`49ers`, `Chiefs`) | standard abbrevs (`SF`, `KC`) |
| Box-score games/season | 272 | 263 / 273 / 272 / 273 / 272 / 273 — **1,626 total** |

**Schema notes for the new Madden file:**

- Headers across all six Madden files are **identical** — one header constant covers every year.
- Box-score headers are identical too; only `box_scores_2024.csv` uses CRLF line endings (the rest are LF). The loader must tolerate both.
- 17 columns are dropped vs the old file (Break Tackle, Run/Pass Block Power & Finesse, Lead Block, Break Sack, Throw Under Pressure, Power/Finesse Moves, Block Shedding, Height, Weight, College, Total Salary, Signing Bonus, Player Handness). The remaining 52 ratings carry over 1:1 (renamed). 4 columns are added: `madden_id`, `season`, `high_pos_group`, `position_group`. The current `feature_config.yaml` only consumes Overall Rating + Archetype, so **no current feature breaks** — but the available-column inventory shrinks.
- The shipped `madden_id` is **not unique** (e.g. `JAYLONJONES_D_FIELD` ×2 in 2024) and is not a PFR code, so it cannot join to box scores. Name-normalization matching is still required.
- PFR team codes are franchise-stable across all years (`was` is used even for the 2020 "Washington Football Team"), so a single static **PFR-code → modern-abbrev** map works for every season.

---

## Project-wide changes

These cut across every phase:

- **Filename suffix.** `*_2024.*` → `*_all.*` for every processed artifact and manifest key.
- **`season` column.** Added to every combined artifact (Phase 1 outputs, feature matrices, predictions, breakdowns) so any row can be traced to its year.
- **Manifest input lists.** Each phase's manifest grows from naming one input file to naming the combined upstream artifact(s); Phase 1's manifest lists all 13 raw inputs.
- **Determinism.** The byte-identical-rerun contract is preserved. Sorting becomes `(season, …)`-keyed wherever it was previously single-season.
- **Tests & fixtures.** Every synthetic fixture and integration test is regenerated/updated for the multi-season shape.
- **Version bumps.** Each phase that changes its output contract bumps its `*_version` tag (`normalization_version`, `splits_version`, `training_version`, `evaluation_version`, vocab version).

---

## Per-phase work (high level)

### Phase 1 — Data Build (largest rewrite)

- Loop the build over seasons 2020–2025; read 12 raw files instead of 2.
- Replace `EXPECTED_MADDEN_HEADER` with the 56-column lowercase schema; make the box-score loader CRLF/LF agnostic.
- Replace `teams.py`'s PFR-code → nickname map with a PFR-code → modern-abbrev map.
- `assign_raw_madden_ids` already accepts a `vintage` argument — pass each season, numbering per-season after a per-season sort.
- Drop the new file's source `madden_id` column on load — the synthetic key is canonical.
- Run matching per season (`box_scores_2020 ↔ madden_2020`, …); the tiered name-normalization logic is unchanged.
- Add a `season` column to `player_overrides.csv` so overrides are season-scoped.
- Emit combined `madden_all.csv` / `box_scores_all.csv` / `player_id_mapping.csv` + a `build_manifest.json` with per-season and total counts.
- Bump `normalization_version`.

### Phase 2 — Feature Engineering

- Read the combined Phase 1 outputs; hash-gate against the new `build_manifest.json`.
- Emit combined `features_flat_all.parquet` / `features_pos_all.parquet` (≈1,626 rows) with a `season` column.
- Rebuild `feature_vocab.json` — the vocab now spans six years (more coaches, officials, stadiums, archetypes).
- Update `feature_config.yaml` (and `ConfigReference.md`) to the new lowercase 56-column Madden inventory.

### Phase 3 — Splits (redesigned)

The week-based strategies are **dropped**. Madden ratings are a static per-season snapshot, so a game's week carries no information and there is no temporal leakage to defend against — the S1 week boundaries and S3 expanding-window CV solved a problem this feature design does not have.

Splitting is now **season-holdout**, for two reasons unrelated to time: (1) a pooled random split puts the same season's roster in both train and test, letting the model memorize "this roster → this many points" instead of learning the ratings→points relationship; (2) the project's goal is predicting an unseen future season, which is by definition a season-holdout evaluation. Season-awareness is therefore confined to the split *boundaries* — it is not a modeling feature, and a feature-matrix row remains "this box score had these players."

- **Default split:** train 2020–2023, validate 2024, test 2025 — every boundary is a whole-season holdout.
- The test season is the honest deployment proxy and must never be pooled. The val season is used only for early stopping / model selection, so pooling 2020–2024 and random-splitting train/val is an acceptable alternative knob.
- **Optional strategy:** leave-one-season-out CV (rotate the holdout season) for a tighter, less noisy estimate, at ~6× training cost.
- `splits_config.yaml` switches from week boundaries to season lists (train/val/test seasons); artifact becomes `splits_all.json`; bump `splits_version`.

### Phase 4 — Training

- Read combined features + splits; hash-gate against the new upstream manifests.
- The rung ladder is unchanged; the trivial baselines (mean, team_mean) must compute their statistics per training fold over the multi-season training set.
- Prediction parquets and `training_manifest.json` carry the `season` column / `_all` suffix; bump `training_version`. Expect longer runtimes (~6× data).

### Phase 5 — Evaluation

- `by_season` is a natural new breakdown dimension alongside the existing five.
- Headline metrics and plots are unchanged in formula; outputs adopt the `_all` suffix; bump `evaluation_version`.

### Phase 6 — Documentation & Diagnostics

- Update `CLAUDE.md`, `ConfigReference.md`, `Overview.md`, every `Spec-Phase*.md` / `Plan-Phase*.md`, and the two notebooks.
- Reconcile `Idea.md` — the old "Phase 7: 2025 test" is now part of the training dataset; the held-out test set is whatever Phase 3 designates.
- Re-execute and re-export the walkthrough and training-dynamics notebooks.

---

## Sequencing

Phases are chained by hash gates, so implementation must proceed strictly 1 → 6. Spec revisions, however, can be done up front. Only the specs for Phases 1–3 are revised up front (their output contracts change materially); the Phase 4–6 specs are patched as those phases are implemented.

| Stage | Work | Notes |
|-------|------|-------|
| 0 | Revise specs for Phases 1–3 | Specs are source of truth |
| 1 | Phase 1 rewrite + tests | Unblocks everything |
| 2 | Phase 2 | Mechanical once Phase 1 lands |
| 3 | Phase 3 + split-strategy redesign | Only phase with a new design |
| 4 | Phase 4 | Longer training runs |
| 5 | Phase 5 | Add `by_season` breakdown |
| 6 | Phase 6 docs + notebook re-runs | Final reconciliation |

---

## Open questions

1. **2020 COVID season** — 263 games and an empty-stadium / attendance anomaly. Keep as-is, or flag it for the modeling phases as a known outlier?
