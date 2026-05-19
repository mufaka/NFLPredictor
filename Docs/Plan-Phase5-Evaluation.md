# Phase 5: Evaluation Implementation Plan

This document defines the phased implementation plan for the Evaluation phase of the NFL Predictor project, based on [Spec-Phase5-Evaluation.md](./Spec-Phase5-Evaluation.md). Each phase builds on the previous one and contains checkbox-tracked work items. Requirement IDs (`EV-*`) reference the corresponding entries in the specification.

This is a single-developer learning project. Phases are sized for one person to complete in sittings of an hour or two, with tests landing in the same phase as the code they cover. Phase 5 is narrower than Phase 4 in surface area — no PyTorch, no training loop, no per-device subtleties — but the plot phase introduces matplotlib and an asymmetric determinism contract (byte-exact metric tables, softer PNG byte-identity), so the plan dedicates one of its six phases to plots specifically.

---

## Progress Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Scaffolding, Config, and Upstream Hash Gates | Complete (2026-05-18) |
| 2 | Sources, Predictions Loader, and Metric Primitives | Complete (2026-05-18) |
| 3 | Headline Metrics, Breakdowns, and Tabular Writers | Complete (2026-05-18) |
| 4 | Plot Rendering | Complete (2026-05-18) |
| 5 | Pipeline Orchestration, Manifest, and Real-Data Smoke Run | Complete (2026-05-18); real-data smoke deferred to the CUDA machine until Phase 4 outputs exist on disk |
| 6 | Determinism Hardening, Integration Tests, and Documentation | Complete (2026-05-18); §6.7 wall-clock measurement deferred behind the real-data smoke |

---

## Current State

- Phases 1–4 are complete; the full Phase 1/2/3 output set plus `Data/processed/training_manifest.json` and `Data/processed/predictions/*.parquet` are produced by their respective entry points.
- On the dev (CPU-only) machine, Phase 4's real-data run has not yet been executed — `Data/processed/predictions/` may be empty. Phase 5's pipeline-run test must skip gracefully when that directory is empty, mirroring `test_train_pipeline_run.py`'s pattern. The full real-data run is expected from the CUDA training machine.
- The Phase 5 specification is final at `Docs/Spec-Phase5-Evaluation.md` (all Phase 5 open questions in `Docs/Idea.md` resolved in-spec).
- No `src/nflpredictor/evaluate/` module exists yet.
- No `Data/raw/evaluation_config.yaml` exists yet; it must be created as part of Phase 1 of this plan.
- One new runtime dependency: `matplotlib` (Agg backend; figure rendering only — no interactive use). `pyyaml`, `pyarrow`, `pandas`, and `numpy` are already declared from earlier phases.

---

## Guiding Principles

1. **Work inside the virtual environment.** All `pip`, `python`, and `pytest` commands run with the `.venv/` venv activated.
2. **Tests accompany every phase.** No phase is complete until the tests it introduces pass.
3. **Vertical slices where practical.** Each phase delivers something runnable or testable on its own; metric primitives in Phase 2 can be exercised before headlines, breakdowns, or plots exist.
4. **One responsibility per module.** Match the proposed `src/nflpredictor/evaluate/` layout from §5 of the spec; do not pile everything into a single file.
5. **Two determinism contracts, kept distinct.** Metric content (JSON + breakdown parquets + manifest scalars) is byte-deterministic with no caveat; PNG byte-identity holds only within the same pinned matplotlib wheel on the same platform. Tests in Phases 3 and 4 must assert the right level for each artifact family.
6. **Phase 5 measures. It does not select winners.** No "best model" field in any output. Headline-metric values are reported; the human or Phase 7 chooses.
7. **Test slice is touched unconditionally.** Per EV-MAN-07 and the §"Choices" decision in the spec, S1.test rows appear in every output on every run with no opt-in flag.
8. **The specification is source of truth.** When the plan and the spec disagree, fix the plan or fix the spec — do not silently improvise.

---

## Proposed Repository Layout (additions)

Phase 5 adds the following files to the existing repo. Files outside this list are untouched (Phases 1–4's packages stay read-only from Phase 5's perspective).

```
NFLPredictor/
├── Data/
│   ├── raw/
│   │   └── evaluation_config.yaml                  # NEW — v1 default config
│   └── processed/
│       └── evaluation/                             # NEW dir
│           ├── metrics_headline.json
│           ├── breakdowns/
│           │   ├── by_team.parquet
│           │   ├── by_week.parquet
│           │   ├── by_home_away.parquet
│           │   ├── by_surface.parquet
│           │   └── by_roof.parquet
│           ├── plots/
│           │   ├── ladder_summary__val.png
│           │   ├── ladder_summary__test.png
│           │   ├── ladder_summary__pooled.png
│           │   ├── <combination_id>__<slice>__scatter.png        # × 12 combinations × slice
│           │   ├── <combination_id>__<slice>__residuals.png      # × 12 combinations × slice
│           │   └── <combination_id>__<slice>__by_week.png        # × 12 combinations × slice (default-on)
│           └── evaluation_manifest.json
├── src/
│   └── nflpredictor/
│       └── evaluate/                               # NEW package
│           ├── __init__.py
│           ├── __main__.py                         # entry: python -m nflpredictor.evaluate
│           ├── pipeline.py                         # top-level orchestration
│           ├── config.py                           # YAML load + validation
│           ├── sources.py                          # Phase 2 + Phase 3 + Phase 4 load + SHA verification
│           ├── metrics.py                          # MAE/RMSE/W-L/spread/total primitives
│           ├── breakdowns.py                       # per-dimension aggregation
│           ├── plots.py                            # matplotlib renderers
│           ├── outputs.py                          # JSON + parquet + PNG writers
│           └── manifest.py                         # evaluation_manifest.json construction
└── tests/
    ├── test_evaluate_smoke.py
    ├── test_evaluate_config.py
    ├── test_evaluate_sources.py
    ├── test_evaluate_metrics.py
    ├── test_evaluate_breakdowns.py
    ├── test_evaluate_outputs.py
    ├── test_evaluate_plots.py
    ├── test_evaluate_manifest.py
    ├── test_evaluate_integration.py
    ├── test_evaluate_determinism.py
    ├── test_evaluate_pipeline_run.py
    └── fixtures/
        └── evaluate/                               # NEW subdir
            ├── raw_phase2/
            │   ├── features_flat_2024.parquet      # sliced (~30 games)
            │   ├── features_pos_2024.parquet       # sliced (matching)
            │   ├── feature_vocab.json
            │   └── feature_manifest.json           # regenerated to match slice SHAs
            ├── raw_phase3/
            │   ├── splits_2024.json                # built from sliced universe
            │   └── splits_manifest.json            # regenerated
            ├── raw_phase4/
            │   ├── predictions/                    # 12 prediction parquets (or trimmed)
            │   │   └── *.parquet
            │   └── training_manifest.json          # regenerated to match parquet SHAs
            ├── raw/
            │   └── evaluation_config.yaml          # mirrors v1 default (or trimmed)
            └── expected/
                ├── metrics_headline.json
                ├── breakdowns/
                │   └── *.parquet                   # one per enabled dimension
                └── evaluation_manifest.json        # build_timestamp_utc blanked
```

Tests use a `test_evaluate_*` prefix to keep them visually separate from earlier phases' test files. All existing tests stay green throughout.

---

## Phase 1: Scaffolding, Config, and Upstream Hash Gates

Establishes the package skeleton, ships the v1 reference `evaluation_config.yaml`, and enforces all three upstream input contracts (Phase 2, Phase 3, Phase 4). Nothing functional yet beyond input validation — but every later phase trusts these invariants.

**Satisfies:** `EV-IN-01` through `EV-IN-09`, `EV-CFG-01` through `EV-CFG-08`, `EV-TEST-01`, partial `EV-TEST-10`.

### 1.1 Runtime Dependency

- [ ] Activate the venv: `source .venv/bin/activate`.
- [ ] Install matplotlib: `pip install matplotlib`. Pin to the resolved version in the venv's `pip freeze`; that wheel becomes the pinned matplotlib for the PNG determinism contract (EV-NF-02).
- [ ] Update `pyproject.toml` to add `matplotlib` as a runtime dep. Do not pin a specific version in source control beyond the major/minor — the exact wheel is captured by `pip freeze` and recorded in the evaluation manifest at run time (EV-MAN-01 `matplotlib_version`).
- [ ] Confirm `matplotlib.use("Agg")` works headlessly: `python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; print('ok')"`.

### 1.2 Package Skeleton

- [ ] Create `src/nflpredictor/evaluate/__init__.py` (empty).
- [ ] Create `src/nflpredictor/evaluate/__main__.py` with a stub `main()` that prints `"evaluation build not implemented yet"` and exits `0`.
- [ ] Verify the entry point: `python -m nflpredictor.evaluate` prints the stub message.
- [ ] Add a smoke test `tests/test_evaluate_smoke.py` that imports `nflpredictor.evaluate` and asserts the import succeeds. Run `pytest` to confirm.

### 1.3 Default Evaluation Config

- [ ] Create `Data/raw/evaluation_config.yaml` with the v1 contents shown verbatim in §4.1 of the spec (`headline_metric: "mae"`; all five `breakdowns` true; `plots.scatter`/`residual_distribution`/`ladder_summary` true; `plots.breakdown_plots: [by_week]`; `plots.dpi: 100`; `plots.figure_width_inches: 8.0`; `plots.figure_height_inches: 5.0`).
- [ ] Confirm the file parses cleanly with `python -c "import yaml; print(yaml.safe_load(open('Data/raw/evaluation_config.yaml')))"`.

### 1.4 Config Loader and Validator

- [ ] In `src/nflpredictor/evaluate/config.py`, define an `EvaluationConfig` frozen dataclass with fields mirroring the YAML schema: `evaluation_version: str`, `headline_metric: str`, `breakdowns: BreakdownToggles`, `plots: PlotConfig`. Nested dataclasses `BreakdownToggles` (with boolean fields per dimension) and `PlotConfig` (with `scatter: bool`, `residual_distribution: bool`, `ladder_summary: bool`, `breakdown_plots: tuple[str, ...]`, `dpi: int`, `figure_width_inches: float`, `figure_height_inches: float`).
- [ ] Implement `load_evaluation_config(path: pathlib.Path) -> EvaluationConfig` that:
  - Reads via `yaml.safe_load` (per `EV-SEC-04`).
  - Rejects unknown top-level keys (`EV-CFG-01`).
  - Validates `evaluation_version` is a non-empty string (`EV-CFG-02`).
  - Validates `headline_metric` is one of `{"mae", "rmse", "wl_accuracy", "spread_mae", "total_mae"}` (`EV-CFG-03`).
  - Validates `breakdowns` keys are a subset of `{by_team, by_week, by_home_away, by_surface, by_roof}` and values are booleans (`EV-CFG-04`).
  - Validates `plots.scatter` / `plots.residual_distribution` / `plots.ladder_summary` are booleans (`EV-CFG-05`).
  - Validates `plots.breakdown_plots` entries are a subset of `{by_week, by_team, by_home_away}` AND every listed entry has its corresponding `breakdowns.<dim>` set to `true`; otherwise fail fast (`EV-CFG-06`).
  - Validates `plots.dpi` is a positive integer; `plots.figure_width_inches` and `plots.figure_height_inches` are positive floats (`EV-CFG-07`).

### 1.5 Upstream Source-Hash Gates

- [ ] In `src/nflpredictor/evaluate/sources.py`, define module-level constants for the Phase 2 basenames (`features_flat_2024.parquet`, `features_pos_2024.parquet`, `feature_vocab.json`, `feature_manifest.json`), Phase 3 basenames (`splits_2024.json`, `splits_manifest.json`), and Phase 4 basenames (`training_manifest.json` plus a wildcard `predictions/*.parquet`).
- [ ] Reuse `compute_sha256` from `src/nflpredictor/databuild/manifest.py`.
- [ ] Implement `verify_phase2_outputs(processed_dir: pathlib.Path) -> dict` analogous to Phase 4's identical helper, raising `Phase2OutputMismatchError` on any divergence (`EV-IN-06`). Returns the parsed Phase 2 manifest.
- [ ] Implement `verify_phase3_outputs(processed_dir: pathlib.Path) -> dict` analogously for `splits_2024.json` (`EV-IN-07`).
- [ ] Implement `verify_phase4_outputs(processed_dir: pathlib.Path) -> dict` that:
  - Loads `training_manifest.json`.
  - Enumerates every entry in `output_sha256` whose key starts with `predictions/`.
  - For each, recomputes the on-disk SHA-256 and compares against the manifest's recorded SHA (`EV-IN-08`).
  - Raises `Phase4OutputMismatchError` naming the divergent file on any mismatch.
  - Returns the parsed Phase 4 manifest for downstream provenance.

### 1.6 Tests

- [ ] `tests/test_evaluate_config.py` (`EV-TEST-01`): cover (a) shipped v1 load (must accept); (b) unknown top-level key (must reject); (c) `headline_metric` outside the allowed set (must reject); (d) unknown breakdown key (must reject); (e) `plots.breakdown_plots` entry whose corresponding `breakdowns.<dim>` is `false` (must reject); (f) non-positive `plots.dpi` (must reject); also exercise valid trim combinations (e.g., `breakdowns.by_team: false` with `breakdown_plots: [by_week]` — must accept).
- [ ] `tests/test_evaluate_sources.py` (partial `EV-TEST-10`): cover (a) happy path against real Phase 2 + Phase 3 + Phase 4 outputs (skip with a clear message when `Data/processed/predictions/` is empty); (b) tampered `features_flat_2024.parquet` → fail-fast naming the file; (c) tampered `splits_2024.json` → fail-fast naming the file; (d) tampered Phase 4 prediction parquet → fail-fast naming the file; (e) missing Phase 4 manifest → fail-fast.

**Definition of done:** Config loading rejects every malformed case enumerated in §3.2; all three upstream hash gates block runs against tampered inputs.

---

## Phase 2: Sources, Predictions Loader, and Metric Primitives

Loads the upstream artifacts into in-memory structures and implements the five core metric arithmetic primitives. Pure data plumbing and numpy math — no aggregation, no JSON, no plots yet.

**Satisfies:** `EV-IN-10`, `EV-MET-01` through `EV-MET-09`, `EV-COMB-01` through `EV-COMB-05`, `EV-NF-03`, `EV-TEST-02`.

### 2.1 Source Loaders

- [ ] In `src/nflpredictor/evaluate/sources.py`, implement `load_features_flat(processed_dir) -> pandas.DataFrame` that reads `features_flat_2024.parquet` via `pyarrow.parquet`. Preserves column dtypes; sorts rows by `GameId` for determinism. (Only `flat` is needed for labels + breakdown columns; `pos` is loaded only for label parity verification per EV-NF-03.)
- [ ] Implement `load_features_pos(processed_dir) -> pandas.DataFrame` analogously for `features_pos_2024.parquet`.
- [ ] Implement `load_vocab(processed_dir) -> dict[str, list[str]]` that reads `feature_vocab.json` (needed for `by_surface` / `by_roof` integer-code → label mapping).
- [ ] Implement `load_splits(processed_dir) -> dict` that reads `splits_2024.json`.
- [ ] Implement `assert_label_parity(flat, pos) -> None` mirroring Phase 4's helper — verifies the two frames share identical `(GameId, home_score, away_score)` triples; raises on mismatch (`EV-NF-03`).

### 2.2 Predictions Loader

- [ ] In `src/nflpredictor/evaluate/sources.py`, implement `enumerate_combinations(phase4_manifest: dict) -> list[CombinationKey]` that reads the prediction filenames listed under `training_manifest.json → output_sha256` and returns one `CombinationKey` per file. `CombinationKey` is a small dataclass with fields `combination_id: str` (filename stem), `strategy: str` (`"S1"` or `"S3"`, derived from the `__s1`/`__s3` suffix), `parquet_path: pathlib.Path`. Sort by `combination_id` lexicographically (`EV-COMB-05`).
- [ ] Implement `load_prediction_parquet(key: CombinationKey) -> pandas.DataFrame` that:
  - Reads the parquet via `pyarrow`.
  - For S1: returns columns `(slice, GameId, pred_home, pred_away)`.
  - For S3: returns columns `(fold_index, GameId, pred_home, pred_away)`.
- [ ] Implement `validate_prediction_coverage(predictions: pd.DataFrame, key: CombinationKey, splits: dict) -> None` (`EV-IN-10`):
  - For S1: assert `slice == "val"` rows' GameId set equals `splits.S1.val`; `slice == "test"` rows' GameId set equals `splits.S1.test`.
  - For S3: per `fold_index = i`, assert GameId set equals `splits.S3.folds[i].val`.
  - Raise with a clear message on mismatch.

### 2.3 Metric Primitives

- [ ] In `src/nflpredictor/evaluate/metrics.py`, implement pure-numpy functions over four 1-D arrays (`pred_home`, `pred_away`, `true_home`, `true_away`):
  - `compute_mae_headline(...) -> float` — `mean(|pred_home - true_home| + |pred_away - true_away|) / 2` (`EV-MET-01`). Identical formula to Phase 4's `TR-MAN-04`.
  - `compute_mae_per_side(...) -> tuple[float, float]` — `(mae_home, mae_away)` (`EV-MET-02`).
  - `compute_rmse_per_side(...) -> tuple[float, float]` — `(rmse_home, rmse_away)` (`EV-MET-03`).
  - `compute_wl_accuracy(...) -> float` — sign-of-difference comparison with the explicit tie-handling rules from `EV-MET-04` (predicted tie counts wrong unless actual is also a tie; actual tie counts correct only when prediction is also a tie). Use `numpy.sign` and explicit equality for ties.
  - `compute_spread_mae(...) -> float` (`EV-MET-05`).
  - `compute_total_mae(...) -> float` (`EV-MET-06`).
- [ ] Implement `compute_all_metrics(pred_home, pred_away, true_home, true_away) -> dict[str, float]` that calls each primitive and returns the canonical metric dict shape: `{"mae", "mae_home", "mae_away", "rmse_home", "rmse_away", "wl_accuracy", "spread_mae", "total_mae"}`. All keys present, all floats.
- [ ] Implement `compute_cell_metrics(pred_home, pred_away, true_home, true_away) -> dict` that returns `{"n_games": int, **metrics}`. When the input arrays are empty (`n_games == 0`), all metric values are `None` (`EV-MET-09`); the dict still carries every key.

### 2.4 Predictions / Labels Joiner

- [ ] In `src/nflpredictor/evaluate/metrics.py` (or a thin helper in `sources.py`), implement `join_predictions_with_labels(predictions: pd.DataFrame, features_flat: pd.DataFrame) -> pd.DataFrame` that left-joins on `GameId` and asserts no nulls in `true_home` / `true_away` (every prediction GameId must exist in features). Returns a frame with `(GameId, slice or fold_index, pred_home, pred_away, true_home, true_away)`.

### 2.5 Slice Iterator

- [ ] In `src/nflpredictor/evaluate/metrics.py`, implement `iter_cells(predictions_with_labels: pd.DataFrame, strategy: str) -> Iterator[Cell]` that yields:
  - For `strategy == "S1"`: two cells with `slice ∈ {"val", "test"}` (`EV-COMB-02`, `EV-COMB-05`).
  - For `strategy == "S3"`: `fold_count` cells with `slice = "fold_<i>"` plus one pooled cell with `slice = "pooled"` (`EV-COMB-03`, `EV-COMB-05`). The pooled cell concatenates all per-fold rows (duplicates if a GameId appears in multiple folds, per `EV-MET-07`).
- [ ] Order matches `EV-COMB-05`: S1 → `["val", "test"]`; S3 → `["fold_0", …, "fold_<n-1>", "pooled"]`.

### 2.6 Tests

- [ ] `tests/test_evaluate_sources.py` (extension): label-parity check accepts a real Phase 2 pair and rejects a synthetic mismatched pair; `validate_prediction_coverage` accepts the real Phase 4 parquets and rejects a synthetic parquet with a missing GameId.
- [ ] `tests/test_evaluate_metrics.py` (`EV-TEST-02`): for a 6-game synthetic fixture with hand-computed expected values, assert every metric primitive matches its expected value to `numpy.float64` round-trip precision. Cover:
  - Standard case (no ties).
  - W/L tie cases: `pred_home == pred_away` with actual non-tie (counts wrong); `true_home == true_away` with predicted non-tie (counts wrong); both ties (counts correct).
  - Empty cell: `compute_cell_metrics` with zero-length arrays returns `n_games == 0` and every metric `None`, no exception (`EV-MET-09`).
  - `compute_all_metrics` returns the canonical 8-key dict in every case.
- [ ] `tests/test_evaluate_metrics.py` (extension): `iter_cells` returns the right sequence for synthetic S1 and S3 prediction frames, including a non-trivial S3 case with 3 folds and one overlapping GameId across folds (pooled `n_games` = sum of per-fold sizes including the repeat).

**Definition of done:** Phase 2/3/4 sources load cleanly with parity and coverage verified; every metric primitive matches its hand-computed expectation; empty-cell handling is exception-free.

---

## Phase 3: Headline Metrics, Breakdowns, and Tabular Writers

Aggregates the metric primitives across `(combination, slice)` cells and breakdown dimensions; emits `metrics_headline.json` and the five breakdown parquets. After this phase, the full tabular contract is in place; only plots and the manifest remain.

**Satisfies:** `EV-BRK-01` through `EV-BRK-07`, `EV-OUT-01`, `EV-OUT-02`, `EV-OUT-03`, `EV-TEST-03`, `EV-TEST-04`, `EV-TEST-05`.

### 3.1 Headline Aggregator

- [ ] In `src/nflpredictor/evaluate/metrics.py`, implement `build_headline_for_combination(combination_id, joined_predictions, strategy) -> dict` that:
  - Calls `iter_cells` to produce per-slice cells.
  - Calls `compute_cell_metrics` on each cell.
  - Returns a dict shaped per the relevant block in §4.2: `{<slice_key>: {<metric_key>: <value or None>, "n_games": <int>}}` with slice keys ordered per `EV-COMB-05`.
- [ ] Implement `build_headline_metrics(combinations: list[CombinationKey], predictions_by_combo: dict[str, pd.DataFrame], features_flat: pd.DataFrame) -> dict` that produces the full top-level structure: `{"combinations": {<combination_id>: <per-combo block>, …}}` with combination keys ordered lexicographically.

### 3.2 Breakdown Aggregators

- [ ] In `src/nflpredictor/evaluate/breakdowns.py`, define a small `BreakdownSpec` dataclass with: `name: str` (e.g., `"by_team"`), `breakdown_key_columns: tuple[str, ...]` (the column(s) added to each output row beyond `combination_id`/`slice`), and a callable `group_keyer(joined_row) -> list[GroupKey]` that returns the breakdown keys a single game contributes to. Most dimensions return one key per game; `by_team` and `by_home_away` return two (per `EV-BRK-01`, `EV-BRK-03`).
- [ ] Implement `aggregate_breakdown(combination_id, joined_predictions, strategy, spec, vocab) -> pd.DataFrame` that:
  - Iterates cells per `iter_cells`.
  - Within each cell, partitions games by `spec.group_keyer(...)`; calls `compute_cell_metrics` per partition.
  - Stacks rows with columns `(combination_id, slice, *spec.breakdown_key_columns, n_games, mae, mae_home, mae_away, rmse_home, rmse_away, wl_accuracy, spread_mae, total_mae)`.
  - For dimensions whose value universe is fixed (week 1–18, the team_codes vocab, the surface/roof vocabs), emit empty cells (`n_games == 0`, nulls per `EV-MET-09`) for breakdown values present in the universe but absent from the slice (`EV-BRK-06`'s "complete-by-construction across the dataset's value universe").
- [ ] Implement the five concrete specs:
  - `BY_TEAM = BreakdownSpec(name="by_team", breakdown_key_columns=("team_code", "home_or_away"), group_keyer=...)` — emits two rows per game keyed by `(team, "home")` and `(team, "away")` per `EV-BRK-01`. The team_codes vocab from Phase 2's `feature_vocab.json` defines the universe (`EV-BRK-07` PFR 3-letter codes).
  - `BY_WEEK = BreakdownSpec(name="by_week", breakdown_key_columns=("week",), …)` per `EV-BRK-02`. Universe: weeks 1–18.
  - `BY_HOME_AWAY = BreakdownSpec(name="by_home_away", breakdown_key_columns=("home_or_away",), …)` per `EV-BRK-03`. Each game contributes two rows; per-side metrics are surfaced via the home/away-specific MAE/RMSE columns.
  - `BY_SURFACE = BreakdownSpec(name="by_surface", breakdown_key_columns=("surface_code", "surface_label"), …)` per `EV-BRK-04`. Universe and labels read from `vocab["surface"]`.
  - `BY_ROOF = BreakdownSpec(name="by_roof", breakdown_key_columns=("roof_code", "roof_label"), …)` per `EV-BRK-05`. Universe and labels read from `vocab["roof"]`.
- [ ] Implement `build_breakdown_table(spec, combinations, predictions_by_combo, features_flat, vocab) -> pd.DataFrame` that concatenates per-combination frames and sorts rows lexicographically by `(combination_id, slice, *breakdown_key_columns)` (`EV-OUT-02`).

### 3.3 JSON and Parquet Writers

- [ ] In `src/nflpredictor/evaluate/outputs.py`, implement `write_metrics_headline(headline: dict, path: pathlib.Path) -> None` using `json.dump(..., sort_keys=True, indent=2)` with a trailing newline (`EV-OUT-01`). UTF-8 explicit; Unix line endings (`EV-NF-09`).
- [ ] Implement `write_breakdown_parquet(df: pd.DataFrame, path: pathlib.Path) -> None`:
  - Coerces dtypes per §4.3 (`combination_id` → string, `slice` → string, `week` → int8 where applicable, `team_code` → string, `home_or_away` → string, `<dim>_code` → int8, `<dim>_label` → string, `n_games` → int32, all metric columns → float64).
  - Writes via `pyarrow.parquet.write_table` with `compression="snappy"`, `row_group_size=1024` (`EV-OUT-03`).
  - Asserts the sort order matches `EV-OUT-02` before writing (cheap defensive check).
- [ ] Implement `ensure_evaluation_dirs(processed_dir) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]` that creates `Data/processed/evaluation/`, `Data/processed/evaluation/breakdowns/`, and `Data/processed/evaluation/plots/` if absent and returns the three paths (`EV-OUT-05`).

### 3.4 Tests

- [ ] `tests/test_evaluate_breakdowns.py` (`EV-TEST-03`):
  - Build a 6-game synthetic joined fixture. Partition by team and assert per-team rows include the expected `(team, "home")` and `(team, "away")` doubling and metric values match hand computations.
  - Inject a team absent from the slice but present in the vocab; assert an `n_games == 0` row appears with null metrics.
  - Repeat for `by_week` with a week absent from the slice; for `by_surface` / `by_roof` with a category absent from the slice.
  - `by_home_away`: assert `mae_home` on the home row reflects only home-side residuals; analogous for away.
- [ ] `tests/test_evaluate_outputs.py` (partial `EV-TEST-04`, `EV-TEST-05`):
  - Round-trip a synthetic headline dict through `write_metrics_headline` → `json.load`; assert combination keys sorted lexicographically; assert slice keys ordered per `EV-COMB-05`; assert metric keys sorted alphabetically within each cell.
  - Round-trip a synthetic breakdown frame through `write_breakdown_parquet` → `pq.read_table`; assert column names, dtypes, and row order match §4.3 + `EV-OUT-02`.
  - Re-run the JSON writer on the same input; assert byte-identical bytes (`pathlib.Path.read_bytes()` equality). Same for the parquet writer.

**Definition of done:** Aggregating any synthetic prediction fixture produces a valid headline JSON + five breakdown parquets; every value universe is honored (empty cells emitted where appropriate); writers honor §4.3 / `EV-OUT-*` contracts byte-identically across re-runs.

---

## Phase 4: Plot Rendering

Adds the matplotlib-based PNG renderers for the three always-on plot families plus the configurable per-breakdown plots. Introduces the asymmetric determinism contract (PNG byte-identity within a pinned matplotlib wheel only).

**Satisfies:** `EV-PLOT-01` through `EV-PLOT-07`, `EV-OUT-04`, `EV-TEST-04` (plot existence assertions).

### 4.1 Matplotlib Setup

- [ ] At the top of `src/nflpredictor/evaluate/plots.py`, do `import matplotlib` immediately followed by `matplotlib.use("Agg")` before importing `matplotlib.pyplot`. Document the reason inline: required for headless determinism (`EV-PLOT-05`).
- [ ] Implement `save_figure(fig, path: pathlib.Path, *, dpi: int) -> None` that:
  - Calls `fig.savefig(path, dpi=dpi, metadata={"Software": None, "Creation Time": None})` (`EV-PLOT-05`; use the equivalent metadata-stripping idiom for the matplotlib version in use — verify against the pinned wheel's documented `metadata=` keys).
  - Calls `plt.close(fig)` after saving.

### 4.2 Plot Renderers

- [ ] In `src/nflpredictor/evaluate/plots.py`, implement `plot_scatter(combination_id, slice_name, joined: pd.DataFrame, plot_cfg: PlotConfig) -> Figure` that:
  - Creates a `figsize=(plot_cfg.figure_width_inches, plot_cfg.figure_height_inches)` figure.
  - Plots two scatter series (home and away) of actual vs predicted scores on a single axis.
  - Draws an identity line `y = x` spanning the data range.
  - Sets title `f"{combination_id} — {slice_name}"`, labels `Actual score` / `Predicted score`, legend `[Home, Away]`.
  - Returns the `Figure` (caller saves and closes via `save_figure`).
  - Implements `EV-PLOT-01`.
- [ ] Implement `plot_residual_distribution(combination_id, slice_name, joined, plot_cfg) -> Figure` per `EV-PLOT-02` — histogram of `pred − true` residuals, home and away overlaid. Use a fixed bin specification (e.g., `bins=numpy.arange(-30, 30, 2)` or an explicit count) so binning is deterministic across runs.
- [ ] Implement `plot_ladder_summary(headline: dict, slice_basis: str, headline_metric: str, plot_cfg) -> Figure` per `EV-PLOT-03`:
  - One bar per combination whose per-combo block contains `slice_basis` (where `slice_basis ∈ {"val", "test", "pooled"}`); skip combinations whose parquet does not contribute to the given slice (e.g., an S3 combination omitted from a `slice_basis == "test"` summary).
  - Bar height = the configured `headline_metric` value for that `(combination, slice_basis)`.
  - Sort combinations lexicographically for deterministic ordering.
  - Title `f"Ladder summary — {slice_basis} {headline_metric}"`.
- [ ] Implement `plot_breakdown(combination_id, slice_name, dim_name, breakdown_rows: pd.DataFrame, headline_metric: str, plot_cfg) -> Figure` per `EV-PLOT-04`:
  - For `dim_name == "by_week"`: line chart with `week` on x-axis and the chosen `headline_metric` on y-axis.
  - For `dim_name == "by_team"`: horizontal bar chart with team codes on y-axis (sorted lexicographically) and the chosen metric on x-axis. Includes both home and away rows (one bar each, distinguished by color or by suffixed team label).
  - For `dim_name == "by_home_away"`: two-bar chart (home and away) on a single axis.

### 4.3 Plot Driver

- [ ] In `src/nflpredictor/evaluate/plots.py`, implement `render_all_plots(headline, breakdown_tables_by_dim, predictions_by_combo, features_flat, combinations, plot_cfg, plots_dir) -> list[pathlib.Path]` that:
  - For each combination, for each emitted slice basis (`{"val", "test"}` for S1; `{"pooled"}` for S3), renders the enabled per-combination plots (`scatter`, `residual_distribution`, plus each entry in `plot_cfg.breakdown_plots`) and saves them with filenames per `EV-PLOT-01`/`EV-PLOT-02`/`EV-PLOT-04` (e.g., `<combination_id>__<slice>__scatter.png`).
  - For each `slice_basis ∈ {"val", "test", "pooled"}`, renders the ladder summary if `plot_cfg.ladder_summary` and saves as `ladder_summary__<slice_basis>.png` (`EV-PLOT-03`).
  - Returns the list of written PNG paths for the manifest's `output_sha256` population.
- [ ] All numeric color choices, font sizes, and styles fall through to matplotlib defaults (`EV-PLOT-06`). No `numpy.random.seed`-based jitter or color sampling.

### 4.4 Tests

- [ ] `tests/test_evaluate_plots.py`:
  - Smoke test: with a tiny synthetic joined frame (3 games), render each plot family and assert: (a) the function returns a `Figure`; (b) `save_figure` writes a file with non-zero size; (c) the file's leading bytes are PNG magic (`b"\x89PNG\r\n\x1a\n"`).
  - PNG byte-determinism within a wheel: render the same plot twice in succession; assert byte-equality (`pathlib.Path.read_bytes()`). This locks down `EV-NF-02`'s within-wheel byte-identity within the test environment. If this test proves flaky across CI runs due to underlying matplotlib non-determinism (e.g., font caching first-run differences), fall back to asserting deterministic content via `PIL.Image.open(...).tobytes()` pixel equality after a warm-up render — and document the relaxation.
  - Title / label content: render a plot and assert the title text matches the expected `f"{combination_id} — {slice}"` format by inspecting the returned `Figure`'s axes title before saving.

**Definition of done:** All four plot families render PNG files for any synthetic joined fixture; rendering is byte-deterministic (or pixel-deterministic after warm-up) within the pinned matplotlib wheel.

---

## Phase 5: Pipeline Orchestration, Manifest, and Real-Data Smoke Run

Wires everything together into a runnable build, emits the evaluation manifest, runs end-to-end against the real Phase 4 outputs (or skips gracefully when absent), and handles the stale-output cleanup contract.

**Satisfies:** `EV-OUT-05`, `EV-OUT-06`, `EV-MAN-01` through `EV-MAN-07`, `EV-NF-04`, `EV-NF-05`, `EV-NF-06`, `EV-NF-07`, `EV-NF-08`, `EV-NF-10`, `EV-NF-11`, `EV-TEST-11`.

### 5.1 Manifest Construction

- [ ] In `src/nflpredictor/evaluate/manifest.py`, reuse `compute_sha256` and `try_get_git_commit` from `databuild.manifest`.
- [ ] Implement `build_evaluation_summary(headline: dict) -> dict` returning the `EV-MAN-02` shape:
  - For S1 combination keys: `{"val": <headline-metric value or None>, "test": <… or None>}`.
  - For S3 combination keys: `{"pooled": <…>, "mean_per_fold": <mean of fold values, excluding None>, "per_fold": [<headline-metric value per fold>, …]}`.
  - The "headline metric" used here is `config.headline_metric`.
- [ ] Implement `build_evaluation_manifest(*, config, evaluation_config_sha256, phase2_source_sha256, phase3_source_sha256, phase4_source_sha256, output_sha256, phase4_manifest_git_commit, combination_ids, evaluation_summary, matplotlib_version, numpy_version, pyarrow_version) -> dict` constructing the full manifest dict per `EV-MAN-01`.
- [ ] Implement `write_evaluation_manifest(manifest: dict, path: pathlib.Path) -> None` using `json.dump(..., sort_keys=True, indent=2)` with a trailing newline (`EV-MAN-05`).
- [ ] Add inline assertions (or a unit test in §5.4) that every key required by `EV-MAN-01` is present.

### 5.2 Stale-Output Cleanup

- [ ] In `src/nflpredictor/evaluate/outputs.py`, implement `cleanup_disabled_outputs(evaluation_dir: pathlib.Path, config: EvaluationConfig) -> None` that (`EV-OUT-06`):
  - For each breakdown dimension whose `config.breakdowns.<dim> == False`, deletes `evaluation/breakdowns/<dim>.parquet` if it exists.
  - For each plot family whose toggle is `False`, deletes the matching PNGs from `evaluation/plots/` (use a per-family filename-pattern matcher).
  - Does not touch any file unrelated to the §3.10 patterns (other files in the directory shall not be deleted).

### 5.3 Pipeline Orchestration

- [ ] In `src/nflpredictor/evaluate/pipeline.py`, implement `run_evaluation_build(raw_dir, processed_dir, *, repo_dir=None) -> None` executing the §5.1 stages from the spec:
  1. Verify upstream outputs (`verify_phase2_outputs`, `verify_phase3_outputs`, `verify_phase4_outputs`).
  2. Load `evaluation_config.yaml` via `load_evaluation_config`.
  3. Load features (`flat` + `pos`), vocab, splits; verify label parity (`assert_label_parity`).
  4. Enumerate combinations from the Phase 4 manifest; for each, load the prediction parquet and validate coverage (`validate_prediction_coverage`).
  5. Build per-combination joined frames (`join_predictions_with_labels`).
  6. Build the headline metrics structure (`build_headline_metrics`).
  7. For each enabled breakdown dimension, build the breakdown table (`build_breakdown_table`).
  8. Render all enabled plots (`render_all_plots`).
  9. Ensure output directories exist (`ensure_evaluation_dirs`); run `cleanup_disabled_outputs`.
  10. Write `metrics_headline.json`, every enabled breakdown parquet, every PNG.
  11. Compute output SHAs against the on-disk artifacts.
  12. Construct and write `evaluation_manifest.json` **last** (`EV-MAN-04`).
- [ ] Compute `evaluation_config_sha256` from the raw `evaluation_config.yaml` bytes (`EV-MAN-06`).
- [ ] Compute `phase2_source_sha256`, `phase3_source_sha256`, `phase4_source_sha256` from the on-disk artifacts; record under their repo-relative paths.
- [ ] Record `matplotlib.__version__`, `numpy.__version__`, `pyarrow.__version__` in the manifest (`EV-MAN-01`).
- [ ] Stdout logging of the per-combination headline-metric matrix (`EV-NF-08`): one line per `(combination, slice)` cell with the configured headline metric. Aligned columns for scannability.
- [ ] Update `src/nflpredictor/evaluate/__main__.py` to call `run_evaluation_build` with default paths and return exit 1 on any exception (`EV-NF-07`).

### 5.4 Smoke Run

- [ ] If `Data/processed/predictions/` is non-empty: run `python -m nflpredictor.evaluate` against the real outputs. Verify:
  - `Data/processed/evaluation/metrics_headline.json` parses cleanly and contains every combination from `training_manifest.json`.
  - Every S1 combination has both `val` and `test` entries; every S3 combination has `fold_0` through `fold_<n-1>` and `pooled` entries.
  - All five breakdown parquets exist (config defaults all on).
  - Plot files exist under `Data/processed/evaluation/plots/` matching the expected count (`~12 × 2 slices × 2 standard plots + 12 × 2 × 1 by_week + 3 ladder_summary` per the §4.5 calculation).
  - `Data/processed/evaluation/evaluation_manifest.json` carries every required key from `EV-MAN-01`; `output_sha256` enumerates every on-disk file.
  - Per-combination headline values on stdout match `metrics_headline.json`.
  - Wall-clock under `EV-NF-05`'s 2-minute limit.
- [ ] If `Data/processed/predictions/` is empty (dev machine): document that the smoke run is deferred to the CUDA machine. The plan checkbox remains unchecked until the smoke run completes on a machine with Phase 4 outputs.

### 5.5 Tests

- [ ] `tests/test_evaluate_manifest.py`:
  - Round-trip `build_evaluation_manifest` over a synthetic input and assert all `EV-MAN-01` keys present.
  - Assert `build_evaluation_summary` returns the exact shape per `EV-MAN-02` for both S1 and S3 examples.
- [ ] `tests/test_evaluate_outputs.py` (extension): `cleanup_disabled_outputs` deletes the right stale files (toggle `breakdowns.by_team: false` from a populated evaluation dir; assert the parquet disappears, other files survive); analogous for a plot family.

**Definition of done:** A real evaluation-build run (when Phase 4 outputs are present) emits the headline JSON, every breakdown parquet, every PNG, and a manifest under `Data/processed/evaluation/`; per-combination metric summary is scannable on stdout; the build runs end-to-end without exceptions in under 2 minutes.

---

## Phase 6: Determinism Hardening, Integration Tests, and Documentation

Locks the evaluation build against silent drift, ships the synthetic fixture and the formal byte-determinism tests, pins the real-data identity test and the cross-phase agreement test, and updates the project docs.

**Satisfies:** `EV-NF-01`, `EV-NF-02`, `EV-NF-04`, `EV-NF-09`, `EV-TEST-06`, `EV-TEST-07`, `EV-TEST-08`, `EV-TEST-09`, closure on any straggling `EV-TEST-*` IDs.

### 6.1 Determinism Audit

- [ ] Review every `sort` and `sorted()` call in `src/nflpredictor/evaluate/`. Each must specify a fully-tie-breaking sort key.
- [ ] Confirm every `json.dump` invocation matches its required option set: `sort_keys=True`, `indent=2`, trailing newline.
- [ ] Confirm every parquet writer uses `compression="snappy"`, `row_group_size=1024`, and `pyarrow`.
- [ ] Confirm no `set()` iteration, dict iteration, or `glob()` order affects output. Where a set or dict is iterated, the output passes through `sorted(...)` at the boundary.
- [ ] Confirm `matplotlib.use("Agg")` runs before any `pyplot` import; verify the metadata-stripping idiom in `save_figure` works against the pinned wheel.
- [ ] Confirm no network access at runtime (`EV-NF-11`) — grep for `urllib`, `requests`, `http`, `download` in the package. matplotlib font caching is acceptable.
- [ ] Confirm all logging goes to `sys.stderr` for diagnostics and `sys.stdout` only for the headline-metric matrix (`EV-NF-08`).

### 6.2 Synthetic Fixture

- [ ] Create `tests/fixtures/evaluate/raw_phase2/features_flat_2024.parquet` — a sliced Phase 2 feature matrix (~30 games covering enough weeks to populate every breakdown universe — at least one game on every surface and roof category, multiple teams, weeks spanning train/val/test).
- [ ] Create `tests/fixtures/evaluate/raw_phase2/features_pos_2024.parquet` — matching pos-shape slice.
- [ ] Create `tests/fixtures/evaluate/raw_phase2/feature_vocab.json` + matching `feature_manifest.json` with `output_sha256` entries equal to the sliced files' SHAs.
- [ ] Create `tests/fixtures/evaluate/raw_phase3/splits_2024.json` + matching `splits_manifest.json` (regenerate from the sliced Phase 2 fixture via Phase 3's `run_split_build`).
- [ ] Generate the Phase 4 fixture inputs: either reuse `tests/fixtures/train/expected/predictions/*.parquet` if their universe matches, or run Phase 4's training build against the Phase 5 fixtures' Phase 2/3 stand-ins (likely the cleaner path — produces 12 internally-consistent prediction parquets + a matching `training_manifest.json`). Save under `tests/fixtures/evaluate/raw_phase4/`.
- [ ] Create `tests/fixtures/evaluate/raw/evaluation_config.yaml` — full v1 default or a trimmed variant (e.g., drop one breakdown to exercise the toggle path).
- [ ] Run the evaluation build against the fixture once; manually inspect outputs; check them in under `tests/fixtures/evaluate/expected/` (headline JSON + breakdown parquets + `evaluation_manifest.json` with `build_timestamp_utc` blanked). Do not check in expected PNGs — assert existence only in tests, per `EV-NF-02`.
- [ ] Add `tests/fixtures/evaluate/_regenerate.py` (companion to earlier phases' regenerators) that rebuilds the fixture on demand. Doc-string notes the fixture is pinned to specific PyTorch + matplotlib wheels.

### 6.3 Integration Test

- [ ] `tests/test_evaluate_integration.py` (`EV-TEST-06`):
  - Runs `run_evaluation_build` against the synthetic fixture into a temp directory.
  - Asserts `metrics_headline.json` matches the checked-in expected file byte-for-byte.
  - Asserts every breakdown parquet matches its checked-in expected file byte-for-byte (`pathlib.Path.read_bytes()` equality).
  - Loads both evaluation manifests, blanks `build_timestamp_utc`, and asserts byte-equality of the remainder.
  - Asserts every expected PNG path exists in the temp directory (byte-equality of PNGs is *not* asserted here — see `EV-NF-02`).

### 6.4 Determinism Test

- [ ] `tests/test_evaluate_determinism.py` (`EV-TEST-07`):
  - Runs `run_evaluation_build` twice in succession against identical fixture inputs into two temp directories.
  - Asserts byte-equality of `metrics_headline.json`, every breakdown parquet, and the manifest (excluding `build_timestamp_utc`) across the two runs.
  - Asserts byte-equality of every PNG across the two runs (within the same pinned matplotlib wheel on the same machine — `EV-NF-02`). If this PNG assertion proves CI-flaky, relax to pixel-equality after warm-up per the relaxation noted in Phase 4.4.

### 6.5 Cross-Phase Agreement Test

- [ ] `tests/test_evaluate_metrics.py` (extension, `EV-TEST-08`):
  - Against the synthetic Phase 4 fixture: load `training_manifest.json`'s `training_summaries`; for every S1 combination, compute Phase 5's headline MAE for the `val` slice and assert it equals `training_summaries.<combo>.val_mae` to float64 precision.
  - For every S3 combination: compute Phase 5's per-fold headline MAE and assert each equals `training_summaries.<combo>.per_fold[i].val_mae` to float64 precision.
  - This test enforces that Phase 5's `EV-MET-01` and Phase 4's `TR-MAN-04` formulas remain numerically identical.

### 6.6 Pinned Real-Data Identities

- [ ] `tests/test_evaluate_pipeline_run.py` (`EV-TEST-09`):
  - Runs `run_evaluation_build` against the actual Phase 2 + Phase 3 + Phase 4 outputs into a temp directory.
  - Skips with a clear message when `Data/processed/predictions/` is empty (mirror the pattern from `test_train_pipeline_run.py`).
  - Asserts (a) every combination in `training_manifest.json` appears in `metrics_headline.json`; (b) every S1 combination has both `val` and `test` entries; (c) every S3 combination has `fold_0` through `fold_<n-1>` and `pooled` entries; (d) every enabled breakdown parquet exists and covers every combination; (e) the manifest's `output_sha256` exactly enumerates the files on disk.

### 6.7 Performance Sanity Check

- [ ] Run `time python -m nflpredictor.evaluate` against the real Phase 4 outputs (on a machine where they exist). Confirm wall-clock is comfortably under `EV-NF-05`'s 2-minute limit. Record the time in a comment.

### 6.8 Documentation Updates

- [ ] Update `CLAUDE.md`'s "Project status" section to note that Phase 5 implementation is complete, the evaluation build is invoked via `python -m nflpredictor.evaluate`, the outputs land in `Data/processed/evaluation/` (headline JSON + five breakdown parquets + ~40 PNGs + manifest), and the determinism contract is split (byte-deterministic metric tables; PNG byte-identity within the pinned matplotlib wheel only).
- [ ] Update `Docs/Idea.md`'s Phase 5 section status block from "Specification complete (YYYY-MM-DD)" to "Implementation complete (YYYY-MM-DD)" with the real-data run summary (per-combination headline MAE for the v1 ladder).
- [ ] Optionally update `README.md` to reference the new build step in the project's run sequence.
- [ ] Optionally update `Docs/Runbook.md` to add a final "Phase 5: evaluation" step covering `python -m nflpredictor.evaluate`.

**Definition of done:** Every `EV-TEST-*` requirement in the spec has a corresponding passing test; re-running the evaluation build produces byte-identical metric outputs and (within the pinned matplotlib wheel) byte-identical PNGs; documentation is updated.

---

## After Phase 6

Phase 5 is complete when:

- All six phases above are checked off.
- `pytest` passes with the venv activated.
- `python -m nflpredictor.evaluate` produces the headline JSON, every enabled breakdown parquet, every PNG, and the evaluation manifest in `Data/processed/evaluation/`.
- Re-running the evaluation build produces byte-identical metric outputs (modulo the manifest timestamp), and byte-identical PNGs within the pinned matplotlib wheel.
- Phase 5's S1.val headline MAE numerically matches Phase 4's `training_summaries.<combo>.val_mae` for every combination.
- The `CLAUDE.md` and `Idea.md` doc updates are in place.

At that point, Phase 7 (Error Analysis & Iteration) can begin scoping against the *actual* Phase 5 outputs on disk — per-combination headline values, per-slice test MAE, per-week and per-team residual patterns — rather than imagined data, which is the same discipline that paced Phases 1–4 against their successors.

---

## References

- [Spec-Phase5-Evaluation.md](./Spec-Phase5-Evaluation.md) — Formal specification for Phase 5.
- [Spec-Phase4-BaselineLadder.md](./Spec-Phase4-BaselineLadder.md) — Phase 4 specification; Phase 5 reads its outputs.
- [Spec-Phase3-Splits.md](./Spec-Phase3-Splits.md) — Phase 3 specification; Phase 5 reads its outputs.
- [Spec-Phase2-FeatureEngineering.md](./Spec-Phase2-FeatureEngineering.md) — Phase 2 specification; Phase 5 reads its outputs.
- [Plan-Phase4-BaselineLadder.md](./Plan-Phase4-BaselineLadder.md) — Stylistic precedent for this document.
- [Plan-Phase3-Splits.md](./Plan-Phase3-Splits.md) — Stylistic precedent for this document.
- [Plan-Phase2-FeatureEngineering.md](./Plan-Phase2-FeatureEngineering.md) — Stylistic precedent for this document.
- [Plan-Phase1-DataBuild.md](./Plan-Phase1-DataBuild.md) — Stylistic precedent for this document.
- [Idea.md](./Idea.md) — Source idea document; §"Phase 5: Evaluation" lists the open questions the spec resolved.
- [CLAUDE.md](../CLAUDE.md) — Repository-level notes; will be updated during Phase 6.8.
