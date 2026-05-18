# Phase 3: Splits Implementation Plan

This document defines the phased implementation plan for the Splits phase of the NFL Predictor project, based on [Spec-Phase3-Splits.md](./Spec-Phase3-Splits.md). Each phase builds on the previous one and contains checkbox-tracked work items. Requirement IDs (`SP-*`) reference the corresponding entries in the specification.

This is a single-developer learning project. Phases are sized for one person to complete in sittings of an hour or two, with tests landing in the same phase as the code they cover. Phase 3 is materially smaller than Phases 1 and 2 — the build's logic is bucketing GameIds by week and writing a single JSON — so the plan is shorter accordingly.

---

## Progress Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Scaffolding, Config Loading, and Phase 2 Hash Gate | Complete |
| 2 | S1 Single-Fold Partition | Complete |
| 3 | S3 Expanding-Window CV | Complete |
| 4 | Output Emission and Pipeline Orchestration | Complete |
| 5 | Determinism Hardening and Integration Tests | Complete |

---

## Current State

- Phase 1 and Phase 2 are complete; `Data/processed/{madden_2024.csv, box_scores_2024.csv, player_id_mapping.csv, build_manifest.json, features_flat_2024.parquet, features_pos_2024.parquet, feature_vocab.json, feature_manifest.json}` are produced by their respective entry points.
- The Phase 3 specification is final at `Docs/Spec-Phase3-Splits.md` (all open questions resolved in-spec).
- No `src/nflpredictor/splits/` module exists yet.
- No `Data/raw/splits_config.yaml` exists yet; it must be created as part of Phase 1 of this plan.
- No new runtime dependencies needed: `pyyaml` and `pyarrow` are already declared from Phase 2.

---

## Guiding Principles

1. **Work inside the virtual environment.** All `pip`, `python`, and `pytest` commands run with the `.venv/` venv activated.
2. **Tests accompany every phase.** No phase is complete until the tests it introduces pass.
3. **Vertical slices where practical.** Each phase delivers something runnable or testable on its own.
4. **One responsibility per module.** Match the proposed `src/nflpredictor/splits/` layout from §5 of the spec; do not pile everything into a single file.
5. **Determinism is non-negotiable.** Every sort, every iteration, every JSON write option is deliberate. Phase 5 exists specifically to validate this.
6. **The specification is source of truth.** When the plan and the spec disagree, fix the plan or fix the spec — do not silently improvise.

---

## Proposed Repository Layout (additions)

Phase 3 adds the following files to the existing repo. Files outside this list are untouched (Phase 1's `src/nflpredictor/databuild/` and Phase 2's `src/nflpredictor/features/` stay read-only from Phase 3's perspective).

```
NFLPredictor/
├── Data/
│   ├── raw/
│   │   └── splits_config.yaml                # NEW — v1 default config
│   └── processed/
│       ├── splits_2024.json                  # NEW — split-assignment artifact
│       └── splits_manifest.json              # NEW — split-build provenance
├── src/
│   └── nflpredictor/
│       └── splits/                           # NEW package
│           ├── __init__.py
│           ├── __main__.py                   # entry: python -m nflpredictor.splits
│           ├── pipeline.py                   # top-level orchestration
│           ├── config.py                     # YAML load + validation
│           ├── s1.py                         # single-fold partition
│           ├── s3.py                         # expanding-window CV folds
│           ├── outputs.py                    # JSON writers
│           └── manifest.py                   # splits_manifest.json construction
└── tests/
    ├── test_splits_config.py
    ├── test_splits_pipeline_input.py
    ├── test_splits_s1.py
    ├── test_splits_s3.py
    ├── test_splits_outputs.py
    ├── test_splits_manifest.py
    ├── test_splits_determinism.py
    ├── test_splits_integration.py
    ├── test_splits_pipeline_run.py
    └── fixtures/
        └── splits/                           # NEW subdir
            ├── raw_phase2/
            │   ├── features_flat_2024.parquet     # sliced
            │   └── feature_manifest.json          # regenerated to match slice
            ├── raw/
            │   └── splits_config.yaml             # mirrors v1 default
            └── expected/
                ├── splits_2024.json
                └── splits_manifest.json
```

Tests use a `test_splits_*` prefix to keep them visually separate from Phase 1's `test_*.py` files and Phase 2's `test_features_*.py` files. All existing tests stay green throughout.

---

## Phase 1: Scaffolding, Config Loading, and Phase 2 Hash Gate

Establishes the package skeleton, ships the v1 reference `splits_config.yaml`, and enforces the Phase 2 input contract. Nothing functional yet beyond input validation — but every later phase trusts these invariants.

**Satisfies:** `SP-IN-01` through `SP-IN-06`, `SP-CFG-01` through `SP-CFG-07`, `SP-TEST-01`, `SP-TEST-07`.

### 1.1 Package Skeleton

- [ ] Create `src/nflpredictor/splits/__init__.py` (empty).
- [ ] Create `src/nflpredictor/splits/__main__.py` with a stub `main()` that prints `"split build not implemented yet"` and exits `0`.
- [ ] Verify the entry point: `python -m nflpredictor.splits` prints the stub message.
- [ ] Add a smoke test `tests/test_splits_smoke.py` that imports `nflpredictor.splits` and asserts the import succeeds. Run `pytest` to confirm.

### 1.2 Default Splits Config

- [ ] Create `Data/raw/splits_config.yaml` with the v1 contents shown verbatim in §4.1 of the spec (boundaries 1–12 / 13–15 / 16–18; strategies `[S1, S3]`; `s3.k_start: 6`; `splits_version: "v1"`).
- [ ] Confirm the file parses cleanly with `python -c "import yaml; print(yaml.safe_load(open('Data/raw/splits_config.yaml')))"`.

### 1.3 Config Loader and Validator

- [ ] In `src/nflpredictor/splits/config.py`, define a `SplitsConfig` frozen dataclass with fields mirroring the YAML schema: `splits_version: str`, `strategies: tuple[str, ...]`, `train_weeks: tuple[int, int]`, `val_weeks: tuple[int, int]`, `test_weeks: tuple[int, int]`, `s3_k_start: int | None` (None when S3 not enabled).
- [ ] Implement `load_splits_config(path: pathlib.Path) -> SplitsConfig` that:
  - Reads via `yaml.safe_load` (per `SP-SEC-04`).
  - Rejects unknown top-level keys (`SP-CFG-01`).
  - Validates `strategies` is a non-empty list of `{S1, S3}` with no duplicates (`SP-CFG-02`).
  - Validates each week range is a `[start, end]` closed range with `1 ≤ start ≤ end ≤ 18` (`SP-CFG-03`).
  - Validates the three ranges are contiguous, non-overlapping, and union to a subset of `[1, 18]` (`SP-CFG-03`).
  - Validates strict temporal ordering: `train_weeks[1] < val_weeks[0]` and `val_weeks[1] < test_weeks[0]` (`SP-CFG-04`).
  - When `S3 ∈ strategies`, validates `s3.k_start` exists and satisfies `1 ≤ k_start < val_weeks[1]` so at least one fold is constructible (`SP-CFG-05`).
  - Validates `splits_version` is a non-empty string (`SP-CFG-06`).

### 1.4 Phase 2 Source-Hash Gate and Week Validation

- [ ] In `src/nflpredictor/splits/pipeline.py`, define module-level constants `PHASE2_FEATURES_FLAT_BASENAME` and `PHASE2_MANIFEST_BASENAME`.
- [ ] Implement `verify_phase2_outputs(processed_dir: pathlib.Path) -> dict` that:
  - Loads `feature_manifest.json`.
  - Recomputes SHA-256 of `features_flat_2024.parquet` on disk.
  - Compares against the manifest's `output_sha256` map.
  - Raises `Phase2OutputMismatchError` naming the divergent file on mismatch (`SP-IN-04`).
  - Returns the parsed manifest dict for downstream provenance.
- [ ] Reuse `compute_sha256` from `src/nflpredictor/databuild/manifest.py` (already imported by Phase 2's pipeline; same dependency direction applies — Phase 3 consumes Phase 2's output).
- [ ] Implement `load_game_universe(parquet_path: pathlib.Path) -> pandas.DataFrame` that reads `GameId` and `week` columns via `pyarrow.parquet`. Validate:
  - `week` column is present (`SP-IN-05`).
  - Every `week` value is an integer in `[1, 18]` — raise on out-of-range or non-integer values.
  - `GameId` values are unique.

### 1.5 Tests

- [ ] `tests/test_splits_config.py` (`SP-TEST-01`): cover (a) shipped v1 load (must accept); (b) overlapping `train_weeks` and `val_weeks` (must reject); (c) non-contiguous gap between `train_weeks[1]` and `val_weeks[0]` (must reject); (d) `s3.k_start ≥ val_weeks[1]` (must reject); (e) unknown top-level key (must reject); (f) `strategies = []` (must reject); (g) duplicate entries in `strategies` (must reject); plus reversed-range and out-of-`[1,18]` boundary rejections.
- [ ] `tests/test_splits_pipeline_input.py` (`SP-TEST-07`): cover (a) happy path against real Phase 2 outputs; (b) tampered parquet → fail-fast with clear error; (c) missing manifest → fail-fast; (d) missing parquet → fail-fast; (e) `week` column absent in a synthetic parquet → fail-fast; (f) `week` value out of `[1, 18]` → fail-fast.

**Definition of done:** Config loading rejects every malformed case enumerated in §3.2; the source-hash gate blocks runs against tampered Phase 2 outputs; the week validator rejects out-of-range or missing weeks.

---

## Phase 2: S1 Single-Fold Partition

Implements the deterministic single-fold partition logic. Pure, side-effect-free code.

**Satisfies:** `SP-S1-01`, `SP-S1-02`, `SP-S1-03`, `SP-TEST-02`.

### 2.1 S1 Partition Function

- [ ] In `src/nflpredictor/splits/s1.py`, implement `assign_s1(universe: pandas.DataFrame, config: SplitsConfig) -> dict[str, list[str]]` that:
  - Returns a dict with keys `"train"`, `"val"`, `"test"`.
  - Each value is a list of `GameId` strings sorted lexicographically ascending (`SP-S1-03`).
  - Every game in the universe appears in exactly one of the three lists, based on its `week` value (`SP-S1-01`).
  - Raises `ValueError` if any game's week falls outside the union of `train_weeks ∪ val_weeks ∪ test_weeks` (defense-in-depth — config validation should already prevent this, but the partitioner re-checks).
- [ ] The function is pure: no I/O, no globals, no mutation of the input DataFrame.

### 2.2 Tests

- [ ] `tests/test_splits_s1.py` (`SP-TEST-02`): build a synthetic `(GameId, week)` DataFrame covering Weeks 1–18 with 2–4 games per week and assert:
  - Every game appears in exactly one role list (`SP-S1-02`).
  - The three role lists are disjoint and union to the full universe.
  - Week assignments match the v1 boundary rules: every `train` GameId has `week ∈ [1, 12]`, every `val` has `week ∈ [13, 15]`, every `test` has `week ∈ [16, 18]`.
  - Each list is sorted lexicographically (`SP-S1-03`).
  - A game with an out-of-config week (e.g., `week = 19` synthetically injected) raises.
  - The function is idempotent: two calls on identical inputs produce identical outputs.

**Definition of done:** S1 partition logic is testable in isolation, deterministic, and protects against malformed inputs.

---

## Phase 3: S3 Expanding-Window CV

Implements the expanding-window cross-validation fold sequence. The S3 test slice is reused from S1; only the `(train, val)` pairs are new.

**Satisfies:** `SP-S3-01` through `SP-S3-06`, `SP-TEST-03`.

### 3.1 S3 Fold Constructor

- [ ] In `src/nflpredictor/splits/s3.py`, define a `Fold` frozen dataclass with fields `fold_index: int`, `k: int`, `train: tuple[str, ...]`, `val: tuple[str, ...]`.
- [ ] Implement `build_s3(universe: pandas.DataFrame, config: SplitsConfig, s1_test: list[str]) -> dict` that:
  - Returns `{"test": [...], "folds": [Fold, ...]}` matching the §4.2 structure.
  - `test` is identical to S1's `test` list (`SP-S3-01`) — passed in to avoid re-deriving.
  - Folds run `k = k_start, k_start + 1, …` until `k + 1 > val_weeks[1]` (`SP-S3-02`, `SP-S3-03`).
  - For each fold: `train` = sorted GameIds whose `week ∈ [1, k]`; `val` = sorted GameIds whose `week == k + 1` (`SP-S3-04`).
  - `fold_index` starts at 0 and increments by 1 (`SP-S3-06`).

### 3.2 Tests

- [ ] `tests/test_splits_s3.py` (`SP-TEST-03`): reuse the same synthetic `(GameId, week)` fixture from Phase 2 and assert:
  - The S3 test list equals S1's test list element-for-element (`SP-S3-01`).
  - For the v1 defaults (`k_start=6`, `val_weeks=[13,15]`), `fold_count == 9` and folds run `k ∈ {6, …, 14}` with val weeks `{7, …, 15}` (`SP-S3-03`).
  - For every fold: `train ∩ val = ∅` and `(train ∪ val) ∩ test = ∅` (`SP-S3-05`).
  - For every fold: each fold's val games are all in exactly one week (`val[i].week == k + 1`).
  - For every fold: train and val lists are sorted lexicographically (`SP-S3-04`).
  - Folds are listed in ascending `fold_index` order (`SP-S3-06`).
  - A different `k_start` (e.g., `k_start = 4`) produces the expected larger fold count.

**Definition of done:** S3 fold construction is testable in isolation, deterministic, and shares the test slice with S1.

---

## Phase 4: Output Emission and Pipeline Orchestration

Writes the two JSON artifacts to `Data/processed/` and wires the pipeline together. This is where the build produces a real result for the first time.

**Satisfies:** `SP-OUT-01` through `SP-OUT-04`, `SP-MAN-01` through `SP-MAN-06`, `SP-NF-02`, `SP-NF-05`, `SP-NF-06`, `SP-NF-07`.

### 4.1 Artifact JSON Writer

- [ ] In `src/nflpredictor/splits/outputs.py`, implement `build_splits_artifact(config, s1, s3) -> dict` that constructs the §4.2 structure with the pinned key order: `splits_version`, then each strategy in the order declared in `config.strategies`. Within S1: `train`, `val`, `test`. Within S3: `test`, `folds`. Per fold: `fold_index`, `k`, `train`, `val`.
- [ ] Implement `write_splits_artifact(artifact: dict, path: pathlib.Path) -> None` using `json.dump(artifact, f, indent=2)` with a trailing `"\n"` (`SP-OUT-02`). Use `sort_keys=False` and rely on the deterministic construction in `build_splits_artifact` for byte-stable output.

### 4.2 Manifest Construction and Writer

- [ ] In `src/nflpredictor/splits/manifest.py`, reuse `compute_sha256` and `try_get_git_commit` from `databuild.manifest` (direct import — chains with Phase 2's reuse of the same helpers).
- [ ] Implement `build_strategy_summaries(s1, s3) -> dict` returning the per-strategy summary structure from `SP-MAN-02`:
  - For S1: `{"train_n": ..., "val_n": ..., "test_n": ...}`.
  - For S3: `{"test_n": ..., "fold_count": ..., "folds": [{"fold_index": ..., "k": ..., "train_n": ..., "val_n": ...}, ...]}`.
- [ ] Implement `build_splits_manifest(*, config, phase2_source_sha256, output_sha256, phase2_manifest_git_commit, strategy_summaries, splits_config_sha256) -> dict` constructing the manifest dict per `SP-MAN-01` with all nine required keys.
- [ ] Implement `write_splits_manifest(manifest: dict, path: pathlib.Path) -> None` using `json.dump(..., sort_keys=True, indent=2)` with a trailing newline (`SP-MAN-04`).

### 4.3 Pipeline Orchestration

- [ ] In `src/nflpredictor/splits/pipeline.py`, implement `run_split_build(raw_dir, processed_dir, *, repo_dir=None) -> None` executing the 7-step pipeline from spec §5.1:
  1. Load and validate `splits_config.yaml`.
  2. Verify Phase 2 outputs (`SP-IN-04`).
  3. Load the `(GameId → week)` universe.
  4. Compute S1 partition (only if `S1 ∈ strategies`).
  5. Compute S3 folds (only if `S3 ∈ strategies`); reuse S1's test list if S1 also enabled, otherwise derive it from the universe and config.
  6. Write `splits_2024.json`.
  7. Write `splits_manifest.json` last (so output SHAs can include the artifact — `SP-MAN-05`).
- [ ] Compute `splits_config_sha256` from the raw `splits_config.yaml` bytes (`SP-MAN-06`).
- [ ] Compute `phase2_source_sha256` from the on-disk `features_flat_2024.parquet`; record under its repo-relative path (`Data/processed/features_flat_2024.parquet`) — mirrors Phase 2's pattern.
- [ ] Compute `output_sha256` for `splits_2024.json` after writing it.
- [ ] Stderr logging of per-strategy counts (`SP-NF-06`): S1 train/val/test sizes, plus S3 fold count and each fold's train/val sizes.
- [ ] Update `src/nflpredictor/splits/__main__.py` to call `run_split_build` with default paths and return exit 1 on any exception (`SP-NF-05`).

### 4.4 Smoke Run

- [ ] Run `python -m nflpredictor.splits` against the real Phase 2 outputs. Verify:
  - `Data/processed/splits_2024.json` exists, parses as valid JSON, and matches the §4.2 structure.
  - S1's three lists union to exactly 272 GameIds with no overlap.
  - S1 train ≈ 180 games, val ≈ 45, test ≈ 44 (exact counts depend on the 2024 NFL schedule's per-week distribution).
  - S3's `fold_count == 9`; S3's `test` list equals S1's `test` list.
  - `Data/processed/splits_manifest.json` has all nine required keys; `output_sha256` includes `splits_2024.json`.
  - Re-running the build produces byte-identical artifacts (modulo timestamp). Capture two runs' SHAs and compare manually for the smoke check.

**Definition of done:** A real split-build run emits two files in `Data/processed/`; per-strategy counts in the manifest are sane; the pipeline runs end-to-end without exceptions.

---

## Phase 5: Determinism Hardening and Integration Tests

Locks the split build against silent drift. The protection layer.

**Satisfies:** `SP-NF-01`, `SP-NF-03`, `SP-NF-04`, `SP-TEST-04`, `SP-TEST-05`, `SP-TEST-06`, `SP-TEST-08`, closure on any straggling `SP-TEST-*` IDs.

### 5.1 Determinism Audit

- [ ] Review every `sort` and `sorted()` call in `splits/`. Each must specify a fully-tie-breaking sort key. Document inline with one-line comments referencing the satisfied spec ID.
- [ ] Confirm every `json.dump` invocation matches its required option set: artifact uses `indent=2` + trailing newline (no `sort_keys`, since key order is intentional); manifest uses `sort_keys=True`, `indent=2`, trailing newline.
- [ ] Confirm no `set()` iteration or dict iteration affects output order. Where a set or dict is iterated, the output passes through `sorted(...)` at the boundary.
- [ ] Confirm all logging goes to `sys.stderr`; no `print()` writes to stdout in the splits package.
- [ ] Confirm `pyarrow.parquet` reads use deterministic column selection (`columns=["GameId", "week"]`).

### 5.2 Synthetic Fixture

- [ ] Create `tests/fixtures/splits/raw_phase2/features_flat_2024.parquet` — a sliced Phase 2 feature matrix small enough to be checked in (a few rows per week × ~6 weeks of coverage, totaling ~30 games). Only the `GameId` and `week` columns matter for Phase 3.
- [ ] Generate a matching `tests/fixtures/splits/raw_phase2/feature_manifest.json` whose `output_sha256` entry for the sliced parquet equals the actual SHA of the sliced file. (Other manifest keys can be minimal stubs.)
- [ ] Create `tests/fixtures/splits/raw/splits_config.yaml` mirroring the v1 default.
- [ ] Run the split build against the fixture once; manually inspect outputs; check them in as `tests/fixtures/splits/expected/{splits_2024.json, splits_manifest.json}` (with `build_timestamp_utc` blanked in the manifest).
- [ ] Add `tests/fixtures/splits/_regenerate.py` (companion to Phase 2's `tests/fixtures/features/_regenerate.py`) that rebuilds the fixture on demand. Doc-string notes the fixture is pinned to the Phase 2 parquet bytes.

### 5.3 Integration Test

- [ ] `tests/test_splits_integration.py` (`SP-TEST-04`):
  - Runs `run_split_build` against the synthetic fixture into a temp directory.
  - Asserts `splits_2024.json` matches the checked-in expected file byte-for-byte.
  - Loads both manifests, blanks `build_timestamp_utc`, and asserts byte-equality of the remainder.

### 5.4 Determinism Test

- [ ] `tests/test_splits_determinism.py` (`SP-TEST-05`):
  - Runs `run_split_build` twice in succession against identical fixture inputs.
  - Asserts byte-equality of `splits_2024.json` across the two runs.
  - For `splits_manifest.json`, blanks `build_timestamp_utc` in both and asserts byte-equality of the remainder.
- [ ] Companion test in `tests/test_splits_pipeline_run.py` exercises the same property on the real 272-game Phase 2 dataset.

### 5.5 Pinned Real-Data Identities

- [ ] `tests/test_splits_pipeline_run.py` (`SP-TEST-06`):
  - Runs `run_split_build` against the actual Phase 2 outputs.
  - Asserts S1's three lists union to exactly 272 GameIds with no duplicates.
  - Asserts S1 train, val, test counts match expected per-week game counts on real 2024 data (pin the exact numbers once first run reveals them).
  - Asserts S3's `fold_count == 9`.
  - Asserts S3's `test` list equals S1's `test` list element-for-element.
- [ ] `tests/test_splits_outputs.py` (`SP-TEST-08`): assert every GameId list in the artifact (S1 train/val/test, S3 test, every fold's train/val) is sorted lexicographically ascending.

### 5.6 Performance Sanity Check

- [ ] Run `time python -m nflpredictor.splits` against the real Phase 2 outputs. Confirm wall-clock is comfortably under `SP-NF-03`'s 5-second limit. Record the time in the plan.

### 5.7 Documentation Updates

- [ ] Update `CLAUDE.md`'s "Project status" section to note that Phase 3 implementation is complete, the split build is invoked via `python -m nflpredictor.splits`, and the two output artifacts in `Data/processed/`.
- [ ] Add a "Status: Implementation complete (YYYY-MM-DD)" header note to `Docs/Idea.md`'s Phase 3 section, mirroring the format used for Phases 1 and 2. Include the real-data run summary (S1 counts, S3 fold count).

**Definition of done:** Every `SP-TEST-*` requirement in the spec has a corresponding passing test; re-running the split build produces byte-identical outputs; documentation is updated.

---

## After Phase 5

Phase 3 is complete when:

- All five phases above are checked off.
- `pytest` passes with the venv activated.
- `python -m nflpredictor.splits` produces the two expected split artifacts in `Data/processed/`.
- Re-running the split build produces byte-identical JSON (modulo the timestamp).
- The `CLAUDE.md` and `Idea.md` doc updates are in place.

At that point, the project is ready to begin scoping Phase 4 (Baseline & Model Ladder), which can now reference the *actual* split assignments (real per-fold game counts, real S3 fold structure) rather than imagined data — which is the same reason Phases 1 and 2 were specified before their successors.

---

## References

- [Spec-Phase3-Splits.md](./Spec-Phase3-Splits.md) — Formal specification for Phase 3.
- [Spec-Phase2-FeatureEngineering.md](./Spec-Phase2-FeatureEngineering.md) — Phase 2 specification; Phase 3 reads its outputs.
- [Plan-Phase2-FeatureEngineering.md](./Plan-Phase2-FeatureEngineering.md) — Phase 2 implementation plan; stylistic precedent for this document.
- [Plan-Phase1-DataBuild.md](./Plan-Phase1-DataBuild.md) — Phase 1 implementation plan; stylistic precedent for this document.
- [Idea.md](./Idea.md) — Source idea document; §"Phase 3: Splits" lists the open questions the spec resolved.
- [CLAUDE.md](../CLAUDE.md) — Repository-level notes; will be updated during Phase 5.7.
