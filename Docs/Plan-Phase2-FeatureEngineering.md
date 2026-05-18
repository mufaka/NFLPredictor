# Phase 2: Feature Engineering & Model Inputs Implementation Plan

This document defines the phased implementation plan for the Feature Engineering & Model Inputs phase of the NFL Predictor project, based on [Spec-Phase2-FeatureEngineering.md](./Spec-Phase2-FeatureEngineering.md). Each phase builds on the previous one and contains checkbox-tracked work items. Requirement IDs (`FE-*`) reference the corresponding entries in the specification.

This is a single-developer learning project. Phases are sized for one person to complete in sittings of an hour or two, with tests landing in the same phase as the code they cover. No CI, no branch-per-phase ceremony — just sequencing.

---

## Progress Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Scaffolding, Dependencies, and Default Config | Complete |
| 2 | Config Loading and Phase 1 Source-Hash Gate | Complete |
| 3 | Vocabulary Builder and Integer Coding | Complete |
| 4 | Game-Level Features, Weather, and Officials | Complete |
| 5 | Position Taxonomy and Madden Column Resolution | Not started |
| 6 | Shape Assembly — B-flat and B-pos | Not started |
| 7 | Output Emission and Pipeline Orchestration | Not started |
| 8 | Determinism Hardening and Integration Tests | Not started |

---

## Current State

- Phase 1 is complete; `Data/processed/{madden_2024.csv, box_scores_2024.csv, player_id_mapping.csv, build_manifest.json}` are produced by `python -m nflpredictor.databuild`.
- The Phase 2 specification is final at `Docs/Spec-Phase2-FeatureEngineering.md` (all nine open questions resolved in-spec).
- No `src/nflpredictor/features/` module exists yet.
- No `Data/raw/feature_config.yaml` exists yet; it must be created as part of Phase 1 of this plan.
- `pyproject.toml` does not yet declare `pyarrow` or `pyyaml`; both must be added.

---

## Guiding Principles

1. **Work inside the virtual environment.** All `pip`, `python`, and `pytest` commands run with the `.venv/` venv activated.
2. **Tests accompany every phase.** No phase is complete until the tests it introduces pass.
3. **Vertical slices where practical.** Each phase delivers something runnable or testable on its own.
4. **One responsibility per module.** Match the proposed `src/nflpredictor/features/` layout from §5 of the spec; do not pile everything into a single file.
5. **Determinism is non-negotiable.** Every sort, every iteration, every parquet write option is deliberate. Phase 8 exists specifically to validate this.
6. **The specification is source of truth.** When the plan and the spec disagree, fix the plan or fix the spec — do not silently improvise.

---

## Proposed Repository Layout (additions)

Phase 2 adds the following files to the existing repo. Files outside this list are untouched (Phase 1's `src/nflpredictor/databuild/` stays read-only from Phase 2's perspective).

```
NFLPredictor/
├── pyproject.toml                         # add: pyarrow, pyyaml
├── Data/
│   ├── raw/
│   │   └── feature_config.yaml            # NEW — v1 default config
│   └── processed/
│       ├── features_flat_2024.parquet     # NEW — B-flat artifact
│       ├── features_pos_2024.parquet      # NEW — B-pos artifact
│       ├── feature_vocab.json             # NEW — categorical vocab sidecar
│       └── feature_manifest.json          # NEW — feature-build provenance
├── src/
│   └── nflpredictor/
│       └── features/                      # NEW package
│           ├── __init__.py
│           ├── __main__.py                # entry: python -m nflpredictor.features
│           ├── pipeline.py                # top-level orchestration
│           ├── config.py                  # YAML load + validation
│           ├── schedule.py                # 2024 NFL week calendar
│           ├── weather.py                 # weather string parser
│           ├── positions.py               # canonical taxonomy + bucket map
│           ├── vocab.py                   # vocabulary builder (sorted, deterministic)
│           ├── flat.py                    # B-flat assembly
│           ├── pos.py                     # B-pos assembly
│           ├── outputs.py                 # parquet + JSON writers
│           └── manifest.py                # feature_manifest.json construction
└── tests/
    ├── test_features_config.py
    ├── test_features_vocab.py
    ├── test_features_schedule.py
    ├── test_features_weather.py
    ├── test_features_positions.py
    ├── test_features_game_level.py
    ├── test_features_madden_resolution.py
    ├── test_features_flat.py
    ├── test_features_pos.py
    ├── test_features_outputs.py
    ├── test_features_manifest.py
    ├── test_features_determinism.py
    ├── test_features_integration.py
    ├── test_features_pinned_identities.py
    └── fixtures/
        └── features/                      # NEW subdir
            ├── tiny_feature_config.yaml
            ├── tiny_madden_phase1.csv     # Phase-1-output-shaped
            ├── tiny_box_scores_phase1.csv
            ├── tiny_build_manifest.json
            └── expected/
                ├── features_flat.parquet
                ├── features_pos.parquet
                ├── feature_vocab.json
                └── feature_manifest.json
```

Tests are named with a `test_features_*` prefix to keep them visually separate from Phase 1's `test_*.py` files. The Phase 1 tests stay green throughout.

---

## Phase 1: Scaffolding, Dependencies, and Default Config

Establishes the package skeleton, declares new runtime dependencies, and ships the v1 reference `feature_config.yaml`. Nothing functional yet — the entry point exists as a stub.

**Satisfies:** `FE-CFG-09` (ships a default config); prerequisites for all subsequent phases.

### 1.1 Dependencies

- [x] Add `pyarrow` (runtime) and `pyyaml` (runtime) to `pyproject.toml`'s `[project.dependencies]`. Pin `pyarrow` to a specific minor version per `FE-NF-08`. _Pinned `pyarrow==24.0.*` (patch versions float, minor/major bump requires explicit upgrade); `pyyaml>=6`._
- [x] Re-install the package in editable mode: `pip install -e ".[dev]"`.
- [x] Verify imports: `python -c "import pyarrow, yaml"`. _pyarrow 24.0.0, pyyaml 6.0.3._

### 1.2 Package Skeleton

- [x] Create `src/nflpredictor/features/__init__.py` (empty).
- [x] Create `src/nflpredictor/features/__main__.py` with a stub `main()` that prints `"feature build not implemented yet"` and exits `0`.
- [x] Verify the entry point: `python -m nflpredictor.features` prints the stub message.
- [x] Add a smoke test `tests/test_features_smoke.py` that imports `nflpredictor.features` and asserts the import succeeds. Run `pytest` to confirm.

### 1.3 Default Feature Config

- [x] Create `Data/raw/feature_config.yaml` with the v1 contents shown verbatim in §4.1 of the spec.
- [x] Confirm the file parses cleanly with `python -c "import yaml; print(yaml.safe_load(open('Data/raw/feature_config.yaml')))"`.

### 1.4 Documentation Touch-Up

- [x] Update `CLAUDE.md`'s "Project status" section to note that Phase 2 implementation has begun and the feature build is invoked via `python -m nflpredictor.features` from an activated venv.
- [x] Add a one-line note that `Data/raw/feature_config.yaml` is the knob for expanding the column inventory.

**Definition of done:** `pytest` passes; `python -m nflpredictor.features` runs the stub successfully; `Data/raw/feature_config.yaml` exists with v1 defaults; dependencies install cleanly.

---

## Phase 2: Config Loading and Phase 1 Source-Hash Gate

Loads and validates `feature_config.yaml` and verifies the on-disk Phase 1 outputs match their recorded hashes. This is the input contract — every later phase trusts these invariants.

**Satisfies:** `FE-IN-01`, `FE-IN-02`, `FE-IN-03`, `FE-IN-04`, `FE-IN-05`, `FE-CFG-01` through `FE-CFG-09`, `FE-TEST-01`, `FE-TEST-08`.

### 2.1 Config Loader and Validator

- [x] In `src/nflpredictor/features/config.py`, define a `FeatureConfig` dataclass with fields mirroring the YAML schema: `normalization_version`, `madden_columns`, `madden_categorical_columns`, `game_features` (sub-dataclass with `weather`, `officials`, `include`), `slot_shapes`. _Both dataclasses are `frozen=True`; collections use tuples for immutability._
- [x] Implement `load_feature_config(path: pathlib.Path) -> FeatureConfig` that:
  - Reads via `yaml.safe_load` (per `FE-SEC-04`).
  - Rejects unknown top-level keys (`FE-CFG-01`).
  - Validates `madden_columns` non-empty and every entry a string (`FE-CFG-02`).
  - Validates `madden_categorical_columns` is a subset of `madden_columns` (`FE-CFG-03`).
  - Validates `game_features.weather ∈ {parsed, skip}` and `game_features.officials ∈ {included, skip}` (`FE-CFG-04`, `FE-CFG-05`).
  - Validates each entry in `game_features.include` matches a known game-level identifier from §3.4 (raise with a clear error otherwise — `FE-GAME-09`).
  - Validates `slot_shapes` is non-empty and a subset of `{flat, pos}` (`FE-CFG-07`).
  - Validates `normalization_version` is a non-empty string (`FE-CFG-08`).
- [x] Implement `validate_madden_columns_exist(config: FeatureConfig, madden_header: list[str]) -> None` that fails fast if any declared Madden column is missing from the Phase 1 Madden header (`FE-CFG-02`). Called from the pipeline after the loader.

### 2.2 Phase 1 Source-Hash Gate

- [x] In `src/nflpredictor/features/pipeline.py`, define module-level constants `PHASE1_MADDEN_BASENAME`, `PHASE1_BOX_SCORES_BASENAME`, `PHASE1_MANIFEST_BASENAME`.
- [x] Implement `verify_phase1_outputs(processed_dir: pathlib.Path) -> dict` that:
  - Loads `build_manifest.json`.
  - Recomputes SHA-256 of `madden_2024.csv` and `box_scores_2024.csv` on disk.
  - Compares against the manifest's `output_sha256` map (keyed by repo-relative paths like `Data/processed/madden_2024.csv`).
  - Raises `Phase1OutputMismatchError` naming the divergent file on mismatch (`FE-IN-04`).
  - Returns the parsed manifest dict for downstream provenance.
- [x] Reuse `compute_sha256` from `src/nflpredictor/databuild/manifest.py` (imported directly — Phase 2 consumes Phase 1's output, so the dependency direction is correct).

### 2.3 Tests

- [x] `tests/test_features_config.py` (`FE-TEST-01`): 13 tests covering shipped v1 load, missing file, unknown top-level key, missing top-level key, empty/invalid `slot_shapes`, invalid weather/officials modes, unknown game-level identifier, categorical-not-in-madden, empty `madden_columns`, and both `validate_madden_columns_exist` paths against the real Madden header.
- [x] `tests/test_features_pipeline_input.py` (`FE-TEST-08`): 5 tests covering happy path against real Phase 1 outputs, tampered Madden, tampered box scores, missing manifest, missing output.

**Definition of done:** Config loading rejects every malformed case enumerated in §3.2; the source-hash gate blocks runs against tampered Phase 1 outputs.

---

## Phase 3: Vocabulary Builder and Integer Coding

Implements the deterministic vocabulary construction used by every categorical column. Pure, side-effect-free code with a high test density.

**Satisfies:** `FE-VOC-01` through `FE-VOC-06`, parts of `FE-TEST-10`.

### 3.1 Vocabulary Construction

- [x] In `src/nflpredictor/features/vocab.py`, define a `Vocabulary` dataclass:
  - `entries: Mapping[str, tuple[str, ...]]` — vocab key → sorted unique string values. _Frozen dataclass; tuples for immutability._
  - Methods: `code(key, value) -> int` (returns `-1` for `None`/empty), `decode(key, code) -> str | None`, `size(key) -> int`, plus `sizes()` for the manifest's `vocab_sizes` payload.
- [x] Implement `build_vocabulary(observations) -> Vocabulary` that sorts observed values lexicographically per key (`FE-VOC-04`), drops `None`/empty so the `-1` sentinel is reserved (`FE-VOC-02`), and rebuilds from scratch per call (`FE-VOC-06`).
- [x] Implement `encode_column(values, key, vocab) -> pandas.Series` returning an `int32` series with `-1` for `None`/`NaN`/empty.

### 3.2 Tests

- [x] `tests/test_features_vocab.py` (parts of `FE-TEST-10`): 13 tests cover lexicographic sort, duplicate collapse + null drop, null sentinel for `None`/`""`, unknown-value raises, unknown-key raises, decode round-trip, deterministic rebuild across input orderings, independent per-key sorting, `sizes()` excluding the sentinel, encode happy path, encode with `NaN`, encode raises on unknown, and an empty-observations edge case.

**Definition of done:** Vocabulary is deterministic, sentinel-aware, and shared by all downstream encoders.

---

## Phase 4: Game-Level Features, Weather, and Officials

Derives every per-game column from the box-score row plus the 2024 calendar. Independent of the per-slot Madden resolution, so it can be tested in isolation.

**Satisfies:** `FE-GAME-01` through `FE-GAME-10`, `FE-WX-01` through `FE-WX-05`, `FE-OFF-01` through `FE-OFF-03`, `FE-TEST-02`, `FE-TEST-04`.

### 4.1 NFL Week Calendar

- [x] In `src/nflpredictor/features/schedule.py`, define `NFL_2024_WEEK_BOUNDARIES: tuple[tuple[date, date, int], ...]` covering Weeks 1–18 of the 2024 regular season. Boundaries were grounded against the 58 distinct `GameDate` values observed in `Data/processed/box_scores_2024.csv`.
- [x] Implement `week_for_date(game_date: date) -> int`; raise with a clear error if the date doesn't fall in any week (`FE-GAME-02`).
- [x] Module docstring notes the table is versioned as code and any change requires bumping `normalization_version`.

### 4.2 Weather Parser

- [x] In `src/nflpredictor/features/weather.py`, define `WEATHER_RX` per `FE-WX-02`, including the `no wind` branch.
- [x] Implement `parse_weather(weather_str, roof) -> tuple[float, float, float, int]`. `is_indoor` is a function of `roof` only (FE-WX-03), so it's set consistently regardless of whether weather is empty.
- [x] Implement `parse_weather_column(box_scores_df) -> pandas.DataFrame` emitting `GameId, weather_temp_f, weather_humidity_pct, weather_wind_mph, weather_is_indoor` in the order pinned by `FE-OUT-06`. Re-raises parse failures with the `GameId` attached.

### 4.3 Days-of-Rest Derivation

- [x] In `src/nflpredictor/features/game_level.py`, implement `compute_days_rest(box_scores_df) -> pandas.DataFrame` returning `GameId`, `days_rest_home`, `days_rest_away` (both `float64`). Walks each team's games in date order; first game of the season → NaN per FE-GAME-08.

### 4.4 Other Game-Level Derivations

- [x] In `game_level.py`, helpers for the remaining columns: `parse_start_hour` (`9:30am` → 9, `12:00am` → 0, `12:30pm` → 12); pass-through for `day_of_week`/`stadium`/`roof`/`surface`/`home_team_code`/`away_team_code`/`home_coach`/`away_coach`.
- [x] Implement `assemble_game_level(box_scores_df, include) -> pandas.DataFrame` emitting `GameId` plus the requested columns in declared order.

### 4.5 Officials Encoding

- [x] In `src/nflpredictor/features/officials.py` (split out from `pipeline.py` for cohesion; not enough to grow into bloat), define `CANONICAL_OFFICIAL_ROLES` and `OFFICIAL_COLUMN_NAMES` in the order pinned by FE-OUT-06.
- [x] Implement `assemble_officials(box_scores_df) -> pandas.DataFrame` walking `Official01..07_{Role,Name}` and routing each name to its canonical column (FE-OFF-02). Names from all seven columns will feed one shared `officials` vocab key in Phase 7 (FE-OFF-03).

### 4.6 Tests

- [x] `tests/test_features_schedule.py`: 5 tests covering 18-week coverage, known-date assertions (incl. Thursday opener, Christmas, final Sunday), out-of-window failures both sides, and real-data sweep.
- [x] `tests/test_features_weather.py` (`FE-TEST-02`): 11 tests covering typical/`no wind`/empty+dome/empty+outdoors/closed-vs-open retractable/malformed/negative-temp/indoor-overrides plus `parse_weather_column` happy path and error wrapping.
- [x] `tests/test_features_game_level.py` (`FE-TEST-04`): 10 tests covering `start_hour` examples + whitespace + failure, days-of-rest first-game NaN + 7-day + 4-day cases + dtype, and `assemble_game_level` ordering / full default set / no-days-rest subset.
- [x] `tests/test_features_officials.py`: 6 tests covering pinned column order, shuffled-role routing, missing-role → None, unknown-role-dropped, real-data smoke (all 7 roles populated across all 272 games), and role count invariant.

**Definition of done:** Every game-level column derivation has a test; weather handles the real-data `no wind` case; days-of-rest emits NaN for openers.

---

## Phase 5: Position Taxonomy and Madden Column Resolution

Wires Phase 1's `madden_id` lookup to per-slot feature reads, and codifies the canonical position taxonomy used by B-pos.

**Satisfies:** `FE-MAD-01` through `FE-MAD-05`, `FE-POS-01`, `FE-POS-03`, `FE-TEST-03`.

### 5.1 Position Bucket Map

- [ ] In `src/nflpredictor/features/positions.py`, define `POSITION_BUCKETS: dict[str, str]` mapping every box-score position label to its bucket per §4.3 (e.g., `"QB" -> "QB"`, `"FB" -> "RB"`, `"T" -> "OL"`, `"NT" -> "DL"`).
- [ ] Define `BUCKET_CAPACITY: dict[str, int]` from §4.3 (`QB:1, RB:2, WR:4, TE:3, OL:5, DL:5, LB:4, DB:5`).
- [ ] Define `CANONICAL_BPOS_SLOTS: list[str]` — the deterministic per-side slot list in taxonomy order: `[QB1, RB1, RB2, WR1..WR4, TE1..TE3, OL1..OL5, DL1..DL5, LB1..LB4, DB1..DB5]`. Per-game both sides emit Home-prefixed and Away-prefixed copies in that order.
- [ ] Implement `bucket_for_position(box_score_position: str) -> str` that raises `KeyError` with a clear message for unmapped labels (`FE-POS-03`).

### 5.2 Madden Column Resolution

- [ ] Decide whether per-slot reads live in `flat.py`/`pos.py` or in a shared helper module. Recommended: a small `slots.py` (added to the §5 layout) with `iter_starters(box_scores_df) -> Iterator[Starter]` and `resolve_slot(starter, madden_lookup, config) -> SlotFeatures` so both shape assemblers share the read path. _If the helper proves trivial, inline it instead — but pick one._
- [ ] Implement `build_madden_lookup(madden_df: pandas.DataFrame) -> dict[str, pandas.Series]` keyed by `madden_id`. The lookup returns the full row (per FE-MAD-01).
- [ ] Implement the per-slot read:
  - For each Madden column in `config.madden_columns`, fetch the row's value.
  - If the column is in `madden_categorical_columns`, leave it as a string (vocab encoding happens later).
  - Otherwise cast to `float64`; raise immediately on a non-numeric value naming the column + `madden_id` (`FE-MAD-03`).
- [ ] Carry the `matched` flag along as a separate per-slot `int32` value (`FE-MAD-04`).

### 5.3 Tests

- [ ] `tests/test_features_positions.py` (`FE-TEST-03`):
  - Every distinct position label observed in `Data/processed/box_scores_2024.csv` maps to a bucket without raising. (Today's observed set: `C, CB, DB, DE, DL, DT, FB, FS, G, LB, MLB, NT, OG, OL, OLB, OT, QB, RB, S, SS, T, TE, WR`.)
  - An invented label `"XYZ"` raises with a clear message.
- [ ] `tests/test_features_madden_resolution.py`:
  - A synthetic 3-row Madden table + a synthetic starter that points to row 2: read returns the right `Overall Rating` and `Archetype`.
  - A starter pointing to a Madden row with `matched=0` returns the null-filled value for numeric columns and the filled `Archetype`.
  - A numeric pass-through column containing `"6'5"` raises (`FE-MAD-03`).

**Definition of done:** Both shape builders can call into a stable, tested per-slot read; the position taxonomy is exhaustively covered against real 2024 labels.

---

## Phase 6: Shape Assembly — B-flat and B-pos

Materializes the two parquet-shaped DataFrames in memory. Output writing (and parquet serialization) is deferred to Phase 7 so this phase is purely transformational and easy to test.

**Satisfies:** `FE-FLAT-01` through `FE-FLAT-04`, `FE-POS-02`, `FE-POS-04`, `FE-TEST-09`.

### 6.1 B-flat Assembly

- [ ] In `src/nflpredictor/features/flat.py`, implement `assemble_flat(box_scores_df, madden_lookup, config, vocab) -> pandas.DataFrame`:
  - Walk the 44 slots in canonical order: `HomeOff01..11, HomeDef01..11, AwayOff01..11, AwayDef01..11`.
  - Emit columns per `(slot, madden_column)` per `FE-FLAT-01`/`FE-FLAT-02` using the pattern `{slot}_madden_{column_snake_case}`.
  - Emit `{slot}_position` integer-coded columns sharing the `positions` vocab key (`FE-FLAT-03`).
  - Emit `{slot}_matched` as a separate `int32` flag column (clarified by the resolved spec — `FE-FLAT-02`, `FE-MAD-04`).
- [ ] Helper: `_snake_case(col: str) -> str` lowercases and replaces spaces with underscores.

### 6.2 B-pos Assembly

- [ ] In `src/nflpredictor/features/pos.py`, implement `assemble_pos(box_scores_df, madden_lookup, config, vocab) -> pandas.DataFrame`:
  - For each game and each side:
    - Walk the 22 box-score slots in canonical order (`{side}Off01..11`, then `{side}Def01..11`).
    - For each starter, look up its bucket (`positions.bucket_for_position`).
    - Within the bucket, assign indices `1, 2, …` in encounter order (`FE-POS-02`).
    - If the bucket overflows its capacity, drop the overflow starter and emit a stderr warning naming the `GameId` and bucket per `FE-POS-02`/§4.3.
  - For each `(side, bucket, index)` slot in `CANONICAL_BPOS_SLOTS`:
    - If a starter was assigned, emit `{slot}_madden_*` columns from the resolved Madden row, `{slot}_present = 1`, `{slot}_matched` from the Madden row.
    - Otherwise emit NaN / `-1` for typed columns and `{slot}_present = 0`, `{slot}_matched = 0` (`FE-MAD-05`).
  - Column naming per `FE-POS-04`: `{side}{bucket}{index}_madden_{column_snake_case}`, `{side}{bucket}{index}_present`, `{side}{bucket}{index}_matched`.

### 6.3 Tests

- [ ] `tests/test_features_flat.py`:
  - Assemble against a 1-game synthetic; assert column inventory matches the expected count (44 slots × `(len(madden_columns) + 1 position)` + 44 `_matched` flags).
  - For a chosen slot, the `{slot}_madden_overall_rating` value matches the synthetic Madden row.
  - `{slot}_position` codes resolve via the `positions` vocab to the right string.
- [ ] `tests/test_features_pos.py` (`FE-TEST-09`):
  - A 1-game synthetic with 3 WRs assigns `WR1`/`WR2`/`WR3` and leaves `WR4` with `present=0` and all per-slot columns at NaN / `-1`.
  - The within-bucket ordering is box-score slot order: a WR appearing in `HomeOff03` becomes `HomeWR1`, a WR in `HomeOff07` becomes `HomeWR2`.
  - A bucket-overflow synthetic (e.g., 6 OL on one side) drops the 6th and emits a warning naming the bucket.

**Definition of done:** Both shapes assemble cleanly from a synthetic fixture; the absent-slot invariant is locked in.

---

## Phase 7: Output Emission and Pipeline Orchestration

Writes the four artifacts to `Data/processed/` and wires the pipeline together. This is where the build produces a real result for the first time.

**Satisfies:** `FE-OUT-01` through `FE-OUT-06`, `FE-VOC-05`, `FE-MAN-01` through `FE-MAN-05`, `FE-NF-02`, `FE-NF-05`, `FE-NF-06`, `FE-NF-07`, `FE-NF-08`.

### 7.1 Parquet Writers

- [ ] In `src/nflpredictor/features/outputs.py`, define `PARQUET_WRITE_OPTIONS` as a single constant capturing the deterministic settings per `FE-OUT-01`: Snappy compression, plain encoding, no dictionary encoding, no statistics, single row group equal to the row count.
- [ ] Implement `write_parquet(df: pandas.DataFrame, path: pathlib.Path) -> None` using `pyarrow.parquet.write_table` with `PARQUET_WRITE_OPTIONS`. Convert the DataFrame to an Arrow Table with explicit types per `FE-OUT-02` (`float64`, `int32`).
- [ ] Implement `finalize_column_order(df, config, *, shape: str) -> pandas.DataFrame` enforcing the order from `FE-OUT-06`: `GameId`, game-level columns in `include` order, weather (if present), officials (if present), slot columns in shape-specific order, then `home_score`, `away_score`.
- [ ] Implement `write_features(df, shape, path)` as the public entry point: pre-sort by `GameId` (`FE-OUT-03`), finalize order, write parquet.

### 7.2 Vocabulary and Manifest Sidecars

- [ ] In `src/nflpredictor/features/outputs.py`, implement `write_vocab(vocab: Vocabulary, path: pathlib.Path) -> None` using `json.dump(..., sort_keys=True, indent=2)` with a trailing `"\n"` (`FE-VOC-05`).
- [ ] In `src/nflpredictor/features/manifest.py`, implement `compute_sha256(path: pathlib.Path) -> str` (reuse from `databuild` if exported; otherwise duplicate).
- [ ] Implement `build_feature_manifest(...)` constructing the dict per `FE-MAN-01`:
  - `build_timestamp_utc` in ISO 8601 UTC (`FE-MAN-02`).
  - `normalization_version` from the config.
  - `feature_config_sha256` of the raw YAML bytes (`FE-MAN-05`).
  - `phase1_source_sha256` for `madden_2024.csv` and `box_scores_2024.csv`.
  - `output_sha256` for whichever parquets were emitted + `feature_vocab.json`.
  - `git_commit` (or `null`) via `try_get_git_commit`.
  - `phase1_manifest_git_commit` from the Phase 1 manifest dict (or `null`).
  - `column_counts` keyed by parquet basename, each with `total` plus the per-section breakdown enumerated in `FE-MAN-01` (`game_id`, `game_level`, `weather`, `officials`, `slot_madden`, `slot_position` (flat only), `slot_present` (pos only), `slot_matched`, `labels`).
  - `vocab_sizes` keyed by vocab key.
- [ ] Implement `write_manifest(manifest_dict, path)` with `json.dump(..., sort_keys=True, indent=2)` and a trailing `"\n"` (`FE-MAN-03`).

### 7.3 Pipeline Orchestration

- [ ] In `src/nflpredictor/features/pipeline.py`, implement `run_feature_build(raw_dir: pathlib.Path, processed_dir: pathlib.Path) -> None`:
  1. Load + validate `feature_config.yaml`.
  2. Verify Phase 1 source hashes; load Phase 1 manifest dict.
  3. Load Phase 1 outputs (Madden + box scores) with `dtype=str`, `keep_default_na=False`.
  4. Validate `config.madden_columns` against the Madden header.
  5. Assemble game-level columns, weather (if `parsed`), officials (if `included`), and days-of-rest.
  6. Build the Madden lookup; resolve per-slot reads.
  7. Collect all categorical observations into the vocabulary builder; build the `Vocabulary`.
  8. Encode every categorical column via the vocab.
  9. Assemble B-flat (if requested) and B-pos (if requested).
  10. Append `home_score` and `away_score` per `FE-OUT-05`.
  11. Write parquets, then `feature_vocab.json`, then `feature_manifest.json` (last, per `FE-MAN-04`).
  12. Delete any stale shape parquet not in `config.slot_shapes` (`FE-CFG-07`).
- [ ] Implement stderr logging of column counts per shape, vocab sizes, and B-pos overflow warnings (`FE-NF-06`).
- [ ] Update `src/nflpredictor/features/__main__.py` to call `run_feature_build` with default paths and exit non-zero on any exception (`FE-NF-05`).

### 7.4 Smoke Run

- [ ] Run `python -m nflpredictor.features` against the real Phase 1 outputs. Inspect the four artifacts:
  - Confirm both parquets exist; load each with `pyarrow.parquet.read_table` and check row count is 272 and the column order matches `FE-OUT-06`.
  - Confirm `feature_vocab.json` has entries for every expected key from `FE-VOC-03`.
  - Confirm `feature_manifest.json` validates against the structure in `FE-MAN-01`.
  - Spot-check one starter end-to-end: pick a known player; assert the Madden join produced the expected `Overall Rating` in both B-flat and B-pos.
- [ ] If anything looks wrong, do not patch the spec — file a note and fix the implementation.

**Definition of done:** A real feature-build run emits four files in `Data/processed/`; column counts and vocab sizes in the manifest are sane; the pipeline runs end-to-end without exceptions.

---

## Phase 8: Determinism Hardening and Integration Tests

Locks the feature build against silent drift and shape regressions. This is the protection layer.

**Satisfies:** `FE-NF-01`, `FE-NF-03`, `FE-NF-04`, `FE-TEST-05`, `FE-TEST-06`, `FE-TEST-07`, `FE-TEST-10`, closure on any straggling `FE-TEST-*` IDs.

### 8.1 Determinism Audit

- [ ] Review every `sort_values()` and `sorted()` call in the features package. Each must specify a fully-tie-breaking key with `kind="stable"` and `ignore_index=True` where applicable.
- [ ] Verify the `Vocabulary` builder iterates `dict`/`set` only after a `sorted(...)` boundary.
- [ ] Confirm parquet writes go through `PARQUET_WRITE_OPTIONS` exclusively (no ad-hoc `write_table` calls).
- [ ] Confirm `json.dump` writes use `sort_keys=True, indent=2` plus a trailing newline.
- [ ] Confirm no `print()` statements end up in any output file (all logging goes to stderr).
- [ ] Verify integer-coded categorical columns serialize as `int32` (no silent `float64` coercion when nulls appear — sentinel `-1` keeps them integer).

### 8.2 Synthetic Fixture

- [ ] Construct `tests/fixtures/features/tiny_madden_phase1.csv`, `tiny_box_scores_phase1.csv`, and `tiny_build_manifest.json` — a small slice (2–3 games, ≤8 Madden rows) shaped exactly like Phase 1's outputs. Easiest path: slice the real Phase 1 outputs to the chosen games and regenerate the manifest's `output_sha256` entries for the slice.
- [ ] Construct `tests/fixtures/features/tiny_feature_config.yaml` mirroring the v1 default.
- [ ] Run the feature build against the fixture once; manually inspect outputs; check them in as `tests/fixtures/features/expected/`.

### 8.3 Integration Test

- [ ] `tests/test_features_integration.py` (`FE-TEST-05`):
  - Run `run_feature_build` against the synthetic fixture into a temp directory.
  - Assert the two parquets are byte-identical to the checked-in expected files.
  - Assert `feature_vocab.json` is byte-identical.
  - Load `feature_manifest.json`, blank the timestamp, and assert the remaining structure matches the expected snapshot.

### 8.4 Determinism Test

- [ ] `tests/test_features_determinism.py` (`FE-TEST-06`):
  - Run `run_feature_build` against identical Phase 1 fixture outputs twice into two temp directories.
  - Assert byte-equality of both parquets and the vocab JSON.
  - For `feature_manifest.json`, load both, blank `build_timestamp_utc`, assert the remainder matches.

### 8.5 Pinned Real-Data Identities

- [ ] `tests/test_features_pinned_identities.py` (`FE-TEST-07`):
  - Run the real Phase 2 build (or load cached outputs).
  - Pick a specific `GameId` (e.g., the Chiefs/Ravens opener `202409050kan`).
  - Assert `home_score` and `away_score` in the parquet match `HomeScore`/`AwayScore` from the raw box score.
  - Assert `HomeOff01_madden_overall_rating` equals the `Overall Rating` of that game's Home Off slot 01 player's Madden row (joined by `madden_id`).
  - Assert the same player's B-pos column (whichever canonical slot they land in) carries the same `Overall Rating`.
  - Failures here mean a derivation, vocab, or shape change has shifted a real-world feature — investigate before acquiescing.

### 8.6 Vocabulary Stability

- [ ] `tests/test_features_vocab_stability.py` (rest of `FE-TEST-10`):
  - Run the build twice; assert the vocab JSON is byte-identical and every entry is lexicographically sorted.

### 8.7 Performance Sanity Check

- [ ] Run `time python -m nflpredictor.features` once and confirm wall-clock is under 30 seconds on the dev laptop (`FE-NF-03`). If not, profile and address before declaring Phase 8 done.

### 8.8 Documentation Updates

- [ ] Update `CLAUDE.md` with a one-paragraph summary of how to invoke the feature build, where its outputs land, and that it requires a clean Phase 1 build first.
- [ ] Append a short "Phase 2 implementation status: complete" note (with date) to `Docs/Idea.md`'s Phase 2 section, so future readers know the feature build is implemented.

**Definition of done:** Every `FE-TEST-*` requirement in the spec has a corresponding passing test; re-running the feature build produces byte-identical outputs; documentation is updated.

---

## After Phase 8

Phase 2 is complete when:

- All eight phases above are checked off.
- `pytest` passes with the venv activated.
- `python -m nflpredictor.features` produces the four expected feature artifacts in `Data/processed/`.
- Re-running the feature build produces byte-identical parquet, vocab, and manifest (modulo the timestamp).
- The `CLAUDE.md` and `Idea.md` doc updates are in place.

At that point, the project is ready to begin scoping Phase 3 (Splits), which can now reference the *actual* assembled feature matrices (real column counts, real vocab sizes, real `matched=0` density per game) rather than imagined data — which is the same reason Phase 1 was specified before Phase 2.

---

## References

- [Spec-Phase2-FeatureEngineering.md](./Spec-Phase2-FeatureEngineering.md) — Formal specification for Phase 2.
- [Spec-Phase1-DataBuild.md](./Spec-Phase1-DataBuild.md) — Phase 1 specification; Phase 2 reads its outputs.
- [Plan-Phase1-DataBuild.md](./Plan-Phase1-DataBuild.md) — Phase 1 implementation plan; stylistic precedent for this document.
- [Idea.md](./Idea.md) — Source idea document; §"Phase 2: Feature Engineering & Model Inputs" lists the open questions the spec resolved.
- [CLAUDE.md](../CLAUDE.md) — Repository-level notes; will be updated during Phases 1.4 and 8.8.
