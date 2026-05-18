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
| 5 | Position Taxonomy and Madden Column Resolution | Complete |
| 6 | Shape Assembly — B-flat and B-pos | Complete |
| 7 | Output Emission and Pipeline Orchestration | Complete |
| 8 | Determinism Hardening and Integration Tests | Complete |

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

- [x] In `src/nflpredictor/features/positions.py`, define `POSITION_BUCKETS: dict[str, str]` covering all 23 box-score position labels observed in 2024 data.
- [x] Define `BUCKET_CAPACITY: dict[str, int]` from §4.3 (`QB:1, RB:2, WR:4, TE:3, OL:5, DL:5, LB:4, DB:5` — sum 29).
- [x] Define `CANONICAL_BPOS_SLOTS_PER_SIDE: tuple[str, ...]` — `[QB1, RB1, RB2, WR1..WR4, TE1..TE3, OL1..OL5, DL1..DL5, LB1..LB4, DB1..DB5]`. The assembler in Phase 6 prefixes `Home`/`Away` for the full B-pos slot list.
- [x] Implement `bucket_for_position(box_score_position: str) -> str` that raises `KeyError` naming the unmapped label and the required normalization-version bump (`FE-POS-03`).

### 5.2 Madden Column Resolution

- [x] Created a small `slots.py` (added to the §5 layout) — shared by Phase 6's B-flat and B-pos assemblers.
- [x] Define `CANONICAL_SLOTS: tuple[str, ...]` — the 44 box-score slot identifiers in FE-FLAT-01 / FE-OUT-06 order.
- [x] Define a frozen `Starter` dataclass (`game_id`, `slot`, `side`, `unit`, `box_score_position`, `madden_id`, `box_score_name`) and `iter_starters(box_scores_df) -> Iterator[Starter]`.
- [x] Implement `build_madden_lookup(madden_df) -> dict[str, dict[str, str]]` keyed by `madden_id`.
- [x] Implement `resolve_madden_features(madden_id, lookup, config) -> tuple[dict, int]`:
  - Categorical columns pass through as `str | None` (empty/`NaN` → `None`).
  - Numeric pass-through columns are cast to `float`; non-numeric raises `ValueError` naming the column and `madden_id` (`FE-MAD-03`).
  - Returns the `matched` flag as a separate `int`.

### 5.3 Tests

- [x] `tests/test_features_positions.py` (`FE-TEST-03`): 7 tests — every observed real label maps without raising, an unmapped label raises, capacity sum invariant, canonical slot count and order spot-checks, tricky-case bucket assertions (FB→RB, OT→OL, NT→DL, MLB→LB, FS/SS→DB), and a dict-size regression alarm.
- [x] `tests/test_features_madden_resolution.py`: 9 tests — canonical slot order + count, happy-path resolve, matched=0 row with null-filled values, non-numeric raises (`"6'5"` for Height), missing `madden_id` raises, empty categorical → None, `iter_starters` yields 44 per game in canonical order, side/unit decomposition correctness.

**Definition of done:** Both shape builders can call into a stable, tested per-slot read; the position taxonomy is exhaustively covered against real 2024 labels.

---

## Phase 6: Shape Assembly — B-flat and B-pos

Materializes the two parquet-shaped DataFrames in memory. Output writing (and parquet serialization) is deferred to Phase 7 so this phase is purely transformational and easy to test.

**Satisfies:** `FE-FLAT-01` through `FE-FLAT-04`, `FE-POS-02`, `FE-POS-04`, `FE-TEST-09`.

### 6.1 B-flat Assembly

- [x] In `src/nflpredictor/features/flat.py`, implement `assemble_flat(box_scores_df, madden_lookup, config, vocab) -> pandas.DataFrame` honoring the FE-OUT-06 section order: GameId, slot-major `{slot}_madden_{column_snake_case}` columns, then 44 `{slot}_position` columns, then 44 `{slot}_matched` columns.
- [x] Categorical Madden columns → `int32` via `vocab`; numeric → `float64`; position and matched flags → `int32` per FE-OUT-02.
- [x] Helper: `snake_case(col)` lowercases and replaces spaces with underscores (FE-FLAT-02). Made public so `pos.py` imports it.

### 6.2 B-pos Assembly

- [x] In `src/nflpredictor/features/pos.py`, define `CANONICAL_BPOS_SLOTS: tuple[str, ...]` — 58 entries (29 home + 29 away) in §4.3 taxonomy order.
- [x] Implement `_assign_one_side(row, side, game_id) -> dict[canonical_slot, (madden_id, position)]` walking `{side}Off01..11` then `{side}Def01..11` and assigning within-bucket indices in encounter order (FE-POS-02). Overflow beyond `BUCKET_CAPACITY` is dropped with a stderr warning naming the `GameId` and bucket.
- [x] Implement `assemble_pos(box_scores_df, madden_lookup, config, vocab) -> pandas.DataFrame`. Absent slots emit NaN / `-1` for typed columns with `present=0` / `matched=0` per FE-MAD-05; assigned slots emit the resolved features with `present=1`. Column order follows FE-OUT-06: slot-major Madden cols, then all `present` flags, then all `matched` flags.

### 6.3 Tests

- [x] `tests/_features_fixtures.py` — shared synthetic-data builders (full 44-slot box-score row + matching Madden frame + vocab) used by Phase 6 and 7 tests.
- [x] `tests/test_features_flat.py`: 6 tests — `snake_case` examples, column inventory (177 cols for v1 default), specific `Overall Rating` values, categorical decoding via vocab, dtype pinning, and matched=0 flag propagation.
- [x] `tests/test_features_pos.py` (`FE-TEST-09`): 8 tests — `CANONICAL_BPOS_SLOTS` count/order, column inventory (233 cols for v1 default), default-lineup TE2/TE3/OL4/OL5 absent, absent-slot NaN/-1/present=0/matched=0 invariant, box-score-order within-bucket assignment (HomeOff03 WR → WR1, HomeOff07 WR → WR2), 6-OL overflow → 5 assigned + stderr warning naming GameId and bucket, categorical decode, dtype pinning.

**Definition of done:** Both shapes assemble cleanly from a synthetic fixture; the absent-slot invariant is locked in.

---

## Phase 7: Output Emission and Pipeline Orchestration

Writes the four artifacts to `Data/processed/` and wires the pipeline together. This is where the build produces a real result for the first time.

**Satisfies:** `FE-OUT-01` through `FE-OUT-06`, `FE-VOC-05`, `FE-MAN-01` through `FE-MAN-05`, `FE-NF-02`, `FE-NF-05`, `FE-NF-06`, `FE-NF-07`, `FE-NF-08`.

### 7.1 Parquet Writers

- [x] In `src/nflpredictor/features/outputs.py`, define `PARQUET_WRITE_OPTIONS` capturing the deterministic settings per `FE-OUT-01`: Snappy compression, `use_dictionary=False`, `write_statistics=False`, `data_page_version='1.0'`, `version='2.6'`.
- [x] Implement `write_parquet(df, path)` using `pyarrow.parquet.write_table` with `PARQUET_WRITE_OPTIONS` and `row_group_size=max(len(df), 1)`. `combine_chunks()` ensures a single row group regardless of pandas chunking.
- [x] Column ordering is handled in `pipeline._combine_sections` rather than a dedicated `finalize_column_order` — simpler than a separate finalize step when the section order is built up incrementally during orchestration.

### 7.2 Vocabulary and Manifest Sidecars

- [x] In `src/nflpredictor/features/outputs.py`, implement `write_vocab(vocab, path)` using `json.dump(..., sort_keys=True, indent=2)` with a trailing `"\n"` (`FE-VOC-05`).
- [x] In `src/nflpredictor/features/manifest.py`, reuse `compute_sha256` and `try_get_git_commit` from `databuild.manifest` (direct import — Phase 2 consumes Phase 1's output).
- [x] Implement `build_feature_manifest(...)` constructing the dict per `FE-MAN-01` with all nine required keys.
- [x] Implement `write_feature_manifest` with `sort_keys=True`, `indent=2`, trailing newline (`FE-MAN-03`).

### 7.3 Pipeline Orchestration

- [x] In `src/nflpredictor/features/pipeline.py`, implement `run_feature_build(raw_dir, processed_dir, *, repo_dir=None) -> None` executing the 12-step pipeline. Pure functions broken out for testability: `collect_observations`, `encode_game_level`, `encode_officials`, `_combine_sections`, `_flat_column_counts`, `_pos_column_counts`, `_with_total`.
- [x] Stderr logging of per-shape row × column counts + vocab sizes (FE-NF-06). B-pos overflow warnings flow up from `pos.assemble_pos` to stderr (FE-POS-02).
- [x] Update `src/nflpredictor/features/__main__.py` to call `run_feature_build` with default paths and return exit 1 on any exception (FE-NF-05).

### 7.4 Smoke Run

- [x] Ran `python -m nflpredictor.features` against the real Phase 1 outputs. Results:
  - `features_flat_2024.parquet`: 272 rows × 202 columns (matches v1 expected total).
  - `features_pos_2024.parquet`: 272 rows × 258 columns (matches v1 expected total).
  - `feature_vocab.json`: 9 keys present (Archetype=46, coaches=35, day_of_week=6, officials=121, positions=23, roof=4, stadium=34, surface=6, team_codes=32).
  - `feature_manifest.json`: validates against FE-MAN-01.
  - Mahomes (HomeOff01 of `202409050kan`) resolves to `Overall Rating=99.0` in B-flat; `home_score=27.0`, `away_score=20.0` propagate correctly.
  - Real-data B-pos emits ~18 OL/LB overflow warnings on games where the box-score lineup exceeds taxonomy capacity — handled per FE-POS-02 without raising.
  - Byte-identical hashes confirmed across two back-to-back runs.

**Definition of done:** A real feature-build run emits four files in `Data/processed/`; column counts and vocab sizes in the manifest are sane; the pipeline runs end-to-end without exceptions.

---

## Phase 8: Determinism Hardening and Integration Tests

Locks the feature build against silent drift and shape regressions. This is the protection layer.

**Satisfies:** `FE-NF-01`, `FE-NF-03`, `FE-NF-04`, `FE-TEST-05`, `FE-TEST-06`, `FE-TEST-07`, `FE-TEST-10`, closure on any straggling `FE-TEST-*` IDs.

### 8.1 Determinism Audit

- [x] Reviewed every `sort_values()` and `sorted()` call in the features package. The single `sort_values` (`pipeline.py:257`, GameId) specifies `kind="stable"` and `ignore_index=True`. All other ordering is `sorted(...)` over deterministic collections (vocab keys/values, manifest dict items, error-message lists).
- [x] Vocabulary builder iterates `dict`/`set` only after `sorted(...)` boundaries (`vocab.build_vocabulary`).
- [x] Parquet writes go through `outputs.write_parquet` exclusively (which applies `PARQUET_WRITE_OPTIONS`). No ad-hoc `write_table` calls anywhere.
- [x] All JSON sidecars use `json.dump(..., sort_keys=True, indent=2)` plus a trailing newline (`outputs.write_vocab`, `manifest.write_feature_manifest`).
- [x] All logging goes to `sys.stderr` (`pipeline.py` row/column counts + vocab sizes; `pos.py` overflow warnings). No `print()` to stdout in the features package.
- [x] Integer-coded categorical columns serialize as `int32` via explicit `pd.array(..., dtype="int32")` calls in `flat.py` / `pos.py` / `pipeline.encode_*`. Numeric columns use explicit `float64`.

### 8.2 Synthetic Fixture

- [x] `tests/fixtures/features/raw_phase1/` holds the sliced Phase 1 outputs: `box_scores_2024.csv` (1 game — Chiefs/Ravens opener `202409050kan`), `madden_2024.csv` (44 Madden rows referenced by that game), and a regenerated `build_manifest.json` whose `output_sha256` entries match the sliced CSVs so `verify_phase1_outputs` accepts them.
- [x] `tests/fixtures/features/raw/feature_config.yaml` mirrors the v1 default.
- [x] `tests/fixtures/features/expected/` holds the four expected artifacts (parquets + vocab + manifest with timestamp blanked).
- [x] `tests/fixtures/features/_regenerate.py` regenerates the fixture on demand (`python -m tests.fixtures.features._regenerate`). Doc-string notes the fixture is pyarrow-version-sensitive.

### 8.3 Integration Test

- [x] `tests/test_features_integration.py` (`FE-TEST-05`): 4 byte-equality tests against checked-in expected outputs (flat parquet, pos parquet, vocab, manifest after timestamp/git-commit blank).

### 8.4 Determinism Test

- [x] `tests/test_features_integration.py::test_fixture_byte_identical_rerun` (`FE-TEST-06`): two independent runs against identical fixture inputs → byte-identical parquets + vocab + manifest minus timestamp. The companion `test_features_pipeline_run.py::test_byte_identical_rerun` exercises the same property on the real 272-game dataset.

### 8.5 Pinned Real-Data Identities

- [x] `tests/test_features_pipeline_run.py` (`FE-TEST-07`): pins Mahomes (HomeOff01 of `202409050kan`) to `Overall Rating=99.0` and `matched=1` in B-flat; pins HomeQB1 in B-pos to the same value; verifies the integer codes round-trip through the vocab (team_codes `kan` and day_of_week `Thursday`).

### 8.6 Vocabulary Stability

- [x] Covered by `test_features_pipeline_run.py::test_vocab_keys_match_spec` (every value list is lexicographically sorted) plus the byte-identical-rerun tests above (vocab bytes stable across runs). No separate stability test file needed.

### 8.7 Performance Sanity Check

- [x] `time python -m nflpredictor.features` against real Phase 1 data: 1.23s wall clock (max RSS ~147 MiB). Comfortably under FE-NF-03's 30-second limit.

### 8.8 Documentation Updates

- [x] `CLAUDE.md`'s Project status section now describes the Phase 2 invocation, the four output artifacts, and the byte-identical-rerun guarantee.
- [x] `Docs/Idea.md`'s Phase 2 section carries a "Status: Implementation complete (2026-05-18)" header with the real-data run summary and the four design-question outcomes.

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
