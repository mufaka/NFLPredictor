# Phase 1: Data Build Implementation Plan

This document defines the phased implementation plan for the Data Build phase of the NFL Predictor project, based on [Spec-Phase1-DataBuild.md](./Spec-Phase1-DataBuild.md). Each phase builds on the previous one and contains checkbox-tracked work items. Requirement IDs (`DB-*`) reference the corresponding entries in the specification.

This is a single-developer learning project. Phases are sized for one person to complete in sittings of an hour or two, with tests landing in the same phase as the code they cover. No CI, no branch-per-phase ceremony — just sequencing.

---

## Progress Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Environment, Scaffolding, and Raw Data Relocation | Complete |
| 2 | Normalization Foundations | Complete |
| 3 | Raw Input Loading and Madden ID Assignment | Complete |
| 4 | Tiered Matching Pipeline and Manual Overrides | Complete |
| 5 | Unmatched Player Handling and Null-Fill | Complete |
| 6 | Output Emission (CSVs and Manifest) | Not Started |
| 7 | Determinism Hardening | Not Started |
| 8 | Integration Tests and Regression Fixtures | Not Started |

---

## Current State

- No Python code exists. The repository is data + documentation only.
- Raw data files are currently at `Data/box_scores_2024.csv` and `Data/maddennfl24fullplayerratings.csv` — they need to move into `Data/raw/`.
- `Data/raw/player_overrides.csv` does not exist yet; it must be created as a header-only stub.
- The Phase 1 specification is final at `Docs/Spec-Phase1-DataBuild.md`.

---

## Guiding Principles

1. **Work inside a virtual environment.** All `pip`, `python`, and `pytest` commands run with the venv activated. The venv directory is `.venv/` at the repo root and is gitignored.
2. **Tests accompany every phase.** No phase is complete until the tests it introduces pass.
3. **Vertical slices where practical.** Each phase delivers something runnable or testable on its own.
4. **One responsibility per module.** Match the proposed `src/nflpredictor/databuild/` layout; do not pile everything into a single file.
5. **Determinism is non-negotiable.** Every sort, every iteration, every floating-point format is deliberate. Phase 7 exists specifically to validate this.
6. **The specification is source of truth.** When the plan and the spec disagree, fix the plan or fix the spec — do not silently improvise.

---

## Proposed Repository Layout

```
NFLPredictor/
├── .venv/                              # gitignored
├── .gitignore
├── pyproject.toml                      # package metadata + dependencies
├── Data/
│   ├── raw/
│   │   ├── box_scores_2024.csv
│   │   ├── maddennfl24fullplayerratings.csv
│   │   └── player_overrides.csv
│   └── processed/                      # build outputs land here; gitignored
├── src/
│   └── nflpredictor/
│       ├── __init__.py
│       └── databuild/
│           ├── __init__.py
│           ├── __main__.py             # entry point: `python -m nflpredictor.databuild`
│           ├── pipeline.py             # top-level orchestration
│           ├── normalization.py        # name + team-code normalization
│           ├── teams.py                # PFR-code <-> Madden-nickname lookup
│           ├── positions.py            # box-score <-> Madden position equivalences
│           ├── ids.py                  # madden_id assignment, sort order, sequence
│           ├── overrides.py            # load and validate player_overrides.csv
│           ├── matching.py             # tier 1-4 matching
│           ├── unmatched.py            # append unmatched rows, null-fill
│           ├── outputs.py              # write processed CSVs deterministically
│           └── manifest.py             # build_manifest.json construction
└── tests/
    ├── __init__.py
    ├── test_normalization.py
    ├── test_teams.py
    ├── test_ids.py
    ├── test_overrides.py
    ├── test_matching.py
    ├── test_unmatched.py
    ├── test_outputs.py
    ├── test_manifest.py
    ├── test_determinism.py
    ├── test_pipeline_integration.py
    └── fixtures/
        ├── tiny_box_scores.csv
        ├── tiny_madden.csv
        ├── tiny_overrides.csv
        └── expected/
            ├── madden_2024.csv
            ├── box_scores_2024.csv
            └── player_id_mapping.csv
```

---

## Phase 1: Environment, Scaffolding, and Raw Data Relocation

Establishes the working environment, the package skeleton, and the file layout the spec assumes. Nothing functional yet — but every later phase depends on this being correct.

**Satisfies:** prerequisites for all `DB-IN-*` requirements; sets up the `Data/raw/` and `Data/processed/` layout (referenced throughout the spec).

### 1.1 Python and Virtual Environment

- [x] Confirm Python 3.11 or newer is available (`python3 --version`). _Python 3.12.3._
- [x] Create the virtual environment at the repo root: `python3 -m venv .venv`.
- [x] Activate the venv (`source .venv/bin/activate`). All subsequent commands assume the venv is active.
- [x] Upgrade pip inside the venv: `pip install --upgrade pip`.
- [x] Add `.venv/` to `.gitignore`.

### 1.2 Project Metadata and Dependencies

- [x] Create `pyproject.toml` declaring:
  - Project name `nflpredictor`, version `0.1.0`, Python `>=3.11`.
  - Runtime dependencies: `pandas`, `rapidfuzz`.
  - Dev dependencies (extras `[dev]`): `pytest`.
  - Build system: `setuptools` (or `hatchling`, picker's choice).
  - `[project.optional-dependencies]` section for the dev extras.
  - `src/` layout via `[tool.setuptools.packages.find]`.
- [x] Install the package and dev deps in editable mode: `pip install -e ".[dev]"`.
- [x] Verify the install: `python -c "import nflpredictor"` (should succeed once `__init__.py` exists; this gets validated in 1.3).

### 1.3 Package Skeleton

- [x] Create `src/nflpredictor/__init__.py` (empty).
- [x] Create `src/nflpredictor/databuild/__init__.py` (empty).
- [x] Create `src/nflpredictor/databuild/__main__.py` with a stub `main()` that prints "build pipeline not implemented yet" and exits `0`.
- [x] Verify the entry point: `python -m nflpredictor.databuild` prints the stub message.
- [x] Create `tests/__init__.py` and `tests/fixtures/__init__.py` (empty).
- [x] Add a smoke test `tests/test_smoke.py` that imports `nflpredictor.databuild` and asserts the import succeeds. Run `pytest` to confirm.

### 1.4 Raw Data Relocation

- [x] Create `Data/raw/` directory.
- [x] Move `Data/box_scores_2024.csv` → `Data/raw/box_scores_2024.csv`.
- [x] Move `Data/maddennfl24fullplayerratings.csv` → `Data/raw/maddennfl24fullplayerratings.csv`.
- [x] Create header-only stub at `Data/raw/player_overrides.csv` with the header `box_score_name,box_score_team_code,box_score_id,madden_id,reason` (satisfies `DB-OVR-01`, allows `DB-TEST-06`).
- [x] Create `Data/processed/` directory (empty for now).
- [x] Add `Data/processed/` to `.gitignore` (build outputs are reproducible; no need to commit).

### 1.5 Update Project Documentation

- [x] Update `CLAUDE.md`'s "Project status" section to note that Phase 1 implementation has begun and the build is invoked via `python -m nflpredictor.databuild` from an activated venv.
- [x] Add a one-line note at the top of `CLAUDE.md` explaining the venv requirement so future contributors don't run commands outside it.

**Definition of done:** `pytest` passes; `python -m nflpredictor.databuild` runs the stub successfully; `Data/raw/` and `Data/processed/` exist; `player_overrides.csv` is a header-only stub; all activated inside the venv.

---

## Phase 2: Normalization Foundations

Implements the deterministic team-code and player-name normalization that every later phase depends on. This is pure, side-effect-free code with a high test density — exactly the right shape to build first.

**Satisfies:** `DB-NORM-01` through `DB-NORM-05`, `DB-TEST-01`, `DB-TEST-02`.

### 2.1 Team-Name Normalization

- [x] In `src/nflpredictor/databuild/teams.py`, define a module-level constant `PFR_CODE_TO_MADDEN_NICKNAME: dict[str, str]` mapping all 32 PFR three-letter codes to their Madden franchise nicknames (`kan → Chiefs`, `rav → Ravens`, `sfo → 49ers`, etc.). Sort the dict literal alphabetically by key for readability.
- [x] Add `normalize_team_code(pfr_code: str) -> str` that looks up the nickname; raise a clear `KeyError` for unknown codes.
- [x] Add `madden_nickname_to_pfr_code(nickname: str) -> str` as the reverse lookup. Useful for diagnostics.

### 2.2 Player-Name Normalization

- [x] In `src/nflpredictor/databuild/normalization.py`, define `NORMALIZATION_VERSION = "v1"` (referenced by `DB-NORM-05`).
- [x] Implement `normalize_name(raw: str) -> str` that performs the five steps in `DB-NORM-03`: strip suffix tokens (`Jr.`, `Sr.`, `II`, `III`, `IV` — case-insensitive, optional trailing period), fold accents via `unicodedata.normalize('NFKD', ...)` + ASCII filter, collapse whitespace, trim, lowercase.
- [x] Document in the function's docstring that the original input is *not* modified anywhere else in the codebase — only this function's output is used for comparison.

### 2.3 Tests for Normalization

- [x] In `tests/test_teams.py`, parametrize a test that walks every entry in `PFR_CODE_TO_MADDEN_NICKNAME` and asserts the round-trip via `madden_nickname_to_pfr_code` returns the original code. Asserts the table is exactly 32 entries.
- [x] In `tests/test_normalization.py`, parametrize a test covering at least these inputs (per `DB-TEST-01`):
  - `"Patrick Mahomes"` → `"patrick mahomes"`
  - `"Odafe Oweh"` → `"odafe oweh"`
  - `"Roquan Smith"` → `"roquan smith"`
  - `"Lamar Jackson"` → `"lamar jackson"`
  - `"A.J. Brown"` → `"a.j. brown"`
  - `"D'Andre Swift"` → `"d'andre swift"`
  - `"DJ Moore"` → `"dj moore"`
  - `"Patrick Mahomes II"` → `"patrick mahomes"`
  - `"Marvin Harrison Jr."` → `"marvin harrison"`
  - `"José Ramírez"` (or similar accented) → ASCII-folded equivalent
  - `"  Travis  Kelce  "` (extra whitespace) → `"travis kelce"`
- [x] Add a property-style test: `normalize_name(normalize_name(x)) == normalize_name(x)` (idempotence) for a handful of representative inputs.

**Definition of done:** All normalization tests pass; `NORMALIZATION_VERSION` is exported and referenced from `manifest.py` later.

---

## Phase 3: Raw Input Loading and Madden ID Assignment

Read the raw files, validate their schemas, and assign deterministic `madden_id` values to the raw Madden rows. Output is purely in-memory at this stage.

**Satisfies:** `DB-IN-01`, `DB-IN-03`, `DB-IN-04`, `DB-IN-05`, `DB-ID-01`, `DB-ID-02`, `DB-ID-04`, `DB-ID-05`.

### 3.1 Schema Validation

- [x] In `src/nflpredictor/databuild/pipeline.py`, define module-level constants `EXPECTED_BOX_SCORES_HEADER: tuple[str, ...]` and `EXPECTED_MADDEN_HEADER: tuple[str, ...]` listing every column name in the order the raw files use. Generate these once from the actual raw headers; commit them as code.
- [x] Implement `load_raw_box_scores(path: pathlib.Path) -> pd.DataFrame` that reads the file with `dtype=str` and `keep_default_na=False`, then asserts the header matches `EXPECTED_BOX_SCORES_HEADER` exactly. Raises a `ValueError` with a useful diff on mismatch.
- [x] Implement `load_raw_madden(path: pathlib.Path) -> pd.DataFrame` that:
  - Reads with `dtype=str` and `keep_default_na=False`.
  - Strips leading/trailing whitespace from column names on read (`DB-IN-05`).
  - Asserts the post-strip header matches `EXPECTED_MADDEN_HEADER`.
- [x] Wire both loaders into a `load_raw_inputs()` function that returns `(box_scores_df, madden_df)`.

### 3.2 Madden ID Assignment

- [x] In `src/nflpredictor/databuild/ids.py`, implement `format_madden_id(vintage: int, sequence: int) -> str` returning `f"{vintage}-{sequence:05d}"`.
- [x] Implement `assign_raw_madden_ids(madden_df: pd.DataFrame, vintage: int = 2024) -> pd.DataFrame` that:
  - Sorts by `(Team, Position, Full Name, Jersey Number)` ascending (`DB-ID-02`).
  - Inserts a `madden_id` column at position 0 with sequential IDs starting at `2024-00001`.
  - Returns the sorted, ID-augmented DataFrame.
- [x] Implement `next_madden_id(last_id: str) -> str` for use when appending unmatched rows in Phase 5.

### 3.3 Tests

- [x] `tests/test_ids.py`:
  - Assert `format_madden_id(2024, 1) == "2024-00001"`, `format_madden_id(2024, 12345) == "2024-12345"`.
  - Assert `next_madden_id("2024-00001") == "2024-00002"`.
  - Assert `assign_raw_madden_ids` on a small synthetic 5-row DataFrame produces a stable, expected ID sequence.
  - Assert running `assign_raw_madden_ids` twice on the same DataFrame produces identical output (precursor to determinism phase).
- [x] `tests/test_pipeline_loaders.py`: header-shape checks, real-file load (272 rows + 2,368 rows), Madden whitespace-stripped headers, schema-mismatch raises, real-file end-to-end ID sweep (`2024-00001` … `2024-02368`).

**Definition of done:** Raw files load without error; schema mismatches raise; ID assignment is deterministic and tested.

---

## Phase 4: Tiered Matching Pipeline and Manual Overrides

Implements the four-tier matching pipeline. This is the heart of the build's identity logic.

**Satisfies:** `DB-MATCH-01` through `DB-MATCH-07`, `DB-OVR-01` through `DB-OVR-05`, `DB-POS-01` through `DB-POS-04`, `DB-IN-02`.

### 4.1 Override Loading

- [x] In `src/nflpredictor/databuild/overrides.py`, define an `Override` dataclass with fields `box_score_name`, `box_score_team_code`, `box_score_id` (`Optional[str]`), `madden_id`, `reason`.
- [x] Implement `load_overrides(path: pathlib.Path) -> list[Override]` that:
  - Accepts a header-only stub file (`DB-TEST-06`).
  - Validates the header against `box_score_name,box_score_team_code,box_score_id,madden_id,reason`.
  - Returns an empty list if there are no data rows.
- [x] Implement `build_override_index(overrides: list[Override], assigned_madden_ids: set[str]) -> dict[OverrideKey, Override]` that:
  - Verifies each override's `madden_id` exists in `assigned_madden_ids` (`DB-OVR-04`); raises if not.
  - Builds two index keys per override: by `box_score_id` (if present) and by `(normalized_name, team_code)`.
  - Raises on ambiguous overrides (two overrides matching the same key — `DB-OVR-05`).

### 4.2 Position Equivalence

- [x] In `src/nflpredictor/databuild/positions.py`, define `POSITION_EQUIVALENCES: dict[str, set[str]]` mapping coarse box-score positions to acceptable Madden positions (e.g., `"OL"` ↔ `{"T", "G", "C"}`, `"DB"` ↔ `{"CB", "S"}`, `"DL"` ↔ `{"DE", "DT"}`). All other positions match by equality. _Map adapted to the actual box-score (`OL`/`OT`/`OG`/`T`/`G`/`DL`/`DE`/`NT`/`DB`/`S`/`LB`/`OLB`/`RB`) and Madden (`C`/`LT`/`RT`/`LG`/`RG`/`LE`/`RE`/`DT`/`CB`/`FS`/`SS`/`LOLB`/`ROLB`/`MLB`/`HB`/`FB`) vocabularies._
- [x] Implement `positions_compatible(box_score_position: str, madden_position: str) -> bool`.

### 4.3 Matching Tiers

- [x] In `src/nflpredictor/databuild/matching.py`, define a `MatchResult` dataclass with fields `madden_id: Optional[str]`, `tier: int` (1–4 or 0 for unmatched), `note_fragment: str`, `position_mismatch: Optional[tuple[str, str]]`.
- [x] Implement `tier1_override(starter, override_index) -> Optional[MatchResult]`. Returns a result with `tier=1` and `note_fragment = f"tier1: manual override ({override.reason or 'no reason'})"`.
- [x] Implement `tier2_team_and_name(starter, madden_df_by_team_and_normname) -> Optional[MatchResult]`.
- [x] Implement `tier3_name_leaguewide(starter, madden_df_by_normname) -> Optional[MatchResult]` — only matches if exactly one Madden row has the normalized name.
- [x] Implement `tier4_fuzzy(starter, madden_df_for_team, threshold=0.85, margin=0.10) -> Optional[MatchResult]` using `rapidfuzz.fuzz.token_set_ratio` (divide by 100 to get 0–1).
- [x] Implement `match_starter(starter, indexes, override_index) -> MatchResult` that runs the four tiers in order and returns the first hit, or `tier=0` for unmatched.

### 4.4 Tests

- [x] `tests/test_overrides.py`:
  - Override with missing `madden_id` reference raises (`DB-OVR-04`).
  - Two overrides targeting the same starter raises (`DB-OVR-05`).
  - Header-only overrides file returns empty list (`DB-TEST-06`).
- [x] `tests/test_matching.py`:
  - Tier 1 wins over a possible Tier 2 match.
  - Tier 2 hit on Patrick Mahomes (Chiefs, QB) returns the expected ID against a fixture Madden table.
  - Tier 3 hit on a single-instance league-wide name (simulate a traded player whose Madden team is stale).
  - Tier 3 *fails* when the normalized name appears for multiple players league-wide.
  - Tier 4 hit at score 0.92 with no close runner-up succeeds.
  - Tier 4 fails when the runner-up is within 0.10 of the leader.
  - All-tiers-fail returns `tier=0`.
- [x] `tests/test_positions.py`: parametrized compatible/incompatible pairs across the equivalence table.

**Definition of done:** Matching tests pass; the four-tier pipeline produces stable results on the synthetic fixture.

---

## Phase 5: Unmatched Player Handling and Null-Fill

Appends rows for box-score players missing from Madden, flags `matched`, and fills nulls per spec.

**Satisfies:** `DB-UNM-01`, `DB-UNM-02`, `DB-UNM-03`, `DB-FILL-01` through `DB-FILL-05`.

### 5.1 Unmatched Append

- [x] In `src/nflpredictor/databuild/unmatched.py`, implement `collect_unmatched_starters(starters_with_match_results) -> list[UnmatchedPlayer]` that deduplicates by `(normalized_name, team_code)` (`DB-UNM-03`) and tracks `first_game_id_seen` per the spec's append-order rule.
- [x] Implement `append_unmatched_rows(madden_df: pd.DataFrame, unmatched: list[UnmatchedPlayer], next_sequence: int, vintage: int = 2024) -> pd.DataFrame` that:
  - Sorts unmatched players by `(team_code, normalized_name, first_game_id_seen)` (`DB-ID-03`).
  - Appends rows with `Team` = Madden nickname, `Position` = box-score position, `Full Name` = original case-preserving name, `matched = 0`, all other columns null.
  - Assigns sequential `madden_id` values continuing from `next_sequence`. _Signature takes the sequence implicitly from the last id in the DataFrame, simplifying the call site._
  - Sets `matched = 1` on the original raw rows.

### 5.2 Null-Fill

- [x] In `src/nflpredictor/databuild/unmatched.py`, implement `compute_fill_values(madden_df: pd.DataFrame) -> dict[str, Any]` that returns one fill value per column:
  - Numeric columns: arithmetic mean of `matched=1` rows. _Birthdate (Excel serial) and the DB-FILL-05 derived columns (Height, Weight, Age, Years Pro, Jersey Number, Total Salary, Signing Bonus) are treated as numeric._
  - Categorical columns (`Archetype`, `Running Style`, `College`, `Player Handness`): mode of `matched=1` rows; tie broken by lexicographically smallest.
  - Skips the `matched` column entirely (`DB-FILL-03`).
- [x] Implement `fill_unmatched_rows(madden_df: pd.DataFrame, fill_values: dict[str, Any]) -> pd.DataFrame` that applies the fill only to `matched=0` rows (`DB-FILL-04`). Leaves nulls on `matched=1` rows untouched.
- [x] Implement column-type classification logic: identify which columns are numeric vs. categorical from the Madden schema. Hardcode the list — do not infer from the data, since data inference is non-deterministic when columns are mixed.

### 5.3 Tests

- [x] `tests/test_unmatched.py`:
  - Same unmatched player seen in 3 games produces 1 appended row (`DB-UNM-03`).
  - Appended row has `matched = 0`; original rows have `matched = 1`.
  - Numeric fill uses mean over matched rows only (synthetic case where matched=0 rows would skew the mean if included incorrectly).
  - Mode tiebreaker picks the lexicographically smallest category.
  - `matched` column is never filled.
  - After fill, no `matched=0` row has any null cell (`DB-TEST-08`).

**Definition of done:** Unmatched rows are stable, null-fill is correct, the in-memory Madden DataFrame is fully populated.

---

## Phase 6: Output Emission (CSVs and Manifest)

Writes the four artifacts to `Data/processed/`. This is where the pipeline produces a real result for the first time.

**Satisfies:** `DB-OUT-01` through `DB-OUT-13`, `DB-MAP-01` through `DB-MAP-06`, `DB-MAN-01` through `DB-MAN-04`, `DB-NF-05`, `DB-NF-06`, `DB-NF-07`.

### 6.1 Processed Madden CSV

- [ ] In `src/nflpredictor/databuild/outputs.py`, implement `write_madden(madden_df, path)` that:
  - Sorts by `madden_id` ascending (`DB-OUT-12`).
  - Reorders columns so `madden_id` is first and `matched` is last (`DB-OUT-10`).
  - Writes with `index=False`, `lineterminator="\n"`, UTF-8 encoding.

### 6.2 Processed Box Scores CSV

- [ ] In `outputs.py`, implement `rewrite_box_score_ids(box_scores_df, slot_to_madden_id_map) -> pd.DataFrame` that walks the 44 per-slot `_ID` columns and replaces each cell with the resolved `madden_id`. The slot map is keyed by `(game_id, slot_column)`.
- [ ] Implement `write_box_scores(processed_box_scores_df, path)` that:
  - Sorts by `GameId` ascending (`DB-OUT-05`).
  - Asserts no `_ID` column contains a blank cell (`DB-OUT-03`); raises if any blank survives.
  - Writes with the same encoding/newline settings.

### 6.3 Mapping File

- [ ] In `outputs.py`, implement `build_mapping_records(match_results) -> list[MappingRecord]` that:
  - Deduplicates to one row per unique `(box_score_id, madden_id)` pair (`DB-MAP-02`).
  - Uses the most informative `note` when a pair has multiple tier hits (`DB-MAP-05`).
  - Appends position-mismatch suffixes per `DB-POS-02`.
  - Records `"blank source _id; resolved by name"` when the box-score `_ID` was blank (`DB-MAP-04`).
- [ ] Implement `write_mapping(records, path)` that:
  - Sorts by `(madden_id, box_score_id)` ascending (`DB-MAP-06`).
  - Writes with `index=False`, `lineterminator="\n"`.

### 6.4 Build Manifest

- [ ] In `src/nflpredictor/databuild/manifest.py`, implement `compute_sha256(path: pathlib.Path) -> str`.
- [ ] Implement `try_get_git_commit() -> Optional[str]` that runs `git rev-parse HEAD` and returns the hash, or `None` if not in a git repo or git is unavailable.
- [ ] Implement `build_manifest(...)` that constructs the manifest dict per `DB-MAN-01` and `DB-MAN-02`, including the counts breakdown.
- [ ] Implement `write_manifest(manifest_dict, path)` that uses `json.dump(..., sort_keys=True, indent=2)` and writes a trailing newline (`DB-MAN-04`).
- [ ] Manifest is written *last* — after the three CSVs — so output SHA-256 hashes can be computed against the on-disk files.

### 6.5 Orchestration

- [ ] In `src/nflpredictor/databuild/pipeline.py`, wire everything together into `run_build(raw_dir: pathlib.Path, processed_dir: pathlib.Path) -> None`:
  1. Load raw inputs.
  2. Assign raw Madden IDs.
  3. Load overrides.
  4. Build matching indexes.
  5. Match every starter; collect match results.
  6. Collect unmatched; append to Madden; fill nulls.
  7. Build slot-to-`madden_id` map; rewrite box-score `_ID` columns.
  8. Write Madden, box scores, mapping (in that order).
  9. Build manifest with output hashes; write manifest last.
- [ ] Add stderr/stdout logging that summarizes counts at the end (`DB-NF-06`): "tier1: X, tier2: Y, tier3: Z, tier4: W, unmatched: U, position mismatches: P".
- [ ] Update `src/nflpredictor/databuild/__main__.py` to call `run_build` with default paths and exit non-zero on any exception (`DB-NF-05`).

### 6.6 Smoke Test

- [ ] Run `python -m nflpredictor.databuild` against the real data. Inspect the four output files manually. Spot-check one starter end-to-end (e.g., confirm Mahomes' `madden_id` is consistent across Madden file, box scores, and mapping).
- [ ] If anything looks wrong, do *not* patch the spec — file a note and fix the implementation.

**Definition of done:** A real build run emits four files in `Data/processed/`; counts in the manifest look reasonable (e.g., total starter slots = 272 × 44 = 11,968); the pipeline runs end-to-end without exceptions.

---

## Phase 7: Determinism Hardening

This phase exists because determinism (`DB-NF-01`, `DB-NF-04`) is the single trickiest invariant the spec demands, and is the easiest to break accidentally during normal development. Treat this as a dedicated audit pass.

**Satisfies:** `DB-NF-01`, `DB-NF-02`, `DB-NF-04`, `DB-NF-07`, `DB-TEST-05`.

### 7.1 Audit Sort Orders

- [ ] Review every `sort_values()` and `sorted()` call in the codebase. Each must specify a fully-tie-breaking sort key — no implicit ordering left over from row order.
- [ ] In `pandas` calls, pass `kind="stable"` explicitly and `ignore_index=True` where appropriate.
- [ ] Document each sort key inline with a one-line comment referencing the spec ID it satisfies.

### 7.2 Audit Float Formatting

- [ ] Identify every numeric column written to CSV. Decide pandas' default float `to_csv` formatting is acceptable, or pin a `float_format` via a single constant in `outputs.py`.
- [ ] Verify integer columns are not silently emitted as floats (the classic pandas gotcha when nulls exist). Use nullable integer dtypes (`Int64`) or string conversion where needed.

### 7.3 Audit Encoding and Newlines

- [ ] Confirm every `to_csv` call uses `lineterminator="\n"` and UTF-8 (no BOM).
- [ ] Confirm `json.dump` writes with `indent=2`, `sort_keys=True`, and a trailing newline.
- [ ] Confirm no `print()` statements end up in the output files.

### 7.4 Audit Dict/Set Iteration

- [ ] Scan for any use of `set()` iteration that influences output ordering. Convert to `sorted(...)` lists at the boundary.
- [ ] Confirm dict iteration order in output construction is either irrelevant or explicitly sorted.

### 7.5 Determinism Test

- [ ] `tests/test_determinism.py`:
  - Run `run_build` against the fixture twice into two temp directories.
  - Assert byte-equality of `madden_2024.csv`, `box_scores_2024.csv`, `player_id_mapping.csv` across the two runs.
  - For `build_manifest.json`, load both, blank out `build_timestamp_utc`, and assert the remainder matches byte-for-byte (`DB-NF-04`).

### 7.6 Reproducibility Smoke

- [ ] Delete `Data/processed/`. Run the build. Capture the SHAs of the four output files.
- [ ] Delete `Data/processed/` again. Run the build. Confirm the SHAs of the three CSVs match the prior run (`DB-NF-01`); the manifest's three output SHA-256 entries should match across runs.

**Definition of done:** The determinism test passes; the manual reproducibility smoke shows identical output SHAs across runs.

---

## Phase 8: Integration Tests and Regression Fixtures

Locks in the build's behavior against synthetic fixtures and a small set of pinned real-data identities. After this phase, the build is protected from silent regression.

**Satisfies:** `DB-TEST-03`, `DB-TEST-04`, `DB-TEST-07`, plus closure on any straggling test IDs from earlier phases.

### 8.1 Synthetic Fixture

- [ ] Construct `tests/fixtures/tiny_box_scores.csv` — 2 games, 4 starters per side per game (so 16 starter slots total instead of 44, to keep the fixture small). Use the same 164-column header for compatibility.
- [ ] Construct `tests/fixtures/tiny_madden.csv` — 8 players covering both teams, with a deliberate name-variation case (e.g., one player with a `Jr.` suffix in box scores but not in Madden).
- [ ] Construct `tests/fixtures/tiny_overrides.csv` — at least one override covering an intentionally ambiguous case.
- [ ] Run the build against the fixture once; manually inspect outputs; check them in as `tests/fixtures/expected/`.
- [ ] *Note:* the fixture is allowed to have a header that's a strict subset of the real schema if header validation can be relaxed in test mode. Alternative: use the real 164-column header with mostly-empty rows.

### 8.2 Integration Test

- [ ] `tests/test_pipeline_integration.py` (`DB-TEST-03`):
  - Runs `run_build` against the synthetic fixture into a temp directory.
  - Asserts the three output CSVs equal the checked-in expected files byte-for-byte.
  - Loads the manifest, blanks the timestamp, and asserts the remaining structure matches an expected snapshot.

### 8.3 Pinned Real-Data Identities (`DB-TEST-04`)

- [ ] `tests/test_pinned_identities.py`:
  - Runs the real build (or loads a cached real-build output).
  - Asserts `Patrick Mahomes` on `kan` resolves to a specific `madden_id`.
  - Asserts `T.J. Watt` resolves to the same `madden_id` across all games in which he appears (even though the raw `_ID` was blank in at least one).
  - Asserts one chosen fuzzy-match case (the developer picks a real example after running the build once and inspecting the mapping file's `tier4` notes) resolves to its expected `madden_id`.
  - Failures here mean a normalization or matching change has shifted a real-world identity — investigate before acquiescing.

### 8.4 Negative-Path Tests

- [ ] `tests/test_overrides.py`: add an integration-style test where `tiny_overrides.csv` references a non-existent `madden_id`; assert the build raises (`DB-TEST-07`).
- [ ] `tests/test_unmatched.py`: add the post-fill no-nulls assertion on `matched=0` rows of the real build's output (`DB-TEST-08`).

### 8.5 Documentation Updates

- [ ] Update `CLAUDE.md` with a one-paragraph summary of how to invoke the build and where outputs land. Note the venv requirement again.
- [ ] Append a short "Phase 1 implementation status: complete" note (with date) to `Docs/Idea.md`'s Phase 1 section, so future readers know the build is implemented.

**Definition of done:** Every `DB-TEST-*` requirement in the spec has a corresponding passing test; the build is locked against regression.

---

## After Phase 8

Phase 1 is complete when:

- All eight phases above are checked off.
- `pytest` passes with the venv activated.
- `python -m nflpredictor.databuild` produces the four expected files in `Data/processed/`.
- Re-running the build produces byte-identical CSV outputs.
- The CLAUDE.md and Idea.md doc updates are in place.

At that point, the project is ready to begin scoping Phase 2 (Feature Engineering & Model Inputs), which can now reference the *actual* assembled data (counts of `matched=0` rows, real distribution of `Archetype` values, etc.) rather than imagined data — which was the entire reason Phase 1 was specified first.

---

## References

- [Spec-Phase1-DataBuild.md](./Spec-Phase1-DataBuild.md) — Formal specification for Phase 1.
- [Idea.md](./Idea.md) — Source idea document; defines Phases 1–7 and records all upstream decisions.
- [Overview.md](./Overview.md) — Superseded; historical context only.
- [CLAUDE.md](../CLAUDE.md) — Repository-level notes; will be updated during Phases 1.5 and 8.5.
