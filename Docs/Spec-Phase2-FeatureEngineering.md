# Phase 2: Feature Engineering & Model Inputs Specification

## 1. Introduction

### 1.1 Purpose

This specification defines the **Feature Engineering & Model Inputs** phase of the NFL Predictor project. Phase 2 converts the Phase 1 processed artifacts (`madden_2024.csv`, `box_scores_2024.csv`) into a small set of deterministic, model-ready feature matrices. The feature matrices are the unit of input every downstream phase (splits, baselines, model ladder, evaluation, error analysis, 2025 test) binds to.

Phase 1 is implemented and frozen; this spec assumes its outputs as given. Phases 3–7 remain exploratory in `Docs/Idea.md` and are intentionally not specified yet — Phase 2 closes the contract one layer up the stack.

### 1.2 Scope

In scope:

- A second deterministic build step (separate from the Phase 1 build) that reads `Data/processed/{madden_2024.csv, box_scores_2024.csv}` and a hand-edited `Data/raw/feature_config.yaml` and writes a small set of feature artifacts into `Data/processed/`.
- Two slot-presentation shapes emitted in parallel: **B-flat** (slot-indexed) and **B-pos** (position-indexed). B-set is deferred.
- A configurable Madden-column selection mechanism whose default v1 list is `Overall Rating` + `Archetype` and that expands by editing one YAML file (no code change).
- Integer-coded categorical encoding with a single shared vocabulary sidecar; final one-hot vs. embedding choice is deferred to training time.
- A parsed weather decomposition (temperature, humidity, wind) and an officials feature group, both included by default.
- A feature manifest that records the source SHAs, the config SHA, the produced output SHAs, and per-shape column counts.
- Testing requirements that protect the contract from silent regression.

Out of scope:

- Train/validation/test split logic. (Phase 3)
- Any model code or training loop. (Phase 4)
- B-set permutation-invariant slot presentation. (Future Phase 2 amendment, gated on Phase 4 rung 4.)
- Final categorical encoding (one-hot vs. embedding vs. target encoding) — Phase 2 ships integer codes; the model code at fit time picks the realization.
- Metrics, error analysis, or 2025-season inference. (Phases 5–7)
- Re-running Phase 1. The feature build assumes Phase 1's outputs are present and trusts their `build_manifest.json`.

### 1.3 Definitions and Acronyms

| Term | Definition |
|------|------------|
| Feature matrix | A 2D table where each row is one NFL game and each column is one model input. Two are emitted: `features_flat_2024.parquet` and `features_pos_2024.parquet`. |
| **B-flat** | Slot-indexed presentation: starter columns are namespaced by their box-score slot (`HomeOff01_…`, `AwayDef11_…`). Model must learn that slot 01 is QB. |
| **B-pos** | Position-indexed presentation: starter columns are grouped by a fixed canonical position taxonomy (`HomeQB_…`, `HomeWR1_…`, `AwayDB3_…`). |
| **B-set** | Permutation-invariant set presentation. Out of scope for v1; named here only to make the deferral explicit. |
| Feature config | The hand-edited `Data/raw/feature_config.yaml` declaring which Madden columns, game-level features, and slot shapes to emit. |
| Vocabulary sidecar | `Data/processed/feature_vocab.json`: maps every categorical column to its integer-code domain (`code → string`). The sole place categorical strings appear in the processed layer. |
| Canonical position taxonomy | The fixed `(position bucket → slot count)` table that defines B-pos's column shape. Versioned in code. |
| Position bucket | A coarse position group used by B-pos: `QB`, `RB`, `WR`, `TE`, `OL`, `DL`, `LB`, `DB`. |
| Slot index | Within a position bucket, the 1-based ordinal of a player after deterministic ordering by jersey number ascending then `madden_id` ascending. |
| Null sentinel (integer codes) | `-1`. Used in integer-coded categorical columns when a value is missing (e.g., empty weather string, B-pos slot unfilled). The vocabulary sidecar never assigns code `-1`. |
| Null sentinel (numeric) | NaN, preserved in the parquet as the parquet-native null. |
| Feature manifest | `Data/processed/feature_manifest.json`: provenance for a feature-build run (input SHAs, config SHA, output SHAs, column counts, build timestamp). |

### 1.4 Design Principles

- **Contract first.** The feature matrices and the vocabulary sidecar are a stable, documented contract. Phases 3+ bind to that contract, not to the build's internals.
- **Determinism above all.** Identical inputs (Phase 1 outputs + `feature_config.yaml`) must produce byte-identical parquet, JSON sidecars, and manifests across runs. The single permitted source of drift is the manifest's build timestamp.
- **Auditable feature provenance.** Every column in the feature matrix can be traced back to a Madden column, a box-score column, or a documented derivation rule. The feature manifest records the column inventory per shape.
- **Loss-less preservation of the Phase 1 layer.** The Phase 1 processed files are read-only. The Phase 2 build never modifies them.
- **Configuration over code edits.** Changing the Madden column list, weather toggle, or officials toggle is a YAML edit. Adding a *new kind of feature* (e.g., a new derived game-level field) is a code change accompanied by a config-schema change.
- **One realization per phase.** Phase 2 emits integer codes for categoricals; one-hot vs. embedding is a Phase 4 decision. Phase 2 emits both B-flat and B-pos; choosing between them at training time is a Phase 4 decision. The line between "the artifact" and "the model input" is drawn deliberately.

### 1.5 Relationship to Phase 1

Phase 1 emits player identity and the joined-but-not-yet-shaped data. Phase 2 picks columns, encodes them, and lays them out in a model-friendly shape. The two builds are independent scripts:

```
Data/raw/{box_scores, madden, player_overrides, feature_config}
    │
    │  python -m nflpredictor.databuild
    ▼
Data/processed/{madden_2024.csv, box_scores_2024.csv,
                player_id_mapping.csv, build_manifest.json}
    │
    │  python -m nflpredictor.features
    ▼
Data/processed/{features_flat_2024.parquet, features_pos_2024.parquet,
                feature_vocab.json, feature_manifest.json}
```

The Phase 2 build refuses to run if the Phase 1 outputs are missing or if their on-disk SHA-256 hashes disagree with `Data/processed/build_manifest.json`'s recorded output hashes. This guarantees that a feature matrix is always pinned to a specific Phase 1 build.

---

## 2. Technology Additions

| Layer | Technology | Purpose |
|-------|------------|---------|
| Build step | Python (same venv as Phase 1) | Reads Phase 1 outputs + `feature_config.yaml`; emits parquet + JSON. |
| Parquet I/O | `pyarrow` | Parquet read/write. Deterministic compression settings are pinned in §3.10. |
| YAML I/O | `pyyaml` (`safe_load`) | Reads `Data/raw/feature_config.yaml`. |
| Storage format (matrices) | Parquet (Snappy compression, plain encoding, no dictionary encoding) | Typed columnar storage for the two feature matrices. Smaller and faster than CSV; preserves NaN semantics for missing numerics. |
| Storage format (vocab and manifest) | JSON (sorted keys, two-space indent, trailing newline) | Human-inspectable sidecars. |

The build is executable from the repository root via `python -m nflpredictor.features` and requires no interactive input.

---

## 3. Functional Requirements

### 3.1 Input Contract

| ID | Requirement |
|----|-------------|
| FE-IN-01 | The build shall read `Data/processed/madden_2024.csv`, `Data/processed/box_scores_2024.csv`, and `Data/processed/build_manifest.json` as primary inputs. |
| FE-IN-02 | The build shall read `Data/raw/feature_config.yaml` as a secondary input. The file is required; the build shall fail fast if it is missing. |
| FE-IN-03 | The build shall not modify any file under `Data/raw/` or any Phase 1 output under `Data/processed/`. |
| FE-IN-04 | The build shall recompute the SHA-256 of `madden_2024.csv` and `box_scores_2024.csv` and compare each to the corresponding `output_sha256` entry in `build_manifest.json`. On mismatch the build shall fail fast with a clear error naming the divergent file. |
| FE-IN-05 | The build shall reject (fail-fast) any `feature_config.yaml` whose schema does not satisfy §4.1. |

### 3.2 Feature Config Schema

| ID | Requirement |
|----|-------------|
| FE-CFG-01 | The `feature_config.yaml` schema shall include the top-level keys defined in §4.1. Unknown top-level keys shall cause the build to fail fast. |
| FE-CFG-02 | `madden_columns` is a non-empty list of column names. Each name shall exist in `madden_2024.csv`'s header (after Phase 1's whitespace-stripping). |
| FE-CFG-03 | `madden_categorical_columns` is a list of column names that the build shall treat as categorical (integer-coded against the vocabulary sidecar). Every entry shall also appear in `madden_columns`. Columns in `madden_columns` but not in `madden_categorical_columns` are treated as numeric pass-through. |
| FE-CFG-04 | `game_features.weather` is one of: `parsed` (default), `skip`. When `parsed`, the build emits the four columns named in §3.5. When `skip`, those four columns are omitted entirely. |
| FE-CFG-05 | `game_features.officials` is one of: `included` (default), `skip`. When `included`, the build emits the seven official-name code columns named in §3.6. When `skip`, those columns are omitted. |
| FE-CFG-06 | `game_features.include` is a list of game-level field identifiers (from §3.4) to include. Identifiers not on the list are omitted. |
| FE-CFG-07 | `slot_shapes` is a list with at least one of `flat`, `pos`. When both are listed, both parquet artifacts are emitted. When only one is listed, the other parquet is not emitted and is removed from `Data/processed/` if present. |
| FE-CFG-08 | `normalization_version` (string) is recorded in the manifest. Any change to derivation rules (weather parser, position taxonomy, days-of-rest logic) requires bumping this version. |
| FE-CFG-09 | A reference default config shall ship in the repository (`Data/raw/feature_config.yaml`) and reproduce the v1 column set declared in §3.3. |

### 3.3 v1 Default Feature Set

The default `feature_config.yaml` shipped in the repo declares the following. Editing this file (and only this file) is how the column inventory expands.

| Field | v1 Default |
|-------|-----------|
| `madden_columns` | `["Overall Rating", "Archetype"]` |
| `madden_categorical_columns` | `["Archetype"]` |
| `game_features.weather` | `parsed` |
| `game_features.officials` | `included` |
| `game_features.include` | `["week", "day_of_week", "start_hour", "stadium", "roof", "surface", "home_team_code", "away_team_code", "home_coach", "away_coach", "days_rest_home", "days_rest_away"]` |
| `slot_shapes` | `["flat", "pos"]` |
| `normalization_version` | `"v1"` |

### 3.4 Game-Level Features

| ID | Requirement |
|----|-------------|
| FE-GAME-01 | The build shall derive and emit the game-level columns listed in §3.3 when present in `game_features.include`. |
| FE-GAME-02 | `week` (numeric): the integer NFL week (1–18). Derived from `GameDate` against the 2024 NFL regular-season calendar; the calendar table shall be versioned as code or a versioned data file. |
| FE-GAME-03 | `day_of_week` (categorical): one of `Mon`, `Tue`, …, `Sun`. Integer-coded via the vocabulary sidecar. |
| FE-GAME-04 | `start_hour` (numeric): the local kickoff hour (0–23) extracted from `StartTime`. Minutes are dropped. |
| FE-GAME-05 | `stadium` (categorical), `roof` (categorical), `surface` (categorical): integer-coded via the vocabulary sidecar from the raw box-score values. |
| FE-GAME-06 | `home_team_code` and `away_team_code` (categorical): integer-coded via the vocabulary sidecar using the 32 PFR three-letter codes as the closed domain. |
| FE-GAME-07 | `home_coach` and `away_coach` (categorical): integer-coded via the vocabulary sidecar from the raw box-score values. |
| FE-GAME-08 | `days_rest_home` and `days_rest_away` (numeric): the number of days between each team's prior game and the current game's `GameDate`, computed across the season. Week-1 games receive a fixed sentinel value of `14` (representing the long offseason gap); the choice is documented in code. |
| FE-GAME-09 | Columns listed in `game_features.include` but not present in any §3.4 requirement shall cause the build to fail fast with a clear error. |
| FE-GAME-10 | Game-level columns appear once per row in both the B-flat and B-pos parquets, with identical values across the two files. |

### 3.5 Weather

| ID | Requirement |
|----|-------------|
| FE-WX-01 | When `game_features.weather` is `parsed`, the build shall emit four columns: `weather_temp_f` (numeric), `weather_humidity_pct` (numeric), `weather_wind_mph` (numeric), `weather_is_indoor` (numeric 0/1). |
| FE-WX-02 | The build shall parse the box-score `Weather` field with the regex `r"(?P<temp>-?\d+)\s+degrees,\s+relative humidity\s+(?P<humidity>\d+)%,\s+wind\s+(?P<wind>\d+)\s+mph"` (case-insensitive). |
| FE-WX-03 | When `Weather` is empty, the build shall emit `weather_temp_f`, `weather_humidity_pct`, `weather_wind_mph` as NaN and set `weather_is_indoor = 1` if `Roof in {dome, retractable roof (closed)}`, else `0`. |
| FE-WX-04 | When `Weather` is non-empty but does not match the regex, the build shall fail fast and report the offending `GameId` and raw weather string. No silent fallback. |
| FE-WX-05 | The four weather columns appear in both B-flat and B-pos parquets with identical values. |

### 3.6 Officials

| ID | Requirement |
|----|-------------|
| FE-OFF-01 | When `game_features.officials` is `included`, the build shall emit seven categorical integer-coded columns named `official_referee`, `official_umpire`, `official_down_judge`, `official_line_judge`, `official_back_judge`, `official_side_judge`, `official_field_judge`. |
| FE-OFF-02 | The build shall assign each official to their column based on the box-score `OfficialNN_Role` field (canonical roles enumerated in §4.2). If a game's officials list does not contain one of the seven canonical roles, the corresponding code shall be the null sentinel `-1`. |
| FE-OFF-03 | The seven official columns share a single vocabulary entry in the sidecar (key `officials`) so that a referee promoted to umpire mid-season carries the same integer code in either column. |

### 3.7 Madden Column Resolution

| ID | Requirement |
|----|-------------|
| FE-MAD-01 | For each starter slot in each game, the build shall look up the slot's `madden_id` in `madden_2024.csv` and read the values of every column listed in `madden_columns`. |
| FE-MAD-02 | Columns in `madden_categorical_columns` shall be integer-coded against the vocabulary sidecar (one vocabulary entry per source column, keyed by the original column name). |
| FE-MAD-03 | Columns not in `madden_categorical_columns` shall be cast to `float64` and passed through. Non-numeric values in numeric pass-through columns shall cause the build to fail fast. |
| FE-MAD-04 | The `matched` flag from the Phase 1 Madden file shall be emitted as a per-slot numeric column (`{slot}_matched`) in both B-flat and B-pos. The flag enables the model to discount null-fill noise per `Docs/Idea.md`'s tradeoff analysis. |
| FE-MAD-05 | When B-pos has no actual player for a canonical slot in a given game (e.g., a `WR4` slot in a 3-WR formation), all per-slot columns for that slot shall be set to NaN (numeric) or `-1` (integer-coded categorical), and an associated `{slot}_present` column (numeric 0/1) shall be set to `0`. When the slot is filled, `{slot}_present` is `1`. |

### 3.8 B-flat Slot Presentation

| ID | Requirement |
|----|-------------|
| FE-FLAT-01 | The B-flat parquet shall include one column per `(slot, madden_column)` pair, where `slot` ranges over the 44 box-score slot identifiers `{HomeOff01..HomeOff11, HomeDef01..HomeDef11, AwayOff01..AwayOff11, AwayDef01..AwayDef11}` and `madden_column` ranges over the resolved Madden columns plus the synthetic `matched` flag. |
| FE-FLAT-02 | B-flat column names shall follow the pattern `{slot}_madden_{column_snake_case}`. `column_snake_case` is the source Madden column name lowercased and with spaces replaced by underscores (e.g., `Overall Rating` → `overall_rating`). |
| FE-FLAT-03 | The 44 box-score slot positions (the `{slot}_Position` field) shall be emitted as 44 additional integer-coded categorical columns (`{slot}_position`) so the model can see what role each slot played in the game. These columns share a single vocabulary entry keyed `positions`. |
| FE-FLAT-04 | B-flat is unaffected by the canonical position taxonomy used by B-pos. Slot 01 is always slot 01 regardless of who played. |

### 3.9 B-pos Slot Presentation

| ID | Requirement |
|----|-------------|
| FE-POS-01 | The B-pos parquet shall use a fixed canonical position taxonomy defined in code (see §4.3): per side (Home/Away), an enumerated list of position-bucket slots with explicit capacities. The taxonomy is version-pinned to `normalization_version`. |
| FE-POS-02 | For each game, the build shall assign each starter to a canonical slot by: (a) mapping the starter's box-score position to a position bucket via the position-bucket map in §4.3; (b) within the bucket, ordering filled slots by `(Jersey Number ascending, madden_id ascending)` to produce stable `WR1`, `WR2`, … assignments; (c) leaving extra slots beyond the canonical capacity unassigned and emitting a stderr warning naming the `GameId` and bucket. |
| FE-POS-03 | If a starter's box-score position is not in the position-bucket map (§4.3), the build shall fail fast with a clear error naming the `GameId`, slot, and unmapped position label. New labels require a code change and a `normalization_version` bump. |
| FE-POS-04 | B-pos column names shall follow the pattern `{side}{bucket}{index}_madden_{column_snake_case}` for the Madden columns (e.g., `HomeWR1_madden_overall_rating`), `{side}{bucket}{index}_present` for the present flag, and `{side}{bucket}{index}_matched` for the matched flag. |

### 3.10 Output Encoding

| ID | Requirement |
|----|-------------|
| FE-OUT-01 | Both parquet files shall be written with: Snappy compression, plain encoding (no dictionary encoding), no statistics, row-group size = total row count (single row group). These settings make the byte output deterministic across `pyarrow` versions that honor them. |
| FE-OUT-02 | Numeric columns shall be written as `float64` for floats and `int32` for integer-coded categoricals (sentinel `-1`). The `matched` and `present` flags are written as `int32`. |
| FE-OUT-03 | Both parquet files shall be sorted by `GameId` ascending and shall include a `GameId` column (string) as the first column. |
| FE-OUT-04 | The set of `GameId` values in both parquets shall equal the set of `GameId` values in `box_scores_2024.csv`. |
| FE-OUT-05 | Label columns `home_score` and `away_score` (both `float64`) shall be present as the last two columns of each parquet. They are the regression targets and are copied verbatim from `HomeScore` / `AwayScore`. |
| FE-OUT-06 | Column order within each parquet shall be: `GameId`, then game-level columns in the order declared in `game_features.include`, then weather columns (if present, in the order temp/humidity/wind/is_indoor), then official columns (if present, in canonical role order), then slot columns (B-flat: by `(slot, madden_column)` with `slot` in the order Home Off 01..11, Home Def 01..11, Away Off 01..11, Away Def 01..11, then `{slot}_position` for the same slots, then `{slot}_matched`; B-pos: by canonical taxonomy order from §4.3, then `{slot}_present` then `{slot}_matched`), then `home_score`, `away_score`. |

### 3.11 Vocabulary Sidecar

| ID | Requirement |
|----|-------------|
| FE-VOC-01 | The build shall emit `Data/processed/feature_vocab.json` containing the integer-code domain for every categorical column. |
| FE-VOC-02 | Each vocabulary entry maps a column key to an ordered list of distinct string values. The integer code for a value is its 0-based index in that list. The null sentinel `-1` is never present in any list. |
| FE-VOC-03 | Vocabulary keys: each Madden categorical column (keyed by its original column name, e.g., `Archetype`), `day_of_week`, `stadium`, `roof`, `surface`, `team_codes` (shared by `home_team_code` and `away_team_code`), `coaches` (shared by `home_coach` and `away_coach`), `officials` (shared by all seven official columns), `positions` (shared by all 44 B-flat `{slot}_position` columns). |
| FE-VOC-04 | Within each vocabulary entry, string values are listed in ascending lexicographic order to make integer codes stable across runs and source orderings. |
| FE-VOC-05 | The vocabulary file shall be written with `json.dump(..., sort_keys=True, indent=2)` and a trailing newline. |
| FE-VOC-06 | When a value appears in input data but is not predictable in advance (e.g., a new official name), it is added to the vocabulary in the same build run. The vocabulary is **fully rebuilt** on every run; the build does not attempt to preserve codes from a previous run. Downstream code that depends on stable codes must re-resolve them via the sidecar each time. |

### 3.12 Feature Manifest

| ID | Requirement |
|----|-------------|
| FE-MAN-01 | The build shall emit `Data/processed/feature_manifest.json` containing at minimum the keys: `build_timestamp_utc`, `normalization_version`, `feature_config_sha256`, `phase1_source_sha256` (object: input filename → SHA), `output_sha256` (object: output filename → SHA), `git_commit` (or `null`), `phase1_manifest_git_commit` (the `git_commit` from `build_manifest.json`, or `null`), `column_counts` (object: per-parquet column count and per-section breakdown), `vocab_sizes` (object: vocabulary key → entry count). |
| FE-MAN-02 | The manifest's `build_timestamp_utc` shall be in ISO 8601 UTC format. |
| FE-MAN-03 | The manifest shall be written with sorted keys and stable two-space indentation; byte-identical inputs shall produce byte-identical manifests modulo the timestamp. |
| FE-MAN-04 | The manifest shall be written **last** — after the parquets and the vocab sidecar — so output SHAs can be computed against the on-disk files. |

---

## 4. Data Model

### 4.1 `Data/raw/feature_config.yaml`

```yaml
normalization_version: "v1"

madden_columns:
  - "Overall Rating"
  - "Archetype"

madden_categorical_columns:
  - "Archetype"

game_features:
  weather: parsed              # parsed | skip
  officials: included          # included | skip
  include:
    - week
    - day_of_week
    - start_hour
    - stadium
    - roof
    - surface
    - home_team_code
    - away_team_code
    - home_coach
    - away_coach
    - days_rest_home
    - days_rest_away

slot_shapes: [flat, pos]
```

### 4.2 Canonical Officials Roles

The seven box-score `OfficialNN_Role` strings, in canonical order: `Referee`, `Umpire`, `Down Judge`, `Line Judge`, `Back Judge`, `Side Judge`, `Field Judge`.

### 4.3 Canonical Position Taxonomy (B-pos)

The position-bucket map collapses the inconsistent box-score position labels into 8 buckets and pins each bucket's capacity:

| Bucket | Box-score labels | Capacity per side |
|--------|------------------|-------------------|
| `QB`   | `QB`             | 1 |
| `RB`   | `RB`, `FB`       | 2 |
| `WR`   | `WR`             | 4 |
| `TE`   | `TE`             | 3 |
| `OL`   | `OL`, `T`, `OT`, `G`, `OG`, `C` | 5 |
| `DL`   | `DL`, `DT`, `NT`, `DE`          | 5 |
| `LB`   | `LB`, `MLB`, `OLB`              | 4 |
| `DB`   | `DB`, `CB`, `S`, `FS`, `SS`     | 5 |

Per-side capacity sum is 29 (offense 15, defense 14). The B-pos parquet therefore has 58 player slots per row (29 home + 29 away). Slots beyond the actually-fielded starter set are marked `{slot}_present = 0`.

If a game's actual lineup contains more players in a bucket than the bucket's capacity, the overflow players are dropped and a stderr warning is emitted naming the `GameId` and bucket. Such overflow is expected to be vanishingly rare on real NFL data; if it becomes routine, the capacity is widened in a Phase 2 amendment with a `normalization_version` bump.

### 4.4 Output Files

| File | Type | Description |
|------|------|-------------|
| `Data/processed/features_flat_2024.parquet` | parquet | B-flat feature matrix. 272 rows × N_flat columns. |
| `Data/processed/features_pos_2024.parquet` | parquet | B-pos feature matrix. 272 rows × N_pos columns. |
| `Data/processed/feature_vocab.json` | JSON | Integer-code vocabularies for all categorical columns. |
| `Data/processed/feature_manifest.json` | JSON | Build provenance. |

`N_flat` and `N_pos` are deterministic functions of the active config; their values are recorded in the manifest's `column_counts`.

---

## 5. Build Pipeline Design

The pipeline is described conceptually; implementation may organize it differently as long as §3's contract is honored. Recommended module layout, mirroring Phase 1's `src/nflpredictor/databuild/`:

```
src/nflpredictor/features/
    __init__.py
    __main__.py        # entry point: python -m nflpredictor.features
    pipeline.py        # top-level orchestration
    config.py          # feature_config.yaml load + validation
    schedule.py        # NFL 2024 week calendar
    weather.py         # weather string parser
    positions.py       # canonical position taxonomy + bucket map
    vocab.py           # vocabulary builder (sorted, deterministic)
    flat.py            # B-flat assembly
    pos.py             # B-pos assembly
    outputs.py         # parquet + JSON writers
    manifest.py        # feature_manifest.json construction
```

### 5.1 Conceptual Stages

1. **Load and validate inputs.** Read `feature_config.yaml`; validate per §3.2 and §4.1. Read Phase 1 outputs; verify their SHAs against `build_manifest.json` (FE-IN-04).
2. **Resolve column inventory.** From the config, compute the final list of game-level columns, weather columns, officials columns, and Madden columns (numeric vs. categorical).
3. **Derive game-level features.** Week from date, day-of-week, start hour, days-of-rest (across the sorted-by-date game sequence per team).
4. **Parse weather.** Apply the regex from FE-WX-02 to every game; raise on unparseable non-empty values.
5. **Build vocabularies.** For every categorical column, collect all observed string values across the entire dataset, sort lexicographically, assign codes by position.
6. **Encode game-level and weather and officials.** Apply numeric casts and integer codes.
7. **Resolve starter slots.** For each game, walk the 44 box-score slots; for each, fetch the slot's `madden_id`, look up the Madden row, read the configured Madden columns.
8. **Assemble B-flat.** One row per game, columns per §3.8 and §3.10.
9. **Assemble B-pos.** One row per game; group starters into canonical buckets per §4.3; assign within-bucket slot indices by `(Jersey Number asc, madden_id asc)`; fill missing slots per FE-MAD-05.
10. **Emit outputs.** Write `features_flat_2024.parquet` (if `flat` in `slot_shapes`), `features_pos_2024.parquet` (if `pos`), `feature_vocab.json`, then `feature_manifest.json` (last).

### 5.2 Determinism Boundaries

The only non-deterministic input is the wall-clock timestamp written to `feature_manifest.json`. All other output bytes — both parquets and the vocabulary JSON — must be byte-identical across re-runs on identical inputs. Phase 2's determinism story is bound to Phase 1's: any drift in Phase 1's CSVs produces an FE-IN-04 hash mismatch and a hard failure rather than silent feature drift.

---

## 6. Integration / Endpoint / Tooling Design

Not applicable. The feature build is a single-shot offline script with no network surface, no API, and no UI.

---

## 7. Changes to Existing Requirements

`Docs/Idea.md` §2A–§2D enumerated open questions for Phase 2. The decisions encoded in this spec are:

- §2A "Which Madden columns": **deferred to `feature_config.yaml`**; v1 default is `Overall Rating` + `Archetype`.
- §2B "How starter slots are presented": **both B-flat and B-pos emitted**; B-set deferred.
- §2C "Game-level features": parsed weather and officials are **both included** by default; the v1 game-level field list is the one in §3.3.
- §2D "Categorical encoding": **integer codes + vocab sidecar**; one-hot vs. embedding deferred to training time.

The Phase 1 spec is unchanged.

---

## 8. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| FE-NF-01 | The build shall be deterministic: identical Phase 1 outputs and `feature_config.yaml` shall produce byte-identical parquet, vocabulary, and manifest (excluding timestamp) across runs. |
| FE-NF-02 | The build shall be re-runnable: it shall produce correct output regardless of whether the four feature artifacts exist, are stale, or are absent. Stale outputs shall be overwritten. |
| FE-NF-03 | The build shall complete in under 30 seconds on a modern laptop for the current data scale (272 games × 44 starters; 2,368 + appended Madden rows). |
| FE-NF-04 | The `build_timestamp_utc` field in `feature_manifest.json` is the only permitted source of run-to-run output drift. |
| FE-NF-05 | The build shall emit a non-zero exit code on any fatal error. Warnings (e.g., B-pos bucket overflow) shall not affect the exit code. |
| FE-NF-06 | The build shall produce sufficient stderr/stdout logging that column counts per shape, vocabulary sizes, and any B-pos overflow warnings are visible without opening the manifest. |
| FE-NF-07 | All JSON sidecars shall use UTF-8 encoding with Unix line endings (`\n`). |
| FE-NF-08 | Parquet writes shall pin `pyarrow` write options to the settings in FE-OUT-01. The `pyarrow` version shall be pinned in `pyproject.toml`; an upgrade requires a manual byte-equality re-check and a `normalization_version` bump if the bytes change. |

---

## 9. UI Requirements

Not applicable.

---

## 10. Testing Requirements

| ID | Requirement |
|----|-------------|
| FE-TEST-01 | A unit test shall verify `feature_config.yaml` validation against (a) the default v1 config (must accept); (b) a config with an unknown top-level key (must reject); (c) a config with a `madden_columns` entry that doesn't exist in the Madden header (must reject); (d) an empty `slot_shapes` (must reject). |
| FE-TEST-02 | A unit test shall verify the weather parser (§3.5) against (a) a typical string (`"67 degrees, relative humidity 53%, wind 8 mph"`); (b) an empty string with `Roof = "dome"` (returns NaN/NaN/NaN/1); (c) an empty string with `Roof = "outdoors"` (returns NaN/NaN/NaN/0); (d) a malformed non-empty string (raises). |
| FE-TEST-03 | A unit test shall verify the position-bucket map (§4.3) against every box-score position label observed in `box_scores_2024.csv`. The test fails if a new unmapped label appears. |
| FE-TEST-04 | A unit test shall verify the days-of-rest derivation (FE-GAME-08) against a synthetic schedule fixture where the expected days-of-rest for each game are known. |
| FE-TEST-05 | An integration test shall run the full Phase 2 build against a synthetic fixture (2–3 games, ≤8 Madden rows) and assert that the two parquets, the vocabulary, and the manifest match a checked-in expected snapshot. |
| FE-TEST-06 | A determinism test shall run `run_feature_build` twice in succession against identical Phase 1 outputs and identical `feature_config.yaml` and assert byte-equality of both parquet files, the vocabulary JSON, and the manifest (excluding `build_timestamp_utc`). |
| FE-TEST-07 | A pinned-identity test shall run the real Phase 2 build and assert: (a) the `home_score` and `away_score` of one chosen game match the raw `HomeScore`/`AwayScore`; (b) `HomeOff01_madden_overall_rating` of the same game equals the `Overall Rating` of that game's Home Off slot 01 player's Madden row, joined by `madden_id`; (c) the same player's B-pos column (whichever canonical slot they fall into) carries the same Overall Rating. |
| FE-TEST-08 | A test shall verify that an FE-IN-04 hash mismatch (e.g., a hand-edited `Data/processed/madden_2024.csv`) causes the build to fail fast with a clear error. |
| FE-TEST-09 | A test shall verify that a B-pos slot with no fielded player has `{slot}_present = 0` and all other per-slot columns at NaN / `-1`. |
| FE-TEST-10 | A test shall verify that vocabulary entries are sorted lexicographically and that integer codes are stable across two independent builds of the same data. |

---

## 11. Security Considerations

| ID | Consideration |
|----|---------------|
| FE-SEC-01 | The build operates exclusively on local files under the repository root. It shall not make any network requests. |
| FE-SEC-02 | The build shall not write to any path outside `Data/processed/`. |
| FE-SEC-03 | No input or output of this build contains credentials, personally identifying information beyond publicly available player and official names, or other sensitive data. |
| FE-SEC-04 | `feature_config.yaml` is user-editable; the build validates its schema (§3.2) but is not required to defend against adversarial input — the only consumer is the build script in the same repository. YAML loading shall use `yaml.safe_load`. |

---

## 12. Future Considerations

The following are explicitly out of scope for Phase 2 v1 and recorded so they are not lost:

- **B-set (permutation-invariant) presentation**: A Phase 4 rung-4 attention model will likely want a permutation-invariant input. The cleanest add is a third parquet (`features_set_2024.parquet`) carrying 22 home-starter rows × N_cols and 22 away-starter rows × N_cols stacked per game, with a `game_id` foreign key. Specifying this requires committing to attention as the model shape.
- **Madden column widening**: After the first Phase 4 run, error analysis will indicate which Madden columns to add. Editing `feature_config.yaml` is the supported mechanism. If categorical column additions appear, append them to `madden_categorical_columns`.
- **Target encoding for high-cardinality categoricals**: Coaches (~64 names), officials (~125 names), stadiums (~30) are large enough that one-hot is wasteful. Target-encoded variants can be produced at training time using the training fold only; the integer codes in the sidecar are the input.
- **Days-of-rest refinement**: The week-1 sentinel of `14` is a placeholder. A future refinement may use a calendar-based "days since last season's last game" derivation. Such a change requires a `normalization_version` bump.
- **Position-taxonomy refinement**: The 8-bucket taxonomy in §4.3 is intentionally coarse. Splitting `OL` into `T`/`G`/`C` (or `DL` into `DE`/`DT`/`NT`) is a Phase 2 amendment with a `normalization_version` bump.
- **2025-season feature builds**: Once `box_scores_2025.csv` exists in `Data/processed/`, a `--season 2025` (or equivalent) flag produces `features_*_2025.parquet`. Multi-season support is a minor extension; the spec stays single-season for v1.
- **Phase 3+ specifications**: Splits, model ladder, evaluation, error analysis, and 2025-test specifications will be drafted in their own documents after Phase 2 is implemented.

---

## 13. References

- [Idea.md](./Idea.md) — Source idea document; §"Phase 2: Feature Engineering & Model Inputs" enumerates the open questions this spec resolves.
- [Spec-Phase1-DataBuild.md](./Spec-Phase1-DataBuild.md) — Phase 1 specification. Phase 2 reads its outputs and binds to its `build_manifest.json`.
- [Plan-Phase1-DataBuild.md](./Plan-Phase1-DataBuild.md) — Phase 1 implementation plan (for stylistic precedent on the Phase 2 plan that will follow).
- `Data/processed/madden_2024.csv` — Phase 1 output; primary Madden input to Phase 2.
- `Data/processed/box_scores_2024.csv` — Phase 1 output; primary box-score input to Phase 2.
- `Data/processed/build_manifest.json` — Phase 1 manifest; consulted for source-hash pinning per FE-IN-04.
- [CLAUDE.md](../CLAUDE.md) — Repository-level notes on venv, dataset shape, and join gotchas.
