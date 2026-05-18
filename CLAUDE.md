# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Always work inside the project's virtual environment.** Activate with `source .venv/bin/activate` before running `python`, `pip`, or `pytest`. The venv lives at `.venv/` and is gitignored.

## Project status

Phase 1 (Data Build) is implemented. Run the build with:

```bash
source .venv/bin/activate
python -m nflpredictor.databuild
```

This reads `Data/raw/{box_scores_2024.csv, maddennfl24fullplayerratings.csv, player_overrides.csv}` and emits four files into `Data/processed/`:

- `madden_2024.csv` — Madden roster with `madden_id` (first column) + `matched` (last column); appended unmatched-starter rows have `matched=0` and null-filled ratings.
- `box_scores_2024.csv` — same shape as raw, but every per-slot `_ID` column now carries a `madden_id` (no blanks).
- `player_id_mapping.csv` — one row per unique `(box_score_id, madden_id)` pair with the tier note that resolved it.
- `build_manifest.json` — SHA-256 hashes of inputs and outputs, normalization version, git commit, and per-tier match counts.

Re-running the build on identical inputs produces byte-identical CSVs (`tests/test_determinism.py` enforces this). Run the test suite with `pytest -q` from the activated venv.

Phase 2 (Feature Engineering) is in progress. The feature build is invoked with:

```bash
source .venv/bin/activate
python -m nflpredictor.features
```

It reads Phase 1's processed outputs plus `Data/raw/feature_config.yaml` and will emit `features_flat_2024.parquet`, `features_pos_2024.parquet`, `feature_vocab.json`, and `feature_manifest.json` into `Data/processed/`. The shipped `feature_config.yaml` is the knob for expanding the feature inventory — adding Madden columns or toggling weather/officials is a YAML edit, not a code change.

The package lives under `src/nflpredictor/`; the data-build module is `src/nflpredictor/databuild/` and the feature module is `src/nflpredictor/features/`. The phase plans and specs are in `Docs/Plan-Phase1-DataBuild.md`, `Docs/Spec-Phase1-DataBuild.md`, `Docs/Plan-Phase2-FeatureEngineering.md`, and `Docs/Spec-Phase2-FeatureEngineering.md`.

## Datasets

Both CSVs live in `Data/raw/` and are the input for whatever modeling work follows.

### `box_scores_2024.csv` (272 games, 164 columns)
One row per NFL 2024 regular-season game. Key column families:
- **Game metadata**: `GameId` (e.g. `202409050kan` — date + home team code), `GameDate`, `DayOfWeek`, `StartTime`, `HomeTeam`/`AwayTeam` (full names) and `HomeTeamCode`/`AwayTeamCode` (3-letter codes like `kan`, `rav`), `HomeScore`/`AwayScore`, `HomeCoach`/`AwayCoach`.
- **Venue/conditions**: `Stadium`, `Attendance`, `Duration`, `Roof`, `Surface`, `Weather` (free-text — may be empty for domes).
- **Starting lineups**: For each team and side of the ball, 11 slots numbered `01`–`11` with `_Position`, `_Name`, `_ID` columns. Naming pattern: `HomeOff01_Position`, `HomeOff01_Name`, `HomeOff01_ID`, …, `HomeDef11_*`, `AwayOff*_*`, `AwayDef*_*`. The `_ID` is a Pro-Football-Reference style player code (e.g. `MahoPa00`) — occasionally empty.
- **Officials**: `Official01_Role`/`_Name` through `Official07_*`.

### `maddennfl24fullplayerratings.csv` (2,368 players, 69 columns)
One row per player from Madden NFL 24. `Team` uses team nicknames (e.g. `49ers`, not the PFR code) — joining to box scores requires a team-name mapping. `Full Name` is the join handle to the box-score lineup names; there is no shared player ID, so name normalization (Jr./Sr., punctuation, accents) will matter. Ratings are 0–99 across general attributes (Speed, Awareness, …) and position-specific skills (Throw Accuracy Short/Mid/Deep, Man/Zone Coverage, etc.). Several columns have leading/trailing spaces in the header (e.g. ` Total Salary `, ` Signing Bonus `) — keep that in mind when reading the CSV.

## Conventions for new work

- Treat `Docs/Overview.md` as the canonical place to capture the project's goals and approach as they crystallize — update it rather than spawning parallel design docs.
- The `.claude/settings.local.json` permission allowlist carries entries inherited from another project (`Nickel.SaaS`). Those are harmless but not signal about this project's stack.
