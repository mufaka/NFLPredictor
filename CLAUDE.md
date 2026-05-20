# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Always work inside the project's virtual environment.** Activate with `source .venv/bin/activate` before running `python`, `pip`, or `pytest`. The venv lives at `.venv/` and is gitignored.

## Project status

The pipeline has been migrated from a single 2024 season to **six seasons, 2020–2025**. `Docs/Plan-MultiYear-Migration.md` is the cross-phase roadmap and records the locked decisions (full multi-year, synthetic `madden_id`, combined output files, season-holdout splits). Phases 1–5 are implemented for multi-year; Phase 6's diagnostics module is migrated, but its prose docs and the two notebooks still need a refresh (see Phase 6 below).

Run each phase from the activated venv:

```bash
python -m nflpredictor.databuild   # Phase 1
python -m nflpredictor.features    # Phase 2
python -m nflpredictor.splits      # Phase 3
python -m nflpredictor.train       # Phase 4
python -m nflpredictor.evaluate    # Phase 5
```

Each phase SHA-verifies the upstream phase's tracked outputs against its manifest before running, so drift fails fast. Re-running any phase on identical inputs produces byte-identical outputs (modulo manifest timestamps). Run the test suite with `pytest -q`.

**Phase 1 (Data Build)** reads the 13 raw files (`box_scores_{2020..2025}.csv`, `madden_{2020..2025}.csv`, `player_overrides.csv`) and emits combined files into `Data/processed/`:
- `madden_all.csv` — six seasons of Madden rows; `madden_id` (first column, `YYYY-NNNNN`, per-season) + 55 source columns incl. `season` + `matched` (last). The raw files' own `madden_id` column is dropped on read. Unmatched-starter rows have `matched=0` and per-season null-filled ratings.
- `box_scores_all.csv` — all 1,622 games, a prepended `season` column, every per-slot `_ID` rewritten to a `madden_id`.
- `player_id_mapping.csv` — one row per `(season, box_score_id, madden_id)` with the resolving tier note.
- `build_manifest.json` — SHA-256 of the 13 inputs + 3 outputs; `normalization_version`; `total` + `by_season` match counts.

Matching is season-scoped (a 2020 starter only matches 2020 Madden rows). `teams.py` maps PFR codes to modern abbreviations (`kan`→`KC`). A guard (DB-IN-07) rejects a `box_scores_<YYYY>.csv` containing a game from another season.

**Phase 2 (Feature Engineering)** emits `features_flat_all.parquet` (1,622 × 502) and `features_pos_all.parquet` (1,622 × 656), each beginning with `GameId` then a `season` identifier column (a split key, never a model feature), plus `feature_vocab.json` (`vocab_version: "v3"`) and `feature_manifest.json`. The shipped `feature_config.yaml` pulls a player-attribute Madden set — `overallrating` plus the athletic ratings (`agility`, `acceleration`, `speed`, `stamina`, `strength`, `toughness`, `awareness`) — with `position` as the lone categorical (`archetype` was dropped). `officials` and coaches are not emitted. Week numbers are derived per season from a six-season Week-1-anchor calendar; days-of-rest is per season.

**Phase 3 (Splits)** emits `splits_all.json` + `splits_manifest.json`. The week-based strategies are gone; splitting is **season-holdout**. The default `splits_config.yaml` partitions by season: `season_holdout` = train 2020–2023 (1,077 games) / val 2024 (272) / test 2025 (273). `loso_cv` (leave-one-season-out CV) is available but opt-in. `splits_version: "v2"`.

**Phase 4 (Baseline & Model Ladder)** emits per-`(rung, shape, strategy)` prediction parquets named `<rung>__<shape>__<season_holdout|loso_cv>.parquet` + `training_manifest.json` + `training_loss_curves.parquet`. The default config (`strategies: [season_holdout]`) produces 6 prediction parquets — 2 trivial rungs (mean, team_mean) + 4 learned (linear/mlp × flat/pos). `training_version: "v4"`. Device-aware, single-device, per-device byte determinism. The encoder excludes the `GameId`/`season` identifier columns and the identity-only categoricals (`NON_MODEL_VOCAB_KEYS` in `train/encoders.py` = team codes, coaches, officials) from the model input: the premise is that a team is its players' attributes, so raw team/coach/official identity is deliberately not a learned feature. Of those, only team codes are still emitted by the shipped `feature_config.yaml` — they stay in the Phase 2 parquet (the `team_mean` baseline and the Phase 5 by-team breakdown read them) but never reach the model.

**Phase 5 (Evaluation)** emits `evaluation/metrics_headline.json` + 5 breakdown parquets + plot PNGs + `evaluation_manifest.json`. Strategies are `season_holdout` (val/test slices) and `loso_cv` (per-fold slices). On the real data the held-out **2025 test MAE** is best for the trivial `mean` baseline (~7.9) — the learned rungs do not yet beat it, which the season-holdout split exposes honestly.

**Phase 6 (Documentation & Diagnostics)** — the diagnostics module `src/nflpredictor/diagnostics/` (`trace.py`, `encoding.py`, `loss_curves.py`) is migrated to the multi-year contract: season-aware `load_raw_game`, `season_holdout`/`loso_cv` split-membership and prediction lookups, `*_all` basenames. `tests/test_diagnostics.py` and `tests/test_loss_curves.py` pass. **Still pending**: the Phase 6 prose docs (`Docs/Phase6-*.md`) and the two `notebooks/phase6_*.ipynb` notebooks were authored against the single-season world and need re-execution / rewording for multi-year.

The package lives under `src/nflpredictor/` (`databuild/`, `features/`, `splits/`, `train/`, `evaluate/`, `diagnostics/`). Phase specs/plans are `Docs/{Spec,Plan}-Phase{1..6}-*.md`; Spec-Phase1–5 carry multi-year revision notes. `Docs/ConfigReference.md` documents the four `Data/raw/` YAML files (it predates the migration — verify against the shipped configs).

## Datasets

`Data/raw/` holds 13 raw inputs: six `box_scores_<YYYY>.csv` (2020–2025), six `madden_<YYYY>.csv`, and `player_overrides.csv`. The old `maddennfl24fullplayerratings.csv` is the superseded single-season Madden source — no longer used by the build.

### `box_scores_<YYYY>.csv` (≈270 games each, 164 columns)
One row per NFL regular-season game. All six share one schema (only `box_scores_2024.csv` uses CRLF line endings; the build tolerates both).
- **Game metadata**: `GameId` (e.g. `202409050kan` — date + home team code), `GameDate`, `DayOfWeek`, `StartTime`, `HomeTeam`/`AwayTeam`, `HomeTeamCode`/`AwayTeamCode` (PFR 3-letter codes like `kan`, `rav`), `HomeScore`/`AwayScore`, `HomeCoach`/`AwayCoach`.
- **Venue/conditions**: `Stadium`, `Attendance`, `Duration`, `Roof`, `Surface`, `Weather` (free-text — may be empty for domes).
- **Starting lineups**: 11 slots per team/side with `_Position`, `_Name`, `_ID` columns (`HomeOff01_*` … `AwayDef11_*`). The `_ID` is a Pro-Football-Reference player code (e.g. `MahoPa00`) — occasionally empty.
- **Officials**: `Official01_Role`/`_Name` through `Official07_*`.

### `madden_<YYYY>.csv` (≈2,300 players each, 56 columns)
One row per player. Lowercase headers, no whitespace. `team` uses modern abbreviations (`KC`, `BAL`). `fullname` is the join handle to box-score names — no shared player ID, so name normalization matters. Carries a `season` column and a source `madden_id` (a non-unique `NAME_POSGROUP` string the build drops). Ratings 0–99. Some columns are entirely empty in some seasons (`midrouterunning`, `birthdate` in 2021–2023, `yearspro` in 2025) — the build leaves them unfilled (DB-FILL-06).

## Conventions for new work

- Treat `Docs/Overview.md` as the canonical place to capture the project's goals and approach as they crystallize — update it rather than spawning parallel design docs.
- The `.claude/settings.local.json` permission allowlist carries entries inherited from another project (`Nickel.SaaS`). Those are harmless but not signal about this project's stack.
