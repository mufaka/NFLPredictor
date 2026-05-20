"""Regenerate the tiny multi-season integration-test fixture from real raw data.

Run from the repo root with the venv active:

    python -m tests.fixtures._regenerate

The script slices one game and its two teams' Madden rosters from each
of two seasons (2024 and 2025), then introduces targeted edits to the
2024 slice so the fixture exercises tier-3 (post-trade), tier-4 (fuzzy),
tier-1 (override), and unmatched paths in addition to the trivial tier-2
majority. The 2025 slice is left clean so the fixture also covers a
plain second season.

Outputs:
    tests/fixtures/raw_tiny/box_scores_{2024,2025}.csv
    tests/fixtures/raw_tiny/madden_{2024,2025}.csv
    tests/fixtures/raw_tiny/player_overrides.csv
    tests/fixtures/expected/{madden_all,box_scores_all,player_id_mapping}.csv
    tests/fixtures/expected/build_manifest.json (timestamp blanked)

The script is run on demand, not from pytest; check in the artifacts.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd

from nflpredictor.databuild.ids import assign_raw_madden_ids
from nflpredictor.databuild.pipeline import run_build
from nflpredictor.databuild.teams import normalize_team_code


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "Data" / "raw"
FIXTURE_DIR = pathlib.Path(__file__).parent
RAW_TINY = FIXTURE_DIR / "raw_tiny"
EXPECTED = FIXTURE_DIR / "expected"

# The 2024 game whose two teams' rosters are sliced and edited.
GAME_2024 = "202409050kan"

# Targets edited in the 2024 slice. Chosen because they're recognizable
# starters in this game.
TIER3_TARGET_NAME = "Marlon Humphrey"   # Ravens CB; move to "BUF" so tier 2 fails
TIER4_TARGET_NAME = "Roquan Smith"      # Ravens LB; misspell for a fuzzy hit
OVERRIDE_TARGET_NAME = "Travis Kelce"   # override onto Mahomes's row (tier-1 + position mismatch)


def _read_box(season: int) -> pd.DataFrame:
    return pd.read_csv(
        RAW_DIR / f"box_scores_{season}.csv", dtype=str, keep_default_na=False
    )


def _read_madden(season: int) -> pd.DataFrame:
    return pd.read_csv(
        RAW_DIR / f"madden_{season}.csv", dtype=str, keep_default_na=False
    )


def _write_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def _slice_season(season: int, game_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (one-game box scores, both teams' Madden rows) for a season."""
    bs = _read_box(season)
    game = bs[bs["GameId"] == game_id].copy().reset_index(drop=True)
    if len(game) != 1:
        raise SystemExit(f"expected exactly one row for game {game_id}")
    abbrevs = {
        normalize_team_code(game.iloc[0]["HomeTeamCode"]),
        normalize_team_code(game.iloc[0]["AwayTeamCode"]),
    }
    mad = _read_madden(season)
    roster = mad[mad["team"].isin(abbrevs)].copy().reset_index(drop=True)
    return game, roster


def build_fixture() -> None:
    # --- 2024 slice: edited to exercise every matching tier. ---
    bs_2024, mad_2024 = _slice_season(2024, GAME_2024)

    mask = mad_2024["fullname"] == TIER3_TARGET_NAME
    if not mask.any():
        raise SystemExit(f"tier-3 target {TIER3_TARGET_NAME!r} not in 2024 slice")
    mad_2024.loc[mask, "team"] = "BUF"

    mask = mad_2024["fullname"] == TIER4_TARGET_NAME
    if not mask.any():
        raise SystemExit(f"tier-4 target {TIER4_TARGET_NAME!r} not in 2024 slice")
    mad_2024.loc[mask, "fullname"] = "Rocquan Smith"

    # --- 2025 slice: a clean second season (first game of the file). ---
    first_game_2025 = _read_box(2025).iloc[0]["GameId"]
    bs_2025, mad_2025 = _slice_season(2025, first_game_2025)

    _write_csv(bs_2024, RAW_TINY / "box_scores_2024.csv")
    _write_csv(mad_2024, RAW_TINY / "madden_2024.csv")
    _write_csv(bs_2025, RAW_TINY / "box_scores_2025.csv")
    _write_csv(mad_2025, RAW_TINY / "madden_2025.csv")

    # Build a tier-1 override targeting Kelce in 2024. The override sends
    # Travis Kelce to Patrick Mahomes's Madden row, a deliberate QB-vs-TE
    # position mismatch so the fixture exercises DB-POS-02 too. The source
    # ``madden_id`` column is dropped first, mirroring ``load_raw_madden``.
    with_ids = assign_raw_madden_ids(mad_2024.drop(columns=["madden_id"]), 2024)
    mahomes = with_ids[
        (with_ids["team"] == "KC") & (with_ids["fullname"] == "Patrick Mahomes")
    ]
    if mahomes.empty:
        raise SystemExit("Mahomes missing from 2024 Madden slice")
    forced_madden_id = mahomes.iloc[0]["madden_id"]

    overrides_path = RAW_TINY / "player_overrides.csv"
    overrides_path.write_text(
        "season,box_score_name,box_score_team_code,box_score_id,madden_id,reason\n"
        f"2024,{OVERRIDE_TARGET_NAME},kan,,{forced_madden_id},"
        "fixture: force tier-1 path\n",
        encoding="utf-8",
    )
    print(f"override sends Kelce -> {forced_madden_id} (Mahomes's 2024 row)")

    # Run the build and capture outputs.
    EXPECTED.mkdir(parents=True, exist_ok=True)
    run_build(RAW_TINY, EXPECTED)

    # Blank the timestamp in the expected manifest so the checked-in copy
    # doesn't drift between regenerations.
    manifest_path = EXPECTED / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["build_timestamp_utc"] = "<blanked-by-regenerate>"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote fixtures under {RAW_TINY} and {EXPECTED}")


if __name__ == "__main__":
    build_fixture()
