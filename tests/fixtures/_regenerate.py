"""Regenerate the tiny integration-test fixture from real raw data.

Run from the repo root with the venv active:

    python -m tests.fixtures._regenerate

The script slices the 2024-09-05 Chiefs/Ravens opener and the Madden
rows for both teams, then introduces three targeted edits so the
fixture exercises tier-3 (post-trade), tier-4 (fuzzy), and unmatched
paths in addition to the trivial tier-2 majority. It also writes an
override file that forces tier-1 for one starter.

Outputs:
    tests/fixtures/raw_tiny/box_scores_2024.csv
    tests/fixtures/raw_tiny/maddennfl24fullplayerratings.csv
    tests/fixtures/raw_tiny/player_overrides.csv
    tests/fixtures/expected/{madden_2024,box_scores_2024,player_id_mapping}.csv
    tests/fixtures/expected/build_manifest.json (timestamp blanked)

The script is run on demand, not from pytest; check in the artifacts.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd

from nflpredictor.databuild.pipeline import run_build


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "Data" / "raw"
FIXTURE_DIR = pathlib.Path(__file__).parent
RAW_TINY = FIXTURE_DIR / "raw_tiny"
EXPECTED = FIXTURE_DIR / "expected"


# Targets we modify in the fixture. Chosen because they're recognizable
# starters in this game.
#
# JuJu Smith-Schuster is naturally unmatched in this slice — Madden 24
# lists him on the Patriots, so he doesn't appear in the Chiefs/Ravens
# slice at all and falls through to the appended-row path organically.
TIER3_TARGET_NAME = "Marlon Humphrey"     # Ravens CB starter; move to "Bills" so tier 2 fails
TIER4_TARGET_NAME = "Roquan Smith"        # Ravens LB starter; misspell to "Rocquan Smith" for fuzzy hit
OVERRIDE_TARGET_NAME = "Travis Kelce"     # override onto Mahomes's row to test tier-1 + position mismatch


def _read_raw_box_scores() -> pd.DataFrame:
    return pd.read_csv(
        RAW_DIR / "box_scores_2024.csv", dtype=str, keep_default_na=False
    )


def _read_raw_madden() -> pd.DataFrame:
    df = pd.read_csv(
        RAW_DIR / "maddennfl24fullplayerratings.csv",
        dtype=str,
        keep_default_na=False,
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def _write_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def build_fixture() -> None:
    bs = _read_raw_box_scores()
    mad = _read_raw_madden()

    tiny_bs = bs[bs["GameId"] == "202409050kan"].copy().reset_index(drop=True)
    if len(tiny_bs) != 1:
        raise SystemExit("expected exactly one row for game 202409050kan")

    tiny_mad = mad[mad["Team"].isin(["Chiefs", "Ravens"])].copy().reset_index(drop=True)

    # Tier-3: move the Madden record to a different team so tier-2 fails
    # but the box-score normalized name remains unique league-wide.
    mask = tiny_mad["Full Name"] == TIER3_TARGET_NAME
    if not mask.any():
        raise SystemExit(f"tier-3 target {TIER3_TARGET_NAME!r} not in Madden slice")
    tiny_mad.loc[mask, "Team"] = "Bills"

    # Tier-4: misspell the Madden name so tier-2/3 fail but fuzzy succeeds within the team.
    mask = tiny_mad["Full Name"] == TIER4_TARGET_NAME
    if not mask.any():
        raise SystemExit(f"tier-4 target {TIER4_TARGET_NAME!r} not in Madden slice")
    tiny_mad.loc[mask, "Full Name"] = "Rocquan Smith"

    _write_csv(tiny_bs, RAW_TINY / "box_scores_2024.csv")
    _write_csv(tiny_mad, RAW_TINY / "maddennfl24fullplayerratings.csv")

    # Build a tier-1 override targeting Kelce. We compute the Madden id by
    # re-running the deterministic ID assignment over the fixture roster.
    from nflpredictor.databuild.ids import assign_raw_madden_ids
    from nflpredictor.databuild.normalization import normalize_name
    with_ids = assign_raw_madden_ids(tiny_mad)
    override_target = with_ids[
        (with_ids["Team"] == "Chiefs") & (with_ids["Full Name"] == "Travis Kelce")
    ]
    if override_target.empty:
        raise SystemExit("override target not present after id assignment")
    override_madden_id = override_target.iloc[0]["madden_id"]

    # The override sends Travis Kelce to Patrick Mahomes's Madden row, which
    # creates a deliberate position mismatch (QB vs TE) so the test exercises
    # DB-POS-02 too.
    mahomes_row = with_ids[
        (with_ids["Team"] == "Chiefs") & (with_ids["Full Name"] == "Patrick Mahomes")
    ]
    if mahomes_row.empty:
        raise SystemExit("Mahomes missing from Madden slice")
    forced_madden_id = mahomes_row.iloc[0]["madden_id"]

    overrides_path = RAW_TINY / "player_overrides.csv"
    overrides_path.write_text(
        "box_score_name,box_score_team_code,box_score_id,madden_id,reason\n"
        f"Travis Kelce,kan,,{forced_madden_id},fixture: force tier-1 path\n",
        encoding="utf-8",
    )
    print(f"override sends Kelce -> {forced_madden_id} (Mahomes's row)")

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
