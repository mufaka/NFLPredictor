"""Identity regression tests against the real build output (DB-TEST-04).

These pin specific real-world player resolutions. If normalization or
matching changes shift any of these, a passing-tests CI signal will
become a failing one and the developer must look.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from nflpredictor.databuild.pipeline import run_build
from nflpredictor.databuild.unmatched import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "Data" / "raw"


@pytest.fixture(scope="module")
def real_outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("real")
    run_build(RAW_DIR, out)
    madden = pd.read_csv(out / "madden_all.csv", dtype=str, keep_default_na=False)
    box = pd.read_csv(out / "box_scores_all.csv", dtype=str, keep_default_na=False)
    mapping = pd.read_csv(out / "player_id_mapping.csv", dtype=str, keep_default_na=False)
    return madden, box, mapping


# IDs captured from a clean run on the current raw data. If any of these
# shift, investigate whether normalization or matching changed by accident.
EXPECTED_MAHOMES_2024_ID = "2024-01131"


def test_mahomes_madden_id(real_outputs):
    madden, _, _ = real_outputs
    rows = madden[
        (madden["team"] == "KC")
        & (madden["fullname"] == "Patrick Mahomes")
        & (madden["season"] == "2024")
    ]
    assert len(rows) == 1
    assert rows.iloc[0]["madden_id"] == EXPECTED_MAHOMES_2024_ID
    assert rows.iloc[0]["matched"] == "1"


def test_mahomes_id_is_season_prefixed_in_every_season(real_outputs):
    """Mahomes appears once per season, each with a season-prefixed id."""
    madden, _, _ = real_outputs
    rows = madden[(madden["team"] == "KC") & (madden["fullname"] == "Patrick Mahomes")]
    seasons = set(rows["season"])
    assert seasons == {"2020", "2021", "2022", "2023", "2024", "2025"}
    for _, row in rows.iterrows():
        assert row["madden_id"].startswith(f"{row['season']}-")


def test_tj_watt_id_consistent_within_each_season(real_outputs):
    """T.J. Watt may be referenced via a blank PFR id in some games; the
    resolved madden_id should be the same across every Steelers game he
    appears in within a season."""
    madden, box, _ = real_outputs
    watt_rows = madden[(madden["team"] == "PIT") & (madden["fullname"] == "T.J. Watt")]
    assert len(watt_rows) >= 1
    id_by_season = dict(zip(watt_rows["season"], watt_rows["madden_id"]))

    def_prefixes = [
        f"{side}Def{n:02d}" for side in ("Home", "Away") for n in range(1, 12)
    ]
    found = 0
    for _, row in box.iterrows():
        season = row["season"]
        for prefix in def_prefixes:
            if row[f"{prefix}_Name"] == "T.J. Watt":
                assert row[f"{prefix}_ID"] == id_by_season.get(season), (
                    f"T.J. Watt in {row['GameId']} resolved to "
                    f"{row[f'{prefix}_ID']}, expected {id_by_season.get(season)}"
                )
                found += 1
    assert found > 0, "T.J. Watt should appear as a starter at least once"


def test_every_box_score_id_resolves_to_a_madden_row(real_outputs):
    """Every per-slot `_ID` in the processed box scores must correspond to a
    real row in the processed Madden file (DB-OUT-02 + DB-OUT-03)."""
    madden, box, _ = real_outputs
    known = set(madden["madden_id"])
    id_cols = [
        c for c in box.columns
        if c.endswith("_ID") and any(
            c.startswith(p) for p in ("HomeOff", "HomeDef", "AwayOff", "AwayDef")
        )
    ]
    for col in id_cols:
        missing = set(box[col]) - known
        assert not missing, f"{col}: {missing} not in Madden file"


def test_no_unexpected_nulls_on_matched_zero_rows(real_outputs):
    """DB-OUT-13: after null-fill, no `matched=0` row has an empty cell, except
    columns skipped per DB-FILL-06 (entirely empty in a season) and the
    identity columns fullname/position when the source box-score slot was
    itself blank."""
    madden, _, _ = real_outputs
    appended = madden[madden["matched"] == "0"]
    assert len(appended) > 0, "real build should produce at least one appended row"
    allowed_blank = {"midrouterunning", "birthdate", "yearspro", "fullname", "position"}
    for col in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS:
        if col in allowed_blank:
            continue
        blanks = (appended[col].str.strip() == "").sum()
        assert blanks == 0, f"matched=0 rows have {blanks} blank cells in {col!r}"
