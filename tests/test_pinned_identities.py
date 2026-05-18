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


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "Data" / "raw"


@pytest.fixture(scope="module")
def real_outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("real")
    run_build(RAW_DIR, out)
    madden = pd.read_csv(out / "madden_2024.csv", dtype=str, keep_default_na=False)
    box = pd.read_csv(out / "box_scores_2024.csv", dtype=str, keep_default_na=False)
    mapping = pd.read_csv(out / "player_id_mapping.csv", dtype=str, keep_default_na=False)
    return madden, box, mapping


# IDs captured from a clean run on the current raw data. If any of these
# shift, investigate whether normalization or matching changed by accident.
EXPECTED_MAHOMES_ID = "2024-00709"


def test_mahomes_madden_id(real_outputs):
    madden, _, _ = real_outputs
    rows = madden[
        (madden["Team"] == "Chiefs") & (madden["Full Name"] == "Patrick Mahomes")
    ]
    assert len(rows) == 1
    assert rows.iloc[0]["madden_id"] == EXPECTED_MAHOMES_ID
    assert rows.iloc[0]["matched"] == "1"


def test_mahomes_resolves_in_every_chiefs_game(real_outputs):
    """Mahomes's box-score slot in every Chiefs home game should carry his id."""
    _, box, _ = real_outputs
    chiefs_home = box[box["HomeTeamCode"] == "kan"]
    for _, row in chiefs_home.iterrows():
        if row["HomeOff01_Name"] == "Patrick Mahomes":
            assert row["HomeOff01_ID"] == EXPECTED_MAHOMES_ID, (
                f"Mahomes in {row['GameId']} resolved to {row['HomeOff01_ID']}"
            )


def test_tj_watt_id_consistent_across_appearances(real_outputs):
    """T.J. Watt may be referenced via a blank PFR id in some games; the
    resolved madden_id should be the same across every Steelers game he
    appears in."""
    madden, box, _ = real_outputs
    watt_rows = madden[
        (madden["Team"] == "Steelers") & (madden["Full Name"] == "T.J. Watt")
    ]
    assert len(watt_rows) == 1, "T.J. Watt should be a single Steelers row"
    expected_id = watt_rows.iloc[0]["madden_id"]

    found = 0
    for _, row in box.iterrows():
        for prefix in (
            "HomeDef01", "HomeDef02", "HomeDef03", "HomeDef04",
            "HomeDef05", "HomeDef06", "HomeDef07", "HomeDef08",
            "HomeDef09", "HomeDef10", "HomeDef11",
            "AwayDef01", "AwayDef02", "AwayDef03", "AwayDef04",
            "AwayDef05", "AwayDef06", "AwayDef07", "AwayDef08",
            "AwayDef09", "AwayDef10", "AwayDef11",
        ):
            if row[f"{prefix}_Name"] == "T.J. Watt":
                assert row[f"{prefix}_ID"] == expected_id, (
                    f"T.J. Watt in {row['GameId']} resolved to "
                    f"{row[f'{prefix}_ID']}, expected {expected_id}"
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
        seen = set(box[col])
        missing = seen - known
        assert not missing, f"{col}: {missing} not in Madden file"


def test_no_nulls_on_matched_zero_rows(real_outputs):
    """DB-OUT-13 / DB-TEST-08: after null-fill, no `matched=0` row has any empty cell."""
    madden, _, _ = real_outputs
    appended = madden[madden["matched"] == "0"]
    assert len(appended) > 0, "real build should produce at least one appended row"
    has_blank = (appended == "").any(axis=None)
    assert not has_blank, "matched=0 row contains a blank cell"
