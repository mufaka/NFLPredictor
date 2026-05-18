"""Negative-path integration tests (DB-TEST-07 + the broken-override case)."""

from __future__ import annotations

import pathlib
import shutil

import pytest

from nflpredictor.databuild.pipeline import run_build


FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures"
RAW_TINY = FIXTURE_DIR / "raw_tiny"


def test_override_referencing_unknown_madden_id_fails(tmp_path: pathlib.Path):
    """DB-OVR-04 / DB-TEST-07: override targeting an unassigned madden_id raises."""
    raw_copy = tmp_path / "raw"
    raw_copy.mkdir()
    shutil.copy(RAW_TINY / "box_scores_2024.csv", raw_copy / "box_scores_2024.csv")
    shutil.copy(
        RAW_TINY / "maddennfl24fullplayerratings.csv",
        raw_copy / "maddennfl24fullplayerratings.csv",
    )
    (raw_copy / "player_overrides.csv").write_text(
        "box_score_name,box_score_team_code,box_score_id,madden_id,reason\n"
        "Patrick Mahomes,kan,,2024-99999,points at a nonexistent row\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="DB-OVR-04"):
        run_build(raw_copy, out)
