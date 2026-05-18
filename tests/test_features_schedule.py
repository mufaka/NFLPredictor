"""Tests for the 2024 NFL week calendar (§3.4 / FE-GAME-02)."""

from __future__ import annotations

import pathlib
from datetime import date

import pandas as pd
import pytest

from nflpredictor.features.schedule import (
    NFL_2024_WEEK_BOUNDARIES,
    week_for_date,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BOX_SCORES = REPO_ROOT / "Data" / "processed" / "box_scores_2024.csv"


def test_eighteen_weeks_covered():
    weeks = [w for _, _, w in NFL_2024_WEEK_BOUNDARIES]
    assert weeks == list(range(1, 19))


def test_known_dates_map_correctly():
    cases = [
        (date(2024, 9, 5),  1),   # Thu opener (Chiefs/Ravens)
        (date(2024, 9, 8),  1),   # Sunday games of Week 1
        (date(2024, 9, 11), 1),   # Last day of Week 1 (Wed)
        (date(2024, 9, 12), 2),   # First day of Week 2 (Thu)
        (date(2024, 11, 28), 13), # Thanksgiving
        (date(2024, 12, 25), 16), # Christmas
        (date(2025, 1, 5),  18),  # Final Sunday
    ]
    for d, expected in cases:
        assert week_for_date(d) == expected, f"{d} should be Week {expected}"


def test_date_before_season_raises():
    with pytest.raises(ValueError, match="outside the 2024 NFL regular-season window"):
        week_for_date(date(2024, 9, 4))


def test_date_after_season_raises():
    with pytest.raises(ValueError, match="outside the 2024 NFL regular-season window"):
        week_for_date(date(2025, 1, 9))


def test_every_real_game_date_maps_to_a_week():
    df = pd.read_csv(BOX_SCORES, dtype=str)
    distinct_dates = sorted({pd.to_datetime(d).date() for d in df["GameDate"]})
    for d in distinct_dates:
        week = week_for_date(d)
        assert 1 <= week <= 18, f"{d} resolved to invalid week {week}"
