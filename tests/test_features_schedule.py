"""Tests for the multi-season NFL week calendar (§3.4 / FE-GAME-02)."""

from __future__ import annotations

import pathlib
from datetime import date

import pandas as pd
import pytest

from nflpredictor.features.schedule import SEASON_WEEK1_THURSDAY, week_for_date


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BOX_SCORES = REPO_ROOT / "Data" / "processed" / "box_scores_all.csv"


def test_all_six_seasons_have_anchors():
    assert sorted(SEASON_WEEK1_THURSDAY) == [2020, 2021, 2022, 2023, 2024, 2025]
    for season, anchor in SEASON_WEEK1_THURSDAY.items():
        assert anchor.weekday() == 3, f"{season} anchor must be a Thursday"


def test_known_dates_map_correctly():
    cases = [
        (date(2024, 9, 5), 2024, 1),    # Thu opener (Chiefs/Ravens)
        (date(2024, 9, 8), 2024, 1),    # Sunday games of Week 1
        (date(2024, 9, 11), 2024, 1),   # Last day of Week 1 (Wed)
        (date(2024, 9, 12), 2024, 2),   # First day of Week 2 (Thu)
        (date(2024, 11, 28), 2024, 13), # Thanksgiving
        (date(2020, 9, 10), 2020, 1),   # 2020 Thu opener
        (date(2025, 9, 4), 2025, 1),    # 2025 Thu opener
    ]
    for d, season, expected in cases:
        assert week_for_date(d, season) == expected


def test_date_before_season_raises():
    with pytest.raises(ValueError, match="before season"):
        week_for_date(date(2024, 9, 4), 2024)


def test_unknown_season_raises():
    with pytest.raises(ValueError, match="no Week-1 anchor"):
        week_for_date(date(2024, 9, 8), 2019)


def test_every_real_game_date_maps_to_a_week():
    df = pd.read_csv(BOX_SCORES, dtype=str, keep_default_na=False)
    for _, row in df.iterrows():
        d = pd.to_datetime(row["GameDate"]).date()
        week = week_for_date(d, int(row["season"]))
        assert week >= 1, f"{d} resolved to invalid week {week}"
