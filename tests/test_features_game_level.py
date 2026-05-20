"""Tests for game-level feature derivation (§3.4 / FE-TEST-04)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from nflpredictor.features.game_level import (
    assemble_game_level,
    compute_days_rest,
    parse_start_hour,
)


def test_start_hour_examples():
    assert parse_start_hour("9:30am") == 9
    assert parse_start_hour("12:30pm") == 12
    assert parse_start_hour("8:20pm") == 20
    assert parse_start_hour("1:00pm") == 13
    assert parse_start_hour("12:00am") == 0


def test_start_hour_strips_whitespace():
    assert parse_start_hour("  4:25pm  ") == 16


def test_start_hour_raises_on_bad_input():
    with pytest.raises(ValueError, match="unparseable StartTime"):
        parse_start_hour("noon")


def test_days_rest_first_game_is_nan():
    """A team's first game of the season produces NaN per FE-GAME-08."""
    df = pd.DataFrame({
        "GameId": ["g1"],
        "season": ["2024"],
        "GameDate": ["2024-09-08"],
        "HomeTeamCode": ["kan"],
        "AwayTeamCode": ["rav"],
    })
    out = compute_days_rest(df)
    assert math.isnan(out.loc[0, "days_rest_home"])
    assert math.isnan(out.loc[0, "days_rest_away"])


def test_days_rest_sunday_to_sunday_is_seven():
    df = pd.DataFrame({
        "GameId": ["g1", "g2"],
        "season": ["2024", "2024"],
        "GameDate": ["2024-09-08", "2024-09-15"],
        "HomeTeamCode": ["kan", "kan"],
        "AwayTeamCode": ["rav", "buf"],
    })
    out = compute_days_rest(df)
    assert math.isnan(out.loc[0, "days_rest_home"])
    assert out.loc[1, "days_rest_home"] == 7.0
    # Buffalo is on its first game in game 2 → NaN.
    assert math.isnan(out.loc[1, "days_rest_away"])


def test_days_rest_sunday_to_thursday_is_four():
    df = pd.DataFrame({
        "GameId": ["g1", "g2"],
        "season": ["2024", "2024"],
        "GameDate": ["2024-09-08", "2024-09-12"],
        "HomeTeamCode": ["kan", "buf"],
        "AwayTeamCode": ["rav", "kan"],
    })
    out = compute_days_rest(df)
    # Kansas City plays both games: Sun-then-Thu = 4 days.
    assert out.loc[1, "days_rest_away"] == 4.0


def test_days_rest_dtype_is_float64():
    df = pd.DataFrame({
        "GameId": ["g1"],
        "season": ["2024"],
        "GameDate": ["2024-09-08"],
        "HomeTeamCode": ["kan"],
        "AwayTeamCode": ["rav"],
    })
    out = compute_days_rest(df)
    assert out["days_rest_home"].dtype == "float64"
    assert out["days_rest_away"].dtype == "float64"


def _three_game_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "GameId": ["g1", "g2", "g3"],
        "season": ["2024", "2024", "2024"],
        "GameDate": ["2024-09-08", "2024-09-15", "2024-09-22"],
        "DayOfWeek": ["Sunday", "Sunday", "Sunday"],
        "StartTime": ["1:00pm", "4:25pm", "8:20pm"],
        "Stadium": ["Arrowhead Stadium", "M&T Bank Stadium", "Highmark Stadium"],
        "Roof": ["outdoors", "outdoors", "outdoors"],
        "Surface": ["grass", "grass", "fieldturf"],
        "HomeTeamCode": ["kan", "rav", "buf"],
        "AwayTeamCode": ["rav", "buf", "kan"],
        "HomeCoach": ["Andy Reid", "John Harbaugh", "Sean McDermott"],
        "AwayCoach": ["John Harbaugh", "Sean McDermott", "Andy Reid"],
    })


def test_assemble_game_level_respects_include_order():
    df = _three_game_frame()
    include = ("start_hour", "week", "day_of_week", "home_team_code")
    out = assemble_game_level(df, include)
    assert list(out.columns) == ["GameId", "start_hour", "week", "day_of_week", "home_team_code"]


def test_assemble_game_level_full_default_set():
    df = _three_game_frame()
    include = (
        "week",
        "day_of_week",
        "start_hour",
        "stadium",
        "roof",
        "surface",
        "home_team_code",
        "away_team_code",
        "home_coach",
        "away_coach",
        "days_rest_home",
        "days_rest_away",
    )
    out = assemble_game_level(df, include)
    assert list(out.columns) == ["GameId", *include]
    # Week 1 -> 1, Week 2 -> 2, Week 3 -> 3.
    assert out["week"].tolist() == [1, 2, 3]
    # Day-of-week carries full names verbatim.
    assert out["day_of_week"].tolist() == ["Sunday", "Sunday", "Sunday"]
    # Start hours.
    assert out["start_hour"].tolist() == [13, 16, 20]
    # Pass-throughs.
    assert out["home_team_code"].tolist() == ["kan", "rav", "buf"]
    assert out["home_coach"].tolist() == ["Andy Reid", "John Harbaugh", "Sean McDermott"]
    # Days-of-rest: g1 first game (NaN); g2 home (rav) was away in g1 → 7; g2 away (buf) first → NaN.
    assert math.isnan(out.loc[0, "days_rest_home"])
    assert out.loc[1, "days_rest_home"] == 7.0


def test_assemble_game_level_omits_days_rest_when_not_requested():
    df = _three_game_frame()
    include = ("week", "stadium")  # no days_rest_*
    out = assemble_game_level(df, include)
    assert "days_rest_home" not in out.columns
    assert "days_rest_away" not in out.columns
