"""Tests for the weather string parser (§3.5 / FE-TEST-02)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from nflpredictor.features.weather import (
    parse_weather,
    parse_weather_column,
)


def test_typical_string():
    temp, hum, wind, indoor = parse_weather(
        "67 degrees, relative humidity 53%, wind 8 mph", "outdoors"
    )
    assert temp == 67.0
    assert hum == 53.0
    assert wind == 8.0
    assert indoor == 0


def test_calm_day_string_yields_wind_zero():
    temp, hum, wind, indoor = parse_weather(
        "54 degrees, relative humidity 90%, no wind", "outdoors"
    )
    assert temp == 54.0
    assert hum == 90.0
    assert wind == 0.0
    assert indoor == 0


def test_empty_with_dome_is_indoor():
    temp, hum, wind, indoor = parse_weather("", "dome")
    assert math.isnan(temp)
    assert math.isnan(hum)
    assert math.isnan(wind)
    assert indoor == 1


def test_empty_with_outdoors_is_not_indoor():
    temp, hum, wind, indoor = parse_weather("", "outdoors")
    assert math.isnan(temp)
    assert math.isnan(hum)
    assert math.isnan(wind)
    assert indoor == 0


def test_empty_with_closed_retractable_is_indoor():
    _, _, _, indoor = parse_weather("", "retractable roof (closed)")
    assert indoor == 1


def test_empty_with_open_retractable_is_not_indoor():
    _, _, _, indoor = parse_weather("", "retractable roof (open)")
    assert indoor == 0


def test_malformed_non_empty_raises():
    with pytest.raises(ValueError, match="unparseable Weather"):
        parse_weather("warm and breezy", "outdoors")


def test_negative_temperature_parses():
    temp, _, _, _ = parse_weather(
        "-5 degrees, relative humidity 80%, wind 12 mph", "outdoors"
    )
    assert temp == -5.0


def test_indoor_overrides_weather_string():
    # Even with a (hypothetical) weather string for a closed dome, the indoor
    # flag is driven by Roof — domes never go to is_indoor=0.
    _, _, _, indoor = parse_weather(
        "72 degrees, relative humidity 50%, wind 0 mph", "dome"
    )
    assert indoor == 1


def test_parse_weather_column_wraps_error_with_game_id():
    df = pd.DataFrame({
        "GameId": ["abc", "def"],
        "Weather": ["67 degrees, relative humidity 53%, wind 8 mph", "garbage"],
        "Roof": ["outdoors", "outdoors"],
    })
    with pytest.raises(ValueError, match="GameId=def"):
        parse_weather_column(df)


def test_parse_weather_column_happy_path():
    df = pd.DataFrame({
        "GameId": ["g1", "g2", "g3"],
        "Weather": [
            "67 degrees, relative humidity 53%, wind 8 mph",
            "",
            "54 degrees, relative humidity 90%, no wind",
        ],
        "Roof": ["outdoors", "dome", "outdoors"],
    })
    out = parse_weather_column(df)
    assert list(out.columns) == [
        "GameId",
        "weather_temp_f",
        "weather_humidity_pct",
        "weather_wind_mph",
        "weather_is_indoor",
    ]
    assert out.loc[0, "weather_temp_f"] == 67.0
    assert out.loc[0, "weather_is_indoor"] == 0
    assert math.isnan(out.loc[1, "weather_temp_f"])
    assert out.loc[1, "weather_is_indoor"] == 1
    assert out.loc[2, "weather_wind_mph"] == 0.0
