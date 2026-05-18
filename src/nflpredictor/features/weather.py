"""Weather string parser (§3.5)."""

from __future__ import annotations

import math
import re

import pandas as pd


# FE-WX-02: temperature, humidity, and either a wind speed or the literal "no wind".
WEATHER_RX = re.compile(
    r"(?P<temp>-?\d+)\s+degrees,\s+relative humidity\s+(?P<humidity>\d+)%,\s+"
    r"(?:wind\s+(?P<wind>\d+)\s+mph|(?P<calm>no wind))",
    re.IGNORECASE,
)

# FE-WX-03: dome + closed retractable roofs are indoor regardless of weather string.
INDOOR_ROOFS: frozenset[str] = frozenset({"dome", "retractable roof (closed)"})


def parse_weather(weather_str: str | None, roof: str | None) -> tuple[float, float, float, int]:
    """Parse a single box-score ``Weather`` cell.

    Returns ``(temp_f, humidity_pct, wind_mph, is_indoor)``. ``is_indoor`` is a
    function of ``roof`` only — domes and closed retractable roofs are indoor
    regardless of whether the source recorded an outdoor weather string.

    Empty ``weather_str`` returns ``(nan, nan, nan, is_indoor)`` per FE-WX-03.
    A non-empty string that doesn't match :data:`WEATHER_RX` raises
    ``ValueError`` per FE-WX-04; the caller is expected to wrap it with the
    offending ``GameId``.
    """
    is_indoor = 1 if (roof in INDOOR_ROOFS) else 0
    if weather_str is None or weather_str == "":
        nan = math.nan
        return nan, nan, nan, is_indoor
    match = WEATHER_RX.search(weather_str)
    if not match:
        raise ValueError(f"unparseable Weather string: {weather_str!r}")
    temp = float(match.group("temp"))
    humidity = float(match.group("humidity"))
    wind = 0.0 if match.group("calm") else float(match.group("wind"))
    return temp, humidity, wind, is_indoor


def parse_weather_column(box_scores_df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`parse_weather` to every game, returning the four weather columns.

    The output DataFrame has columns ``GameId``, ``weather_temp_f``,
    ``weather_humidity_pct``, ``weather_wind_mph``, ``weather_is_indoor`` —
    in the order pinned by FE-OUT-06. Failures are re-raised with the offending
    ``GameId`` attached per FE-WX-04.
    """
    rows: list[tuple] = []
    for row in box_scores_df.itertuples(index=False):
        try:
            temp, humidity, wind, indoor = parse_weather(row.Weather, row.Roof)
        except ValueError as exc:
            raise ValueError(
                f"weather parse failed for GameId={row.GameId}: {exc}"
            ) from exc
        rows.append((row.GameId, temp, humidity, wind, indoor))
    return pd.DataFrame(
        rows,
        columns=[
            "GameId",
            "weather_temp_f",
            "weather_humidity_pct",
            "weather_wind_mph",
            "weather_is_indoor",
        ],
    )
