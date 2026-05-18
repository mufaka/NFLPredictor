"""Game-level feature derivation (§3.4)."""

from __future__ import annotations

import math
import re
from datetime import date

import pandas as pd

from .schedule import week_for_date


_START_TIME_RX = re.compile(r"^(\d{1,2}):(\d{2})\s*(am|pm)$", re.IGNORECASE)


def _to_date(value: object) -> date:
    """Convert a box-score ``GameDate`` cell (string or datetime-like) to ``date``."""
    return pd.to_datetime(value).date()


def parse_start_hour(start_time: str) -> int:
    """Stadium-local kickoff hour (0–23) from a ``StartTime`` cell (FE-GAME-04).

    Examples: ``"9:30am"`` → ``9``, ``"12:30pm"`` → ``12``, ``"8:20pm"`` → ``20``,
    ``"12:30am"`` → ``0``. Minutes are dropped.
    """
    match = _START_TIME_RX.match(start_time.strip())
    if not match:
        raise ValueError(f"unparseable StartTime: {start_time!r}")
    hour = int(match.group(1))
    meridiem = match.group(3).lower()
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    return hour


def compute_days_rest(box_scores_df: pd.DataFrame) -> pd.DataFrame:
    """Days of rest for the home and away teams (FE-GAME-08).

    Walks each team's games in chronological order. For a team's first game of
    the season the corresponding ``days_rest_*`` cell is NaN (parquet-native
    null) per the resolved spec — no arbitrary sentinel.

    Returns a DataFrame with columns ``GameId``, ``days_rest_home``,
    ``days_rest_away`` (both ``float64``) in the same row order as
    ``box_scores_df``.
    """
    dates = box_scores_df["GameDate"].map(_to_date)
    home = box_scores_df["HomeTeamCode"].astype(str)
    away = box_scores_df["AwayTeamCode"].astype(str)
    game_ids = box_scores_df["GameId"].astype(str)

    indexed = list(enumerate(zip(game_ids, dates, home, away)))
    # Sort by (date, original_index) so games on the same day stay stable.
    indexed.sort(key=lambda t: (t[1][1], t[0]))

    last_seen: dict[str, date] = {}
    home_rest: dict[int, float] = {}
    away_rest: dict[int, float] = {}
    for idx, (_gid, d, h, a) in indexed:
        home_rest[idx] = (d - last_seen[h]).days if h in last_seen else math.nan
        away_rest[idx] = (d - last_seen[a]).days if a in last_seen else math.nan
        last_seen[h] = d
        last_seen[a] = d

    return pd.DataFrame({
        "GameId": game_ids.values,
        "days_rest_home": [home_rest[i] for i in range(len(box_scores_df))],
        "days_rest_away": [away_rest[i] for i in range(len(box_scores_df))],
    }).astype({"days_rest_home": "float64", "days_rest_away": "float64"})


_BOX_SCORE_COLUMN_FOR_FIELD: dict[str, str] = {
    "day_of_week": "DayOfWeek",
    "stadium": "Stadium",
    "roof": "Roof",
    "surface": "Surface",
    "home_team_code": "HomeTeamCode",
    "away_team_code": "AwayTeamCode",
    "home_coach": "HomeCoach",
    "away_coach": "AwayCoach",
}


def assemble_game_level(
    box_scores_df: pd.DataFrame, include: tuple[str, ...]
) -> pd.DataFrame:
    """Build the game-level frame in declared ``include`` order (FE-GAME-01/10).

    Returns a DataFrame with ``GameId`` first and one column per identifier in
    ``include``, in the same order. Categorical columns are emitted as raw
    strings — the vocabulary encoder converts them to integer codes later.
    """
    out = pd.DataFrame({"GameId": box_scores_df["GameId"].astype(str).values})

    days_rest = None
    if "days_rest_home" in include or "days_rest_away" in include:
        days_rest = compute_days_rest(box_scores_df)

    for field in include:
        if field == "week":
            out["week"] = box_scores_df["GameDate"].map(
                lambda v: week_for_date(_to_date(v))
            ).astype("int64").values
        elif field == "start_hour":
            out["start_hour"] = box_scores_df["StartTime"].map(
                parse_start_hour
            ).astype("int64").values
        elif field == "days_rest_home":
            out["days_rest_home"] = days_rest["days_rest_home"].values  # type: ignore[index]
        elif field == "days_rest_away":
            out["days_rest_away"] = days_rest["days_rest_away"].values  # type: ignore[index]
        elif field in _BOX_SCORE_COLUMN_FOR_FIELD:
            out[field] = box_scores_df[_BOX_SCORE_COLUMN_FOR_FIELD[field]].values
        else:
            # config.py rejects unknown identifiers, so this is defense-in-depth.
            raise ValueError(f"unknown game-level identifier: {field!r}")
    return out
