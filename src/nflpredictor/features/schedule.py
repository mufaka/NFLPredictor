"""NFL regular-season week calendar for 2020-2025 (FE-GAME-02).

Each season's Week 1 begins on the Thursday of the NFL Kickoff Game. NFL
weeks run Thursday through the following Wednesday and are exactly seven
days apart, so a game's week is a pure arithmetic offset from that
season's Week-1 Thursday. The six anchors below were grounded against the
earliest ``GameDate`` in each ``Data/raw/box_scores_<YYYY>.csv`` on
2026-05-20; all six are Thursdays.

Playoff games carry week numbers above 18 (e.g. wild-card weekend is week
19). Bumping ``normalization_version`` is required if these anchors change.
"""

from __future__ import annotations

from datetime import date


SEASON_WEEK1_THURSDAY: dict[int, date] = {
    2020: date(2020, 9, 10),
    2021: date(2021, 9, 9),
    2022: date(2022, 9, 8),
    2023: date(2023, 9, 7),
    2024: date(2024, 9, 5),
    2025: date(2025, 9, 4),
}

# Generous upper bound: 18 regular-season weeks + playoff rounds. A week
# beyond this signals a bad GameDate rather than a real game.
_MAX_WEEK = 25


def week_for_date(game_date: date, season: int) -> int:
    """Return the NFL week number for ``game_date`` in ``season``.

    Weeks 1-18 are the regular season; higher numbers are playoff rounds.
    Raises ``ValueError`` if the season is unknown or the date falls before
    that season's Week 1.
    """
    anchor = SEASON_WEEK1_THURSDAY.get(season)
    if anchor is None:
        raise ValueError(
            f"no Week-1 anchor for season {season!r}; "
            f"known seasons: {sorted(SEASON_WEEK1_THURSDAY)}"
        )
    delta_days = (game_date - anchor).days
    if delta_days < 0:
        raise ValueError(
            f"{game_date.isoformat()} is before season {season}'s "
            f"Week 1 ({anchor.isoformat()})"
        )
    week = delta_days // 7 + 1
    if week > _MAX_WEEK:
        raise ValueError(
            f"{game_date.isoformat()} resolves to week {week} in season "
            f"{season}, beyond the {_MAX_WEEK}-week sanity bound"
        )
    return week
