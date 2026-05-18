"""2024 NFL regular-season week calendar (FE-GAME-02).

Each row is ``(week_start, week_end_inclusive, week_number)`` covering Thursday
through the following Wednesday. The boundaries were grounded against the
58 distinct ``GameDate`` values observed in ``Data/processed/box_scores_2024.csv``
on 2026-05-18 — every observed date falls inside exactly one bucket below.

Bumping ``normalization_version`` is required if this table ever changes.
"""

from __future__ import annotations

from datetime import date


NFL_2024_WEEK_BOUNDARIES: tuple[tuple[date, date, int], ...] = (
    (date(2024, 9, 5),   date(2024, 9, 11),  1),
    (date(2024, 9, 12),  date(2024, 9, 18),  2),
    (date(2024, 9, 19),  date(2024, 9, 25),  3),
    (date(2024, 9, 26),  date(2024, 10, 2),  4),
    (date(2024, 10, 3),  date(2024, 10, 9),  5),
    (date(2024, 10, 10), date(2024, 10, 16), 6),
    (date(2024, 10, 17), date(2024, 10, 23), 7),
    (date(2024, 10, 24), date(2024, 10, 30), 8),
    (date(2024, 10, 31), date(2024, 11, 6),  9),
    (date(2024, 11, 7),  date(2024, 11, 13), 10),
    (date(2024, 11, 14), date(2024, 11, 20), 11),
    (date(2024, 11, 21), date(2024, 11, 27), 12),
    (date(2024, 11, 28), date(2024, 12, 4),  13),
    (date(2024, 12, 5),  date(2024, 12, 11), 14),
    (date(2024, 12, 12), date(2024, 12, 18), 15),
    (date(2024, 12, 19), date(2024, 12, 25), 16),
    (date(2024, 12, 26), date(2025, 1, 1),   17),
    (date(2025, 1, 2),   date(2025, 1, 8),   18),
)


def week_for_date(game_date: date) -> int:
    """Return the NFL 2024 regular-season week number for ``game_date`` (1–18).

    Raises ``ValueError`` if the date falls outside the regular-season window.
    """
    for start, end, week in NFL_2024_WEEK_BOUNDARIES:
        if start <= game_date <= end:
            return week
    raise ValueError(
        f"{game_date.isoformat()} is outside the 2024 NFL regular-season window "
        f"({NFL_2024_WEEK_BOUNDARIES[0][0].isoformat()} to "
        f"{NFL_2024_WEEK_BOUNDARIES[-1][1].isoformat()})"
    )
