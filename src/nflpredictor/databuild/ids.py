"""Madden ID assignment and formatting (DB-ID-01..05).

A ``madden_id`` is a stable per-season handle for a Madden roster slot.
Format: ``"<season>-<5-digit-sequence>"``, e.g. ``"2024-00001"``.

IDs are assigned per season by sorting that season's raw Madden table by
``(team, position, fullname, jerseynumber)`` ascending and numbering from
1 (DB-ID-02). Numbering restarts at 1 for each season. Unmatched starters
appended later get their IDs from :func:`next_madden_id` continuing the
same season's sequence.
"""

from __future__ import annotations

import re

import pandas as pd


_ID_RE = re.compile(r"^(\d+)-(\d{5})$")


def format_madden_id(season: int, sequence: int) -> str:
    """Format a ``madden_id`` from a season and 1-based sequence number."""
    return f"{season}-{sequence:05d}"


def next_madden_id(last_id: str) -> str:
    """Return the next ``madden_id`` after ``last_id`` (same season)."""
    match = _ID_RE.match(last_id)
    if not match:
        raise ValueError(f"malformed madden_id: {last_id!r}")
    season = int(match.group(1))
    sequence = int(match.group(2))
    return format_madden_id(season, sequence + 1)


def assign_raw_madden_ids(madden_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Sort one season's raw Madden table and assign sequential ``madden_id`` values.

    Sort order (DB-ID-02): ``(team, position, fullname, jerseynumber)``
    ascending. The sort is stable and produces a deterministic numbering
    across runs. ``season`` is the vintage prefix for every assigned ID.
    """
    sorted_df = madden_df.sort_values(
        by=["team", "position", "fullname", "jerseynumber"],
        kind="stable",
        ignore_index=True,
    )
    ids = [format_madden_id(season, i + 1) for i in range(len(sorted_df))]
    sorted_df.insert(0, "madden_id", ids)
    return sorted_df
