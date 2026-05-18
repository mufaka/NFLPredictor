"""Madden ID assignment and formatting (DB-ID-01..05).

A ``madden_id`` is a stable per-vintage handle for a Madden roster slot.
Format: ``"<vintage>-<5-digit-sequence>"``, e.g. ``"2024-00001"``.

IDs are assigned by sorting the raw Madden table by
``(Team, Position, Full Name, Jersey Number)`` ascending and numbering
from 1 (DB-ID-02). Unmatched starters appended later get their IDs from
:func:`next_madden_id` continuing the sequence.
"""

from __future__ import annotations

import re

import pandas as pd


_ID_RE = re.compile(r"^(\d+)-(\d{5})$")


def format_madden_id(vintage: int, sequence: int) -> str:
    """Format a ``madden_id`` from a vintage and 1-based sequence number."""
    return f"{vintage}-{sequence:05d}"


def next_madden_id(last_id: str) -> str:
    """Return the next ``madden_id`` after ``last_id`` (same vintage)."""
    match = _ID_RE.match(last_id)
    if not match:
        raise ValueError(f"malformed madden_id: {last_id!r}")
    vintage = int(match.group(1))
    sequence = int(match.group(2))
    return format_madden_id(vintage, sequence + 1)


def assign_raw_madden_ids(
    madden_df: pd.DataFrame, vintage: int = 2024
) -> pd.DataFrame:
    """Sort the raw Madden table and assign sequential ``madden_id`` values.

    Sort order (DB-ID-02): ``(Team, Position, Full Name, Jersey Number)``
    ascending. The sort is stable and produces a deterministic numbering
    across runs.
    """
    sorted_df = madden_df.sort_values(
        by=["Team", "Position", "Full Name", "Jersey Number"],
        kind="stable",
        ignore_index=True,
    )
    ids = [format_madden_id(vintage, i + 1) for i in range(len(sorted_df))]
    sorted_df.insert(0, "madden_id", ids)
    return sorted_df
