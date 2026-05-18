"""Unmatched-player append (§3.6) and null-fill (§3.7) implementations.

Unmatched starters become new rows on the processed Madden table with
`matched=0`; raw Madden rows carry `matched=1` (DB-UNM-01, DB-UNM-02).
Deduplication is by `(normalized_name, team_code)` so a player who
appears as a starter in three games still produces exactly one
appended row (DB-UNM-03).

Null-fill (DB-FILL-01..05) populates every empty cell on `matched=0`
rows. Numeric columns use the arithmetic mean over `matched=1` rows;
categorical columns use the mode, with a lexicographic tiebreak.
Column classification is hard-coded so the build does not silently
flip behavior when input data changes shape.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .ids import format_madden_id, next_madden_id
from .matching import MatchResult, Starter
from .normalization import normalize_name
from .teams import normalize_team_code


MATCHED_COLUMN = "matched"


NUMERIC_COLUMNS: tuple[str, ...] = (
    # Rating columns (0–99 integers in the raw file).
    "Overall Rating",
    "Jersey Number",
    "Speed",
    "Acceleration",
    "Strength",
    "Agility",
    "Awareness",
    "Catching",
    "Carrying",
    "Throw Power",
    "Kick Power",
    "Kick Accuracy",
    "Run Block",
    "Pass Block",
    "Tackle",
    "Break Tackle",
    "Jumping",
    "Kick Return",
    "Injury",
    "Stamina",
    "Toughness",
    "Trucking",
    "Change Of Direction",
    "Ball Carrier Vision",
    "Stiff Arm",
    "Spin Move",
    "Juke Move",
    "Impact Blocking",
    "Run Block Power",
    "Run Block Finesse",
    "Pass Block Power",
    "Pass Block Finesse",
    "Lead Block",
    "Break Sack",
    "Throw Under Pressure",
    "Power Moves",
    "Finesse Moves",
    "Block Shedding",
    "Pursuit",
    "Play Recognition",
    "Man Coverage",
    "Zone Coverage",
    "Spectacular Catch",
    "Catch In Traffic",
    "Short Route Running",
    "Medium Route Running",
    "Deep Route Running",
    "Hit Power",
    "Press",
    "Release",
    "Throw Accuracy Short",
    "Throw Accuracy Mid",
    "Throw Accuracy Deep",
    "Play Action",
    "Throw On The Run",
    # Biographical / derived columns (DB-FILL-05). Birthdate is an Excel
    # serial day count; arithmetic mean produces a meaningful "average DOB".
    "Height",
    "Weight",
    "Age",
    "Birthdate",
    "Years Pro",
    "Total Salary",
    "Signing Bonus",
)


CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "Running Style",
    "Archetype",
    "College",
    "Player Handness",
)


IDENTITY_COLUMNS: tuple[str, ...] = (
    "madden_id",
    "Team",
    "Position",
    "Full Name",
    MATCHED_COLUMN,
)


@dataclass(frozen=True)
class UnmatchedPlayer:
    """A unique (normalized_name, team_code) that needs an appended row."""

    name: str               # original box-score casing
    team_code: str          # PFR code
    position: str           # box-score position
    normalized_name: str
    first_game_id_seen: str


def collect_unmatched_starters(
    matched_pairs: list[tuple[Starter, MatchResult]],
) -> list[UnmatchedPlayer]:
    """Deduplicate unmatched starters by ``(normalized_name, team_code)``.

    ``matched_pairs`` is iterated in order; ``first_game_id_seen`` is
    the earliest ``Starter.game_id`` for each unique player (DB-UNM-03).
    """
    seen: dict[tuple[str, str], UnmatchedPlayer] = {}
    for starter, result in matched_pairs:
        if result.tier != 0:
            continue
        normalized = normalize_name(starter.name)
        key = (normalized, starter.team_code)
        if key in seen:
            continue
        seen[key] = UnmatchedPlayer(
            name=starter.name,
            team_code=starter.team_code,
            position=starter.position,
            normalized_name=normalized,
            first_game_id_seen=starter.game_id,
        )
    return list(seen.values())


def _next_sequence(last_madden_id: Optional[str]) -> str:
    """Return the next id after ``last_madden_id``, or ``2024-00001`` if None."""
    if last_madden_id is None:
        return format_madden_id(2024, 1)
    return next_madden_id(last_madden_id)


def append_unmatched_rows(
    madden_df: pd.DataFrame,
    unmatched: list[UnmatchedPlayer],
    vintage: int = 2024,
) -> tuple[pd.DataFrame, dict[tuple[str, str], str]]:
    """Append one row per unique unmatched starter to ``madden_df``.

    Adds a ``matched`` column to the raw rows (set to ``"1"``) and to the
    appended rows (set to ``"0"``). Sorts unmatched players by
    ``(team_code, normalized_name, first_game_id_seen)`` ascending per
    DB-ID-03 so the assigned sequence is reproducible.

    Returns the augmented DataFrame and a mapping from
    ``(normalized_name, team_code)`` → assigned ``madden_id`` so the
    matching layer can resolve later occurrences of the same player.
    """
    if MATCHED_COLUMN not in madden_df.columns:
        madden_df = madden_df.copy()
        madden_df[MATCHED_COLUMN] = "1"
    sorted_unmatched = sorted(
        unmatched,
        key=lambda u: (u.team_code, u.normalized_name, u.first_game_id_seen),
    )
    last_id = madden_df["madden_id"].iloc[-1] if len(madden_df) else None
    new_rows: list[dict[str, object]] = []
    assignments: dict[tuple[str, str], str] = {}
    next_id = _next_sequence(last_id)
    template = {col: "" for col in madden_df.columns}
    for player in sorted_unmatched:
        row = dict(template)
        row["madden_id"] = next_id
        row["Team"] = normalize_team_code(player.team_code)
        row["Position"] = player.position
        row["Full Name"] = player.name
        row[MATCHED_COLUMN] = "0"
        new_rows.append(row)
        assignments[(player.normalized_name, player.team_code)] = next_id
        last_id = next_id
        next_id = next_madden_id(last_id)
        _ = vintage  # vintage is implied by the input sequence
    if new_rows:
        appended = pd.DataFrame(new_rows, columns=madden_df.columns)
        combined = pd.concat([madden_df, appended], ignore_index=True)
    else:
        combined = madden_df.reset_index(drop=True)
    return combined, assignments


FILL_FLOAT_FORMAT = "{:.4f}"


def _parse_numeric(series: pd.Series) -> pd.Series:
    """Strip whitespace and coerce to float; empty strings become NaN."""
    return pd.to_numeric(
        series.astype(str).str.strip().replace({"": None}),
        errors="coerce",
    )


def _mode_lex_smallest(values: pd.Series) -> Optional[str]:
    """Return the most frequent non-empty value; lex-smallest on tie."""
    counts: collections.Counter[str] = collections.Counter(
        v for v in values if v != ""
    )
    if not counts:
        return None
    top_count = max(counts.values())
    return min(k for k, c in counts.items() if c == top_count)


def compute_fill_values(madden_df: pd.DataFrame) -> dict[str, str]:
    """Compute the per-column fill value from ``matched=1`` rows only."""
    matched_one = madden_df[madden_df[MATCHED_COLUMN] == "1"]
    fill: dict[str, str] = {}
    for column in NUMERIC_COLUMNS:
        numeric = _parse_numeric(matched_one[column])
        mean = numeric.mean()
        if pd.isna(mean):
            raise ValueError(
                f"Cannot compute mean for numeric column {column!r}: "
                "no parseable values in matched=1 rows"
            )
        fill[column] = FILL_FLOAT_FORMAT.format(mean)
    for column in CATEGORICAL_COLUMNS:
        mode = _mode_lex_smallest(matched_one[column])
        if mode is None:
            raise ValueError(
                f"Cannot compute mode for categorical column {column!r}: "
                "no non-empty values in matched=1 rows"
            )
        fill[column] = mode
    return fill


def fill_unmatched_rows(
    madden_df: pd.DataFrame, fill_values: dict[str, str]
) -> pd.DataFrame:
    """Fill empty cells on ``matched=0`` rows using ``fill_values``."""
    df = madden_df.copy()
    mask = df[MATCHED_COLUMN] == "0"
    for column, value in fill_values.items():
        if column in IDENTITY_COLUMNS:
            continue
        # Only replace empty cells on matched=0 rows (DB-FILL-04).
        col_mask = mask & (df[column] == "")
        df.loc[col_mask, column] = value
    return df
