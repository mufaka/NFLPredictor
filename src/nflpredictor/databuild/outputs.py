"""CSV emission helpers (§3.8-§3.10).

Outputs land in ``Data/processed/`` as combined six-season files:
  - ``madden_all.csv`` — augmented Madden roster (madden_id + matched).
  - ``box_scores_all.csv`` — same shape as raw + season, _ID columns rewritten.
  - ``player_id_mapping.csv`` — one row per unique (season, box_score_id, madden_id).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from .matching import MatchResult, Starter
from .unmatched import IDENTITY_COLUMNS, MATCHED_COLUMN


# Tier rank used by DB-MAP-05: when a triple has multiple notes across the
# season, the higher rank wins (later tier carries more nuance).
_TIER_RANK = {1: 1, 2: 2, 3: 3, 4: 4, 0: 5}


@dataclass(frozen=True)
class MappingRecord:
    """One row of `player_id_mapping.csv` (DB-MAP-01..06)."""

    season: str
    box_score_id: str
    madden_id: str
    note: str


def _slot_id_columns(columns: Iterable[str]) -> list[str]:
    """Return the 44 per-slot ``_ID`` column names present in ``columns``."""
    return [
        c for c in columns
        if c.endswith("_ID") and (
            c.startswith("HomeOff") or c.startswith("HomeDef")
            or c.startswith("AwayOff") or c.startswith("AwayDef")
        )
    ]


def write_madden(madden_df: pd.DataFrame, path: pathlib.Path) -> None:
    """Write the processed Madden table sorted by ``madden_id`` (DB-OUT-12)."""
    ordered_cols = ["madden_id"] + [
        c for c in madden_df.columns if c not in {"madden_id", MATCHED_COLUMN}
    ] + [MATCHED_COLUMN]
    out = madden_df[ordered_cols].sort_values(
        "madden_id", kind="stable", ignore_index=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def rewrite_box_score_ids(
    box_scores_df: pd.DataFrame,
    slot_to_madden_id: dict[tuple[str, str], str],
) -> pd.DataFrame:
    """Replace every per-slot ``_ID`` cell with the resolved ``madden_id``.

    ``slot_to_madden_id`` is keyed by ``(GameId, "<Side><NN>_ID")``,
    e.g. ``("202409050kan", "HomeOff01_ID")``.
    """
    result = box_scores_df.copy()
    for col in _slot_id_columns(result.columns):
        result[col] = [
            slot_to_madden_id[(game_id, col)]
            for game_id in result["GameId"]
        ]
    return result


def write_box_scores(
    processed_box_scores_df: pd.DataFrame, path: pathlib.Path
) -> None:
    """Write the processed box scores sorted by ``GameId`` (DB-OUT-05)."""
    out = processed_box_scores_df.sort_values(
        "GameId", kind="stable", ignore_index=True
    )
    for col in _slot_id_columns(out.columns):
        blanks = (out[col] == "").sum()
        if blanks:
            raise ValueError(
                f"DB-OUT-03 violated: column {col!r} has {blanks} blank cells "
                "after matching pipeline; every _ID must resolve to a madden_id"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def build_mapping_records(
    pairs: Iterable[tuple[Starter, MatchResult]],
    season: str,
) -> list[MappingRecord]:
    """Build one record per unique ``(box_score_id, madden_id)`` pair for a season.

    Pairs with the same key but different tiers across the season collapse
    to a single record carrying the most informative note (DB-MAP-05).
    Blank source ``_ID`` cells get the ``blank source _id; resolved by name``
    suffix (DB-MAP-04). A position-mismatch suffix is appended per DB-POS-02.
    """
    best: dict[tuple[str, str], MatchResult] = {}
    best_blank_seen: dict[tuple[str, str], bool] = {}
    best_pos_mismatch: dict[tuple[str, str], Optional[tuple[str, str]]] = {}
    for starter, result in pairs:
        assert result.madden_id is not None, "caller must resolve unmatched first"
        key = (starter.box_score_id, result.madden_id)
        current = best.get(key)
        if current is None or _TIER_RANK[result.tier] > _TIER_RANK[current.tier]:
            best[key] = result
        # Blank-source flag and position-mismatch are "any occurrence" flags.
        if starter.box_score_id == "":
            best_blank_seen[key] = True
        if result.position_mismatch is not None:
            best_pos_mismatch[key] = result.position_mismatch
    records: list[MappingRecord] = []
    for (box_id, madden_id), result in best.items():
        note = result.note_fragment
        if best_pos_mismatch.get((box_id, madden_id)):
            box_pos, madden_pos = best_pos_mismatch[(box_id, madden_id)]
            note += f"; position mismatch box={box_pos} madden={madden_pos}"
        if best_blank_seen.get((box_id, madden_id)):
            note += "; blank source _id; resolved by name"
        records.append(
            MappingRecord(
                season=season, box_score_id=box_id, madden_id=madden_id, note=note
            )
        )
    return records


def write_mapping(records: list[MappingRecord], path: pathlib.Path) -> None:
    """Write the mapping CSV sorted by ``(madden_id, box_score_id)`` (DB-MAP-06)."""
    df = pd.DataFrame(
        [(r.season, r.box_score_id, r.madden_id, r.note) for r in records],
        columns=["season", "box_score_id", "madden_id", "note"],
    )
    df = df.sort_values(
        ["madden_id", "box_score_id"], kind="stable", ignore_index=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


# IDENTITY_COLUMNS is re-exported here so callers don't have to reach into
# the unmatched module just to know the column ordering.
_ = IDENTITY_COLUMNS
