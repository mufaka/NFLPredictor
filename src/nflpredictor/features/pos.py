"""B-pos (position-indexed) feature matrix assembly (§3.9)."""

from __future__ import annotations

import math
import sys

import pandas as pd

from .config import FeatureConfig
from .flat import snake_case
from .positions import (
    BUCKET_CAPACITY,
    CANONICAL_BPOS_SLOTS_PER_SIDE,
    bucket_for_position,
)
from .slots import resolve_madden_features
from .vocab import NULL_SENTINEL, Vocabulary


# Full B-pos slot list — 58 entries: 29 home + 29 away in §4.3 taxonomy order.
CANONICAL_BPOS_SLOTS: tuple[str, ...] = tuple(
    f"{side}{slot}"
    for side in ("Home", "Away")
    for slot in CANONICAL_BPOS_SLOTS_PER_SIDE
)


def _assign_one_side(
    row, side: str, game_id: str
) -> dict[str, tuple[str, str]]:
    """Walk box-score slots for one side and return canonical_slot → (madden_id, position).

    Box-score slots are walked in canonical order (``{side}Off01..11`` then
    ``{side}Def01..11``) — this is the within-bucket tiebreaker that replaces
    the original ``Jersey Number``-based ordering (resolved in the spec because
    Phase 1 null-fills jersey numbers for unmatched players).

    Overflow beyond a bucket's capacity is dropped with a stderr warning.
    """
    next_index: dict[str, int] = {}
    assigned: dict[str, tuple[str, str]] = {}
    overflow_count: dict[str, int] = {}

    for unit in ("Off", "Def"):
        for i in range(1, 12):
            box_slot = f"{side}{unit}{i:02d}"
            position = str(getattr(row, f"{box_slot}_Position"))
            madden_id = str(getattr(row, f"{box_slot}_ID"))
            if not position or not madden_id:
                continue
            bucket = bucket_for_position(position)
            idx = next_index.get(bucket, 0) + 1
            next_index[bucket] = idx
            if idx > BUCKET_CAPACITY[bucket]:
                overflow_count[bucket] = overflow_count.get(bucket, 0) + 1
                continue
            assigned[f"{side}{bucket}{idx}"] = (madden_id, position)

    for bucket, n_dropped in overflow_count.items():
        print(
            f"warning: GameId={game_id} side={side} bucket={bucket} overflow "
            f"({n_dropped} starter(s) dropped beyond capacity {BUCKET_CAPACITY[bucket]})",
            file=sys.stderr,
        )

    return assigned


def assemble_pos(
    box_scores_df: pd.DataFrame,
    madden_lookup: dict[str, dict[str, str]],
    config: FeatureConfig,
    vocab: Vocabulary,
) -> pd.DataFrame:
    """Build the B-pos feature matrix in memory (§3.9 / FE-POS-01..04, FE-MAD-05).

    Returns one row per game with columns in the FE-OUT-06 order:
    ``GameId``; then per-slot Madden columns slot-major across all 58
    canonical slots in §4.3 taxonomy order; then 58 ``{slot}_present`` flags;
    then 58 ``{slot}_matched`` flags. Absent slots emit NaN/``-1`` for typed
    columns with ``present=0`` and ``matched=0`` per FE-MAD-05.
    """
    snake_cols: list[str] = [snake_case(c) for c in config.madden_columns]
    is_categorical: dict[str, bool] = {
        c: (c in config.madden_categorical_columns) for c in config.madden_columns
    }

    madden_buf: dict[str, list[object]] = {
        f"{slot}_madden_{snake}": []
        for slot in CANONICAL_BPOS_SLOTS
        for snake in snake_cols
    }
    present_buf: dict[str, list[int]] = {
        f"{slot}_present": [] for slot in CANONICAL_BPOS_SLOTS
    }
    matched_buf: dict[str, list[int]] = {
        f"{slot}_matched": [] for slot in CANONICAL_BPOS_SLOTS
    }
    game_ids: list[str] = []

    for row in box_scores_df.itertuples(index=False):
        game_id = str(row.GameId)
        game_ids.append(game_id)

        assigned: dict[str, tuple[str, str]] = {}
        for side in ("Home", "Away"):
            assigned.update(_assign_one_side(row, side, game_id))

        for slot in CANONICAL_BPOS_SLOTS:
            if slot in assigned:
                madden_id, _position = assigned[slot]
                features, matched = resolve_madden_features(
                    madden_id, madden_lookup, config
                )
                for col, snake in zip(config.madden_columns, snake_cols):
                    value = features[col]
                    if is_categorical[col]:
                        value = vocab.code(col, value)  # type: ignore[arg-type]
                    madden_buf[f"{slot}_madden_{snake}"].append(value)
                present_buf[f"{slot}_present"].append(1)
                matched_buf[f"{slot}_matched"].append(matched)
            else:
                for col, snake in zip(config.madden_columns, snake_cols):
                    name = f"{slot}_madden_{snake}"
                    if is_categorical[col]:
                        madden_buf[name].append(NULL_SENTINEL)
                    else:
                        madden_buf[name].append(math.nan)
                present_buf[f"{slot}_present"].append(0)
                matched_buf[f"{slot}_matched"].append(0)

    df_data: dict[str, object] = {"GameId": game_ids}
    for slot in CANONICAL_BPOS_SLOTS:
        for col, snake in zip(config.madden_columns, snake_cols):
            name = f"{slot}_madden_{snake}"
            if is_categorical[col]:
                df_data[name] = pd.array(madden_buf[name], dtype="int32")
            else:
                df_data[name] = pd.array(madden_buf[name], dtype="float64")
    for name, buf in present_buf.items():
        df_data[name] = pd.array(buf, dtype="int32")
    for name, buf in matched_buf.items():
        df_data[name] = pd.array(buf, dtype="int32")

    return pd.DataFrame(df_data)
