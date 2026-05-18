"""B-flat (slot-indexed) feature matrix assembly (§3.8)."""

from __future__ import annotations

import pandas as pd

from .config import FeatureConfig
from .slots import CANONICAL_SLOTS, resolve_madden_features
from .vocab import Vocabulary


def snake_case(col: str) -> str:
    """Convert a Madden column name to its snake_case suffix (FE-FLAT-02)."""
    return col.lower().replace(" ", "_")


def assemble_flat(
    box_scores_df: pd.DataFrame,
    madden_lookup: dict[str, dict[str, str]],
    config: FeatureConfig,
    vocab: Vocabulary,
) -> pd.DataFrame:
    """Build the B-flat feature matrix in memory (§3.8 / FE-FLAT-01..04).

    Returns one row per game with columns in the FE-OUT-06 order:
    ``GameId``; then per-slot Madden columns slot-major
    (``HomeOff01_madden_overall_rating``, ``HomeOff01_madden_archetype``,
    ``HomeOff02_madden_overall_rating``, …); then 44 ``{slot}_position``
    columns; then 44 ``{slot}_matched`` columns. Categorical columns
    integer-coded via ``vocab``; numeric columns as ``float64``; flags as
    ``int32``.
    """
    snake_cols: list[str] = [snake_case(c) for c in config.madden_columns]
    is_categorical: dict[str, bool] = {
        c: (c in config.madden_categorical_columns) for c in config.madden_columns
    }

    madden_buf: dict[str, list[object]] = {
        f"{slot}_madden_{snake}": []
        for slot in CANONICAL_SLOTS
        for snake in snake_cols
    }
    position_buf: dict[str, list[int]] = {
        f"{slot}_position": [] for slot in CANONICAL_SLOTS
    }
    matched_buf: dict[str, list[int]] = {
        f"{slot}_matched": [] for slot in CANONICAL_SLOTS
    }
    game_ids: list[str] = []

    for row in box_scores_df.itertuples(index=False):
        game_ids.append(str(row.GameId))
        for slot in CANONICAL_SLOTS:
            position = str(getattr(row, f"{slot}_Position"))
            madden_id = str(getattr(row, f"{slot}_ID"))
            features, matched = resolve_madden_features(madden_id, madden_lookup, config)
            for col, snake in zip(config.madden_columns, snake_cols):
                value = features[col]
                if is_categorical[col]:
                    value = vocab.code(col, value)  # type: ignore[arg-type]
                madden_buf[f"{slot}_madden_{snake}"].append(value)
            position_buf[f"{slot}_position"].append(vocab.code("positions", position))
            matched_buf[f"{slot}_matched"].append(matched)

    df_data: dict[str, object] = {"GameId": game_ids}
    for slot in CANONICAL_SLOTS:
        for col, snake in zip(config.madden_columns, snake_cols):
            name = f"{slot}_madden_{snake}"
            if is_categorical[col]:
                df_data[name] = pd.array(madden_buf[name], dtype="int32")
            else:
                df_data[name] = pd.array(madden_buf[name], dtype="float64")
    for name, buf in position_buf.items():
        df_data[name] = pd.array(buf, dtype="int32")
    for name, buf in matched_buf.items():
        df_data[name] = pd.array(buf, dtype="int32")

    return pd.DataFrame(df_data)
