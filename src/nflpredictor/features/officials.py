"""Officials feature group (§3.6, §4.2)."""

from __future__ import annotations

import pandas as pd


# Canonical roles per §4.2, in the order pinned by FE-OUT-06.
CANONICAL_OFFICIAL_ROLES: tuple[str, ...] = (
    "Referee",
    "Umpire",
    "Down Judge",
    "Line Judge",
    "Back Judge",
    "Side Judge",
    "Field Judge",
)


def _column_name(role: str) -> str:
    return "official_" + role.lower().replace(" ", "_")


# Public column-name order — used by the assembler and by the pipeline for
# vocabulary collection.
OFFICIAL_COLUMN_NAMES: tuple[str, ...] = tuple(_column_name(r) for r in CANONICAL_OFFICIAL_ROLES)


def assemble_officials(box_scores_df: pd.DataFrame) -> pd.DataFrame:
    """Build the seven officials columns from ``Official01..07_{Role,Name}`` (FE-OFF-02).

    For each game, walk the seven ``Official{NN}_Role``/``Official{NN}_Name``
    pairs and route each name to the column matching its role. Roles outside
    :data:`CANONICAL_OFFICIAL_ROLES` are silently dropped; missing roles for a
    game leave the corresponding cell as ``None`` (the vocabulary encoder turns
    those into the ``-1`` sentinel later per FE-OFF-02).
    """
    role_to_col = dict(zip(CANONICAL_OFFICIAL_ROLES, OFFICIAL_COLUMN_NAMES))
    columns: dict[str, list[str | None]] = {col: [] for col in OFFICIAL_COLUMN_NAMES}

    for row in box_scores_df.itertuples(index=False):
        per_game: dict[str, str | None] = {col: None for col in OFFICIAL_COLUMN_NAMES}
        for i in range(1, 8):
            role = getattr(row, f"Official{i:02d}_Role", None)
            name = getattr(row, f"Official{i:02d}_Name", None)
            if not role or not name:
                continue
            target = role_to_col.get(role)
            if target is not None:
                per_game[target] = name
        for col in OFFICIAL_COLUMN_NAMES:
            columns[col].append(per_game[col])

    out = pd.DataFrame(columns)
    out.insert(0, "GameId", box_scores_df["GameId"].astype(str).values)
    return out
