"""Per-slot Madden resolution shared by B-flat and B-pos (§3.7)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from .config import FeatureConfig


# Canonical 44 starter slots in the order pinned by FE-FLAT-01/FE-OUT-06:
# Home Off 01..11, Home Def 01..11, Away Off 01..11, Away Def 01..11.
_SIDES: tuple[str, ...] = ("Home", "Away")
_UNITS: tuple[str, ...] = ("Off", "Def")
_INDICES: tuple[str, ...] = tuple(f"{i:02d}" for i in range(1, 12))

CANONICAL_SLOTS: tuple[str, ...] = tuple(
    f"{side}{unit}{idx}"
    for side in _SIDES
    for unit in _UNITS
    for idx in _INDICES
)


@dataclass(frozen=True)
class Starter:
    """One starter slot in one game (44 per game = 11,968 over a 272-game season)."""

    game_id: str
    slot: str               # e.g., "HomeOff01"
    side: str               # "Home" or "Away"
    unit: str               # "Off" or "Def"
    box_score_position: str  # e.g., "QB"
    madden_id: str          # e.g., "2024-00709"; never blank in Phase 1 output
    box_score_name: str     # e.g., "Patrick Mahomes"


def iter_starters(box_scores_df: pd.DataFrame) -> Iterator[Starter]:
    """Yield a :class:`Starter` per slot per game, in canonical slot order."""
    for row in box_scores_df.itertuples(index=False):
        game_id = str(getattr(row, "GameId"))
        for slot in CANONICAL_SLOTS:
            yield Starter(
                game_id=game_id,
                slot=slot,
                side=slot[:4],
                unit=slot[4:7],
                box_score_position=str(getattr(row, f"{slot}_Position")),
                madden_id=str(getattr(row, f"{slot}_ID")),
                box_score_name=str(getattr(row, f"{slot}_Name")),
            )


def build_madden_lookup(madden_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Return ``madden_id → {column → value}`` for fast per-slot reads (FE-MAD-01).

    The DataFrame is assumed to come from Phase 1's ``madden_2024.csv`` with
    ``dtype=str`` and ``keep_default_na=False`` — every cell is already a
    string, including the ``matched`` flag and any null-filled numeric values.
    """
    return madden_df.set_index("madden_id").to_dict(orient="index")  # type: ignore[return-value]


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if value == "":
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return False


def resolve_madden_features(
    madden_id: str,
    lookup: dict[str, dict[str, str]],
    config: FeatureConfig,
) -> tuple[dict[str, object], int]:
    """Read the configured Madden columns for a single ``madden_id`` (FE-MAD-01..04).

    Returns ``(features, matched_flag)``:
      * ``features`` maps each name in ``config.madden_columns`` to its value.
        Categorical columns pass through as ``str | None``; numeric pass-through
        columns are cast to ``float`` (FE-MAD-03) and raise ``ValueError`` with
        the offending value when the source is non-numeric.
      * ``matched_flag`` is the integer ``matched`` cell from the Madden row.

    Raises ``KeyError`` if ``madden_id`` is not in the lookup — Phase 1
    guarantees every box-score slot resolves to a real Madden row, so a miss
    here always indicates a Phase 1 / Phase 2 contract break.
    """
    if madden_id not in lookup:
        raise KeyError(f"madden_id {madden_id!r} not present in Madden lookup")
    row = lookup[madden_id]

    try:
        matched = int(row["matched"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(
            f"missing or non-integer 'matched' flag for madden_id={madden_id!r}: "
            f"{row.get('matched')!r}"
        ) from exc

    features: dict[str, object] = {}
    for col in config.madden_columns:
        raw = row[col]
        if col in config.madden_categorical_columns:
            features[col] = None if _is_missing(raw) else raw
        else:
            try:
                features[col] = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"non-numeric value in Madden column {col!r} for "
                    f"madden_id={madden_id!r}: {raw!r}"
                ) from exc
    return features, matched
