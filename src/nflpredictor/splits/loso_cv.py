"""Leave-one-season-out cross-validation folds (§3.5 / SP-LOSO-01..06)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import SplitsConfig


@dataclass(frozen=True)
class Fold:
    fold_index: int
    val_season: int
    train: tuple[str, ...]
    val: tuple[str, ...]


def build_loso_cv(
    universe: pd.DataFrame,
    config: SplitsConfig,
    test_list: list[str],
) -> dict[str, Any]:
    """Build the leave-one-season-out fold sequence (SP-LOSO-01..06).

    Returns ``{"test": [...], "folds": [Fold, ...]}``. ``test`` is identical to
    season_holdout's test list (passed in to avoid re-deriving). One fold per
    season in the rotation pool ``train_seasons ∪ {val_season}``, in ascending
    ``val_season`` order: that season's games are ``val`` and the rest of the
    pool is ``train``. The test season is never part of any fold.
    """
    pool = config.rotation_pool  # already sorted ascending
    folds: list[Fold] = []
    for fold_index, held_out in enumerate(pool):
        train_seasons = [s for s in pool if s != held_out]
        val_ids = tuple(
            sorted(
                universe.loc[universe["season"] == held_out, "GameId"]
                .astype(str)
                .tolist()
            )
        )
        train_ids = tuple(
            sorted(
                universe.loc[universe["season"].isin(train_seasons), "GameId"]
                .astype(str)
                .tolist()
            )
        )
        folds.append(
            Fold(
                fold_index=fold_index,
                val_season=held_out,
                train=train_ids,
                val=val_ids,
            )
        )

    return {"test": list(test_list), "folds": folds}
