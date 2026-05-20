"""season_holdout single-fold partition (§3.4 / SP-SH-01..03)."""

from __future__ import annotations

import pandas as pd

from .config import SplitsConfig


def assign_season_holdout(
    universe: pd.DataFrame, config: SplitsConfig
) -> dict[str, list[str]]:
    """Partition ``universe`` into train / val / test by season (SP-SH-01).

    ``universe`` is a DataFrame with columns ``GameId`` and ``season`` (int).
    The returned dict has keys ``"train"``, ``"val"``, ``"test"`` whose values
    are lists of ``GameId`` strings sorted lexicographically ascending
    (SP-SH-03). Pure function: no I/O, no globals, no mutation of the input.
    """
    train_seasons = set(config.train_seasons)

    def _bucket(mask: pd.Series) -> list[str]:
        # GameIds are unique (verified upstream), so the default string sort
        # is fully tie-breaking (SP-SH-03).
        return sorted(universe.loc[mask, "GameId"].astype(str).tolist())

    return {
        "train": _bucket(universe["season"].isin(train_seasons)),
        "val": _bucket(universe["season"] == config.val_season),
        "test": _bucket(universe["season"] == config.test_season),
    }
