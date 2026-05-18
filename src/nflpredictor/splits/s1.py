"""S1 single-fold partition (§3.4 / SP-S1-01..03)."""

from __future__ import annotations

import pandas as pd

from .config import SplitsConfig


def assign_s1(
    universe: pd.DataFrame, config: SplitsConfig
) -> dict[str, list[str]]:
    """Partition ``universe`` into train / val / test by week (SP-S1-01).

    ``universe`` is a DataFrame with columns ``GameId`` and ``week``. The
    returned dict has keys ``"train"``, ``"val"``, ``"test"`` whose values are
    lists of ``GameId`` strings sorted lexicographically ascending (SP-S1-03).
    Pure function: no I/O, no globals, no mutation of the input.
    """
    train_lo, train_hi = config.train_weeks
    val_lo, val_hi = config.val_weeks
    test_lo, test_hi = config.test_weeks
    valid_min = min(train_lo, val_lo, test_lo)
    valid_max = max(train_hi, val_hi, test_hi)

    out_of_config = universe[
        (universe["week"] < valid_min) | (universe["week"] > valid_max)
    ]
    if not out_of_config.empty:
        bad = sorted(set(out_of_config["week"].tolist()))
        raise ValueError(
            f"universe contains games with weeks outside the configured "
            f"train/val/test ranges: {bad}"
        )

    def _bucket(lo: int, hi: int) -> list[str]:
        mask = (universe["week"] >= lo) & (universe["week"] <= hi)
        # GameIds are unique (verified upstream), so the default string sort
        # is fully tie-breaking (SP-S1-03).
        return sorted(universe.loc[mask, "GameId"].astype(str).tolist())

    return {
        "train": _bucket(train_lo, train_hi),
        "val": _bucket(val_lo, val_hi),
        "test": _bucket(test_lo, test_hi),
    }
