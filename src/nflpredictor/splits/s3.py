"""S3 expanding-window cross-validation folds (§3.5 / SP-S3-01..06)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import SplitsConfig


@dataclass(frozen=True)
class Fold:
    fold_index: int
    k: int
    train: tuple[str, ...]
    val: tuple[str, ...]


def build_s3(
    universe: pd.DataFrame,
    config: SplitsConfig,
    s1_test: list[str],
) -> dict[str, Any]:
    """Build the expanding-window fold sequence (SP-S3-01..06).

    Returns ``{"test": [...], "folds": [Fold, ...]}``. ``test`` is identical to
    S1's test list (passed in to avoid re-deriving). Folds run ``k = k_start,
    k_start + 1, ...`` until ``k + 1 > val_weeks[1]``; for each fold, ``train``
    are GameIds with ``week ∈ [1, k]`` and ``val`` are GameIds with
    ``week == k + 1``, both sorted lexicographically ascending.
    """
    if config.s3_k_start is None:
        raise ValueError(
            "build_s3 called but config has no s3_k_start "
            "(S3 not in strategies)"
        )

    val_hi = config.val_weeks[1]
    k_start = config.s3_k_start

    folds: list[Fold] = []
    fold_index = 0
    for k in range(k_start, val_hi):  # stops when k + 1 > val_hi
        train_mask = universe["week"] <= k
        val_mask = universe["week"] == k + 1
        # GameIds are unique (verified upstream), so string sort is fully
        # tie-breaking (SP-S3-04).
        train_ids = tuple(
            sorted(universe.loc[train_mask, "GameId"].astype(str).tolist())
        )
        val_ids = tuple(
            sorted(universe.loc[val_mask, "GameId"].astype(str).tolist())
        )
        folds.append(
            Fold(
                fold_index=fold_index,
                k=k,
                train=train_ids,
                val=val_ids,
            )
        )
        fold_index += 1

    return {
        "test": list(s1_test),
        "folds": folds,
    }
