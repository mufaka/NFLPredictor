"""Splits artifact JSON writer (§4.2 / SP-OUT-01..02)."""

from __future__ import annotations

import json
import pathlib
from collections import OrderedDict
from typing import Any

from .config import SplitsConfig
from .loso_cv import Fold


def build_splits_artifact(
    config: SplitsConfig,
    season_holdout: dict[str, list[str]] | None,
    loso_cv: dict[str, Any] | None,
) -> "OrderedDict[str, Any]":
    """Construct the splits artifact dict with pinned key order (§4.2).

    Top level: ``splits_version`` then each strategy in declaration order.
    season_holdout: ``train``, ``val``, ``test``. loso_cv: ``test`` then
    ``folds``. Per-fold: ``fold_index``, ``val_season``, ``train``, ``val``.
    """
    artifact: "OrderedDict[str, Any]" = OrderedDict()
    artifact["splits_version"] = config.splits_version

    for strategy in config.strategies:
        if strategy == "season_holdout":
            if season_holdout is None:
                raise ValueError(
                    "season_holdout strategy enabled but no partition supplied"
                )
            artifact["season_holdout"] = OrderedDict(
                [
                    ("train", list(season_holdout["train"])),
                    ("val", list(season_holdout["val"])),
                    ("test", list(season_holdout["test"])),
                ]
            )
        elif strategy == "loso_cv":
            if loso_cv is None:
                raise ValueError("loso_cv strategy enabled but no folds supplied")
            folds_serialised = [_serialise_fold(f) for f in loso_cv["folds"]]
            artifact["loso_cv"] = OrderedDict(
                [
                    ("test", list(loso_cv["test"])),
                    ("folds", folds_serialised),
                ]
            )
        else:  # defensive: config validator should already reject this.
            raise ValueError(f"unknown strategy in config: {strategy!r}")

    return artifact


def _serialise_fold(fold: Fold) -> "OrderedDict[str, Any]":
    return OrderedDict(
        [
            ("fold_index", fold.fold_index),
            ("val_season", fold.val_season),
            ("train", list(fold.train)),
            ("val", list(fold.val)),
        ]
    )


def write_splits_artifact(artifact: dict, path: pathlib.Path) -> None:
    """Write the artifact JSON with indent=2 and a trailing newline (SP-OUT-02).

    ``newline="\\n"`` keeps the file byte-identical across OSes (so Phase 4's
    source-hash gate accepts a Windows-built artifact on a Linux machine).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        # sort_keys=False — key order is intentional, set by build_splits_artifact.
        json.dump(artifact, f, indent=2)
        f.write("\n")
