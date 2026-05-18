"""Splits artifact JSON writer (§4.2 / SP-OUT-01..02)."""

from __future__ import annotations

import json
import pathlib
from collections import OrderedDict
from typing import Any

from .config import SplitsConfig
from .s3 import Fold


def build_splits_artifact(
    config: SplitsConfig,
    s1: dict[str, list[str]] | None,
    s3: dict[str, Any] | None,
) -> "OrderedDict[str, Any]":
    """Construct the splits artifact dict with pinned key order (§4.2).

    Top level: ``splits_version`` then each strategy in declaration order.
    S1: ``train``, ``val``, ``test``. S3: ``test`` then ``folds``.
    Per-fold: ``fold_index``, ``k``, ``train``, ``val``.
    """
    artifact: "OrderedDict[str, Any]" = OrderedDict()
    artifact["splits_version"] = config.splits_version

    for strategy in config.strategies:
        if strategy == "S1":
            if s1 is None:
                raise ValueError("S1 strategy enabled but no S1 partition supplied")
            artifact["S1"] = OrderedDict(
                [
                    ("train", list(s1["train"])),
                    ("val", list(s1["val"])),
                    ("test", list(s1["test"])),
                ]
            )
        elif strategy == "S3":
            if s3 is None:
                raise ValueError("S3 strategy enabled but no S3 folds supplied")
            folds_serialised = [_serialise_fold(f) for f in s3["folds"]]
            artifact["S3"] = OrderedDict(
                [
                    ("test", list(s3["test"])),
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
            ("k", fold.k),
            ("train", list(fold.train)),
            ("val", list(fold.val)),
        ]
    )


def write_splits_artifact(artifact: dict, path: pathlib.Path) -> None:
    """Write the artifact JSON with indent=2 and a trailing newline (SP-OUT-02)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        # sort_keys=False — key order is intentional, set by build_splits_artifact.
        json.dump(artifact, f, indent=2)
        f.write("\n")
