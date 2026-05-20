"""Splits manifest construction (§3.7 / SP-MAN-01..06)."""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
from typing import Any, Optional

# Reuse Phase 1/2's helpers — same SHA logic and git-commit probe.
from nflpredictor.databuild.manifest import (
    compute_sha256 as compute_sha256,
    try_get_git_commit as try_get_git_commit,
)

from .config import SplitsConfig
from .loso_cv import Fold


def utc_timestamp(now: Optional[_dt.datetime] = None) -> str:
    """ISO 8601 UTC timestamp formatted as ``...Z`` (SP-MAN-03)."""
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_strategy_summaries(
    season_holdout: dict[str, list[str]] | None,
    loso_cv: dict[str, Any] | None,
) -> dict[str, Any]:
    """Per-strategy summary block for the manifest (SP-MAN-02)."""
    summaries: dict[str, Any] = {}
    if season_holdout is not None:
        summaries["season_holdout"] = {
            "train_n": len(season_holdout["train"]),
            "val_n": len(season_holdout["val"]),
            "test_n": len(season_holdout["test"]),
        }
    if loso_cv is not None:
        folds: list[Fold] = list(loso_cv["folds"])
        summaries["loso_cv"] = {
            "test_n": len(loso_cv["test"]),
            "fold_count": len(folds),
            "folds": [
                {
                    "fold_index": f.fold_index,
                    "val_season": f.val_season,
                    "train_n": len(f.train),
                    "val_n": len(f.val),
                }
                for f in folds
            ],
        }
    return summaries


def build_splits_manifest(
    *,
    config: SplitsConfig,
    phase2_source_sha256: dict[str, str],
    output_sha256: dict[str, str],
    phase2_manifest_git_commit: Optional[str],
    strategy_summaries: dict[str, Any],
    splits_config_sha256: str,
    repo_dir: pathlib.Path,
    now: Optional[_dt.datetime] = None,
) -> dict:
    """Assemble the splits manifest dict per SP-MAN-01."""
    return {
        "build_timestamp_utc": utc_timestamp(now),
        "splits_version": config.splits_version,
        "splits_config_sha256": splits_config_sha256,
        "season_assignment": {
            "train_seasons": list(config.train_seasons),
            "val_season": config.val_season,
            "test_season": config.test_season,
        },
        "phase2_source_sha256": dict(sorted(phase2_source_sha256.items())),
        "output_sha256": dict(sorted(output_sha256.items())),
        "git_commit": try_get_git_commit(repo_dir),
        "phase2_manifest_git_commit": phase2_manifest_git_commit,
        "strategy_summaries": strategy_summaries,
    }


def write_splits_manifest(manifest_dict: dict, path: pathlib.Path) -> None:
    """Write the manifest with sorted keys + trailing newline (SP-MAN-04).

    ``newline="\\n"`` keeps the file byte-identical across OSes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest_dict, f, sort_keys=True, indent=2)
        f.write("\n")
