"""Feature build manifest construction (§3.12 / FE-MAN-01..05)."""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
from typing import Optional

# Reuse Phase 1's helpers — same SHA logic and git-commit probe.
from nflpredictor.databuild.manifest import (
    compute_sha256 as compute_sha256,
    try_get_git_commit as try_get_git_commit,
)

from .vocab import Vocabulary


def utc_timestamp(now: Optional[_dt.datetime] = None) -> str:
    """ISO 8601 UTC timestamp formatted as ``...Z`` (FE-MAN-02)."""
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_feature_manifest(
    *,
    config_path: pathlib.Path,
    normalization_version: str,
    phase1_outputs: dict[str, pathlib.Path],
    phase1_manifest: dict,
    feature_outputs: dict[str, pathlib.Path],
    column_counts: dict[str, dict[str, int]],
    row_count: int,
    vocab: Vocabulary,
    repo_dir: pathlib.Path,
    now: Optional[_dt.datetime] = None,
) -> dict:
    """Assemble the feature manifest dict per FE-MAN-01."""
    return {
        "build_timestamp_utc": utc_timestamp(now),
        "normalization_version": normalization_version,
        "row_count": row_count,
        "feature_config_sha256": compute_sha256(config_path),
        "phase1_source_sha256": {
            name: compute_sha256(path)
            for name, path in sorted(phase1_outputs.items())
        },
        "output_sha256": {
            name: compute_sha256(path)
            for name, path in sorted(feature_outputs.items())
        },
        "git_commit": try_get_git_commit(repo_dir),
        "phase1_manifest_git_commit": phase1_manifest.get("git_commit"),
        "column_counts": column_counts,
        "vocab_sizes": vocab.sizes(),
    }


def write_feature_manifest(manifest_dict: dict, path: pathlib.Path) -> None:
    """Write the manifest with sorted keys + trailing newline (FE-MAN-03).

    ``newline="\\n"`` keeps the file byte-identical across OSes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest_dict, f, sort_keys=True, indent=2)
        f.write("\n")
