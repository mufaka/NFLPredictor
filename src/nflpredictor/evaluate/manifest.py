"""Evaluation manifest construction (§3.9, §4.4, EV-MAN-01..07)."""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import statistics
from typing import Any, Optional

# Reuse Phase 1 helpers (same SHA logic, same git-commit probe).
from nflpredictor.databuild.manifest import (  # noqa: F401  (re-exported for callers)
    compute_sha256 as compute_sha256,
    try_get_git_commit as try_get_git_commit,
)


def utc_timestamp(now: Optional[_dt.datetime] = None) -> str:
    """ISO 8601 UTC timestamp formatted as ``...Z`` (EV-MAN-03)."""
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_evaluation_summary(
    headline: dict[str, Any], headline_metric: str
) -> dict[str, Any]:
    """Per-combination summary block (EV-MAN-02).

    For season_holdout combos (combination id ends with ``__season_holdout``):
        ``{"val": <float or None>, "test": <float or None>}``
    For loso_cv combos (ends with ``__loso_cv``):
        ``{"pooled": <float or None>, "mean_per_fold": <float or None>,
           "per_fold": [<float or None>, ...]}``

    ``mean_per_fold`` excludes None values; if every per-fold value is None
    the field is None.
    """
    combo_block = headline.get("combinations", {})
    summary: dict[str, Any] = {}
    for combo_id in sorted(combo_block.keys()):
        per_combo = combo_block[combo_id]
        if combo_id.endswith("__season_holdout"):
            summary[combo_id] = {
                "val": _maybe_value(per_combo.get("val"), headline_metric),
                "test": _maybe_value(per_combo.get("test"), headline_metric),
            }
        elif combo_id.endswith("__loso_cv"):
            per_fold: list[Optional[float]] = []
            for slice_name in sorted(per_combo.keys()):
                if slice_name.startswith("fold_"):
                    per_fold.append(_maybe_value(per_combo[slice_name], headline_metric))
            non_null = [v for v in per_fold if v is not None]
            mean_per_fold = float(statistics.fmean(non_null)) if non_null else None
            summary[combo_id] = {
                "pooled": _maybe_value(per_combo.get("pooled"), headline_metric),
                "mean_per_fold": mean_per_fold,
                "per_fold": per_fold,
            }
        else:
            # Defense: combination id should always carry an __season_holdout / __loso_cv suffix
            # (enforced by enumerate_combinations).
            raise ValueError(
                f"unrecognized combination id {combo_id!r}: expected __season_holdout or __loso_cv suffix"
            )
    return summary


def _maybe_value(cell: Any, headline_metric: str) -> Optional[float]:
    if not isinstance(cell, dict):
        return None
    value = cell.get(headline_metric)
    if value is None:
        return None
    return float(value)


def build_evaluation_manifest(
    *,
    evaluation_version: str,
    headline_metric: str,
    evaluation_config_sha256: str,
    phase2_source_sha256: dict[str, str],
    phase3_source_sha256: dict[str, str],
    phase4_source_sha256: dict[str, str],
    output_sha256: dict[str, str],
    phase4_manifest_git_commit: Optional[str],
    combination_ids: list[str],
    evaluation_summary: dict[str, Any],
    matplotlib_version: str,
    numpy_version: str,
    pyarrow_version: str,
    repo_dir: pathlib.Path,
    now: Optional[_dt.datetime] = None,
) -> dict[str, Any]:
    """Assemble the evaluation manifest dict per EV-MAN-01."""
    return {
        "build_timestamp_utc": utc_timestamp(now),
        "evaluation_version": evaluation_version,
        "headline_metric": headline_metric,
        "evaluation_config_sha256": evaluation_config_sha256,
        "phase2_source_sha256": dict(sorted(phase2_source_sha256.items())),
        "phase3_source_sha256": dict(sorted(phase3_source_sha256.items())),
        "phase4_source_sha256": dict(sorted(phase4_source_sha256.items())),
        "output_sha256": dict(sorted(output_sha256.items())),
        "git_commit": try_get_git_commit(repo_dir),
        "phase4_manifest_git_commit": phase4_manifest_git_commit,
        "matplotlib_version": matplotlib_version,
        "numpy_version": numpy_version,
        "pyarrow_version": pyarrow_version,
        "combination_ids": sorted(combination_ids),
        "evaluation_summary": evaluation_summary,
    }


def write_evaluation_manifest(manifest_dict: dict[str, Any], path: pathlib.Path) -> None:
    """Write the manifest with sorted keys + trailing newline (EV-MAN-05, EV-NF-09)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest_dict, f, sort_keys=True, indent=2)
        f.write("\n")
