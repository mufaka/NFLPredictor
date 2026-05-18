"""Training manifest construction (§3.11 / TR-MAN-01..08)."""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import statistics
from typing import Any, Optional

# Reuse Phase 1's helpers — same SHA logic and git-commit probe.
from nflpredictor.databuild.manifest import (
    compute_sha256 as compute_sha256,
    try_get_git_commit as try_get_git_commit,
)

from .config import TrainingConfig
from .train_loop import ResolvedDevice, TrainingResult


def utc_timestamp(now: Optional[_dt.datetime] = None) -> str:
    """ISO 8601 UTC timestamp formatted as ``...Z`` (TR-MAN-05)."""
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_training_summary_s1(result: Optional[TrainingResult], val_mae: float) -> dict[str, Any]:
    """Per-combination S1 summary block (TR-MAN-02).

    ``result`` is ``None`` for trivial rungs — the three epoch fields then
    collapse to ``null`` per TR-MAN-02. ``val_mae`` is always recorded.
    """
    if result is None:
        return {
            "val_mae": float(val_mae),
            "epochs_trained": None,
            "best_epoch": None,
            "stopped_early": None,
        }
    return {
        "val_mae": float(val_mae),
        "epochs_trained": int(result.epochs_trained),
        "best_epoch": int(result.best_epoch),
        "stopped_early": bool(result.stopped_early),
    }


def build_training_summary_s3(
    per_fold_results: list[Optional[TrainingResult]],
    per_fold_val_mae: list[float],
) -> dict[str, Any]:
    """Per-combination S3 summary block (TR-MAN-02)."""
    fold_count = len(per_fold_val_mae)
    mean_val_mae = float(statistics.fmean(per_fold_val_mae)) if per_fold_val_mae else 0.0
    per_fold: list[dict[str, Any]] = []
    for i, (result, mae) in enumerate(zip(per_fold_results, per_fold_val_mae)):
        entry: dict[str, Any] = {"fold_index": i, "val_mae": float(mae)}
        if result is None:
            entry["epochs_trained"] = None
            entry["best_epoch"] = None
            entry["stopped_early"] = None
        else:
            entry["epochs_trained"] = int(result.epochs_trained)
            entry["best_epoch"] = int(result.best_epoch)
            entry["stopped_early"] = bool(result.stopped_early)
        per_fold.append(entry)
    return {
        "fold_count": fold_count,
        "mean_val_mae": mean_val_mae,
        "per_fold": per_fold,
    }


def build_training_manifest(
    *,
    config: TrainingConfig,
    resolved_device: ResolvedDevice,
    torch_version: str,
    phase2_source_sha256: dict[str, str],
    phase3_source_sha256: dict[str, str],
    output_sha256: dict[str, str],
    training_config_sha256: str,
    phase2_manifest_git_commit: Optional[str],
    phase3_manifest_git_commit: Optional[str],
    training_summaries: dict[str, Any],
    repo_dir: pathlib.Path,
    now: Optional[_dt.datetime] = None,
) -> dict[str, Any]:
    """Assemble the training manifest dict per TR-MAN-01.

    ``training_summaries`` shall not contain any ``test_mae`` keys
    (TR-MAN-03); the caller is responsible for that contract.
    """
    return {
        "build_timestamp_utc": utc_timestamp(now),
        "training_version": config.training_version,
        "seed": int(config.seed),
        "torch_version": torch_version,
        "device_requested": config.device,
        "device_resolved": resolved_device.name,
        "cuda_device_name": resolved_device.cuda_device_name,
        "cuda_version": resolved_device.cuda_version,
        "training_config_sha256": training_config_sha256,
        "phase2_source_sha256": dict(sorted(phase2_source_sha256.items())),
        "phase3_source_sha256": dict(sorted(phase3_source_sha256.items())),
        "output_sha256": dict(sorted(output_sha256.items())),
        "git_commit": try_get_git_commit(repo_dir),
        "phase2_manifest_git_commit": phase2_manifest_git_commit,
        "phase3_manifest_git_commit": phase3_manifest_git_commit,
        "training_summaries": training_summaries,
    }


def write_training_manifest(manifest_dict: dict[str, Any], path: pathlib.Path) -> None:
    """Write the manifest with sorted keys + trailing newline (TR-MAN-07)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest_dict, f, sort_keys=True, indent=2)
        f.write("\n")
