"""Loader and filter helpers for ``Data/processed/training_loss_curves.parquet``.

Powers the Phase 6 training-dynamics notebook (DD-TD-02). The loader verifies
the on-disk parquet against ``training_manifest.json → output_sha256`` so
stale sidecars surface as a clear error rather than silently misleading
downstream plots (mirrors the pinning discipline of Phase 5's source
contracts in ``nflpredictor.evaluate.sources``).
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import pandas as pd

from nflpredictor.databuild.manifest import compute_sha256


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "Data" / "processed"

LOSS_CURVES_BASENAME = "training_loss_curves.parquet"
TRAINING_MANIFEST_BASENAME = "training_manifest.json"


class LossCurvesIntegrityError(RuntimeError):
    """Raised when the parquet's SHA-256 does not match training_manifest.json."""


def load_loss_curves(
    processed_dir: pathlib.Path = DEFAULT_PROCESSED_DIR,
    *,
    verify_sha: bool = True,
) -> pd.DataFrame:
    """Read ``training_loss_curves.parquet`` from ``processed_dir``.

    When ``verify_sha=True`` (the default), the on-disk file's SHA-256 must
    equal ``training_manifest.json → output_sha256["training_loss_curves.parquet"]``,
    or :class:`LossCurvesIntegrityError` is raised. Set ``verify_sha=False`` to
    skip the check (useful when reading a hand-crafted fixture without a
    matching manifest).

    Returns the parquet as-is — six columns in the order documented in
    Spec-Phase6 §4.1 (``combination_id, fold, epoch, train_loss, val_loss,
    val_mae``), rows sorted by ``(combination_id, fold, epoch)``.
    """
    parquet_path = processed_dir / LOSS_CURVES_BASENAME
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} does not exist")

    if verify_sha:
        manifest_path = processed_dir / TRAINING_MANIFEST_BASENAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} is missing; cannot verify loss-curves SHA. "
                f"Pass verify_sha=False to skip."
            )
        manifest = json.loads(manifest_path.read_text())
        expected = manifest.get("output_sha256", {}).get(LOSS_CURVES_BASENAME)
        if expected is None:
            raise LossCurvesIntegrityError(
                f"{manifest_path} does not record an output_sha256 entry for "
                f"{LOSS_CURVES_BASENAME!r}"
            )
        actual = compute_sha256(parquet_path)
        if actual != expected:
            raise LossCurvesIntegrityError(
                f"SHA-256 mismatch for {parquet_path}: manifest expects "
                f"{expected}, on-disk file is {actual}"
            )

    return pd.read_parquet(parquet_path)


def filter_curves(
    df: pd.DataFrame,
    *,
    combination_id: Optional[str] = None,
    fold: Optional[int] = None,
) -> pd.DataFrame:
    """Filter a loss-curves frame by ``combination_id`` and/or ``fold``.

    Both arguments are optional; an unset filter passes through. Returns a
    new frame with the original column order preserved and a reset index.
    """
    out = df
    if combination_id is not None:
        out = out[out["combination_id"] == combination_id]
    if fold is not None:
        out = out[out["fold"] == int(fold)]
    return out.reset_index(drop=True)
