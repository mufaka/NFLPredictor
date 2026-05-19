"""Upstream source loaders + hash gates (§3.1, EV-IN-01..10).

Phase 5 reads Phase 2's feature matrices + vocab + manifest, Phase 3's splits
artifact + manifest, and Phase 4's prediction parquets + training manifest.
Every tracked output is SHA-pinned against its upstream manifest before any
metric work happens — drift fails fast.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

# Reuse the SHA helper that has carried through Phases 1, 2, 3, and 4.
from nflpredictor.databuild.manifest import compute_sha256


PHASE2_FEATURES_FLAT_BASENAME = "features_flat_2024.parquet"
PHASE2_FEATURES_POS_BASENAME = "features_pos_2024.parquet"
PHASE2_VOCAB_BASENAME = "feature_vocab.json"
PHASE2_MANIFEST_BASENAME = "feature_manifest.json"

PHASE3_SPLITS_BASENAME = "splits_2024.json"
PHASE3_MANIFEST_BASENAME = "splits_manifest.json"

PHASE4_MANIFEST_BASENAME = "training_manifest.json"
PHASE4_PREDICTIONS_DIRNAME = "predictions"

PHASE2_TRACKED_OUTPUTS: tuple[str, ...] = (
    PHASE2_FEATURES_FLAT_BASENAME,
    PHASE2_FEATURES_POS_BASENAME,
    PHASE2_VOCAB_BASENAME,
)


class Phase2OutputMismatchError(ValueError):
    """Raised when an on-disk Phase 2 output diverges from feature_manifest.json (EV-IN-06)."""


class Phase3OutputMismatchError(ValueError):
    """Raised when splits_2024.json diverges from splits_manifest.json (EV-IN-07)."""


class Phase4OutputMismatchError(ValueError):
    """Raised when a Phase 4 prediction parquet diverges from training_manifest.json (EV-IN-08)."""


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_phase2_outputs(processed_dir: pathlib.Path) -> dict[str, Any]:
    """Verify the three tracked Phase 2 outputs against ``feature_manifest.json`` (EV-IN-06).

    Returns the parsed Phase 2 manifest so downstream code can copy provenance.
    """
    manifest_path = processed_dir / PHASE2_MANIFEST_BASENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Phase 2 manifest not found at {manifest_path}; "
            "run `python -m nflpredictor.features` first"
        )
    manifest = _load_json(manifest_path)

    expected = manifest.get("output_sha256")
    if not isinstance(expected, dict):
        raise Phase2OutputMismatchError(
            f"{manifest_path} is missing the 'output_sha256' map"
        )

    for basename in PHASE2_TRACKED_OUTPUTS:
        on_disk = processed_dir / basename
        if not on_disk.exists():
            raise FileNotFoundError(
                f"Phase 2 output {on_disk} not found; "
                "run `python -m nflpredictor.features` first"
            )
        manifest_key = f"Data/processed/{basename}"
        if manifest_key not in expected:
            raise Phase2OutputMismatchError(
                f"feature_manifest.json output_sha256 does not record {manifest_key}"
            )
        actual = compute_sha256(on_disk)
        if actual != expected[manifest_key]:
            raise Phase2OutputMismatchError(
                f"Phase 2 output {manifest_key} hash mismatch: "
                f"manifest={expected[manifest_key]!r}, disk={actual!r}. "
                "Re-run `python -m nflpredictor.features` to regenerate."
            )
    return manifest


def verify_phase3_outputs(processed_dir: pathlib.Path) -> dict[str, Any]:
    """Verify ``splits_2024.json`` against ``splits_manifest.json`` (EV-IN-07).

    Returns the parsed Phase 3 manifest so downstream code can copy provenance.
    """
    manifest_path = processed_dir / PHASE3_MANIFEST_BASENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Phase 3 manifest not found at {manifest_path}; "
            "run `python -m nflpredictor.splits` first"
        )
    manifest = _load_json(manifest_path)

    expected = manifest.get("output_sha256")
    if not isinstance(expected, dict):
        raise Phase3OutputMismatchError(
            f"{manifest_path} is missing the 'output_sha256' map"
        )

    splits_path = processed_dir / PHASE3_SPLITS_BASENAME
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Phase 3 output {splits_path} not found; "
            "run `python -m nflpredictor.splits` first"
        )
    manifest_key = f"Data/processed/{PHASE3_SPLITS_BASENAME}"
    if manifest_key not in expected:
        raise Phase3OutputMismatchError(
            f"splits_manifest.json output_sha256 does not record {manifest_key}"
        )
    actual = compute_sha256(splits_path)
    if actual != expected[manifest_key]:
        raise Phase3OutputMismatchError(
            f"Phase 3 output {manifest_key} hash mismatch: "
            f"manifest={expected[manifest_key]!r}, disk={actual!r}. "
            "Re-run `python -m nflpredictor.splits` to regenerate."
        )
    return manifest


def verify_phase4_outputs(processed_dir: pathlib.Path) -> dict[str, Any]:
    """Verify every Phase 4 prediction parquet against ``training_manifest.json`` (EV-IN-08).

    Returns the parsed Phase 4 manifest so downstream code can copy provenance
    (e.g., ``phase4_manifest_git_commit``).
    """
    manifest_path = processed_dir / PHASE4_MANIFEST_BASENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Phase 4 manifest not found at {manifest_path}; "
            "run `python -m nflpredictor.train` first"
        )
    manifest = _load_json(manifest_path)

    expected = manifest.get("output_sha256")
    if not isinstance(expected, dict):
        raise Phase4OutputMismatchError(
            f"{manifest_path} is missing the 'output_sha256' map"
        )

    prediction_keys = sorted(
        k for k in expected.keys() if k.startswith(f"{PHASE4_PREDICTIONS_DIRNAME}/")
    )
    if not prediction_keys:
        raise Phase4OutputMismatchError(
            f"{manifest_path} output_sha256 records no prediction parquets; "
            "expected at least one entry under 'predictions/'"
        )

    for manifest_key in prediction_keys:
        on_disk = processed_dir / manifest_key
        if not on_disk.exists():
            raise FileNotFoundError(
                f"Phase 4 output {on_disk} listed in training_manifest.json is missing; "
                "re-run `python -m nflpredictor.train` to regenerate."
            )
        actual = compute_sha256(on_disk)
        if actual != expected[manifest_key]:
            raise Phase4OutputMismatchError(
                f"Phase 4 output {manifest_key} hash mismatch: "
                f"manifest={expected[manifest_key]!r}, disk={actual!r}. "
                "Re-run `python -m nflpredictor.train` to regenerate."
            )
    return manifest


# ---------------------------------------------------------------------------
# Phase 2 / Phase 3 / Phase 4 source loaders
# ---------------------------------------------------------------------------


class LabelParityError(ValueError):
    """Raised when features_flat and features_pos disagree on (GameId, labels) (EV-NF-03)."""


class PredictionCoverageError(ValueError):
    """Raised when a prediction parquet's GameId coverage diverges from splits (EV-IN-10)."""


def load_features_flat(processed_dir: pathlib.Path) -> pd.DataFrame:
    """Read ``features_flat_2024.parquet`` (labels + breakdown columns).

    Returned frame is sorted by ``GameId`` for stable downstream iteration.
    """
    path = processed_dir / PHASE2_FEATURES_FLAT_BASENAME
    df = pq.read_table(path).to_pandas()
    return df.sort_values("GameId", kind="mergesort").reset_index(drop=True)


def load_features_pos(processed_dir: pathlib.Path) -> pd.DataFrame:
    """Read ``features_pos_2024.parquet`` (consulted only for EV-NF-03 parity)."""
    path = processed_dir / PHASE2_FEATURES_POS_BASENAME
    df = pq.read_table(path).to_pandas()
    return df.sort_values("GameId", kind="mergesort").reset_index(drop=True)


def load_vocab(processed_dir: pathlib.Path) -> dict[str, list[str]]:
    """Read ``feature_vocab.json`` (used for surface/roof/team label lookups)."""
    return _load_json(processed_dir / PHASE2_VOCAB_BASENAME)


def load_splits(processed_dir: pathlib.Path) -> dict[str, Any]:
    """Read ``splits_2024.json``."""
    return _load_json(processed_dir / PHASE3_SPLITS_BASENAME)


def assert_label_parity(flat: pd.DataFrame, pos: pd.DataFrame) -> None:
    """Verify ``flat`` and ``pos`` agree on ``(GameId, home_score, away_score)`` (EV-NF-03)."""
    if list(flat["GameId"]) != list(pos["GameId"]):
        raise LabelParityError(
            "features_flat and features_pos have different GameId sets / orderings"
        )
    for col in ("home_score", "away_score"):
        if not (flat[col].to_numpy() == pos[col].to_numpy()).all():
            raise LabelParityError(
                f"features_flat and features_pos disagree on label column {col!r}"
            )


# ---------------------------------------------------------------------------
# Phase 4 predictions enumeration + loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CombinationKey:
    """Identifies one Phase 4 prediction parquet (EV-COMB-01, EV-COMB-05)."""

    combination_id: str   # parquet filename stem (e.g., "rung0_mean__none__s1")
    strategy: str         # "S1" or "S3"
    parquet_path: pathlib.Path


def enumerate_combinations(
    phase4_manifest: dict[str, Any], processed_dir: pathlib.Path
) -> list[CombinationKey]:
    """Enumerate prediction parquets from ``training_manifest.json`` (EV-COMB-01, EV-COMB-05).

    Combinations are returned in lexicographic order of ``combination_id`` so
    downstream iteration is deterministic.
    """
    output_sha256 = phase4_manifest.get("output_sha256")
    if not isinstance(output_sha256, dict):
        raise Phase4OutputMismatchError(
            "training_manifest.json is missing the 'output_sha256' map"
        )
    prediction_keys = sorted(
        k for k in output_sha256.keys() if k.startswith(f"{PHASE4_PREDICTIONS_DIRNAME}/")
    )
    combinations: list[CombinationKey] = []
    for manifest_key in prediction_keys:
        basename = manifest_key[len(PHASE4_PREDICTIONS_DIRNAME) + 1 :]
        if not basename.endswith(".parquet"):
            raise Phase4OutputMismatchError(
                f"unexpected non-parquet entry in predictions/: {manifest_key!r}"
            )
        stem = basename[: -len(".parquet")]
        if stem.endswith("__s1"):
            strategy = "S1"
        elif stem.endswith("__s3"):
            strategy = "S3"
        else:
            raise Phase4OutputMismatchError(
                f"prediction filename {basename!r} does not end with __s1 or __s3"
            )
        combinations.append(
            CombinationKey(
                combination_id=stem,
                strategy=strategy,
                parquet_path=processed_dir / manifest_key,
            )
        )
    return combinations


def load_prediction_parquet(key: CombinationKey) -> pd.DataFrame:
    """Read a prediction parquet into a frame with the per-strategy columns.

    S1 parquets carry ``(slice, GameId, pred_home, pred_away)``; S3 parquets
    carry ``(fold_index, GameId, pred_home, pred_away)``. Phase 4's writers
    (TR-OUT-02, TR-OUT-03) emit these schemas; this loader is a thin pass-through.
    """
    return pq.read_table(key.parquet_path).to_pandas()


def validate_prediction_coverage(
    predictions: pd.DataFrame,
    key: CombinationKey,
    splits: dict[str, Any],
) -> None:
    """Verify a prediction parquet's GameId membership matches the splits artifact (EV-IN-10).

    S1: ``slice == "val"`` rows' GameIds equal ``splits.S1.val``; ``slice == "test"``
        rows equal ``splits.S1.test``.
    S3: per ``fold_index = i``, rows' GameIds equal ``splits.S3.folds[i].val``.
    """
    if key.strategy == "S1":
        if "slice" not in predictions.columns:
            raise PredictionCoverageError(
                f"{key.combination_id}: S1 parquet missing required 'slice' column"
            )
        s1 = splits.get("S1")
        if not isinstance(s1, dict):
            raise PredictionCoverageError(
                "splits artifact missing 'S1' block; cannot verify coverage"
            )
        for slice_name in ("val", "test"):
            expected = set(s1.get(slice_name, []))
            actual = set(
                predictions.loc[predictions["slice"] == slice_name, "GameId"].astype(str)
            )
            if expected != actual:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise PredictionCoverageError(
                    f"{key.combination_id}: slice={slice_name!r} GameId set diverges "
                    f"from splits.S1.{slice_name}; missing={missing[:5]} extra={extra[:5]}"
                )
        return

    # S3
    if "fold_index" not in predictions.columns:
        raise PredictionCoverageError(
            f"{key.combination_id}: S3 parquet missing required 'fold_index' column"
        )
    s3 = splits.get("S3")
    if not isinstance(s3, dict) or not isinstance(s3.get("folds"), list):
        raise PredictionCoverageError(
            "splits artifact missing 'S3.folds' block; cannot verify coverage"
        )
    folds = s3["folds"]
    for fold in folds:
        i = int(fold["fold_index"])
        expected = set(fold.get("val", []))
        actual = set(
            predictions.loc[predictions["fold_index"] == i, "GameId"].astype(str)
        )
        if expected != actual:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise PredictionCoverageError(
                f"{key.combination_id}: fold_index={i} GameId set diverges "
                f"from splits.S3.folds[{i}].val; missing={missing[:5]} extra={extra[:5]}"
            )
    # Also assert no rogue fold_index values appear in the parquet.
    valid_indices = {int(f["fold_index"]) for f in folds}
    actual_indices = set(int(x) for x in predictions["fold_index"].unique())
    extra_folds = sorted(actual_indices - valid_indices)
    if extra_folds:
        raise PredictionCoverageError(
            f"{key.combination_id}: prediction parquet has unknown fold_index values "
            f"not present in splits.S3.folds: {extra_folds}"
        )
