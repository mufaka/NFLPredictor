"""Upstream source loaders + hash gates (§3.1, TR-IN-01..08).

Phase 4 reads Phase 2's feature matrices, vocab, and manifest; and Phase 3's
splits artifact and manifest. Every tracked output is SHA-pinned against its
upstream manifest before any other work happens — drift fails fast.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq

# Reuse the SHA helper that has carried through Phases 1, 2, and 3.
from nflpredictor.databuild.manifest import compute_sha256


PHASE2_FEATURES_FLAT_BASENAME = "features_flat_2024.parquet"
PHASE2_FEATURES_POS_BASENAME = "features_pos_2024.parquet"
PHASE2_VOCAB_BASENAME = "feature_vocab.json"
PHASE2_MANIFEST_BASENAME = "feature_manifest.json"

PHASE3_SPLITS_BASENAME = "splits_2024.json"
PHASE3_MANIFEST_BASENAME = "splits_manifest.json"

# Order is stable so the manifest's source_sha256 maps come out the same shape.
PHASE2_TRACKED_OUTPUTS: tuple[str, ...] = (
    PHASE2_FEATURES_FLAT_BASENAME,
    PHASE2_FEATURES_POS_BASENAME,
    PHASE2_VOCAB_BASENAME,
)


class Phase2OutputMismatchError(ValueError):
    """Raised when an on-disk Phase 2 output diverges from feature_manifest.json (TR-IN-05)."""


class Phase3OutputMismatchError(ValueError):
    """Raised when splits_2024.json diverges from splits_manifest.json (TR-IN-06)."""


class StrategyUnavailableError(ValueError):
    """Raised when training_config.yaml requests a strategy not present in splits_2024.json (TR-IN-08)."""


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_phase2_outputs(processed_dir: pathlib.Path) -> dict[str, Any]:
    """Verify the three tracked Phase 2 outputs against ``feature_manifest.json`` (TR-IN-05).

    Returns the parsed Phase 2 manifest so downstream code can copy provenance
    (e.g., ``git_commit``) into Phase 4's own manifest.
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
    """Verify ``splits_2024.json`` against ``splits_manifest.json`` (TR-IN-06).

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


def load_splits_artifact(processed_dir: pathlib.Path) -> dict[str, Any]:
    """Load the on-disk splits_2024.json (after verification by ``verify_phase3_outputs``)."""
    return _load_json(processed_dir / PHASE3_SPLITS_BASENAME)


def verify_strategy_availability(
    splits: dict[str, Any],
    required_strategies: Iterable[str],
) -> None:
    """Confirm every requested strategy is present in the splits artifact (TR-IN-08)."""
    requested = list(required_strategies)
    missing = [s for s in requested if s not in splits]
    if missing:
        raise StrategyUnavailableError(
            f"training_config requests strategies {requested} but splits_2024.json "
            f"only contains {sorted(k for k in splits.keys() if k != 'splits_version')}; "
            f"missing: {missing}"
        )


def load_vocab(processed_dir: pathlib.Path) -> dict[str, list[str]]:
    """Load feature_vocab.json (used both by config validation and the model layer)."""
    return _load_json(processed_dir / PHASE2_VOCAB_BASENAME)


def high_card_vocab_keys(vocab: dict[str, list[str]], threshold: int = 8) -> frozenset[str]:
    """Vocab keys whose size exceeds ``threshold`` (TR-CAT-02 boundary).

    Phase 4's spec fixes the boundary at 8 (``size > 8`` → embedding;
    ``size <= 8`` → one-hot). The threshold parameter exists for testing.
    """
    return frozenset(k for k, values in vocab.items() if len(values) > threshold)


_SHAPE_TO_BASENAME: dict[str, str] = {
    "flat": PHASE2_FEATURES_FLAT_BASENAME,
    "pos": PHASE2_FEATURES_POS_BASENAME,
}


class LabelParityError(ValueError):
    """Raised when features_flat and features_pos disagree on (GameId, labels) (TR-NF-02)."""


def load_features(processed_dir: pathlib.Path, shape: str) -> pd.DataFrame:
    """Read the Phase 2 feature parquet for ``shape`` (TR-SHAPE-02).

    Returns a DataFrame sorted by ``GameId`` so any downstream slicing inherits
    a deterministic row order.
    """
    if shape not in _SHAPE_TO_BASENAME:
        raise ValueError(f"unknown shape {shape!r}; expected one of {sorted(_SHAPE_TO_BASENAME)}")
    path = processed_dir / _SHAPE_TO_BASENAME[shape]
    df = pq.read_table(path).to_pandas()
    return df.sort_values("GameId", kind="mergesort").reset_index(drop=True)


def assert_label_parity(flat: pd.DataFrame, pos: pd.DataFrame) -> None:
    """Verify the two feature shapes share identical ``(GameId, home_score, away_score)`` triples (TR-NF-02).

    Both frames must already be sorted by ``GameId`` (which ``load_features`` does).
    Mismatch in either the GameId set or the per-game labels raises ``LabelParityError``.
    """
    if list(flat["GameId"]) != list(pos["GameId"]):
        raise LabelParityError(
            "features_flat and features_pos have different GameId sets / orderings"
        )
    for col in ("home_score", "away_score"):
        if not (flat[col].to_numpy() == pos[col].to_numpy()).all():
            raise LabelParityError(
                f"features_flat and features_pos disagree on label column {col!r}"
            )
