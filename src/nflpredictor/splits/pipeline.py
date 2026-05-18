"""Split build pipeline (§5)."""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from nflpredictor.databuild.manifest import compute_sha256

from .config import MAX_WEEK, MIN_WEEK, load_splits_config
from .manifest import (
    build_splits_manifest,
    build_strategy_summaries,
    write_splits_manifest,
)
from .outputs import build_splits_artifact, write_splits_artifact
from .s1 import assign_s1
from .s3 import build_s3


PHASE2_FEATURES_FLAT_BASENAME = "features_flat_2024.parquet"
PHASE2_MANIFEST_BASENAME = "feature_manifest.json"

SPLITS_CONFIG_BASENAME = "splits_config.yaml"
SPLITS_ARTIFACT_BASENAME = "splits_2024.json"
SPLITS_MANIFEST_BASENAME = "splits_manifest.json"


class Phase2OutputMismatchError(ValueError):
    """Raised when an on-disk Phase 2 output diverges from the manifest hash (SP-IN-04)."""


def verify_phase2_outputs(processed_dir: pathlib.Path) -> dict:
    """Verify Phase 2 outputs against ``feature_manifest.json`` (SP-IN-04).

    Returns the parsed manifest dict so the caller can propagate provenance into
    Phase 3's own manifest.
    """
    manifest_path = processed_dir / PHASE2_MANIFEST_BASENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Phase 2 manifest not found at {manifest_path}; "
            "run `python -m nflpredictor.features` first"
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected = manifest.get("output_sha256")
    if not isinstance(expected, dict):
        raise Phase2OutputMismatchError(
            "feature_manifest.json is missing the 'output_sha256' map"
        )

    parquet_path = processed_dir / PHASE2_FEATURES_FLAT_BASENAME
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Phase 2 output {parquet_path} not found; "
            "run `python -m nflpredictor.features` first"
        )
    manifest_key = f"Data/processed/{PHASE2_FEATURES_FLAT_BASENAME}"
    if manifest_key not in expected:
        raise Phase2OutputMismatchError(
            f"feature_manifest.json output_sha256 does not record {manifest_key}"
        )
    actual = compute_sha256(parquet_path)
    if actual != expected[manifest_key]:
        raise Phase2OutputMismatchError(
            f"Phase 2 output {manifest_key} hash mismatch: "
            f"manifest={expected[manifest_key]!r}, disk={actual!r}. "
            "Re-run `python -m nflpredictor.features` to regenerate."
        )
    return manifest


def load_game_universe(parquet_path: pathlib.Path) -> pd.DataFrame:
    """Read ``GameId`` and ``week`` columns from the Phase 2 feature matrix (SP-IN-05).

    Validates that ``week`` is an integer in ``[1, 18]`` and that ``GameId`` values
    are unique. Returns a 2-column DataFrame ``[GameId, week]``.
    """
    schema_names = pq.ParquetFile(parquet_path).schema_arrow.names
    if "week" not in schema_names:
        raise ValueError(
            f"Phase 2 feature matrix {parquet_path} is missing required 'week' column"
        )
    if "GameId" not in schema_names:
        raise ValueError(
            f"Phase 2 feature matrix {parquet_path} is missing required 'GameId' column"
        )

    table = pq.read_table(parquet_path, columns=["GameId", "week"])
    df = table.to_pandas()

    if df["week"].isna().any():
        raise ValueError("Phase 2 'week' column contains null values")

    week_series = df["week"]
    if not pd.api.types.is_integer_dtype(week_series):
        # Allow numeric-but-fractional rejection alongside non-integer values.
        try:
            coerced = week_series.astype(int)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Phase 2 'week' column must be integer; got dtype {week_series.dtype}"
            ) from exc
        if not (coerced == week_series).all():
            raise ValueError(
                "Phase 2 'week' column contains non-integer values"
            )
        df = df.assign(week=coerced)

    out_of_range = df[(df["week"] < MIN_WEEK) | (df["week"] > MAX_WEEK)]
    if not out_of_range.empty:
        bad = sorted(set(out_of_range["week"].tolist()))
        raise ValueError(
            f"Phase 2 'week' column contains values outside [{MIN_WEEK}, {MAX_WEEK}]: {bad}"
        )

    duplicates = df["GameId"][df["GameId"].duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(
            f"Phase 2 'GameId' column contains duplicates: {sorted(duplicates)[:5]}"
        )

    return df


def _log_strategy_counts(
    s1: dict[str, list[str]] | None,
    s3: dict[str, object] | None,
) -> None:
    """Emit per-strategy counts to stderr (SP-NF-06)."""
    if s1 is not None:
        print(
            f"S1: train={len(s1['train'])} val={len(s1['val'])} test={len(s1['test'])}",
            file=sys.stderr,
        )
    if s3 is not None:
        folds = s3["folds"]  # list[Fold]
        print(
            f"S3: fold_count={len(folds)} test={len(s3['test'])}",
            file=sys.stderr,
        )
        for fold in folds:  # type: ignore[assignment]
            print(
                f"  fold {fold.fold_index}: k={fold.k} "
                f"train={len(fold.train)} val={len(fold.val)}",
                file=sys.stderr,
            )


def run_split_build(
    raw_dir: pathlib.Path,
    processed_dir: pathlib.Path,
    *,
    repo_dir: Optional[pathlib.Path] = None,
) -> None:
    """Run the Phase 3 split build end-to-end (§5.1)."""
    if repo_dir is None:
        repo_dir = processed_dir.parent.parent  # Data/processed/.. -> repo root

    # 1. Load and validate splits_config.yaml.
    config_path = raw_dir / SPLITS_CONFIG_BASENAME
    config = load_splits_config(config_path)

    # 2. Verify Phase 2 outputs (SP-IN-04).
    phase2_manifest = verify_phase2_outputs(processed_dir)

    # 3. Load the (GameId -> week) universe.
    parquet_path = processed_dir / PHASE2_FEATURES_FLAT_BASENAME
    universe = load_game_universe(parquet_path)

    # 4. S1.
    s1 = assign_s1(universe, config) if "S1" in config.strategies else None

    # 5. S3 — reuse S1's test list when available; otherwise derive it.
    s3 = None
    if "S3" in config.strategies:
        if s1 is not None:
            s3_test = s1["test"]
        else:
            test_lo, test_hi = config.test_weeks
            mask = (universe["week"] >= test_lo) & (universe["week"] <= test_hi)
            s3_test = sorted(universe.loc[mask, "GameId"].astype(str).tolist())
        s3 = build_s3(universe, config, s3_test)

    # 6. Write splits_2024.json.
    artifact = build_splits_artifact(config, s1, s3)
    artifact_path = processed_dir / SPLITS_ARTIFACT_BASENAME
    write_splits_artifact(artifact, artifact_path)

    _log_strategy_counts(s1, s3)

    # 7. Write manifest LAST so output SHAs include the artifact (SP-MAN-05).
    summaries = build_strategy_summaries(s1, s3)
    splits_config_sha256 = compute_sha256(config_path)  # raw bytes (SP-MAN-06)
    phase2_source_sha256 = {
        f"Data/processed/{PHASE2_FEATURES_FLAT_BASENAME}": compute_sha256(parquet_path),
    }
    output_sha256 = {
        f"Data/processed/{SPLITS_ARTIFACT_BASENAME}": compute_sha256(artifact_path),
    }
    manifest = build_splits_manifest(
        config=config,
        phase2_source_sha256=phase2_source_sha256,
        output_sha256=output_sha256,
        phase2_manifest_git_commit=phase2_manifest.get("git_commit"),
        strategy_summaries=summaries,
        splits_config_sha256=splits_config_sha256,
        repo_dir=repo_dir,
    )
    write_splits_manifest(manifest, processed_dir / SPLITS_MANIFEST_BASENAME)
