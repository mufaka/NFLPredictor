"""Split build pipeline (§5)."""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pyarrow.parquet as pq

from nflpredictor.databuild.manifest import compute_sha256

from .config import MAX_WEEK, MIN_WEEK


PHASE2_FEATURES_FLAT_BASENAME = "features_flat_2024.parquet"
PHASE2_MANIFEST_BASENAME = "feature_manifest.json"


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
