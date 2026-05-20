"""Split build pipeline (§5)."""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from nflpredictor.databuild.manifest import compute_sha256

from .config import MAX_SEASON, MIN_SEASON, SplitsConfig, load_splits_config
from .loso_cv import build_loso_cv
from .manifest import (
    build_splits_manifest,
    build_strategy_summaries,
    write_splits_manifest,
)
from .outputs import build_splits_artifact, write_splits_artifact
from .season_holdout import assign_season_holdout


PHASE2_FEATURES_FLAT_BASENAME = "features_flat_all.parquet"
PHASE2_MANIFEST_BASENAME = "feature_manifest.json"

SPLITS_CONFIG_BASENAME = "splits_config.yaml"
SPLITS_ARTIFACT_BASENAME = "splits_all.json"
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
    """Read ``GameId`` and ``season`` from the Phase 2 feature matrix (SP-IN-05).

    Validates that ``season`` is an integer in ``[2020, 2025]`` and that
    ``GameId`` values are unique. Returns a 2-column DataFrame ``[GameId, season]``.
    """
    schema_names = pq.ParquetFile(parquet_path).schema_arrow.names
    for col in ("GameId", "season"):
        if col not in schema_names:
            raise ValueError(
                f"Phase 2 feature matrix {parquet_path} is missing "
                f"required {col!r} column"
            )

    df = pq.read_table(parquet_path, columns=["GameId", "season"]).to_pandas()

    if df["season"].isna().any():
        raise ValueError("Phase 2 'season' column contains null values")

    season_series = df["season"]
    if not pd.api.types.is_integer_dtype(season_series):
        try:
            coerced = season_series.astype(int)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Phase 2 'season' column must be integer; "
                f"got dtype {season_series.dtype}"
            ) from exc
        if not (coerced == season_series).all():
            raise ValueError("Phase 2 'season' column contains non-integer values")
        df = df.assign(season=coerced)

    out_of_range = df[(df["season"] < MIN_SEASON) | (df["season"] > MAX_SEASON)]
    if not out_of_range.empty:
        bad = sorted(set(out_of_range["season"].tolist()))
        raise ValueError(
            f"Phase 2 'season' column has values outside "
            f"[{MIN_SEASON}, {MAX_SEASON}]: {bad}"
        )

    duplicates = df["GameId"][df["GameId"].duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(
            f"Phase 2 'GameId' column contains duplicates: {sorted(duplicates)[:5]}"
        )

    return df


def _assert_config_covers_universe(
    universe: pd.DataFrame, config: SplitsConfig
) -> None:
    """SP-CFG-04: the config's role assignment must cover exactly the data's seasons."""
    data_seasons = set(int(s) for s in universe["season"].unique())
    config_seasons = {*config.train_seasons, config.val_season, config.test_season}
    if data_seasons != config_seasons:
        raise ValueError(
            "splits_config season assignment does not match the seasons present "
            f"in the feature matrix: config={sorted(config_seasons)}, "
            f"data={sorted(data_seasons)} (SP-CFG-04)"
        )


def _log_strategy_counts(
    season_holdout: dict[str, list[str]] | None,
    loso_cv: dict[str, object] | None,
) -> None:
    """Emit per-strategy counts to stderr (SP-NF-06)."""
    if season_holdout is not None:
        print(
            f"season_holdout: train={len(season_holdout['train'])} "
            f"val={len(season_holdout['val'])} test={len(season_holdout['test'])}",
            file=sys.stderr,
        )
    if loso_cv is not None:
        folds = loso_cv["folds"]  # list[Fold]
        print(
            f"loso_cv: fold_count={len(folds)} test={len(loso_cv['test'])}",
            file=sys.stderr,
        )
        for fold in folds:  # type: ignore[assignment]
            print(
                f"  fold {fold.fold_index}: val_season={fold.val_season} "
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

    # 3. Load the (GameId -> season) universe and confirm coverage (SP-CFG-04).
    parquet_path = processed_dir / PHASE2_FEATURES_FLAT_BASENAME
    universe = load_game_universe(parquet_path)
    _assert_config_covers_universe(universe, config)

    # 4. season_holdout.
    season_holdout = (
        assign_season_holdout(universe, config)
        if "season_holdout" in config.strategies
        else None
    )

    # 5. loso_cv — reuse season_holdout's test list when available.
    loso_cv = None
    if "loso_cv" in config.strategies:
        if season_holdout is not None:
            test_list = season_holdout["test"]
        else:
            test_list = sorted(
                universe.loc[
                    universe["season"] == config.test_season, "GameId"
                ].astype(str).tolist()
            )
        loso_cv = build_loso_cv(universe, config, test_list)

    # 6. Write splits_all.json.
    artifact = build_splits_artifact(config, season_holdout, loso_cv)
    artifact_path = processed_dir / SPLITS_ARTIFACT_BASENAME
    write_splits_artifact(artifact, artifact_path)

    _log_strategy_counts(season_holdout, loso_cv)

    # 7. Write manifest LAST so output SHAs include the artifact (SP-MAN-05).
    summaries = build_strategy_summaries(season_holdout, loso_cv)
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
