"""Per-combination prediction parquet writers (§3.10, §4.2, TR-OUT-01..07)."""

from __future__ import annotations

import pathlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PREDICTIONS_DIRNAME: str = "predictions"

# Phase 6 sidecar artifact (DD-LC-02). Lives at the top level of Data/processed/.
TRAINING_LOSS_CURVES_BASENAME: str = "training_loss_curves.parquet"

# Per TR-OUT-06: pyarrow writer with snappy compression and row_group_size=1024.
PARQUET_COMPRESSION: str = "snappy"
PARQUET_ROW_GROUP_SIZE: int = 1024


# Mapping of plan/spec rung names to the file-naming prefix (TR-OUT-01).
RUNG_FILE_PREFIX: dict[str, str] = {
    "mean": "rung0_mean",
    "team_mean": "rung1_team_mean",
    "linear": "rung2_linear",
    "mlp": "rung3_mlp",
}


def combination_filename(rung: str, shape: str, strategy: str) -> str:
    """Build the canonical ``<rung_id>__<shape>__<strategy>.parquet`` name (TR-OUT-01)."""
    if rung not in RUNG_FILE_PREFIX:
        raise ValueError(f"unknown rung {rung!r}; expected one of {sorted(RUNG_FILE_PREFIX)}")
    if shape not in {"none", "flat", "pos"}:
        raise ValueError(f"unknown shape {shape!r}")
    if strategy not in {"season_holdout", "loso_cv"}:
        raise ValueError(f"unknown strategy {strategy!r}")
    return f"{RUNG_FILE_PREFIX[rung]}__{shape}__{strategy}.parquet"


def ensure_predictions_dir(processed_dir: pathlib.Path) -> pathlib.Path:
    """Create ``Data/processed/predictions/`` if absent and return it (TR-OUT-07)."""
    path = processed_dir / PREDICTIONS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_parquet(table: pa.Table, path: pathlib.Path) -> None:
    pq.write_table(
        table,
        path,
        compression=PARQUET_COMPRESSION,
        row_group_size=PARQUET_ROW_GROUP_SIZE,
    )


def write_holdout_predictions(df: pd.DataFrame, path: pathlib.Path) -> None:
    """Write a season_holdout combination parquet (TR-OUT-02).

    Input ``df`` must have columns ``slice`` (``"val"`` / ``"test"``),
    ``GameId``, ``pred_home``, ``pred_away``. Rows are sorted by
    ``(slice, GameId)`` with ``val`` ordered before ``test``.
    """
    required = {"slice", "GameId", "pred_home", "pred_away"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"season_holdout predictions frame is missing required columns: "
            f"{sorted(missing)}"
        )

    # Stable, explicit slice ordering: val < test.
    slice_rank = df["slice"].map({"val": 0, "test": 1})
    if slice_rank.isna().any():
        bad = sorted(set(df["slice"].tolist()) - {"val", "test"})
        raise ValueError(
            f"season_holdout predictions frame has unknown slice values: {bad}"
        )
    ordered = (
        df.assign(_rank=slice_rank.astype("int8"))
        .sort_values(["_rank", "GameId"], kind="mergesort")
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )

    table = pa.Table.from_pydict({
        "slice": pa.array(ordered["slice"].astype(str), type=pa.string()),
        "GameId": pa.array(ordered["GameId"].astype(str), type=pa.string()),
        "pred_home": pa.array(ordered["pred_home"].astype("float64"), type=pa.float64()),
        "pred_away": pa.array(ordered["pred_away"].astype("float64"), type=pa.float64()),
    })
    _write_parquet(table, path)


LOSS_CURVES_COLUMNS: tuple[str, ...] = (
    "combination_id", "fold", "epoch", "train_loss", "val_loss", "val_mae",
)


def write_loss_curves(df: pd.DataFrame, path: pathlib.Path) -> None:
    """Write Phase 6's per-epoch loss-curve sidecar parquet (DD-LC-02 / §4.1).

    Input ``df`` must have the six columns listed in :data:`LOSS_CURVES_COLUMNS`.
    Rows are sorted lexicographically by ``(combination_id, fold, epoch)``
    for byte-determinism. When ``df`` is empty (e.g., a config with only
    trivial rungs) the writer still emits a parquet with the correct schema
    and zero rows.
    """
    missing = set(LOSS_CURVES_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"loss-curve frame is missing required columns: {sorted(missing)}"
        )

    ordered = (
        df.sort_values(
            ["combination_id", "fold", "epoch"], kind="mergesort"
        ).reset_index(drop=True)
    )

    table = pa.Table.from_pydict({
        "combination_id": pa.array(ordered["combination_id"].astype(str), type=pa.string()),
        "fold": pa.array(ordered["fold"].astype("int32"), type=pa.int32()),
        "epoch": pa.array(ordered["epoch"].astype("int32"), type=pa.int32()),
        "train_loss": pa.array(ordered["train_loss"].astype("float64"), type=pa.float64()),
        "val_loss": pa.array(ordered["val_loss"].astype("float64"), type=pa.float64()),
        "val_mae": pa.array(ordered["val_mae"].astype("float64"), type=pa.float64()),
    })
    _write_parquet(table, path)


def write_cv_predictions(df: pd.DataFrame, path: pathlib.Path) -> None:
    """Write a loso_cv combination parquet (TR-OUT-03).

    Input ``df`` must have columns ``fold_index`` (int), ``GameId``,
    ``pred_home``, ``pred_away``. Rows are sorted by ``(fold_index, GameId)``.
    """
    required = {"fold_index", "GameId", "pred_home", "pred_away"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"loso_cv predictions frame is missing required columns: {sorted(missing)}"
        )

    ordered = (
        df.sort_values(["fold_index", "GameId"], kind="mergesort").reset_index(drop=True)
    )

    table = pa.Table.from_pydict({
        "fold_index": pa.array(ordered["fold_index"].astype("int8"), type=pa.int8()),
        "GameId": pa.array(ordered["GameId"].astype(str), type=pa.string()),
        "pred_home": pa.array(ordered["pred_home"].astype("float64"), type=pa.float64()),
        "pred_away": pa.array(ordered["pred_away"].astype("float64"), type=pa.float64()),
    })
    _write_parquet(table, path)
