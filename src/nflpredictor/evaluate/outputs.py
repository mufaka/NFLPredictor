"""Output writers: headline JSON + breakdown parquets + plot dirs (§4.2..§4.5, EV-OUT-01..06)."""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .breakdowns import BreakdownSpec
from .metrics import METRIC_KEYS


EVALUATION_DIRNAME: str = "evaluation"
BREAKDOWNS_DIRNAME: str = "breakdowns"
PLOTS_DIRNAME: str = "plots"

METRICS_HEADLINE_BASENAME: str = "metrics_headline.json"
EVALUATION_MANIFEST_BASENAME: str = "evaluation_manifest.json"

# Per EV-OUT-03: pyarrow writer with snappy compression and row_group_size=1024.
PARQUET_COMPRESSION: str = "snappy"
PARQUET_ROW_GROUP_SIZE: int = 1024


def ensure_evaluation_dirs(
    processed_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Create ``evaluation/``, ``evaluation/breakdowns/``, ``evaluation/plots/`` (EV-OUT-05).

    Returns ``(evaluation_dir, breakdowns_dir, plots_dir)``.
    """
    evaluation = processed_dir / EVALUATION_DIRNAME
    breakdowns = evaluation / BREAKDOWNS_DIRNAME
    plots = evaluation / PLOTS_DIRNAME
    for d in (evaluation, breakdowns, plots):
        d.mkdir(parents=True, exist_ok=True)
    return evaluation, breakdowns, plots


def write_metrics_headline(headline: dict, path: pathlib.Path) -> None:
    """Write ``metrics_headline.json`` with sorted keys, indent=2, trailing newline (EV-OUT-01).

    ``newline="\\n"`` ensures byte identity across OSes (EV-NF-09).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(headline, f, sort_keys=True, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Breakdown parquet writers (§4.3)
# ---------------------------------------------------------------------------


# Spec § 4.3 dtype contract.
# Per-key columns are dimension-specific; metric columns are always the same.
_PER_KEY_TYPES: dict[str, pa.DataType] = {
    "team_code": pa.string(),
    "home_or_away": pa.string(),
    "week": pa.int8(),
    "surface_code": pa.int8(),
    "surface_label": pa.string(),
    "roof_code": pa.int8(),
    "roof_label": pa.string(),
}


def _build_arrow_table(df: pd.DataFrame, spec: BreakdownSpec) -> pa.Table:
    """Coerce ``df`` to the parquet schema declared in §4.3."""
    expected_columns = (
        ["combination_id", "slice"]
        + list(spec.breakdown_key_columns)
        + ["n_games"]
        + list(METRIC_KEYS)
    )
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"breakdown frame for {spec.name!r} is missing columns: {sorted(missing)}"
        )

    arrays: dict[str, pa.Array] = {}
    arrays["combination_id"] = pa.array(
        df["combination_id"].astype(str), type=pa.string()
    )
    arrays["slice"] = pa.array(df["slice"].astype(str), type=pa.string())
    for col in spec.breakdown_key_columns:
        col_type = _PER_KEY_TYPES[col]
        if pa.types.is_string(col_type):
            arrays[col] = pa.array(df[col].astype(str), type=col_type)
        else:
            # Integer codes — coerce to plain int via numpy first to avoid
            # pyarrow's mixed-type pandas conversion path picking up dtype=object.
            arrays[col] = pa.array(df[col].astype("int64"), type=col_type)
    arrays["n_games"] = pa.array(df["n_games"].astype("int64"), type=pa.int32())
    for metric in METRIC_KEYS:
        # compute_cell_metrics returns None for empty cells; pyarrow translates
        # None -> null in a float64 array.
        arrays[metric] = pa.array(list(df[metric]), type=pa.float64())
    return pa.table(arrays)


def write_breakdown_parquet(
    df: pd.DataFrame, spec: BreakdownSpec, path: pathlib.Path
) -> None:
    """Write a breakdown parquet per EV-OUT-02 / EV-OUT-03 / §4.3.

    Asserts the input frame is already sorted lexicographically on
    ``(combination_id, slice, *breakdown_key_columns)``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sort_columns = ["combination_id", "slice", *spec.breakdown_key_columns]
    if not df.empty:
        sorted_df = df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
        if not df.reset_index(drop=True).equals(sorted_df):
            raise ValueError(
                f"breakdown frame for {spec.name!r} is not sorted by "
                f"{sort_columns}; sort before calling write_breakdown_parquet"
            )

    table = _build_arrow_table(df, spec)
    pq.write_table(
        table,
        path,
        compression=PARQUET_COMPRESSION,
        row_group_size=PARQUET_ROW_GROUP_SIZE,
    )
