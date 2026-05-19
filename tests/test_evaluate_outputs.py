"""EV-TEST-04 / EV-TEST-05 (partial): JSON + parquet writer contracts."""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pyarrow.parquet as pq
import pytest

from nflpredictor.evaluate.breakdowns import (
    BY_HOME_AWAY,
    BY_ROOF,
    BY_SURFACE,
    BY_TEAM,
    BY_WEEK,
)
from nflpredictor.evaluate.metrics import METRIC_KEYS
from nflpredictor.evaluate.outputs import (
    BREAKDOWNS_DIRNAME,
    EVALUATION_DIRNAME,
    PARQUET_COMPRESSION,
    PLOTS_DIRNAME,
    ensure_evaluation_dirs,
    write_breakdown_parquet,
    write_metrics_headline,
)


# ---------------------------------------------------------------------------
# ensure_evaluation_dirs
# ---------------------------------------------------------------------------


def test_ensure_evaluation_dirs_creates_three_dirs(tmp_path: pathlib.Path) -> None:
    eval_d, br_d, pl_d = ensure_evaluation_dirs(tmp_path)
    assert eval_d == tmp_path / EVALUATION_DIRNAME
    assert br_d == tmp_path / EVALUATION_DIRNAME / BREAKDOWNS_DIRNAME
    assert pl_d == tmp_path / EVALUATION_DIRNAME / PLOTS_DIRNAME
    assert eval_d.is_dir()
    assert br_d.is_dir()
    assert pl_d.is_dir()


def test_ensure_evaluation_dirs_idempotent(tmp_path: pathlib.Path) -> None:
    ensure_evaluation_dirs(tmp_path)
    # Second call shouldn't raise.
    ensure_evaluation_dirs(tmp_path)


# ---------------------------------------------------------------------------
# write_metrics_headline
# ---------------------------------------------------------------------------


def _headline_fixture() -> dict:
    """A 2-combination headline dict for round-trip testing."""
    return {
        "combinations": {
            # Insertion order intentionally not lexicographic to verify sort_keys=True.
            "rung2_linear__flat__s1": {
                "val": {
                    "n_games": 3,
                    "mae": 1.0,
                    "mae_home": 1.5,
                    "mae_away": 0.5,
                    "rmse_home": 2.0,
                    "rmse_away": 1.0,
                    "wl_accuracy": 0.66,
                    "spread_mae": 1.2,
                    "total_mae": 2.4,
                },
                "test": {
                    "n_games": 2,
                    "mae": 2.0,
                    "mae_home": 2.0,
                    "mae_away": 2.0,
                    "rmse_home": 3.0,
                    "rmse_away": 3.0,
                    "wl_accuracy": 0.5,
                    "spread_mae": 1.0,
                    "total_mae": 2.0,
                },
            },
            "rung0_mean__none__s1": {
                "val": {
                    "n_games": 3,
                    "mae": 5.0,
                    "mae_home": 5.0,
                    "mae_away": 5.0,
                    "rmse_home": 6.0,
                    "rmse_away": 6.0,
                    "wl_accuracy": 0.5,
                    "spread_mae": 3.0,
                    "total_mae": 6.0,
                },
                "test": {
                    "n_games": 2,
                    "mae": 5.5,
                    "mae_home": 5.5,
                    "mae_away": 5.5,
                    "rmse_home": 6.5,
                    "rmse_away": 6.5,
                    "wl_accuracy": 0.5,
                    "spread_mae": 3.5,
                    "total_mae": 7.0,
                },
            },
        }
    }


def test_headline_round_trip_and_combination_order(tmp_path: pathlib.Path) -> None:
    headline = _headline_fixture()
    path = tmp_path / "metrics_headline.json"
    write_metrics_headline(headline, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == headline
    # Combination keys sorted lexicographically.
    combo_keys = list(loaded["combinations"].keys())
    assert combo_keys == sorted(combo_keys)
    # Per-combo metric keys sorted alphabetically (EV-OUT-01 sort_keys=True).
    val_keys = list(loaded["combinations"]["rung0_mean__none__s1"]["val"].keys())
    assert val_keys == sorted(val_keys)


def test_headline_json_byte_identical_across_writes(tmp_path: pathlib.Path) -> None:
    headline = _headline_fixture()
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_metrics_headline(headline, a)
    write_metrics_headline(headline, b)
    assert a.read_bytes() == b.read_bytes()
    # Confirms trailing newline (EV-OUT-01).
    assert a.read_bytes().endswith(b"\n")


def test_headline_json_indent_is_two_spaces(tmp_path: pathlib.Path) -> None:
    headline = _headline_fixture()
    path = tmp_path / "x.json"
    write_metrics_headline(headline, path)
    text = path.read_text(encoding="utf-8")
    # First nested key uses 2-space indent.
    assert '\n  "combinations": {' in text


# ---------------------------------------------------------------------------
# write_breakdown_parquet
# ---------------------------------------------------------------------------


def _breakdown_fixture_week() -> pd.DataFrame:
    """A 4-row by_week breakdown frame across 2 combinations."""
    rows = []
    for combo in ("rung0_mean__none__s1", "rung2_linear__flat__s1"):
        for week in (1, 2):
            rows.append({
                "combination_id": combo,
                "slice": "val",
                "week": week,
                "n_games": 2,
                "mae": 1.0 * week,
                "mae_home": 1.5,
                "mae_away": 0.5,
                "rmse_home": 2.0,
                "rmse_away": 1.0,
                "wl_accuracy": 0.5,
                "spread_mae": 1.0,
                "total_mae": 2.0,
            })
    return pd.DataFrame(rows).sort_values(
        ["combination_id", "slice", "week"], kind="mergesort"
    ).reset_index(drop=True)


def test_breakdown_parquet_round_trip_schema(tmp_path: pathlib.Path) -> None:
    df = _breakdown_fixture_week()
    path = tmp_path / "by_week.parquet"
    write_breakdown_parquet(df, BY_WEEK, path)
    table = pq.read_table(path)
    columns = table.column_names
    assert columns == [
        "combination_id", "slice", "week",
        "n_games", *METRIC_KEYS,
    ]
    # Spec § 4.3 dtypes
    schema = table.schema
    assert schema.field("combination_id").type == "string"
    assert schema.field("slice").type == "string"
    assert schema.field("week").type == "int8"
    assert schema.field("n_games").type == "int32"
    for metric in METRIC_KEYS:
        assert schema.field(metric).type == "double"


def test_breakdown_parquet_row_order_preserved(tmp_path: pathlib.Path) -> None:
    df = _breakdown_fixture_week()
    path = tmp_path / "by_week.parquet"
    write_breakdown_parquet(df, BY_WEEK, path)
    rt = pq.read_table(path).to_pandas()
    assert list(rt["combination_id"]) == sorted(rt["combination_id"])


def test_breakdown_parquet_unsorted_input_rejected(tmp_path: pathlib.Path) -> None:
    df = _breakdown_fixture_week().iloc[::-1].reset_index(drop=True)
    path = tmp_path / "by_week.parquet"
    with pytest.raises(ValueError, match="not sorted"):
        write_breakdown_parquet(df, BY_WEEK, path)


def test_breakdown_parquet_byte_identical_across_writes(tmp_path: pathlib.Path) -> None:
    df = _breakdown_fixture_week()
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    write_breakdown_parquet(df, BY_WEEK, a)
    write_breakdown_parquet(df, BY_WEEK, b)
    assert a.read_bytes() == b.read_bytes()


def test_breakdown_parquet_null_metrics_round_trip(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame([
        {
            "combination_id": "c", "slice": "val", "week": 7,
            "n_games": 0,
            "mae": None, "mae_home": None, "mae_away": None,
            "rmse_home": None, "rmse_away": None,
            "wl_accuracy": None, "spread_mae": None, "total_mae": None,
        }
    ])
    path = tmp_path / "null.parquet"
    write_breakdown_parquet(df, BY_WEEK, path)
    rt = pq.read_table(path).to_pandas()
    # Pandas reads nulls as NaN for float columns.
    assert int(rt["n_games"].iloc[0]) == 0
    for metric in METRIC_KEYS:
        assert pd.isna(rt[metric].iloc[0])


def test_by_team_parquet_dtypes(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame([
        {
            "combination_id": "c", "slice": "val",
            "team_code": "atl", "home_or_away": "home",
            "n_games": 2,
            "mae": 1.0, "mae_home": 1.0, "mae_away": 1.0,
            "rmse_home": 1.0, "rmse_away": 1.0,
            "wl_accuracy": 0.5, "spread_mae": 1.0, "total_mae": 2.0,
        }
    ])
    path = tmp_path / "by_team.parquet"
    write_breakdown_parquet(df, BY_TEAM, path)
    schema = pq.read_table(path).schema
    assert schema.field("team_code").type == "string"
    assert schema.field("home_or_away").type == "string"


def test_by_surface_parquet_dtypes(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame([
        {
            "combination_id": "c", "slice": "val",
            "surface_code": 0, "surface_label": "grass",
            "n_games": 2,
            "mae": 1.0, "mae_home": 1.0, "mae_away": 1.0,
            "rmse_home": 1.0, "rmse_away": 1.0,
            "wl_accuracy": 0.5, "spread_mae": 1.0, "total_mae": 2.0,
        }
    ])
    path = tmp_path / "by_surface.parquet"
    write_breakdown_parquet(df, BY_SURFACE, path)
    schema = pq.read_table(path).schema
    assert schema.field("surface_code").type == "int8"
    assert schema.field("surface_label").type == "string"


def test_by_roof_parquet_dtypes(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame([
        {
            "combination_id": "c", "slice": "val",
            "roof_code": 0, "roof_label": "dome",
            "n_games": 2,
            "mae": 1.0, "mae_home": 1.0, "mae_away": 1.0,
            "rmse_home": 1.0, "rmse_away": 1.0,
            "wl_accuracy": 0.5, "spread_mae": 1.0, "total_mae": 2.0,
        }
    ])
    path = tmp_path / "by_roof.parquet"
    write_breakdown_parquet(df, BY_ROOF, path)
    schema = pq.read_table(path).schema
    assert schema.field("roof_code").type == "int8"
    assert schema.field("roof_label").type == "string"


def test_by_home_away_parquet_schema(tmp_path: pathlib.Path) -> None:
    df = pd.DataFrame([
        {
            "combination_id": "c", "slice": "val",
            "home_or_away": "home",
            "n_games": 6,
            "mae": 1.0, "mae_home": 1.0, "mae_away": 1.0,
            "rmse_home": 1.0, "rmse_away": 1.0,
            "wl_accuracy": 0.5, "spread_mae": 1.0, "total_mae": 2.0,
        },
        {
            "combination_id": "c", "slice": "val",
            "home_or_away": "away",
            "n_games": 6,
            "mae": 1.0, "mae_home": 1.0, "mae_away": 1.0,
            "rmse_home": 1.0, "rmse_away": 1.0,
            "wl_accuracy": 0.5, "spread_mae": 1.0, "total_mae": 2.0,
        },
    ]).sort_values(["combination_id", "slice", "home_or_away"], kind="mergesort").reset_index(drop=True)
    path = tmp_path / "by_home_away.parquet"
    write_breakdown_parquet(df, BY_HOME_AWAY, path)
    schema = pq.read_table(path).schema
    assert schema.field("home_or_away").type == "string"
    assert "combination_id" in schema.names
    assert "n_games" in schema.names


def test_breakdown_parquet_uses_snappy(tmp_path: pathlib.Path) -> None:
    df = _breakdown_fixture_week()
    path = tmp_path / "by_week.parquet"
    write_breakdown_parquet(df, BY_WEEK, path)
    pf = pq.ParquetFile(path)
    # Row group metadata records the column-level codec; we assert snappy at
    # least on one column to confirm the writer parameter is in effect.
    md = pf.metadata.row_group(0).column(0)
    assert md.compression.lower() == PARQUET_COMPRESSION
