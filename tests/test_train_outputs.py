"""Partial TR-TEST-05: prediction parquet writers honor schema, sort, and dtype contracts."""

from __future__ import annotations

import pathlib

import pandas as pd
import pyarrow.parquet as pq
import pytest

from nflpredictor.train.outputs import (
    RUNG_FILE_PREFIX,
    combination_filename,
    ensure_predictions_dir,
    write_cv_predictions,
    write_holdout_predictions,
)


# --------------------------- combination_filename ---------------------------


def test_combination_filename_matches_spec_pattern() -> None:
    """TR-OUT-01: filenames are <rung_id>__<shape>__<strategy>.parquet."""
    assert combination_filename("mean", "none", "season_holdout") == (
        "rung0_mean__none__season_holdout.parquet"
    )
    assert combination_filename("team_mean", "none", "loso_cv") == (
        "rung1_team_mean__none__loso_cv.parquet"
    )
    assert combination_filename("linear", "flat", "season_holdout") == (
        "rung2_linear__flat__season_holdout.parquet"
    )
    assert combination_filename("mlp", "pos", "loso_cv") == (
        "rung3_mlp__pos__loso_cv.parquet"
    )


def test_combination_filename_rejects_unknown_inputs() -> None:
    with pytest.raises(ValueError, match="rung"):
        combination_filename("rocket", "flat", "season_holdout")
    with pytest.raises(ValueError, match="shape"):
        combination_filename("mlp", "set", "season_holdout")
    with pytest.raises(ValueError, match="strategy"):
        combination_filename("mlp", "flat", "S5")


def test_rung_file_prefix_covers_every_spec_rung() -> None:
    assert set(RUNG_FILE_PREFIX.keys()) == {"mean", "team_mean", "linear", "mlp"}


# --------------------------- ensure_predictions_dir -------------------------


def test_ensure_predictions_dir_creates_dir(tmp_path: pathlib.Path) -> None:
    out = ensure_predictions_dir(tmp_path)
    assert out == tmp_path / "predictions"
    assert out.is_dir()


def test_ensure_predictions_dir_idempotent(tmp_path: pathlib.Path) -> None:
    out1 = ensure_predictions_dir(tmp_path)
    out2 = ensure_predictions_dir(tmp_path)
    assert out1 == out2


# ----------------------- season_holdout writer ----------------------------


def _holdout_frame() -> pd.DataFrame:
    """A small unordered frame so we can exercise the sort rule."""
    return pd.DataFrame({
        "slice":     ["test", "val",  "val",  "test", "val"],
        "GameId":    ["G06",  "G03",  "G01",  "G04",  "G02"],
        "pred_home": [10.0,    20.0,   22.0,   8.0,    23.0],
        "pred_away": [13.0,    21.0,   17.0,   14.0,   20.0],
    })


def test_write_holdout_sorts_val_before_test_then_by_gameid(tmp_path: pathlib.Path) -> None:
    """TR-OUT-02 row order: slice asc with val < test, then GameId asc."""
    path = tmp_path / "out.parquet"
    write_holdout_predictions(_holdout_frame(), path)
    rt = pq.read_table(path).to_pandas()
    assert rt["slice"].tolist() == ["val", "val", "val", "test", "test"]
    assert rt["GameId"].tolist() == ["G01", "G02", "G03", "G04", "G06"]


def test_write_holdout_dtype_contract(tmp_path: pathlib.Path) -> None:
    """TR-OUT-02 dtypes: slice/GameId string, pred_* float64."""
    path = tmp_path / "out.parquet"
    write_holdout_predictions(_holdout_frame(), path)
    table = pq.read_table(path)
    assert table.schema.field("slice").type == "string"
    assert table.schema.field("GameId").type == "string"
    assert table.schema.field("pred_home").type == "double"
    assert table.schema.field("pred_away").type == "double"


def test_write_holdout_rejects_unknown_slice_value(tmp_path: pathlib.Path) -> None:
    bad = _holdout_frame()
    bad.loc[0, "slice"] = "train"
    with pytest.raises(ValueError, match="unknown slice values"):
        write_holdout_predictions(bad, tmp_path / "out.parquet")


def test_write_holdout_byte_identical_across_runs(tmp_path: pathlib.Path) -> None:
    """Re-running the writer on the same input produces a byte-identical parquet."""
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    write_holdout_predictions(_holdout_frame(), p1)
    write_holdout_predictions(_holdout_frame(), p2)
    assert p1.read_bytes() == p2.read_bytes()


# ----------------------------- loso_cv writer -----------------------------


def _cv_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "fold_index": [1, 0, 1, 0, 2, 2],
        "GameId":     ["G05", "G01", "G06", "G02", "G07", "G03"],
        "pred_home":  [18.0,  21.0,  19.0,  24.0,  17.0,  22.0],
        "pred_away":  [14.0,  17.0,  20.0,  21.0,  24.0,  19.0],
    })


def test_write_cv_sorts_by_fold_then_gameid(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "out.parquet"
    write_cv_predictions(_cv_frame(), path)
    rt = pq.read_table(path).to_pandas()
    assert rt["fold_index"].tolist() == [0, 0, 1, 1, 2, 2]
    assert rt["GameId"].tolist() == ["G01", "G02", "G05", "G06", "G03", "G07"]


def test_write_cv_dtype_contract(tmp_path: pathlib.Path) -> None:
    """TR-OUT-03 dtypes: fold_index int8, GameId string, pred_* float64."""
    path = tmp_path / "out.parquet"
    write_cv_predictions(_cv_frame(), path)
    table = pq.read_table(path)
    assert table.schema.field("fold_index").type == "int8"
    assert table.schema.field("GameId").type == "string"
    assert table.schema.field("pred_home").type == "double"
    assert table.schema.field("pred_away").type == "double"


def test_write_cv_byte_identical_across_runs(tmp_path: pathlib.Path) -> None:
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    write_cv_predictions(_cv_frame(), p1)
    write_cv_predictions(_cv_frame(), p2)
    assert p1.read_bytes() == p2.read_bytes()


def test_write_cv_missing_required_columns_rejected(tmp_path: pathlib.Path) -> None:
    bad = _cv_frame().drop(columns=["pred_away"])
    with pytest.raises(ValueError, match="missing required columns"):
        write_cv_predictions(bad, tmp_path / "out.parquet")
