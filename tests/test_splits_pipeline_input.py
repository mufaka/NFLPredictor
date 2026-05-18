"""Tests for the Phase 2 source-hash gate and game-universe loader (SP-TEST-07)."""

from __future__ import annotations

import json
import pathlib
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nflpredictor.splits.pipeline import (
    PHASE2_FEATURES_FLAT_BASENAME,
    PHASE2_MANIFEST_BASENAME,
    Phase2OutputMismatchError,
    load_game_universe,
    verify_phase2_outputs,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL_PROCESSED = REPO_ROOT / "Data" / "processed"


def _copy_real_phase2(dst: pathlib.Path) -> pathlib.Path:
    dst.mkdir(parents=True, exist_ok=True)
    for name in (PHASE2_FEATURES_FLAT_BASENAME, PHASE2_MANIFEST_BASENAME):
        shutil.copy2(REAL_PROCESSED / name, dst / name)
    return dst


def test_happy_path_against_real_phase2_outputs():
    manifest = verify_phase2_outputs(REAL_PROCESSED)
    assert isinstance(manifest, dict)
    assert "output_sha256" in manifest
    assert (
        f"Data/processed/{PHASE2_FEATURES_FLAT_BASENAME}"
        in manifest["output_sha256"]
    )


def test_tampered_parquet_fails_fast(tmp_path):
    proc = _copy_real_phase2(tmp_path / "processed")
    parquet_path = proc / PHASE2_FEATURES_FLAT_BASENAME
    with parquet_path.open("ab") as f:
        f.write(b"\x00")
    with pytest.raises(Phase2OutputMismatchError, match="hash mismatch"):
        verify_phase2_outputs(proc)


def test_missing_manifest_fails_fast(tmp_path):
    proc = _copy_real_phase2(tmp_path / "processed")
    (proc / PHASE2_MANIFEST_BASENAME).unlink()
    with pytest.raises(FileNotFoundError, match="Phase 2 manifest not found"):
        verify_phase2_outputs(proc)


def test_missing_parquet_fails_fast(tmp_path):
    proc = _copy_real_phase2(tmp_path / "processed")
    (proc / PHASE2_FEATURES_FLAT_BASENAME).unlink()
    with pytest.raises(FileNotFoundError, match="not found"):
        verify_phase2_outputs(proc)


def test_manifest_missing_output_sha_map_fails_fast(tmp_path):
    proc = _copy_real_phase2(tmp_path / "processed")
    manifest_path = proc / PHASE2_MANIFEST_BASENAME
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    del manifest["output_sha256"]
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with pytest.raises(Phase2OutputMismatchError, match="missing the 'output_sha256' map"):
        verify_phase2_outputs(proc)


def _write_parquet(path: pathlib.Path, table: pa.Table) -> None:
    pq.write_table(table, path)


def test_load_game_universe_rejects_missing_week_column(tmp_path):
    parquet = tmp_path / "features.parquet"
    _write_parquet(parquet, pa.table({"GameId": ["a", "b"], "x": [1, 2]}))
    with pytest.raises(ValueError, match="missing required 'week' column"):
        load_game_universe(parquet)


def test_load_game_universe_rejects_missing_gameid_column(tmp_path):
    parquet = tmp_path / "features.parquet"
    _write_parquet(parquet, pa.table({"week": [1, 2]}))
    with pytest.raises(ValueError, match="missing required 'GameId' column"):
        load_game_universe(parquet)


def test_load_game_universe_rejects_out_of_range_week(tmp_path):
    parquet = tmp_path / "features.parquet"
    _write_parquet(parquet, pa.table({"GameId": ["a", "b"], "week": [1, 19]}))
    with pytest.raises(ValueError, match=r"outside \[1, 18\]"):
        load_game_universe(parquet)


def test_load_game_universe_rejects_zero_week(tmp_path):
    parquet = tmp_path / "features.parquet"
    _write_parquet(parquet, pa.table({"GameId": ["a", "b"], "week": [0, 5]}))
    with pytest.raises(ValueError, match=r"outside \[1, 18\]"):
        load_game_universe(parquet)


def test_load_game_universe_rejects_duplicate_gameids(tmp_path):
    parquet = tmp_path / "features.parquet"
    _write_parquet(parquet, pa.table({"GameId": ["a", "a", "b"], "week": [1, 2, 3]}))
    with pytest.raises(ValueError, match="duplicates"):
        load_game_universe(parquet)


def test_load_game_universe_happy_path_real_data():
    df = load_game_universe(REAL_PROCESSED / PHASE2_FEATURES_FLAT_BASENAME)
    assert list(df.columns) == ["GameId", "week"]
    assert len(df) == 272
    assert df["week"].between(1, 18).all()
    assert df["GameId"].is_unique
