"""Tests for splits_config.yaml loading + validation (SP-TEST-01)."""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from nflpredictor.splits.config import (
    SplitsConfig,
    SplitsConfigError,
    load_splits_config,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = REPO_ROOT / "Data" / "raw" / "splits_config.yaml"


def _write(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    path = tmp_path / "splits_config.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_shipped_v1_config_loads():
    cfg = load_splits_config(SHIPPED_CONFIG)
    assert isinstance(cfg, SplitsConfig)
    assert cfg.splits_version == "v1"
    assert cfg.strategies == ("S1", "S3")
    assert cfg.train_weeks == (1, 12)
    assert cfg.val_weeks == (13, 15)
    assert cfg.test_weeks == (16, 18)
    assert cfg.s3_k_start == 6


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="splits_config.yaml not found"):
        load_splits_config(tmp_path / "does_not_exist.yaml")


def test_unknown_top_level_key_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v1"
        strategies: [S1]
        train_weeks: [1, 12]
        val_weeks: [13, 15]
        test_weeks: [16, 18]
        bogus_key: 42
    """)
    with pytest.raises(SplitsConfigError, match="unknown top-level keys"):
        load_splits_config(path)


def test_missing_top_level_key_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v1"
        strategies: [S1]
        train_weeks: [1, 12]
        val_weeks: [13, 15]
    """)
    with pytest.raises(SplitsConfigError, match="missing required top-level keys"):
        load_splits_config(path)


def test_empty_strategies_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v1"
        strategies: []
        train_weeks: [1, 12]
        val_weeks: [13, 15]
        test_weeks: [16, 18]
    """)
    with pytest.raises(SplitsConfigError, match="strategies must be a non-empty list"):
        load_splits_config(path)


def test_duplicate_strategies_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v1"
        strategies: [S1, S1]
        train_weeks: [1, 12]
        val_weeks: [13, 15]
        test_weeks: [16, 18]
    """)
    with pytest.raises(SplitsConfigError, match="duplicate entry in strategies"):
        load_splits_config(path)


def test_unknown_strategy_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v1"
        strategies: [S1, S7]
        train_weeks: [1, 12]
        val_weeks: [13, 15]
        test_weeks: [16, 18]
    """)
    with pytest.raises(SplitsConfigError, match="must be one of"):
        load_splits_config(path)


def test_overlapping_train_and_val_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v1"
        strategies: [S1]
        train_weeks: [1, 13]
        val_weeks: [13, 15]
        test_weeks: [16, 18]
    """)
    with pytest.raises(SplitsConfigError, match="strictly less than"):
        load_splits_config(path)


def test_gap_between_train_and_val_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v1"
        strategies: [S1]
        train_weeks: [1, 11]
        val_weeks: [13, 15]
        test_weeks: [16, 18]
    """)
    with pytest.raises(SplitsConfigError, match="train_weeks and val_weeks must be contiguous"):
        load_splits_config(path)


def test_gap_between_val_and_test_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v1"
        strategies: [S1]
        train_weeks: [1, 12]
        val_weeks: [13, 15]
        test_weeks: [17, 18]
    """)
    with pytest.raises(SplitsConfigError, match="val_weeks and test_weeks must be contiguous"):
        load_splits_config(path)


def test_reversed_range_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v1"
        strategies: [S1]
        train_weeks: [12, 1]
        val_weeks: [13, 15]
        test_weeks: [16, 18]
    """)
    with pytest.raises(SplitsConfigError, match="start <= end"):
        load_splits_config(path)


def test_week_out_of_bounds_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v1"
        strategies: [S1]
        train_weeks: [0, 12]
        val_weeks: [13, 15]
        test_weeks: [16, 18]
    """)
    with pytest.raises(SplitsConfigError, match=r"start <= end <= 18"):
        load_splits_config(path)


def test_week_above_18_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v1"
        strategies: [S1]
        train_weeks: [1, 12]
        val_weeks: [13, 15]
        test_weeks: [16, 19]
    """)
    with pytest.raises(SplitsConfigError, match=r"start <= end <= 18"):
        load_splits_config(path)


def test_s3_k_start_too_large_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v1"
        strategies: [S1, S3]
        train_weeks: [1, 12]
        val_weeks: [13, 15]
        test_weeks: [16, 18]
        s3:
          k_start: 15
    """)
    with pytest.raises(SplitsConfigError, match=r"1 <= k_start < val_weeks\[1\]"):
        load_splits_config(path)


def test_s3_k_start_at_val_weeks_end_raises(tmp_path):
    # k_start == val_weeks[1] would yield zero folds; must reject.
    path = _write(tmp_path, """
        splits_version: "v1"
        strategies: [S3]
        train_weeks: [1, 12]
        val_weeks: [13, 15]
        test_weeks: [16, 18]
        s3:
          k_start: 15
    """)
    with pytest.raises(SplitsConfigError):
        load_splits_config(path)


def test_s3_block_missing_when_required_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v1"
        strategies: [S1, S3]
        train_weeks: [1, 12]
        val_weeks: [13, 15]
        test_weeks: [16, 18]
    """)
    with pytest.raises(SplitsConfigError, match="s3 block is required"):
        load_splits_config(path)


def test_s3_block_present_when_not_in_strategies_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v1"
        strategies: [S1]
        train_weeks: [1, 12]
        val_weeks: [13, 15]
        test_weeks: [16, 18]
        s3:
          k_start: 6
    """)
    with pytest.raises(SplitsConfigError, match="not in strategies"):
        load_splits_config(path)


def test_empty_splits_version_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: ""
        strategies: [S1]
        train_weeks: [1, 12]
        val_weeks: [13, 15]
        test_weeks: [16, 18]
    """)
    with pytest.raises(SplitsConfigError, match="splits_version must be a non-empty string"):
        load_splits_config(path)


def test_s3_only_minimal_config_loads(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v1"
        strategies: [S3]
        train_weeks: [1, 12]
        val_weeks: [13, 15]
        test_weeks: [16, 18]
        s3:
          k_start: 6
    """)
    cfg = load_splits_config(path)
    assert cfg.strategies == ("S3",)
    assert cfg.s3_k_start == 6


def test_s1_only_config_loads(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v1"
        strategies: [S1]
        train_weeks: [1, 12]
        val_weeks: [13, 15]
        test_weeks: [16, 18]
    """)
    cfg = load_splits_config(path)
    assert cfg.strategies == ("S1",)
    assert cfg.s3_k_start is None
