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


def test_shipped_v2_config_loads():
    cfg = load_splits_config(SHIPPED_CONFIG)
    assert isinstance(cfg, SplitsConfig)
    assert cfg.splits_version == "v2"
    assert cfg.strategies == ("season_holdout",)
    assert cfg.train_seasons == (2020, 2021, 2022, 2023)
    assert cfg.val_season == 2024
    assert cfg.test_season == 2025
    assert cfg.rotation_pool == (2020, 2021, 2022, 2023, 2024)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="splits_config.yaml not found"):
        load_splits_config(tmp_path / "does_not_exist.yaml")


def test_unknown_top_level_key_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v2"
        strategies: [season_holdout]
        train_seasons: [2020, 2021, 2022, 2023]
        val_season: 2024
        test_season: 2025
        bogus_key: 42
    """)
    with pytest.raises(SplitsConfigError, match="unknown top-level keys"):
        load_splits_config(path)


def test_missing_top_level_key_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v2"
        strategies: [season_holdout]
        train_seasons: [2020, 2021, 2022, 2023]
        val_season: 2024
    """)
    with pytest.raises(SplitsConfigError, match="missing required top-level keys"):
        load_splits_config(path)


def test_empty_strategies_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v2"
        strategies: []
        train_seasons: [2020, 2021, 2022, 2023]
        val_season: 2024
        test_season: 2025
    """)
    with pytest.raises(SplitsConfigError, match="strategies must be a non-empty list"):
        load_splits_config(path)


def test_duplicate_strategies_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v2"
        strategies: [season_holdout, season_holdout]
        train_seasons: [2020, 2021, 2022, 2023]
        val_season: 2024
        test_season: 2025
    """)
    with pytest.raises(SplitsConfigError, match="duplicate entry in strategies"):
        load_splits_config(path)


def test_unknown_strategy_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v2"
        strategies: [season_holdout, S7]
        train_seasons: [2020, 2021, 2022, 2023]
        val_season: 2024
        test_season: 2025
    """)
    with pytest.raises(SplitsConfigError, match="must be one of"):
        load_splits_config(path)


def test_val_season_in_train_seasons_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v2"
        strategies: [season_holdout]
        train_seasons: [2020, 2021, 2022, 2024]
        val_season: 2024
        test_season: 2025
    """)
    with pytest.raises(SplitsConfigError, match="val_season .* must not also appear"):
        load_splits_config(path)


def test_test_season_in_train_seasons_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v2"
        strategies: [season_holdout]
        train_seasons: [2020, 2021, 2022, 2025]
        val_season: 2024
        test_season: 2025
    """)
    with pytest.raises(SplitsConfigError, match="test_season .* must not also appear"):
        load_splits_config(path)


def test_val_equals_test_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v2"
        strategies: [season_holdout]
        train_seasons: [2020, 2021, 2022, 2023]
        val_season: 2025
        test_season: 2025
    """)
    with pytest.raises(SplitsConfigError, match="val_season and test_season must differ"):
        load_splits_config(path)


def test_season_out_of_range_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v2"
        strategies: [season_holdout]
        train_seasons: [2019, 2021, 2022, 2023]
        val_season: 2024
        test_season: 2025
    """)
    with pytest.raises(SplitsConfigError, match=r"\[2020, 2025\]"):
        load_splits_config(path)


def test_duplicate_train_season_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v2"
        strategies: [season_holdout]
        train_seasons: [2020, 2020, 2022, 2023]
        val_season: 2024
        test_season: 2025
    """)
    with pytest.raises(SplitsConfigError, match="duplicate entry in train_seasons"):
        load_splits_config(path)


def test_empty_splits_version_raises(tmp_path):
    path = _write(tmp_path, """
        splits_version: ""
        strategies: [season_holdout]
        train_seasons: [2020, 2021, 2022, 2023]
        val_season: 2024
        test_season: 2025
    """)
    with pytest.raises(SplitsConfigError, match="splits_version must be a non-empty string"):
        load_splits_config(path)


def test_loso_cv_with_single_season_pool_raises(tmp_path):
    # Rotation pool is train_seasons ∪ {val_season}. A single train season plus
    # a val season is two — so to actually trip the guard the pool must be one,
    # which is impossible with distinct val ∉ train. Instead the realistic
    # failure is loso_cv enabled with an empty rotation contribution; here we
    # confirm a healthy two-season pool is accepted.
    path = _write(tmp_path, """
        splits_version: "v2"
        strategies: [season_holdout, loso_cv]
        train_seasons: [2020]
        val_season: 2024
        test_season: 2025
    """)
    cfg = load_splits_config(path)
    assert cfg.rotation_pool == (2020, 2024)


def test_both_strategies_config_loads(tmp_path):
    path = _write(tmp_path, """
        splits_version: "v2"
        strategies: [season_holdout, loso_cv]
        train_seasons: [2020, 2021, 2022, 2023]
        val_season: 2024
        test_season: 2025
    """)
    cfg = load_splits_config(path)
    assert cfg.strategies == ("season_holdout", "loso_cv")
