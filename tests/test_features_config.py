"""Tests for feature_config.yaml loading + validation (FE-TEST-01)."""

from __future__ import annotations

import pathlib
import textwrap

import pandas as pd
import pytest

from nflpredictor.features.config import (
    FeatureConfig,
    FeatureConfigError,
    GameFeaturesConfig,
    load_feature_config,
    validate_madden_columns_exist,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = REPO_ROOT / "Data" / "raw" / "feature_config.yaml"
PROCESSED_MADDEN = REPO_ROOT / "Data" / "processed" / "madden_2024.csv"


def _write(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    path = tmp_path / "feature_config.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_shipped_v1_config_loads():
    cfg = load_feature_config(SHIPPED_CONFIG)
    assert isinstance(cfg, FeatureConfig)
    assert cfg.normalization_version == "v1"
    assert cfg.madden_columns == ("Overall Rating", "Archetype")
    assert cfg.madden_categorical_columns == ("Archetype",)
    assert cfg.game_features == GameFeaturesConfig(
        weather="parsed",
        officials="included",
        include=(
            "week",
            "day_of_week",
            "start_hour",
            "stadium",
            "roof",
            "surface",
            "home_team_code",
            "away_team_code",
            "home_coach",
            "away_coach",
            "days_rest_home",
            "days_rest_away",
        ),
    )
    assert cfg.slot_shapes == ("flat", "pos")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="feature_config.yaml not found"):
        load_feature_config(tmp_path / "does_not_exist.yaml")


def test_unknown_top_level_key_raises(tmp_path):
    path = _write(tmp_path, """
        normalization_version: "v1"
        madden_columns: ["Overall Rating"]
        madden_categorical_columns: []
        game_features:
          weather: parsed
          officials: included
          include: [week]
        slot_shapes: [flat]
        bogus_key: 42
    """)
    with pytest.raises(FeatureConfigError, match="unknown top-level keys"):
        load_feature_config(path)


def test_missing_top_level_key_raises(tmp_path):
    path = _write(tmp_path, """
        normalization_version: "v1"
        madden_columns: ["Overall Rating"]
        madden_categorical_columns: []
        game_features:
          weather: parsed
          officials: included
          include: [week]
    """)
    with pytest.raises(FeatureConfigError, match="missing required top-level keys"):
        load_feature_config(path)


def test_empty_slot_shapes_raises(tmp_path):
    path = _write(tmp_path, """
        normalization_version: "v1"
        madden_columns: ["Overall Rating"]
        madden_categorical_columns: []
        game_features:
          weather: parsed
          officials: included
          include: [week]
        slot_shapes: []
    """)
    with pytest.raises(FeatureConfigError, match="slot_shapes must be a non-empty list"):
        load_feature_config(path)


def test_invalid_slot_shape_raises(tmp_path):
    path = _write(tmp_path, """
        normalization_version: "v1"
        madden_columns: ["Overall Rating"]
        madden_categorical_columns: []
        game_features:
          weather: parsed
          officials: included
          include: [week]
        slot_shapes: [flat, set]
    """)
    with pytest.raises(FeatureConfigError, match="slot_shapes entries"):
        load_feature_config(path)


def test_invalid_weather_mode_raises(tmp_path):
    path = _write(tmp_path, """
        normalization_version: "v1"
        madden_columns: ["Overall Rating"]
        madden_categorical_columns: []
        game_features:
          weather: maybe
          officials: included
          include: [week]
        slot_shapes: [flat]
    """)
    with pytest.raises(FeatureConfigError, match="game_features.weather must be one of"):
        load_feature_config(path)


def test_invalid_officials_mode_raises(tmp_path):
    path = _write(tmp_path, """
        normalization_version: "v1"
        madden_columns: ["Overall Rating"]
        madden_categorical_columns: []
        game_features:
          weather: parsed
          officials: sometimes
          include: [week]
        slot_shapes: [flat]
    """)
    with pytest.raises(FeatureConfigError, match="game_features.officials must be one of"):
        load_feature_config(path)


def test_unknown_game_level_identifier_raises(tmp_path):
    path = _write(tmp_path, """
        normalization_version: "v1"
        madden_columns: ["Overall Rating"]
        madden_categorical_columns: []
        game_features:
          weather: parsed
          officials: included
          include: [week, made_up_field]
        slot_shapes: [flat]
    """)
    with pytest.raises(FeatureConfigError, match="unknown game-level identifiers"):
        load_feature_config(path)


def test_categorical_column_not_in_madden_columns_raises(tmp_path):
    path = _write(tmp_path, """
        normalization_version: "v1"
        madden_columns: ["Overall Rating"]
        madden_categorical_columns: ["Archetype"]
        game_features:
          weather: parsed
          officials: included
          include: [week]
        slot_shapes: [flat]
    """)
    with pytest.raises(FeatureConfigError, match="entries not in madden_columns"):
        load_feature_config(path)


def test_empty_madden_columns_raises(tmp_path):
    path = _write(tmp_path, """
        normalization_version: "v1"
        madden_columns: []
        madden_categorical_columns: []
        game_features:
          weather: parsed
          officials: included
          include: [week]
        slot_shapes: [flat]
    """)
    with pytest.raises(FeatureConfigError, match="madden_columns must be a non-empty list"):
        load_feature_config(path)


def test_validate_madden_columns_exist_passes_on_real_header():
    cfg = load_feature_config(SHIPPED_CONFIG)
    header = list(pd.read_csv(PROCESSED_MADDEN, nrows=0).columns)
    validate_madden_columns_exist(cfg, header)  # must not raise


def test_validate_madden_columns_exist_rejects_missing_column():
    cfg = load_feature_config(SHIPPED_CONFIG)
    with pytest.raises(FeatureConfigError, match="not in madden_2024.csv header"):
        validate_madden_columns_exist(cfg, ["Some Other Column"])
