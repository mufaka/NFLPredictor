"""feature_config.yaml loader + validator (§3.2, §4.1)."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

import yaml


# Closed set of game-level identifiers from spec §3.4 (FE-GAME-02..08).
KNOWN_GAME_LEVEL_FIELDS: frozenset[str] = frozenset({
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
})

ALLOWED_WEATHER_MODES: frozenset[str] = frozenset({"parsed", "skip"})
ALLOWED_OFFICIALS_MODES: frozenset[str] = frozenset({"included", "skip"})
ALLOWED_SLOT_SHAPES: frozenset[str] = frozenset({"flat", "pos"})

_TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    "normalization_version",
    "madden_columns",
    "madden_categorical_columns",
    "game_features",
    "slot_shapes",
})

_GAME_FEATURES_KEYS: frozenset[str] = frozenset({"weather", "officials", "include"})


@dataclass(frozen=True)
class GameFeaturesConfig:
    weather: str
    officials: str
    include: tuple[str, ...]


@dataclass(frozen=True)
class FeatureConfig:
    normalization_version: str
    madden_columns: tuple[str, ...]
    madden_categorical_columns: tuple[str, ...]
    game_features: GameFeaturesConfig
    slot_shapes: tuple[str, ...]


class FeatureConfigError(ValueError):
    """Raised for malformed feature_config.yaml content (§3.2)."""


def load_feature_config(path: pathlib.Path) -> FeatureConfig:
    """Read, parse, and validate feature_config.yaml (FE-IN-02, FE-CFG-01..09)."""
    if not path.exists():
        raise FileNotFoundError(
            f"feature_config.yaml not found at {path}"
        )
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise FeatureConfigError(
            f"feature_config.yaml must be a mapping at the top level; "
            f"got {type(raw).__name__}"
        )
    return _parse(raw)


def _parse(raw: dict[str, Any]) -> FeatureConfig:
    unknown = set(raw.keys()) - _TOP_LEVEL_KEYS
    if unknown:
        raise FeatureConfigError(
            f"unknown top-level keys in feature_config.yaml: {sorted(unknown)}"
        )
    missing = _TOP_LEVEL_KEYS - set(raw.keys())
    if missing:
        raise FeatureConfigError(
            f"missing required top-level keys: {sorted(missing)}"
        )

    norm_ver = raw["normalization_version"]
    if not isinstance(norm_ver, str) or not norm_ver:
        raise FeatureConfigError(
            f"normalization_version must be a non-empty string; got {norm_ver!r}"
        )

    madden_columns = raw["madden_columns"]
    if not isinstance(madden_columns, list) or not madden_columns:
        raise FeatureConfigError("madden_columns must be a non-empty list")
    if not all(isinstance(c, str) and c for c in madden_columns):
        raise FeatureConfigError(
            "madden_columns entries must all be non-empty strings"
        )

    cat_columns = raw["madden_categorical_columns"]
    if not isinstance(cat_columns, list):
        raise FeatureConfigError("madden_categorical_columns must be a list")
    if not all(isinstance(c, str) for c in cat_columns):
        raise FeatureConfigError(
            "madden_categorical_columns entries must all be strings"
        )
    extras = set(cat_columns) - set(madden_columns)
    if extras:
        raise FeatureConfigError(
            f"madden_categorical_columns has entries not in madden_columns: "
            f"{sorted(extras)}"
        )

    gf_raw = raw["game_features"]
    if not isinstance(gf_raw, dict):
        raise FeatureConfigError(
            f"game_features must be a mapping; got {type(gf_raw).__name__}"
        )
    unknown_gf = set(gf_raw.keys()) - _GAME_FEATURES_KEYS
    if unknown_gf:
        raise FeatureConfigError(
            f"unknown game_features keys: {sorted(unknown_gf)}"
        )
    missing_gf = _GAME_FEATURES_KEYS - set(gf_raw.keys())
    if missing_gf:
        raise FeatureConfigError(
            f"missing required game_features keys: {sorted(missing_gf)}"
        )

    weather = gf_raw["weather"]
    if weather not in ALLOWED_WEATHER_MODES:
        raise FeatureConfigError(
            f"game_features.weather must be one of "
            f"{sorted(ALLOWED_WEATHER_MODES)}; got {weather!r}"
        )

    officials = gf_raw["officials"]
    if officials not in ALLOWED_OFFICIALS_MODES:
        raise FeatureConfigError(
            f"game_features.officials must be one of "
            f"{sorted(ALLOWED_OFFICIALS_MODES)}; got {officials!r}"
        )

    include = gf_raw["include"]
    if not isinstance(include, list):
        raise FeatureConfigError("game_features.include must be a list")
    if not all(isinstance(c, str) for c in include):
        raise FeatureConfigError(
            "game_features.include entries must all be strings"
        )
    unknown_fields = [c for c in include if c not in KNOWN_GAME_LEVEL_FIELDS]
    if unknown_fields:
        raise FeatureConfigError(
            f"unknown game-level identifiers in game_features.include: "
            f"{unknown_fields}"
        )

    slot_shapes = raw["slot_shapes"]
    if not isinstance(slot_shapes, list) or not slot_shapes:
        raise FeatureConfigError("slot_shapes must be a non-empty list")
    bad_shapes = [s for s in slot_shapes if s not in ALLOWED_SLOT_SHAPES]
    if bad_shapes:
        raise FeatureConfigError(
            f"slot_shapes entries must each be one of "
            f"{sorted(ALLOWED_SLOT_SHAPES)}; got {bad_shapes}"
        )

    return FeatureConfig(
        normalization_version=norm_ver,
        madden_columns=tuple(madden_columns),
        madden_categorical_columns=tuple(cat_columns),
        game_features=GameFeaturesConfig(
            weather=weather,
            officials=officials,
            include=tuple(include),
        ),
        slot_shapes=tuple(slot_shapes),
    )


def validate_madden_columns_exist(
    config: FeatureConfig, madden_header: list[str]
) -> None:
    """Reject configs that reference Madden columns absent from the header (FE-CFG-02)."""
    header_set = set(madden_header)
    missing = [c for c in config.madden_columns if c not in header_set]
    if missing:
        raise FeatureConfigError(
            f"madden_columns references columns not in madden_2024.csv header: "
            f"{missing}"
        )
