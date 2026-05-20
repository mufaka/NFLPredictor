"""splits_config.yaml loader + validator (§3.2, §4.1)."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

import yaml


ALLOWED_STRATEGIES: frozenset[str] = frozenset({"season_holdout", "loso_cv"})

_TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    "splits_version",
    "strategies",
    "train_seasons",
    "val_season",
    "test_season",
})

MIN_SEASON = 2020
MAX_SEASON = 2025


@dataclass(frozen=True)
class SplitsConfig:
    splits_version: str
    strategies: tuple[str, ...]
    train_seasons: tuple[int, ...]
    val_season: int
    test_season: int

    @property
    def rotation_pool(self) -> tuple[int, ...]:
        """The loso_cv rotation pool — train seasons plus the val season, sorted."""
        return tuple(sorted((*self.train_seasons, self.val_season)))


class SplitsConfigError(ValueError):
    """Raised for malformed splits_config.yaml content (§3.2)."""


def load_splits_config(path: pathlib.Path) -> SplitsConfig:
    """Read, parse, and validate splits_config.yaml (SP-IN-02, SP-IN-06, SP-CFG-01..07)."""
    if not path.exists():
        raise FileNotFoundError(f"splits_config.yaml not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise SplitsConfigError(
            f"splits_config.yaml must be a mapping at the top level; "
            f"got {type(raw).__name__}"
        )
    return _parse(raw)


def _parse(raw: dict[str, Any]) -> SplitsConfig:
    unknown = set(raw.keys()) - _TOP_LEVEL_KEYS
    if unknown:
        raise SplitsConfigError(
            f"unknown top-level keys in splits_config.yaml: {sorted(unknown)}"
        )
    missing = _TOP_LEVEL_KEYS - set(raw.keys())
    if missing:
        raise SplitsConfigError(f"missing required top-level keys: {sorted(missing)}")

    splits_version = raw["splits_version"]
    if not isinstance(splits_version, str) or not splits_version:
        raise SplitsConfigError(
            f"splits_version must be a non-empty string; got {splits_version!r}"
        )

    strategies = _parse_strategies(raw["strategies"])
    train_seasons = _parse_train_seasons(raw["train_seasons"])
    val_season = _parse_season("val_season", raw["val_season"])
    test_season = _parse_season("test_season", raw["test_season"])

    _validate_role_assignment(train_seasons, val_season, test_season)

    if "loso_cv" in strategies:
        pool = sorted({*train_seasons, val_season})
        if len(pool) < 2:
            raise SplitsConfigError(
                "loso_cv needs a rotation pool of at least two seasons "
                f"(train_seasons ∪ {{val_season}}); got {pool}"
            )

    return SplitsConfig(
        splits_version=splits_version,
        strategies=strategies,
        train_seasons=train_seasons,
        val_season=val_season,
        test_season=test_season,
    )


def _parse_strategies(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SplitsConfigError(
            "strategies must be a non-empty list of strategy names"
        )
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, str):
            raise SplitsConfigError(
                f"strategies entries must be strings; got {entry!r}"
            )
        if entry not in ALLOWED_STRATEGIES:
            raise SplitsConfigError(
                f"strategies entry {entry!r} must be one of "
                f"{sorted(ALLOWED_STRATEGIES)}"
            )
        if entry in seen:
            raise SplitsConfigError(f"duplicate entry in strategies: {entry!r}")
        seen.add(entry)
    return tuple(value)


def _parse_season(field: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SplitsConfigError(f"{field} must be an integer season; got {value!r}")
    if not (MIN_SEASON <= value <= MAX_SEASON):
        raise SplitsConfigError(
            f"{field} must be in [{MIN_SEASON}, {MAX_SEASON}]; got {value}"
        )
    return value


def _parse_train_seasons(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise SplitsConfigError("train_seasons must be a non-empty list of seasons")
    seasons: list[int] = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise SplitsConfigError(
                f"train_seasons entries must be integers; got {entry!r}"
            )
        if not (MIN_SEASON <= entry <= MAX_SEASON):
            raise SplitsConfigError(
                f"train_seasons entry {entry} must be in "
                f"[{MIN_SEASON}, {MAX_SEASON}]"
            )
        if entry in seasons:
            raise SplitsConfigError(f"duplicate entry in train_seasons: {entry}")
        seasons.append(entry)
    return tuple(sorted(seasons))


def _validate_role_assignment(
    train_seasons: tuple[int, ...], val_season: int, test_season: int
) -> None:
    """SP-CFG-04 (intra-config part): the three role assignments are disjoint."""
    if val_season in train_seasons:
        raise SplitsConfigError(
            f"val_season ({val_season}) must not also appear in train_seasons"
        )
    if test_season in train_seasons:
        raise SplitsConfigError(
            f"test_season ({test_season}) must not also appear in train_seasons"
        )
    if val_season == test_season:
        raise SplitsConfigError(
            f"val_season and test_season must differ; both are {val_season}"
        )
