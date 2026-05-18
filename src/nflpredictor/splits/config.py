"""splits_config.yaml loader + validator (§3.2, §4.1)."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Optional

import yaml


ALLOWED_STRATEGIES: frozenset[str] = frozenset({"S1", "S3"})

_TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    "splits_version",
    "strategies",
    "train_weeks",
    "val_weeks",
    "test_weeks",
    "s3",
})

_S3_KEYS: frozenset[str] = frozenset({"k_start"})

MIN_WEEK = 1
MAX_WEEK = 18


@dataclass(frozen=True)
class SplitsConfig:
    splits_version: str
    strategies: tuple[str, ...]
    train_weeks: tuple[int, int]
    val_weeks: tuple[int, int]
    test_weeks: tuple[int, int]
    s3_k_start: Optional[int]


class SplitsConfigError(ValueError):
    """Raised for malformed splits_config.yaml content (§3.2)."""


def load_splits_config(path: pathlib.Path) -> SplitsConfig:
    """Read, parse, and validate splits_config.yaml (SP-IN-02, SP-IN-06, SP-CFG-01..06)."""
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

    required = {"splits_version", "strategies", "train_weeks", "val_weeks", "test_weeks"}
    missing = required - set(raw.keys())
    if missing:
        raise SplitsConfigError(
            f"missing required top-level keys: {sorted(missing)}"
        )

    splits_version = raw["splits_version"]
    if not isinstance(splits_version, str) or not splits_version:
        raise SplitsConfigError(
            f"splits_version must be a non-empty string; got {splits_version!r}"
        )

    strategies = _parse_strategies(raw["strategies"])
    train_weeks = _parse_week_range("train_weeks", raw["train_weeks"])
    val_weeks = _parse_week_range("val_weeks", raw["val_weeks"])
    test_weeks = _parse_week_range("test_weeks", raw["test_weeks"])

    _validate_range_layout(train_weeks, val_weeks, test_weeks)

    s3_k_start: Optional[int] = None
    if "S3" in strategies:
        s3_k_start = _parse_s3_block(raw.get("s3"), val_weeks)
    elif "s3" in raw and raw["s3"] is not None:
        # An s3 block is allowed-but-meaningless when S3 isn't enabled; reject so
        # the config can't quietly drift from the strategies list.
        raise SplitsConfigError(
            "s3 block present but 'S3' is not in strategies"
        )

    return SplitsConfig(
        splits_version=splits_version,
        strategies=strategies,
        train_weeks=train_weeks,
        val_weeks=val_weeks,
        test_weeks=test_weeks,
        s3_k_start=s3_k_start,
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
            raise SplitsConfigError(
                f"duplicate entry in strategies: {entry!r}"
            )
        seen.add(entry)
    return tuple(value)


def _parse_week_range(field: str, value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(v, bool) for v in value)
        or not all(isinstance(v, int) for v in value)
    ):
        raise SplitsConfigError(
            f"{field} must be a [start, end] pair of integers; got {value!r}"
        )
    start, end = value
    if not (MIN_WEEK <= start <= end <= MAX_WEEK):
        raise SplitsConfigError(
            f"{field} must satisfy {MIN_WEEK} <= start <= end <= {MAX_WEEK}; "
            f"got [{start}, {end}]"
        )
    return (start, end)


def _validate_range_layout(
    train: tuple[int, int],
    val: tuple[int, int],
    test: tuple[int, int],
) -> None:
    if not (train[1] < val[0]):
        raise SplitsConfigError(
            f"train_weeks[1] ({train[1]}) must be strictly less than "
            f"val_weeks[0] ({val[0]})"
        )
    if not (val[1] < test[0]):
        raise SplitsConfigError(
            f"val_weeks[1] ({val[1]}) must be strictly less than "
            f"test_weeks[0] ({test[0]})"
        )
    if val[0] != train[1] + 1:
        raise SplitsConfigError(
            f"train_weeks and val_weeks must be contiguous: "
            f"train_weeks ends at {train[1]}, val_weeks starts at {val[0]}"
        )
    if test[0] != val[1] + 1:
        raise SplitsConfigError(
            f"val_weeks and test_weeks must be contiguous: "
            f"val_weeks ends at {val[1]}, test_weeks starts at {test[0]}"
        )


def _parse_s3_block(value: Any, val_weeks: tuple[int, int]) -> int:
    if not isinstance(value, dict):
        raise SplitsConfigError(
            "s3 block is required when 'S3' is in strategies"
        )
    unknown = set(value.keys()) - _S3_KEYS
    if unknown:
        raise SplitsConfigError(
            f"unknown keys in s3 block: {sorted(unknown)}"
        )
    if "k_start" not in value:
        raise SplitsConfigError("s3.k_start is required when 'S3' is in strategies")
    k_start = value["k_start"]
    if isinstance(k_start, bool) or not isinstance(k_start, int):
        raise SplitsConfigError(
            f"s3.k_start must be an integer; got {k_start!r}"
        )
    if not (1 <= k_start < val_weeks[1]):
        raise SplitsConfigError(
            f"s3.k_start must satisfy 1 <= k_start < val_weeks[1] "
            f"({val_weeks[1]}); got {k_start}"
        )
    return k_start
