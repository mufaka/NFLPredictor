"""training_config.yaml loader + validator (§3.2, §4.1)."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Iterable

import yaml


ALLOWED_RUNGS: frozenset[str] = frozenset({"mean", "team_mean", "linear", "mlp"})
ALLOWED_SHAPES: frozenset[str] = frozenset({"flat", "pos"})
ALLOWED_STRATEGIES: frozenset[str] = frozenset({"S1", "S3"})
ALLOWED_DEVICES: frozenset[str] = frozenset({"auto", "cpu", "cuda"})
ALLOWED_ACTIVATIONS: frozenset[str] = frozenset({"gelu", "relu"})

_TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    "training_version",
    "seed",
    "device",
    "rungs",
    "shapes",
    "strategies",
    "linear",
    "mlp",
    "embedding_dims",
})

_LINEAR_KEYS: frozenset[str] = frozenset({
    "lr", "batch_size", "max_epochs", "early_stop_patience",
})

_MLP_KEYS: frozenset[str] = frozenset({
    "lr", "batch_size", "max_epochs", "early_stop_patience",
    "hidden_dim", "activation", "dropout",
})


@dataclass(frozen=True)
class LinearHyperparams:
    lr: float
    batch_size: int
    max_epochs: int
    early_stop_patience: int


@dataclass(frozen=True)
class MlpHyperparams:
    lr: float
    batch_size: int
    max_epochs: int
    early_stop_patience: int
    hidden_dim: int
    activation: str
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    training_version: str
    seed: int
    device: str
    rungs: tuple[str, ...]
    shapes: tuple[str, ...]
    strategies: tuple[str, ...]
    linear: LinearHyperparams
    mlp: MlpHyperparams
    embedding_dims: dict[str, int]


class TrainingConfigError(ValueError):
    """Raised for malformed training_config.yaml content (§3.2)."""


def load_training_config(
    path: pathlib.Path,
    high_card_vocab_keys: Iterable[str],
) -> TrainingConfig:
    """Read, parse, and validate training_config.yaml (TR-IN-03, TR-IN-07, TR-CFG-01..10).

    ``high_card_vocab_keys`` is the set of vocab keys with size > 8 (per
    TR-CAT-02) that ``embedding_dims`` must cover. The caller computes this
    from ``feature_vocab.json``.
    """
    if not path.exists():
        raise FileNotFoundError(f"training_config.yaml not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)  # TR-SEC-04
    if not isinstance(raw, dict):
        raise TrainingConfigError(
            f"training_config.yaml must be a mapping at the top level; "
            f"got {type(raw).__name__}"
        )
    return _parse(raw, frozenset(high_card_vocab_keys))


def _parse(raw: dict[str, Any], required_embedding_keys: frozenset[str]) -> TrainingConfig:
    unknown = set(raw.keys()) - _TOP_LEVEL_KEYS
    if unknown:
        raise TrainingConfigError(
            f"unknown top-level keys in training_config.yaml: {sorted(unknown)}"
        )
    required = {
        "training_version", "seed", "device",
        "rungs", "shapes", "strategies",
        "linear", "mlp", "embedding_dims",
    }
    missing = required - set(raw.keys())
    if missing:
        raise TrainingConfigError(
            f"missing required top-level keys: {sorted(missing)}"
        )

    return TrainingConfig(
        training_version=_parse_training_version(raw["training_version"]),
        seed=_parse_seed(raw["seed"]),
        device=_parse_device(raw["device"]),
        rungs=_parse_enum_list("rungs", raw["rungs"], ALLOWED_RUNGS),
        shapes=_parse_enum_list("shapes", raw["shapes"], ALLOWED_SHAPES),
        strategies=_parse_enum_list("strategies", raw["strategies"], ALLOWED_STRATEGIES),
        linear=_parse_linear(raw["linear"]),
        mlp=_parse_mlp(raw["mlp"]),
        embedding_dims=_parse_embedding_dims(raw["embedding_dims"], required_embedding_keys),
    )


def _parse_training_version(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TrainingConfigError(
            f"training_version must be a non-empty string; got {value!r}"
        )
    return value


def _parse_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingConfigError(f"seed must be an integer; got {value!r}")
    if value < 0:
        raise TrainingConfigError(f"seed must be non-negative; got {value}")
    return value


def _parse_device(value: Any) -> str:
    if not isinstance(value, str) or value not in ALLOWED_DEVICES:
        raise TrainingConfigError(
            f"device must be one of {sorted(ALLOWED_DEVICES)}; got {value!r}"
        )
    return value


def _parse_enum_list(field: str, value: Any, allowed: frozenset[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TrainingConfigError(f"{field} must be a non-empty list")
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, str):
            raise TrainingConfigError(
                f"{field} entries must be strings; got {entry!r}"
            )
        if entry not in allowed:
            raise TrainingConfigError(
                f"{field} entry {entry!r} must be one of {sorted(allowed)}"
            )
        if entry in seen:
            raise TrainingConfigError(f"duplicate entry in {field}: {entry!r}")
        seen.add(entry)
    return tuple(value)


def _require_keys(block_name: str, block: Any, allowed: frozenset[str]) -> dict[str, Any]:
    if not isinstance(block, dict):
        raise TrainingConfigError(
            f"{block_name} must be a mapping; got {type(block).__name__}"
        )
    unknown = set(block.keys()) - allowed
    if unknown:
        raise TrainingConfigError(
            f"unknown keys in {block_name}: {sorted(unknown)}"
        )
    missing = allowed - set(block.keys())
    if missing:
        raise TrainingConfigError(
            f"missing required keys in {block_name}: {sorted(missing)}"
        )
    return block


def _positive_int(field: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TrainingConfigError(f"{field} must be an integer; got {value!r}")
    if value <= 0:
        raise TrainingConfigError(f"{field} must be > 0; got {value}")
    return value


def _positive_float(field: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigError(f"{field} must be a number; got {value!r}")
    fvalue = float(value)
    if not (fvalue > 0):
        raise TrainingConfigError(f"{field} must be > 0; got {value}")
    return fvalue


def _parse_linear(block: Any) -> LinearHyperparams:
    b = _require_keys("linear", block, _LINEAR_KEYS)
    return LinearHyperparams(
        lr=_positive_float("linear.lr", b["lr"]),
        batch_size=_positive_int("linear.batch_size", b["batch_size"]),
        max_epochs=_positive_int("linear.max_epochs", b["max_epochs"]),
        early_stop_patience=_positive_int(
            "linear.early_stop_patience", b["early_stop_patience"]
        ),
    )


def _parse_mlp(block: Any) -> MlpHyperparams:
    b = _require_keys("mlp", block, _MLP_KEYS)
    activation = b["activation"]
    if not isinstance(activation, str) or activation not in ALLOWED_ACTIVATIONS:
        raise TrainingConfigError(
            f"mlp.activation must be one of {sorted(ALLOWED_ACTIVATIONS)}; "
            f"got {activation!r}"
        )
    dropout = b["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise TrainingConfigError(f"mlp.dropout must be a number; got {dropout!r}")
    dropout_f = float(dropout)
    if not (0.0 <= dropout_f < 1.0):
        raise TrainingConfigError(
            f"mlp.dropout must be in [0.0, 1.0); got {dropout}"
        )
    return MlpHyperparams(
        lr=_positive_float("mlp.lr", b["lr"]),
        batch_size=_positive_int("mlp.batch_size", b["batch_size"]),
        max_epochs=_positive_int("mlp.max_epochs", b["max_epochs"]),
        early_stop_patience=_positive_int(
            "mlp.early_stop_patience", b["early_stop_patience"]
        ),
        hidden_dim=_positive_int("mlp.hidden_dim", b["hidden_dim"]),
        activation=activation,
        dropout=dropout_f,
    )


def _parse_embedding_dims(
    block: Any,
    required_keys: frozenset[str],
) -> dict[str, int]:
    if not isinstance(block, dict):
        raise TrainingConfigError(
            f"embedding_dims must be a mapping; got {type(block).__name__}"
        )
    out: dict[str, int] = {}
    for key, value in block.items():
        if not isinstance(key, str) or not key:
            raise TrainingConfigError(
                f"embedding_dims keys must be non-empty strings; got {key!r}"
            )
        if isinstance(value, bool) or not isinstance(value, int):
            raise TrainingConfigError(
                f"embedding_dims[{key!r}] must be an integer; got {value!r}"
            )
        if value <= 0:
            raise TrainingConfigError(
                f"embedding_dims[{key!r}] must be > 0; got {value}"
            )
        out[key] = value
    missing = required_keys - set(out.keys())
    if missing:
        raise TrainingConfigError(
            f"embedding_dims is missing required high-cardinality vocab keys: "
            f"{sorted(missing)}"
        )
    return out
