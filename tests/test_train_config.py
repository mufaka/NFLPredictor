"""TR-TEST-01: validate training_config.yaml schema enforcement."""

from __future__ import annotations

import pathlib

import pytest
import yaml

from nflpredictor.train.config import (
    TrainingConfigError,
    load_training_config,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "Data" / "raw" / "training_config.yaml"

# Mirrors the high-cardinality vocab keys from Data/processed/feature_vocab.json.
# Keeping this static keeps the test independent of Phase 2 outputs.
HIGH_CARD_KEYS = frozenset({
    "Archetype", "coaches", "officials", "positions", "stadium", "team_codes",
})


def _baseline() -> dict:
    """A valid config in dict form — mutate, then write back to YAML to test rejections."""
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write(tmp_path: pathlib.Path, data: dict) -> pathlib.Path:
    p = tmp_path / "training_config.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return p


# (a) the default v1 config (must accept)
def test_default_v1_config_accepts() -> None:
    cfg = load_training_config(DEFAULT_CONFIG_PATH, HIGH_CARD_KEYS)
    assert cfg.training_version == "v1"
    assert cfg.seed == 1729
    assert cfg.device == "auto"
    assert cfg.rungs == ("mean", "team_mean", "linear", "mlp")
    assert cfg.shapes == ("flat", "pos")
    assert cfg.strategies == ("S1", "S3")
    assert cfg.linear.lr == 0.001
    assert cfg.mlp.activation == "gelu"
    assert set(cfg.embedding_dims.keys()) >= HIGH_CARD_KEYS


# (b) unknown top-level key
def test_unknown_top_level_key_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["surprise_key"] = 42
    with pytest.raises(TrainingConfigError, match="unknown top-level keys"):
        load_training_config(_write(tmp_path, data), HIGH_CARD_KEYS)


# (c) rungs = [] rejected
def test_empty_rungs_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["rungs"] = []
    with pytest.raises(TrainingConfigError, match="rungs"):
        load_training_config(_write(tmp_path, data), HIGH_CARD_KEYS)


# (d) duplicate entry in rungs/shapes/strategies
@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("rungs", ["mean", "mean", "linear"]),
        ("shapes", ["flat", "flat"]),
        ("strategies", ["S1", "S1"]),
    ],
)
def test_duplicate_in_list_rejected(
    tmp_path: pathlib.Path, field: str, bad_value: list[str]
) -> None:
    data = _baseline()
    data[field] = bad_value
    with pytest.raises(TrainingConfigError, match="duplicate"):
        load_training_config(_write(tmp_path, data), HIGH_CARD_KEYS)


# (e) rungs entry outside the allowed set
def test_bad_rungs_entry_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["rungs"] = ["mean", "rocket"]
    with pytest.raises(TrainingConfigError, match="rungs"):
        load_training_config(_write(tmp_path, data), HIGH_CARD_KEYS)


# (f) missing embedding_dims for a high-card categorical
def test_missing_embedding_dim_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["embedding_dims"].pop("Archetype")
    with pytest.raises(
        TrainingConfigError,
        match="embedding_dims is missing required high-cardinality vocab keys",
    ):
        load_training_config(_write(tmp_path, data), HIGH_CARD_KEYS)


# (g) negative seed
def test_negative_seed_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["seed"] = -1
    with pytest.raises(TrainingConfigError, match="seed must be non-negative"):
        load_training_config(_write(tmp_path, data), HIGH_CARD_KEYS)


# (h) unsupported mlp.activation
def test_bad_activation_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["mlp"]["activation"] = "tanh"
    with pytest.raises(TrainingConfigError, match="mlp.activation"):
        load_training_config(_write(tmp_path, data), HIGH_CARD_KEYS)


# (i) device value outside the allowed set
@pytest.mark.parametrize("bad_device", ["gpu", "mps", "AUTO", "", None])
def test_bad_device_rejected(tmp_path: pathlib.Path, bad_device: object) -> None:
    data = _baseline()
    data["device"] = bad_device
    with pytest.raises(TrainingConfigError, match="device"):
        load_training_config(_write(tmp_path, data), HIGH_CARD_KEYS)


# Additional spot-check rejections — not in TR-TEST-01 enumeration but cheap.
def test_missing_top_level_key_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data.pop("device")
    with pytest.raises(TrainingConfigError, match="missing required top-level keys"):
        load_training_config(_write(tmp_path, data), HIGH_CARD_KEYS)


def test_bad_mlp_dropout_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["mlp"]["dropout"] = 1.0
    with pytest.raises(TrainingConfigError, match="mlp.dropout"):
        load_training_config(_write(tmp_path, data), HIGH_CARD_KEYS)


def test_nonpositive_lr_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["linear"]["lr"] = 0
    with pytest.raises(TrainingConfigError, match="linear.lr"):
        load_training_config(_write(tmp_path, data), HIGH_CARD_KEYS)
