"""LinearRung / MlpRung shape and parameter-count assertions."""

from __future__ import annotations

import pandas as pd
import torch

from nflpredictor.train.encoders import (
    FeatureEncoder,
    classify_columns,
    prepare_batch,
)
from nflpredictor.train.models import LinearRung, MlpRung, build_model
from nflpredictor.train.config import MlpHyperparams


def _setup(d_in_target: int | None = None) -> tuple[FeatureEncoder, pd.DataFrame, "object"]:
    """Build a small encoder + matching synthetic DataFrame.

    Returns (encoder, df, classification).
    """
    vocab = {"stadium": [f"s_{i}" for i in range(10)]}
    cols = ["GameId", "x", "stadium", "home_score", "away_score"]
    cls = classify_columns(cols, vocab)
    enc = FeatureEncoder(cls, vocab, embedding_dims={"stadium": 4})
    # d_in = 1 numeric + 0 low-card + 4 (stadium embedding) = 5
    assert enc.d_in == 5
    df = pd.DataFrame({
        "GameId": ["G1", "G2", "G3", "G4"],
        "x": [1.0, 2.0, 3.0, 4.0],
        "stadium": [0, 1, 2, 3],
        "home_score": [10.0, 14.0, 17.0, 21.0],
        "away_score": [7.0,  10.0, 14.0, 17.0],
    })
    return enc, df, cls


def test_linear_rung_forward_shape() -> None:
    enc, df, cls = _setup()
    model = LinearRung(enc)
    batch = prepare_batch(df, cls)
    out = model(batch["numeric"], batch["low_card"], batch["high_card"])
    assert out.shape == (4, 2)
    assert out.dtype == torch.float32


def test_mlp_rung_forward_shape() -> None:
    enc, df, cls = _setup()
    model = MlpRung(enc, hidden_dim=16, activation="gelu", dropout=0.1)
    batch = prepare_batch(df, cls)
    out = model(batch["numeric"], batch["low_card"], batch["high_card"])
    assert out.shape == (4, 2)


def test_linear_rung_head_param_count() -> None:
    enc, _, _ = _setup()
    model = LinearRung(enc)
    # nn.Linear(d_in, 2) → d_in*2 weights + 2 biases.
    head_params = sum(p.numel() for p in model.head.parameters())
    assert head_params == enc.d_in * 2 + 2


def test_mlp_rung_head_param_count() -> None:
    enc, _, _ = _setup()
    hidden = 16
    model = MlpRung(enc, hidden_dim=hidden, activation="gelu", dropout=0.0)
    head_params = sum(p.numel() for p in model.head.parameters())
    # Linear(d_in, hidden) + activation + dropout + Linear(hidden, 2)
    expected = (enc.d_in * hidden + hidden) + (hidden * 2 + 2)
    assert head_params == expected


def test_mlp_rung_relu_path() -> None:
    enc, df, cls = _setup()
    model = MlpRung(enc, hidden_dim=8, activation="relu", dropout=0.0)
    batch = prepare_batch(df, cls)
    out = model(batch["numeric"], batch["low_card"], batch["high_card"])
    assert out.shape == (4, 2)
    # Confirm the activation layer is ReLU, not GELU.
    activation_layer = list(model.head)[1]
    assert isinstance(activation_layer, torch.nn.ReLU)


def test_build_model_dispatch() -> None:
    enc, _, _ = _setup()
    hp = MlpHyperparams(
        lr=0.001, batch_size=32, max_epochs=10, early_stop_patience=3,
        hidden_dim=8, activation="gelu", dropout=0.1,
    )
    assert isinstance(build_model("linear", enc, hp), LinearRung)
    assert isinstance(build_model("mlp", enc, hp), MlpRung)
