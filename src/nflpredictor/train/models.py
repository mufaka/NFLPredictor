"""Learned-rung model definitions (TR-RUNG-03, TR-RUNG-04, TR-MODEL-01..05)."""

from __future__ import annotations

from torch import nn

from .config import MlpHyperparams
from .encoders import FeatureEncoder


class LinearRung(nn.Module):
    """Rung 2: ``encoder → nn.Linear(d_in, 2)`` (TR-RUNG-03)."""

    def __init__(self, encoder: FeatureEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.d_in, 2)  # TR-MODEL-01

    def forward(self, numeric, low_card, high_card):  # type: ignore[no-untyped-def]
        x = self.encoder(numeric, low_card, high_card)
        return self.head(x)


class MlpRung(nn.Module):
    """Rung 3: ``encoder → Linear(d_in, h) → activation → Dropout → Linear(h, 2)`` (TR-RUNG-04)."""

    def __init__(
        self,
        encoder: FeatureEncoder,
        *,
        hidden_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        act: nn.Module
        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        else:
            raise ValueError(f"unsupported activation {activation!r}")  # config.py rejects, defense-in-depth
        self.head = nn.Sequential(
            nn.Linear(encoder.d_in, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, numeric, low_card, high_card):  # type: ignore[no-untyped-def]
        x = self.encoder(numeric, low_card, high_card)
        return self.head(x)


def build_model(
    rung_id: str,
    encoder: FeatureEncoder,
    mlp_hyperparams: MlpHyperparams,
) -> nn.Module:
    """Construct a learned-rung model for ``rung_id`` over a fresh encoder."""
    if rung_id == "linear":
        return LinearRung(encoder)
    if rung_id == "mlp":
        return MlpRung(
            encoder,
            hidden_dim=mlp_hyperparams.hidden_dim,
            activation=mlp_hyperparams.activation,
            dropout=mlp_hyperparams.dropout,
        )
    raise ValueError(f"build_model called for non-learned rung {rung_id!r}")
