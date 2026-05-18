"""Categorical encoding policy + FeatureEncoder (§3.6, TR-CAT-01..06, TR-SHAPE-*).

Phase 2 emits integer-coded categoricals; this module decides which Phase 2
columns route to which vocab key, classifies each column as numeric / one-hot /
embedding, and assembles the model-side ``nn.Module`` that turns a batch of
rows into the flat ``(B, d_in)`` tensor the regression head expects.

Sharing convention: a single ``nn.Embedding`` (or one-hot template) is built
per **vocab key**, not per physical parquet column. Physical columns that
share a vocab (e.g., ``home_team_code`` and ``away_team_code`` both use
``team_codes``) all look up against the same table — standard ML practice and
what the vocab-keyed ``embedding_dims`` config field implies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


# Reserved label columns; never passed to the model as features.
LABEL_COLUMNS: tuple[str, ...] = ("home_score", "away_score")

# GameId is the join key; never a feature.
GAME_ID_COLUMN: str = "GameId"

# Cardinality boundary for low-card vs high-card (TR-CAT-01 / TR-CAT-02).
LOW_CARD_THRESHOLD: int = 8

# Phase 2 emits -1 as a NULL_SENTINEL for legitimately-missing categorical cells
# (e.g., a roster slot a team doesn't fill — features_pos has empty AwayWR4 for
# 3-WR sets). The encoder bumps every categorical index by +1 so:
#   - input  -1  → bumped index 0  → reserved "null" slot
#   - input   0  → bumped index 1  → vocab entry 0
#   - input N-1  → bumped index N  → vocab entry N-1
# Embedding tables and one-hot widths are sized to ``vocab_size + 1`` to fit the
# extra null slot. The null embedding is a learnable representation of "missing."
NULL_BUMP: int = 1


def column_to_vocab_key(col_name: str, vocab_keys: frozenset[str]) -> Optional[str]:
    """Map a Phase 2 parquet column to a vocab key, or ``None`` if numeric.

    The rules are derived from the Phase 2 column-naming convention:

    - Exact match against a vocab key (``day_of_week``, ``roof``, ``surface``,
      ``stadium``).
    - Suffix ``_team_code`` → ``team_codes`` (``home_team_code``, ``away_team_code``).
    - Suffix ``_coach`` → ``coaches`` (``home_coach``, ``away_coach``).
    - Prefix ``official_`` → ``officials`` (the 7 ``official_*`` columns).
    - Suffix ``_madden_archetype`` → ``Archetype`` (per-slot Madden columns).
    - Suffix ``_position`` → ``positions`` (per-slot position columns, B-flat only).

    Any other column is numeric. The rule set is pinned to v1 Phase 2 outputs;
    column-naming changes in Phase 2 need a corresponding update here.
    """
    if col_name in vocab_keys:
        return col_name
    if col_name.endswith("_team_code") and "team_codes" in vocab_keys:
        return "team_codes"
    if col_name.endswith("_coach") and "coaches" in vocab_keys:
        return "coaches"
    if col_name.startswith("official_") and "officials" in vocab_keys:
        return "officials"
    if col_name.endswith("_madden_archetype") and "Archetype" in vocab_keys:
        return "Archetype"
    if col_name.endswith("_position") and "positions" in vocab_keys:
        return "positions"
    return None


@dataclass(frozen=True)
class ColumnClassification:
    """Routing decision for every column in a Phase 2 feature parquet.

    All four tuples are sorted lexicographically; this is the canonical order
    the encoder uses when concatenating its output tensor (TR-CAT-05).
    """

    numeric: tuple[str, ...]
    low_card_categorical: tuple[str, ...]
    high_card_categorical: tuple[str, ...]
    labels: tuple[str, ...]

    # Per-column vocab key (only populated for entries in the two categorical
    # tuples). Lets the encoder look up the right shared embedding/one-hot.
    column_vocab_key: Mapping[str, str]


def classify_columns(
    feature_columns: list[str],
    vocab: Mapping[str, list[str]],
) -> ColumnClassification:
    """Partition a Phase 2 feature parquet's columns into the four roles (TR-CAT-01..06)."""
    vocab_keys = frozenset(vocab.keys())
    numeric: list[str] = []
    low: list[str] = []
    high: list[str] = []
    labels: list[str] = []
    column_vocab_key: dict[str, str] = {}

    for col in feature_columns:
        if col == GAME_ID_COLUMN:
            continue
        if col in LABEL_COLUMNS:
            labels.append(col)
            continue
        vocab_key = column_to_vocab_key(col, vocab_keys)
        if vocab_key is None:
            numeric.append(col)
            continue
        column_vocab_key[col] = vocab_key
        size = len(vocab[vocab_key])
        if size <= LOW_CARD_THRESHOLD:
            low.append(col)
        else:
            high.append(col)

    return ColumnClassification(
        numeric=tuple(sorted(numeric)),
        low_card_categorical=tuple(sorted(low)),
        high_card_categorical=tuple(sorted(high)),
        labels=tuple(sorted(labels)),
        column_vocab_key=dict(column_vocab_key),
    )


class FeatureEncoder(nn.Module):
    """Turns a batch of rows into a ``(B, d_in)`` flat tensor (TR-CAT-05, TR-MODEL-05).

    ``forward`` accepts three pre-split inputs the caller builds via
    :func:`prepare_batch`:

    - ``numeric`` — ``(B, len(numeric))`` ``float32`` tensor.
    - ``low_card`` — dict of ``col_name → (B,)`` ``int64`` index tensors.
    - ``high_card`` — dict of ``col_name → (B,)`` ``int64`` index tensors.

    Embedding weights are shared across physical columns that point at the
    same vocab key (e.g., ``home_team_code`` and ``away_team_code`` share one
    ``team_codes`` embedding). One-hot widths come straight from the vocab.
    """

    def __init__(
        self,
        classification: ColumnClassification,
        vocab: Mapping[str, list[str]],
        embedding_dims: Mapping[str, int],
    ) -> None:
        super().__init__()
        self.classification = classification
        # Snapshot the values we'll need without holding a reference to mutable inputs.
        self._vocab_sizes: dict[str, int] = {k: len(v) for k, v in vocab.items()}
        self._embedding_dims = dict(embedding_dims)

        # Unique vocab keys actually in use for each policy. Sorted for determinism.
        self._low_card_vocab_keys: tuple[str, ...] = tuple(sorted({
            classification.column_vocab_key[c] for c in classification.low_card_categorical
        }))
        self._high_card_vocab_keys: tuple[str, ...] = tuple(sorted({
            classification.column_vocab_key[c] for c in classification.high_card_categorical
        }))

        # One embedding per high-card vocab key, shared across all physical columns.
        # +1 entry for the null slot (NULL_SENTINEL = -1 bumped to 0).
        self.embeddings = nn.ModuleDict({
            key: nn.Embedding(
                num_embeddings=self._vocab_sizes[key] + NULL_BUMP,
                embedding_dim=self._embedding_dims[key],
            )
            for key in self._high_card_vocab_keys
        })

    @property
    def d_in(self) -> int:
        """Total flat-vector width: numeric + one-hots (vocab+1) + embeddings (TR-CAT-06).

        Low-card one-hots include the NULL slot, so each contributes
        ``vocab_size + 1`` columns. High-card embeddings have a fixed output
        dim regardless of vocab size; the extra null entry only widens the
        embedding table, not the model's input.
        """
        width = len(self.classification.numeric)
        for col in self.classification.low_card_categorical:
            width += self._vocab_sizes[self.classification.column_vocab_key[col]] + NULL_BUMP
        for col in self.classification.high_card_categorical:
            width += self._embedding_dims[self.classification.column_vocab_key[col]]
        return width

    def forward(
        self,
        numeric: torch.Tensor,
        low_card: Mapping[str, torch.Tensor],
        high_card: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build the ``(B, d_in)`` flat input vector in stable concat order (TR-CAT-05).

        Every categorical index is bumped by +1 so the Phase 2 NULL_SENTINEL
        (-1) maps to the reserved null slot at index 0.
        """
        parts: list[torch.Tensor] = [numeric.to(torch.float32)]
        # Sorted column order ensures byte-stable d_in layout — matches classify_columns.
        for col in self.classification.low_card_categorical:
            key = self.classification.column_vocab_key[col]
            size = self._vocab_sizes[key] + NULL_BUMP
            idx = (low_card[col].to(torch.long) + NULL_BUMP)
            parts.append(F.one_hot(idx, num_classes=size).to(torch.float32))
        for col in self.classification.high_card_categorical:
            key = self.classification.column_vocab_key[col]
            idx = (high_card[col].to(torch.long) + NULL_BUMP)
            parts.append(self.embeddings[key](idx))
        return torch.cat(parts, dim=1)


def prepare_batch(
    df: pd.DataFrame,
    classification: ColumnClassification,
    device: Optional[torch.device] = None,
) -> dict[str, object]:
    """Convert a DataFrame slice into the three tensors :meth:`FeatureEncoder.forward` expects.

    Returns a dict with keys ``"numeric"``, ``"low_card"``, ``"high_card"``,
    ``"labels"`` (a ``(B, 2)`` ``float32`` tensor of ``[home_score, away_score]``),
    and ``"game_ids"`` (a tuple of ``str`` in row order). Numeric NaNs are
    filled with ``0.0`` so the model never sees ``nan`` in its input vector
    (per TR-SHAPE-04 trust the upstream parquet's column types, but ``days_rest_*``
    is parquet-native nullable per Phase 2's contract).
    """
    if device is None:
        device = torch.device("cpu")

    # Numeric block — fill any NaN with 0.0 so downstream matmuls don't NaN-poison.
    if classification.numeric:
        num_df = df[list(classification.numeric)].astype("float64").fillna(0.0)
        numeric = torch.tensor(num_df.to_numpy(dtype="float32"), device=device)
    else:
        numeric = torch.zeros((len(df), 0), dtype=torch.float32, device=device)

    low: dict[str, torch.Tensor] = {}
    for col in classification.low_card_categorical:
        idx = df[col].to_numpy()
        low[col] = torch.tensor(idx, dtype=torch.long, device=device)

    high: dict[str, torch.Tensor] = {}
    for col in classification.high_card_categorical:
        idx = df[col].to_numpy()
        high[col] = torch.tensor(idx, dtype=torch.long, device=device)

    if all(c in df.columns for c in classification.labels) and classification.labels:
        labels = torch.tensor(
            df[list(classification.labels)].to_numpy(dtype="float32"),
            device=device,
        )
    else:
        labels = torch.zeros((len(df), len(classification.labels)), dtype=torch.float32, device=device)

    game_ids = tuple(df[GAME_ID_COLUMN].astype(str).tolist())

    return {
        "numeric": numeric,
        "low_card": low,
        "high_card": high,
        "labels": labels,
        "game_ids": game_ids,
    }
