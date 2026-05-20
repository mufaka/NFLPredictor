"""Encoded-vector reconstruction helpers for a single game (B-flat / B-pos rows + categorical decode).

Powers the Phase 6 walkthrough's Section 3 cells (DD-WT-02). The categorical
explainer walks the raw → integer-code → embedding-table chain that Phase 4's
``train.encoders`` produces (DD-INT-01 / TR-CAT-05).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "Data" / "processed"

FEATURES_FLAT_BASENAME = "features_flat_all.parquet"
FEATURES_POS_BASENAME = "features_pos_all.parquet"
FEATURE_VOCAB_BASENAME = "feature_vocab.json"

# Phase 4 encoder constants (TR-CAT-07). The encoder reserves index 0 for the
# null sentinel, so a vocab entry at list index i lands at integer code i + 1.
NULL_BUMP: int = 1
LOW_CARD_THRESHOLD: int = 8


def encode_one_game_flat(
    game_id: str,
    processed_dir: pathlib.Path = DEFAULT_PROCESSED_DIR,
) -> pd.Series:
    """Return the row of ``features_flat_all.parquet`` for ``game_id``."""
    return _read_one_game(processed_dir / FEATURES_FLAT_BASENAME, game_id)


def encode_one_game_pos(
    game_id: str,
    processed_dir: pathlib.Path = DEFAULT_PROCESSED_DIR,
) -> pd.Series:
    """Return the row of ``features_pos_all.parquet`` for ``game_id``."""
    return _read_one_game(processed_dir / FEATURES_POS_BASENAME, game_id)


def _read_one_game(path: pathlib.Path, game_id: str) -> pd.Series:
    df = pd.read_parquet(path)
    matched = df[df["GameId"] == game_id]
    if matched.empty:
        raise KeyError(f"GameId {game_id!r} not found in {path}")
    if len(matched) > 1:
        raise RuntimeError(
            f"GameId {game_id!r} matches {len(matched)} rows in {path}; expected one"
        )
    return matched.iloc[0]


def explain_categorical(
    column: str,
    raw_value: str,
    processed_dir: pathlib.Path = DEFAULT_PROCESSED_DIR,
) -> dict[str, Any]:
    """Walk a categorical column's raw → vocab key → integer code → routing chain.

    Returns ``{"raw", "vocab_key", "vocab_size", "integer_code", "routing",
    "embedding_table"}``. ``integer_code`` follows Phase 4's ``NULL_BUMP = 1``
    convention so the value present in the encoded vector matches the Phase 4
    encoder's input. ``routing`` is ``"one_hot"`` for low-cardinality vocabs
    (size ≤ 8) and ``"embedding"`` for high-cardinality vocabs (size > 8).
    ``embedding_table`` is the shared-table name (= ``vocab_key``); columns
    pointing at the same vocab key share a single ``nn.Embedding`` per
    TR-CAT-05.

    Raises ``ValueError`` if the column does not route to any vocab key
    (i.e., it is numeric in Phase 4's classifier), or if ``raw_value`` is
    absent from that vocab.
    """
    payload = _load_feature_vocab(processed_dir)
    entries_by_key = payload["entries"]
    column_vocab_keys = payload["column_vocab_keys"]
    vocab_key = column_vocab_keys.get(column)
    if vocab_key is None:
        raise ValueError(
            f"column {column!r} is not categorical (not present in "
            f"feature_vocab.json's column_vocab_keys map)"
        )
    entries = entries_by_key[vocab_key]
    try:
        idx = entries.index(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"raw value {raw_value!r} is not present in vocab[{vocab_key!r}] "
            f"(size {len(entries)})"
        ) from exc

    size = len(entries)
    routing = "one_hot" if size <= LOW_CARD_THRESHOLD else "embedding"
    return {
        "raw": raw_value,
        "vocab_key": vocab_key,
        "vocab_size": size,
        "integer_code": idx + NULL_BUMP,
        "routing": routing,
        "embedding_table": vocab_key,
    }


def _load_feature_vocab(processed_dir: pathlib.Path) -> dict[str, Any]:
    with (processed_dir / FEATURE_VOCAB_BASENAME).open("r", encoding="utf-8") as f:
        return json.load(f)
