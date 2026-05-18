"""TR-TEST-02: categorical encoding policy + FeatureEncoder behavior."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest
import torch

from nflpredictor.train.encoders import (
    LOW_CARD_THRESHOLD,
    NULL_BUMP,
    ColumnClassification,
    FeatureEncoder,
    classify_columns,
    column_to_vocab_key,
    prepare_batch,
)
from nflpredictor.train.sources import (
    assert_label_parity,
    high_card_vocab_keys,
    load_features,
    load_vocab,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL_PROCESSED = REPO_ROOT / "Data" / "processed"


def _synthetic_vocab() -> dict[str, list[str]]:
    """A small vocab spanning both sides of the LOW_CARD_THRESHOLD boundary."""
    return {
        "roof": ["a", "b", "c", "d"],                # size 4 → low-card
        "stadium": [f"stadium_{i}" for i in range(20)],  # size 20 → high-card
        "team_codes": [f"team_{i}" for i in range(12)],  # size 12 → high-card
    }


def _synthetic_feature_columns() -> list[str]:
    return [
        "GameId",
        "week",
        "days_rest_home",
        "roof",
        "stadium",
        "home_team_code",
        "away_team_code",
        "home_score",
        "away_score",
    ]


def _synthetic_df() -> pd.DataFrame:
    return pd.DataFrame({
        "GameId": ["G1", "G2", "G3", "G4", "G5"],
        "week": [1, 2, 3, 4, 5],
        "days_rest_home": [7.0, 7.0, 4.0, 6.0, 7.0],
        "roof": [0, 1, 0, 2, 3],
        "stadium": [0, 5, 12, 19, 3],
        "home_team_code": [1, 2, 3, 4, 5],
        "away_team_code": [6, 7, 8, 9, 10],
        "home_score": [21.0, 24.0, 17.0, 30.0, 14.0],
        "away_score": [14.0, 27.0, 20.0, 23.0, 24.0],
    })


# ------------------------------ column routing ------------------------------


def test_column_to_vocab_key_rules() -> None:
    vk = frozenset({"day_of_week", "roof", "stadium", "team_codes", "coaches",
                    "officials", "Archetype", "positions"})
    assert column_to_vocab_key("day_of_week", vk) == "day_of_week"
    assert column_to_vocab_key("roof", vk) == "roof"
    assert column_to_vocab_key("home_team_code", vk) == "team_codes"
    assert column_to_vocab_key("away_team_code", vk) == "team_codes"
    assert column_to_vocab_key("home_coach", vk) == "coaches"
    assert column_to_vocab_key("official_referee", vk) == "officials"
    assert column_to_vocab_key("HomeOff01_madden_archetype", vk) == "Archetype"
    assert column_to_vocab_key("HomeOff01_position", vk) == "positions"
    # Numeric: not routed.
    assert column_to_vocab_key("days_rest_home", vk) is None
    assert column_to_vocab_key("week", vk) is None
    assert column_to_vocab_key("HomeOff01_madden_overall_rating", vk) is None
    assert column_to_vocab_key("HomeOff01_matched", vk) is None


def test_classify_routes_by_threshold() -> None:
    vocab = _synthetic_vocab()
    cls = classify_columns(_synthetic_feature_columns(), vocab)
    # GameId dropped; labels split out; week + days_rest_home numeric.
    assert cls.numeric == ("days_rest_home", "week")
    assert cls.low_card_categorical == ("roof",)
    assert sorted(cls.high_card_categorical) == ["away_team_code", "home_team_code", "stadium"]
    assert cls.labels == ("away_score", "home_score")
    # Vocab routing for each cat column.
    assert cls.column_vocab_key["roof"] == "roof"
    assert cls.column_vocab_key["stadium"] == "stadium"
    assert cls.column_vocab_key["home_team_code"] == "team_codes"
    assert cls.column_vocab_key["away_team_code"] == "team_codes"


def test_classify_sort_order_is_independent_of_input_order() -> None:
    vocab = _synthetic_vocab()
    cols_a = _synthetic_feature_columns()
    cols_b = list(reversed(cols_a))
    cls_a = classify_columns(cols_a, vocab)
    cls_b = classify_columns(cols_b, vocab)
    assert cls_a == cls_b


# ------------------------------ FeatureEncoder ------------------------------


def test_d_in_arithmetic() -> None:
    """d_in = numeric_width + sum(low_card_vocab_size + 1) + sum(embedding_dim_per_high_card_col).

    The +1 on each low-card one-hot is the NULL slot (see NULL_BUMP / TR-CAT-07).
    """
    vocab = _synthetic_vocab()
    cls = classify_columns(_synthetic_feature_columns(), vocab)
    emb_dims = {"stadium": 6, "team_codes": 4}
    enc = FeatureEncoder(cls, vocab, emb_dims)
    expected = (
        2                           # numeric: days_rest_home, week
        + (4 + 1)                   # roof one-hot (vocab size 4 + 1 null)
        + 6                         # stadium embedding (dim 6) × 1 col
        + 4 + 4                     # team_codes embedding (dim 4) × 2 cols
    )
    assert enc.d_in == expected


def test_null_sentinel_lookup() -> None:
    """Phase 2's NULL_SENTINEL (-1) maps to the reserved null slot in both encodings.

    For high-card: the embedding table has vocab_size+1 entries; idx=-1 bumps to 0.
    For low-card: the one-hot has vocab_size+1 columns; idx=-1 bumps to 0.
    """
    vocab = _synthetic_vocab()
    cls = classify_columns(_synthetic_feature_columns(), vocab)
    enc = FeatureEncoder(cls, vocab, {"stadium": 6, "team_codes": 4})
    df = pd.DataFrame({
        "GameId": ["null1", "null2"],
        "week": [1, 2],
        "days_rest_home": [7.0, 7.0],
        "roof": [-1, 0],            # first row has NULL roof
        "stadium": [-1, 5],         # first row has NULL stadium
        "home_team_code": [-1, 1],  # first row has NULL home team
        "away_team_code": [0, 1],
        "home_score": [0.0, 0.0],
        "away_score": [0.0, 0.0],
    })
    batch = prepare_batch(df, cls)
    # Forward must not crash even with -1 indices.
    out = enc(batch["numeric"], batch["low_card"], batch["high_card"])
    assert out.shape == (2, enc.d_in)
    assert torch.isfinite(out).all()


def test_null_bump_constant() -> None:
    assert NULL_BUMP == 1


def test_forward_shape() -> None:
    vocab = _synthetic_vocab()
    cls = classify_columns(_synthetic_feature_columns(), vocab)
    enc = FeatureEncoder(cls, vocab, {"stadium": 6, "team_codes": 4})
    batch = prepare_batch(_synthetic_df(), cls)
    out = enc(batch["numeric"], batch["low_card"], batch["high_card"])
    assert out.shape == (5, enc.d_in)
    assert out.dtype == torch.float32


def test_shared_embedding_across_columns_uses_same_table() -> None:
    """home_team_code and away_team_code share the team_codes embedding (TR-CAT-02 / sharing)."""
    vocab = _synthetic_vocab()
    cls = classify_columns(_synthetic_feature_columns(), vocab)
    enc = FeatureEncoder(cls, vocab, {"stadium": 6, "team_codes": 4})
    # Only one nn.Embedding per high-card vocab key in use (not one per physical column).
    expected_keys = {"stadium", "team_codes"}
    assert set(enc.embeddings.keys()) == expected_keys


def test_seeded_init_is_reproducible() -> None:
    """Identical seeds produce identical embedding weights (TR-MODEL-05 default init + manual_seed)."""
    vocab = _synthetic_vocab()
    cls = classify_columns(_synthetic_feature_columns(), vocab)

    torch.manual_seed(1729)
    enc_a = FeatureEncoder(cls, vocab, {"stadium": 6, "team_codes": 4})
    weights_a = {k: enc_a.embeddings[k].weight.detach().clone() for k in enc_a.embeddings}

    torch.manual_seed(1729)
    enc_b = FeatureEncoder(cls, vocab, {"stadium": 6, "team_codes": 4})
    for k in enc_b.embeddings:
        assert torch.equal(weights_a[k], enc_b.embeddings[k].weight)


def test_low_card_threshold_constant() -> None:
    """The cardinality boundary is fixed at 8 per TR-CAT-01 / TR-CAT-02."""
    assert LOW_CARD_THRESHOLD == 8


# -------------------- real-data integration sanity checks --------------------


def test_real_features_classify_as_expected_for_flat() -> None:
    vocab = load_vocab(REAL_PROCESSED)
    flat = load_features(REAL_PROCESSED, "flat")
    cls = classify_columns(list(flat.columns), vocab)
    # All low-card vocab keys (size ≤ 8) appear in low_card columns.
    assert set(cls.low_card_categorical) == {"day_of_week", "roof", "surface"}
    # High-card cols include all six high-card vocab keys.
    high_card_keys_in_use = {cls.column_vocab_key[c] for c in cls.high_card_categorical}
    assert high_card_keys_in_use == high_card_vocab_keys(vocab)
    # Labels routed out of feature inputs.
    assert set(cls.labels) == {"home_score", "away_score"}


def test_real_features_pos_lacks_positions_columns() -> None:
    """features_pos encodes position in column names, so the 'positions' vocab is unused."""
    vocab = load_vocab(REAL_PROCESSED)
    pos = load_features(REAL_PROCESSED, "pos")
    cls = classify_columns(list(pos.columns), vocab)
    high_card_keys_in_use = {cls.column_vocab_key[c] for c in cls.high_card_categorical}
    assert "positions" not in high_card_keys_in_use
    # The other 5 high-card vocabs are present.
    assert high_card_keys_in_use == high_card_vocab_keys(vocab) - {"positions"}


def test_real_label_parity_passes() -> None:
    flat = load_features(REAL_PROCESSED, "flat")
    pos = load_features(REAL_PROCESSED, "pos")
    assert_label_parity(flat, pos)


def test_label_parity_detects_score_mismatch() -> None:
    flat = load_features(REAL_PROCESSED, "flat")
    pos = load_features(REAL_PROCESSED, "pos").copy()
    pos.loc[0, "home_score"] = -999.0
    from nflpredictor.train.sources import LabelParityError
    with pytest.raises(LabelParityError, match="home_score"):
        assert_label_parity(flat, pos)
