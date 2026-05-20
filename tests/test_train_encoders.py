"""TR-TEST-02: categorical encoding policy + FeatureEncoder behavior."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest
import torch

from nflpredictor.train.encoders import (
    LOW_CARD_THRESHOLD,
    NON_MODEL_VOCAB_KEYS,
    NULL_BUMP,
    ColumnClassification,
    FeatureEncoder,
    classify_columns,
    prepare_batch,
)
from nflpredictor.train.sources import (
    assert_label_parity,
    high_card_vocab_keys,
    load_column_vocab_keys,
    load_features,
    load_vocab,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL_PROCESSED = REPO_ROOT / "Data" / "processed"


def _synthetic_vocab() -> dict[str, list[str]]:
    """A small vocab spanning both sides of the LOW_CARD_THRESHOLD boundary.

    ``archetype`` is the representative high-card vocab shared across two
    physical columns (a real model feature). ``team_codes`` is high-card too
    but listed in NON_MODEL_VOCAB_KEYS — it must never reach the model.
    """
    return {
        "roof": ["a", "b", "c", "d"],                    # size 4 → low-card
        "stadium": [f"stadium_{i}" for i in range(20)],  # size 20 → high-card
        "archetype": [f"arch_{i}" for i in range(12)],   # size 12 → high-card feature
        "team_codes": [f"team_{i}" for i in range(12)],  # high-card, NON-MODEL
        "coaches": [f"coach_{i}" for i in range(40)],    # high-card, NON-MODEL
        "officials": [f"ref_{i}" for i in range(50)],    # high-card, NON-MODEL
    }


def _synthetic_feature_columns() -> list[str]:
    return [
        "GameId",
        "season",
        "week",
        "days_rest_home",
        "roof",
        "stadium",
        "home_qb_archetype",
        "away_qb_archetype",
        "home_team_code",
        "away_team_code",
        "home_coach",
        "away_coach",
        "official_referee",
        "home_score",
        "away_score",
    ]


def _synthetic_column_vocab_keys() -> dict[str, str]:
    """Mirrors the column→vocab_key map Phase 2 emits for the synthetic columns."""
    return {
        "roof": "roof",
        "stadium": "stadium",
        "home_qb_archetype": "archetype",
        "away_qb_archetype": "archetype",
        "home_team_code": "team_codes",
        "away_team_code": "team_codes",
        "home_coach": "coaches",
        "away_coach": "coaches",
        "official_referee": "officials",
    }


# Just the categoricals that survive as model features (NON_MODEL keys dropped).
_MODEL_COLUMN_VOCAB_KEYS: dict[str, str] = {
    "roof": "roof",
    "stadium": "stadium",
    "home_qb_archetype": "archetype",
    "away_qb_archetype": "archetype",
}


def _synthetic_df() -> pd.DataFrame:
    return pd.DataFrame({
        "GameId": ["G1", "G2", "G3", "G4", "G5"],
        "season": [2020, 2021, 2022, 2023, 2024],
        "week": [1, 2, 3, 4, 5],
        "days_rest_home": [7.0, 7.0, 4.0, 6.0, 7.0],
        "roof": [0, 1, 0, 2, 3],
        "stadium": [0, 5, 12, 19, 3],
        "home_qb_archetype": [0, 1, 2, 3, 4],
        "away_qb_archetype": [5, 6, 7, 8, 9],
        "home_team_code": [1, 2, 3, 4, 5],
        "away_team_code": [6, 7, 8, 9, 10],
        "home_coach": [0, 1, 2, 3, 4],
        "away_coach": [5, 6, 7, 8, 9],
        "official_referee": [0, 1, 2, 3, 4],
        "home_score": [21.0, 24.0, 17.0, 30.0, 14.0],
        "away_score": [14.0, 27.0, 20.0, 23.0, 24.0],
    })


# ------------------------------ column routing ------------------------------


def test_classify_uses_column_vocab_keys_map() -> None:
    """classify_columns reads the data-driven column→vocab_key map (no suffix matching)."""
    vocab = _synthetic_vocab()
    cvk = _synthetic_column_vocab_keys()
    cls = classify_columns(_synthetic_feature_columns(), vocab, cvk)
    # GameId + season dropped (identifiers); labels split out; week +
    # days_rest_home numeric (absent from map).
    assert cls.numeric == ("days_rest_home", "week")
    # roof has vocab size 4 → low-card. stadium=20, archetype=12 → high-card.
    # team_codes/coaches/officials are NON-MODEL and never classified.
    assert cls.low_card_categorical == ("roof",)
    assert sorted(cls.high_card_categorical) == ["away_qb_archetype", "home_qb_archetype", "stadium"]
    assert cls.labels == ("away_score", "home_score")
    # Vocab routing covers only the surviving model categoricals.
    assert cls.column_vocab_key == _MODEL_COLUMN_VOCAB_KEYS


def test_classify_excludes_identity_categoricals() -> None:
    """Team / coach / official columns stay in the parquet but never reach the model.

    They encode raw identity, not a rated attribute, so feeding them to the
    learned rungs would just memorize team-level scoring.
    """
    vocab = _synthetic_vocab()
    cvk = _synthetic_column_vocab_keys()
    cls = classify_columns(_synthetic_feature_columns(), vocab, cvk)
    classified = (
        set(cls.numeric)
        | set(cls.low_card_categorical)
        | set(cls.high_card_categorical)
        | set(cls.labels)
    )
    for identity_col in (
        "home_team_code", "away_team_code",
        "home_coach", "away_coach",
        "official_referee",
    ):
        assert identity_col not in classified
        assert identity_col not in cls.column_vocab_key
    # The excluded vocab keys are exactly the documented NON_MODEL set.
    assert NON_MODEL_VOCAB_KEYS == frozenset({"team_codes", "coaches", "officials"})


def test_classify_unknown_columns_become_numeric() -> None:
    """A column absent from column_vocab_keys lands in numeric, never raises."""
    vocab = _synthetic_vocab()
    cvk = _synthetic_column_vocab_keys()
    cols = _synthetic_feature_columns() + ["mystery_column", "another_numeric"]
    cls = classify_columns(cols, vocab, cvk)
    assert "mystery_column" in cls.numeric
    assert "another_numeric" in cls.numeric


def test_classify_threshold_is_configurable() -> None:
    """one_hot_threshold flips a vocab from low-card to high-card or back."""
    vocab = _synthetic_vocab()
    cvk = _synthetic_column_vocab_keys()
    # Default threshold 8: roof (size 4) is low-card.
    cls_default = classify_columns(_synthetic_feature_columns(), vocab, cvk)
    assert "roof" in cls_default.low_card_categorical
    # Threshold 3: roof (size 4) becomes high-card.
    cls_low_threshold = classify_columns(
        _synthetic_feature_columns(), vocab, cvk, one_hot_threshold=3
    )
    assert "roof" in cls_low_threshold.high_card_categorical


def test_classify_raises_when_map_points_to_missing_vocab_key() -> None:
    """A column→vocab_key entry pointing at an absent vocab is a fail-fast bug."""
    vocab = _synthetic_vocab()
    cvk = {"roof": "roof", "phantom_col": "no_such_vocab_key"}
    cols = _synthetic_feature_columns() + ["phantom_col"]
    with pytest.raises(KeyError, match="no_such_vocab_key"):
        classify_columns(cols, vocab, cvk)


def test_classify_sort_order_is_independent_of_input_order() -> None:
    vocab = _synthetic_vocab()
    cvk = _synthetic_column_vocab_keys()
    cols_a = _synthetic_feature_columns()
    cols_b = list(reversed(cols_a))
    cls_a = classify_columns(cols_a, vocab, cvk)
    cls_b = classify_columns(cols_b, vocab, cvk)
    assert cls_a == cls_b


# ------------------------------ FeatureEncoder ------------------------------


def test_d_in_arithmetic() -> None:
    """d_in = numeric_width + sum(low_card_vocab_size + 1) + sum(embedding_dim_per_high_card_col).

    The +1 on each low-card one-hot is the NULL slot (see NULL_BUMP / TR-CAT-07).
    """
    vocab = _synthetic_vocab()
    cls = classify_columns(_synthetic_feature_columns(), vocab, _synthetic_column_vocab_keys())
    emb_dims = {"stadium": 6, "archetype": 4}
    enc = FeatureEncoder(cls, vocab, emb_dims)
    expected = (
        2                           # numeric: days_rest_home, week
        + (4 + 1)                   # roof one-hot (vocab size 4 + 1 null)
        + 6                         # stadium embedding (dim 6) × 1 col
        + 4 + 4                     # archetype embedding (dim 4) × 2 cols
    )
    assert enc.d_in == expected


def test_null_sentinel_lookup() -> None:
    """Phase 2's NULL_SENTINEL (-1) maps to the reserved null slot in both encodings.

    For high-card: the embedding table has vocab_size+1 entries; idx=-1 bumps to 0.
    For low-card: the one-hot has vocab_size+1 columns; idx=-1 bumps to 0.
    """
    vocab = _synthetic_vocab()
    cls = classify_columns(_synthetic_feature_columns(), vocab, _synthetic_column_vocab_keys())
    enc = FeatureEncoder(cls, vocab, {"stadium": 6, "archetype": 4})
    df = pd.DataFrame({
        "GameId": ["null1", "null2"],
        "week": [1, 2],
        "days_rest_home": [7.0, 7.0],
        "roof": [-1, 0],                  # first row has NULL roof
        "stadium": [-1, 5],               # first row has NULL stadium
        "home_qb_archetype": [-1, 1],     # first row has NULL home archetype
        "away_qb_archetype": [0, 1],
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
    cls = classify_columns(_synthetic_feature_columns(), vocab, _synthetic_column_vocab_keys())
    enc = FeatureEncoder(cls, vocab, {"stadium": 6, "archetype": 4})
    batch = prepare_batch(_synthetic_df(), cls)
    out = enc(batch["numeric"], batch["low_card"], batch["high_card"])
    assert out.shape == (5, enc.d_in)
    assert out.dtype == torch.float32


def test_shared_embedding_across_columns_uses_same_table() -> None:
    """home_qb_archetype and away_qb_archetype share the archetype embedding (TR-CAT-02 / sharing)."""
    vocab = _synthetic_vocab()
    cls = classify_columns(_synthetic_feature_columns(), vocab, _synthetic_column_vocab_keys())
    enc = FeatureEncoder(cls, vocab, {"stadium": 6, "archetype": 4})
    # Only one nn.Embedding per high-card vocab key in use (not one per physical column).
    expected_keys = {"stadium", "archetype"}
    assert set(enc.embeddings.keys()) == expected_keys


def test_seeded_init_is_reproducible() -> None:
    """Identical seeds produce identical embedding weights (TR-MODEL-05 default init + manual_seed)."""
    vocab = _synthetic_vocab()
    cls = classify_columns(_synthetic_feature_columns(), vocab, _synthetic_column_vocab_keys())

    torch.manual_seed(1729)
    enc_a = FeatureEncoder(cls, vocab, {"stadium": 6, "archetype": 4})
    weights_a = {k: enc_a.embeddings[k].weight.detach().clone() for k in enc_a.embeddings}

    torch.manual_seed(1729)
    enc_b = FeatureEncoder(cls, vocab, {"stadium": 6, "archetype": 4})
    for k in enc_b.embeddings:
        assert torch.equal(weights_a[k], enc_b.embeddings[k].weight)


def test_low_card_threshold_constant() -> None:
    """The cardinality boundary is fixed at 8 per TR-CAT-01 / TR-CAT-02."""
    assert LOW_CARD_THRESHOLD == 8


# -------------------- real-data integration sanity checks --------------------


def test_real_features_classify_as_expected_for_flat() -> None:
    vocab = load_vocab(REAL_PROCESSED)
    cvk = load_column_vocab_keys(REAL_PROCESSED)
    flat = load_features(REAL_PROCESSED, "flat")
    cls = classify_columns(list(flat.columns), vocab, cvk)
    # All low-card vocab keys (size ≤ 8) appear in low_card columns.
    assert set(cls.low_card_categorical) == {"day_of_week", "roof", "surface"}
    # High-card cols cover every high-card vocab key except the NON-MODEL ones
    # (team codes, coaches, officials are identity-only — never model features).
    high_card_keys_in_use = {cls.column_vocab_key[c] for c in cls.high_card_categorical}
    assert high_card_keys_in_use == high_card_vocab_keys(vocab) - NON_MODEL_VOCAB_KEYS
    # Labels routed out of feature inputs.
    assert set(cls.labels) == {"home_score", "away_score"}


def test_real_features_exclude_identity_categoricals() -> None:
    """team_code / coach / official columns are present in the parquet but not classified."""
    vocab = load_vocab(REAL_PROCESSED)
    cvk = load_column_vocab_keys(REAL_PROCESSED)
    flat = load_features(REAL_PROCESSED, "flat")
    # The identity columns physically exist (the team_mean baseline reads them).
    assert {"home_team_code", "away_team_code", "home_coach", "away_coach"} <= set(flat.columns)
    cls = classify_columns(list(flat.columns), vocab, cvk)
    classified = (
        set(cls.numeric)
        | set(cls.low_card_categorical)
        | set(cls.high_card_categorical)
        | set(cls.labels)
    )
    # No column routed to a NON_MODEL vocab key survives into the classification.
    for col, key in cvk.items():
        if key in NON_MODEL_VOCAB_KEYS:
            assert col not in classified


def test_real_features_pos_lacks_positions_columns() -> None:
    """features_pos encodes position in column names, so the 'positions' vocab is unused."""
    vocab = load_vocab(REAL_PROCESSED)
    cvk = load_column_vocab_keys(REAL_PROCESSED)
    pos = load_features(REAL_PROCESSED, "pos")
    cls = classify_columns(list(pos.columns), vocab, cvk)
    high_card_keys_in_use = {cls.column_vocab_key[c] for c in cls.high_card_categorical}
    assert "positions" not in high_card_keys_in_use
    # Every other high-card vocab key is present except positions + NON-MODEL keys.
    assert high_card_keys_in_use == (
        high_card_vocab_keys(vocab) - {"positions"} - NON_MODEL_VOCAB_KEYS
    )


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
