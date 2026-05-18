"""Tests for B-flat assembly (§3.8 / FE-FLAT-01..04)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from nflpredictor.features.flat import assemble_flat, snake_case
from nflpredictor.features.slots import CANONICAL_SLOTS, build_madden_lookup

from ._features_fixtures import (
    build_box_scores,
    build_madden,
    build_vocab_from,
    default_config,
)


def test_snake_case_examples():
    assert snake_case("Overall Rating") == "overall_rating"
    assert snake_case("Archetype") == "archetype"
    assert snake_case("Throw Accuracy Short") == "throw_accuracy_short"


def test_assemble_flat_column_inventory():
    box = build_box_scores(n_games=1)
    madden = build_madden(box)
    config = default_config()
    vocab = build_vocab_from(box, madden)

    out = assemble_flat(box, build_madden_lookup(madden), config, vocab)

    # 1 row per game.
    assert len(out) == 1
    # Column count: 1 (GameId) + 44 * 2 madden + 44 position + 44 matched = 177.
    assert len(out.columns) == 1 + 88 + 44 + 44

    # FE-OUT-06 section order: GameId, slot-madden cols, slot-position cols, slot-matched cols.
    assert out.columns[0] == "GameId"
    # First 88 after GameId are slot-major madden columns.
    assert out.columns[1] == "HomeOff01_madden_overall_rating"
    assert out.columns[2] == "HomeOff01_madden_archetype"
    assert out.columns[3] == "HomeOff02_madden_overall_rating"
    # Positions section comes after all madden cols.
    assert out.columns[89] == "HomeOff01_position"
    assert out.columns[132] == "AwayDef11_position"
    # Matched section last.
    assert out.columns[133] == "HomeOff01_matched"
    assert out.columns[-1] == "AwayDef11_matched"


def test_assemble_flat_overall_rating_values():
    box = build_box_scores(n_games=1)
    madden = build_madden(box)
    config = default_config()
    vocab = build_vocab_from(box, madden)

    out = assemble_flat(box, build_madden_lookup(madden), config, vocab)

    # build_madden encodes rating as 50 + (slot_index * 2) + 10 for offense.
    assert out.loc[0, "HomeOff01_madden_overall_rating"] == 50 + 2 + 10  # 62.0
    assert out.loc[0, "HomeOff11_madden_overall_rating"] == 50 + 22 + 10  # 82.0
    assert out.loc[0, "HomeDef01_madden_overall_rating"] == 50 + 2  # 52.0


def test_assemble_flat_categorical_codes_decode():
    box = build_box_scores(n_games=1)
    madden = build_madden(box, archetypes_by_position={"QB": "QB_Field_General"})
    config = default_config()
    vocab = build_vocab_from(box, madden)

    out = assemble_flat(box, build_madden_lookup(madden), config, vocab)
    archetype_code = int(out.loc[0, "HomeOff01_madden_archetype"])
    assert vocab.decode("Archetype", archetype_code) == "QB_Field_General"
    position_code = int(out.loc[0, "HomeOff01_position"])
    assert vocab.decode("positions", position_code) == "QB"


def test_assemble_flat_dtypes():
    box = build_box_scores(n_games=1)
    madden = build_madden(box)
    config = default_config()
    vocab = build_vocab_from(box, madden)

    out = assemble_flat(box, build_madden_lookup(madden), config, vocab)

    # Numeric Madden columns are float64.
    assert out["HomeOff01_madden_overall_rating"].dtype == "float64"
    # Categorical madden columns are int32.
    assert out["HomeOff01_madden_archetype"].dtype == "int32"
    # Position columns are int32.
    assert out["HomeOff01_position"].dtype == "int32"
    # Matched flag is int32.
    assert out["HomeOff01_matched"].dtype == "int32"


def test_assemble_flat_matched_zero_propagates():
    """A Madden row with matched=0 produces a 0 in the per-slot matched flag."""
    box = build_box_scores(n_games=1)
    madden = build_madden(box)
    # Flip the matched flag for the first slot's Madden row.
    target_id = str(box.loc[0, "HomeOff01_ID"])
    madden.loc[madden["madden_id"] == target_id, "matched"] = "0"

    config = default_config()
    vocab = build_vocab_from(box, madden)
    out = assemble_flat(box, build_madden_lookup(madden), config, vocab)

    assert out.loc[0, "HomeOff01_matched"] == 0
    assert out.loc[0, "HomeOff02_matched"] == 1
