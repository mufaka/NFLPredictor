"""Tests for B-pos assembly (§3.9 / FE-POS-01..04, FE-MAD-05, FE-TEST-09)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from nflpredictor.features.pos import CANONICAL_BPOS_SLOTS, assemble_pos
from nflpredictor.features.slots import build_madden_lookup
from nflpredictor.features.vocab import NULL_SENTINEL

from ._features_fixtures import (
    DEFAULT_DEF_POSITIONS,
    DEFAULT_OFF_POSITIONS,
    build_box_scores,
    build_madden,
    build_vocab_from,
    default_config,
)


def test_canonical_bpos_slots_count():
    # 29 per side × 2 sides.
    assert len(CANONICAL_BPOS_SLOTS) == 58
    assert CANONICAL_BPOS_SLOTS[0] == "HomeQB1"
    assert CANONICAL_BPOS_SLOTS[28] == "HomeDB5"
    assert CANONICAL_BPOS_SLOTS[29] == "AwayQB1"
    assert CANONICAL_BPOS_SLOTS[-1] == "AwayDB5"


def test_assemble_pos_column_inventory():
    box = build_box_scores(n_games=1)
    madden = build_madden(box)
    config = default_config()
    vocab = build_vocab_from(box, madden)

    out = assemble_pos(box, build_madden_lookup(madden), config, vocab)

    assert len(out) == 1
    # Column count: 1 (GameId) + 58 * 2 madden + 58 present + 58 matched = 233.
    assert len(out.columns) == 1 + 116 + 58 + 58
    assert out.columns[0] == "GameId"
    assert out.columns[1] == "HomeQB1_madden_overall_rating"
    assert out.columns[2] == "HomeQB1_madden_archetype"


def test_assemble_pos_default_lineup_fills_all_offensive_slots():
    """The default lineup has 1 QB, 2 RBs, 4 WRs, 1 TE, 3 OL — TE2/TE3, OL4/OL5 unfilled."""
    box = build_box_scores(n_games=1)
    madden = build_madden(box)
    config = default_config()
    vocab = build_vocab_from(box, madden)

    out = assemble_pos(box, build_madden_lookup(madden), config, vocab)

    # Filled.
    assert out.loc[0, "HomeQB1_present"] == 1
    assert out.loc[0, "HomeRB1_present"] == 1
    assert out.loc[0, "HomeRB2_present"] == 1
    assert out.loc[0, "HomeWR4_present"] == 1
    assert out.loc[0, "HomeTE1_present"] == 1
    # Default offense has only 1 TE and 3 OL.
    assert out.loc[0, "HomeTE2_present"] == 0
    assert out.loc[0, "HomeTE3_present"] == 0
    assert out.loc[0, "HomeOL3_present"] == 1
    assert out.loc[0, "HomeOL4_present"] == 0
    assert out.loc[0, "HomeOL5_present"] == 0


def test_assemble_pos_absent_slot_has_nan_and_sentinel():
    """FE-TEST-09: an unfilled canonical slot emits NaN / -1 / present=0 / matched=0."""
    box = build_box_scores(n_games=1)
    madden = build_madden(box)
    config = default_config()
    vocab = build_vocab_from(box, madden)

    out = assemble_pos(box, build_madden_lookup(madden), config, vocab)

    # TE2 is absent in the default lineup.
    assert out.loc[0, "HomeTE2_present"] == 0
    assert out.loc[0, "HomeTE2_matched"] == 0
    assert math.isnan(out.loc[0, "HomeTE2_madden_overall_rating"])
    assert out.loc[0, "HomeTE2_madden_archetype"] == NULL_SENTINEL


def test_assemble_pos_within_bucket_ordering_is_box_score_slot_order():
    """A WR in HomeOff03 becomes WR1; a later WR (HomeOff07) becomes WR2 (FE-POS-02)."""
    # Custom lineup: WR at HomeOff03 and HomeOff07, other slots non-WR.
    overrides = {
        "HomeOff01_Position": "QB",
        "HomeOff02_Position": "RB",
        "HomeOff03_Position": "WR",   # → WR1
        "HomeOff04_Position": "RB",
        "HomeOff05_Position": "TE",
        "HomeOff06_Position": "OL",
        "HomeOff07_Position": "WR",   # → WR2
        "HomeOff08_Position": "OL",
        "HomeOff09_Position": "OL",
        "HomeOff10_Position": "OL",
        "HomeOff11_Position": "OL",
    }
    box = build_box_scores(games=[overrides])
    madden = build_madden(box)
    config = default_config()
    vocab = build_vocab_from(box, madden)

    out = assemble_pos(box, build_madden_lookup(madden), config, vocab)

    # The WR in HomeOff03 has Overall Rating 50 + 3*2 + 10 = 66.
    # The WR in HomeOff07 has Overall Rating 50 + 7*2 + 10 = 74.
    assert out.loc[0, "HomeWR1_madden_overall_rating"] == 66.0
    assert out.loc[0, "HomeWR2_madden_overall_rating"] == 74.0
    assert out.loc[0, "HomeWR3_present"] == 0


def test_assemble_pos_bucket_overflow_drops_and_warns(capsys):
    """6 OL on one side → 5 assigned, 1 dropped with stderr warning."""
    overrides = {
        # Pile 6 OL into the home offense, plus QB + RB to keep the row well-formed.
        "HomeOff01_Position": "QB",
        "HomeOff02_Position": "RB",
        "HomeOff03_Position": "WR",
        "HomeOff04_Position": "WR",
        "HomeOff05_Position": "WR",
        "HomeOff06_Position": "OL",
        "HomeOff07_Position": "OL",
        "HomeOff08_Position": "OL",
        "HomeOff09_Position": "OL",
        "HomeOff10_Position": "OL",
        "HomeOff11_Position": "OL",  # 6th OL → overflow
    }
    box = build_box_scores(games=[overrides])
    madden = build_madden(box)
    config = default_config()
    vocab = build_vocab_from(box, madden)

    out = assemble_pos(box, build_madden_lookup(madden), config, vocab)

    # OL1..OL5 are filled; the 6th OL was dropped.
    for idx in range(1, 6):
        assert out.loc[0, f"HomeOL{idx}_present"] == 1
    # Warning message names the GameId and bucket.
    captured = capsys.readouterr()
    assert "GameId=g1" in captured.err
    assert "bucket=OL" in captured.err
    assert "overflow" in captured.err


def test_assemble_pos_categorical_decode():
    box = build_box_scores(n_games=1)
    madden = build_madden(box, archetypes_by_position={"QB": "QB_Field_General"})
    config = default_config()
    vocab = build_vocab_from(box, madden)

    out = assemble_pos(box, build_madden_lookup(madden), config, vocab)
    qb_code = int(out.loc[0, "HomeQB1_madden_archetype"])
    assert vocab.decode("Archetype", qb_code) == "QB_Field_General"


def test_assemble_pos_dtypes():
    box = build_box_scores(n_games=1)
    madden = build_madden(box)
    config = default_config()
    vocab = build_vocab_from(box, madden)

    out = assemble_pos(box, build_madden_lookup(madden), config, vocab)

    assert out["HomeQB1_madden_overall_rating"].dtype == "float64"
    assert out["HomeQB1_madden_archetype"].dtype == "int32"
    assert out["HomeQB1_present"].dtype == "int32"
    assert out["HomeQB1_matched"].dtype == "int32"
