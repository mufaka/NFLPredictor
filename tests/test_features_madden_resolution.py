"""Tests for the per-slot Madden read (§3.7 / FE-MAD-01..04)."""

from __future__ import annotations

import pandas as pd
import pytest

from nflpredictor.features.config import FeatureConfig, GameFeaturesConfig
from nflpredictor.features.slots import (
    CANONICAL_SLOTS,
    Starter,
    build_madden_lookup,
    iter_starters,
    resolve_madden_features,
)


def _tiny_config(
    madden_columns: tuple[str, ...] = ("Overall Rating", "Archetype"),
    madden_categorical_columns: tuple[str, ...] = ("Archetype",),
) -> FeatureConfig:
    return FeatureConfig(
        normalization_version="v1",
        madden_columns=madden_columns,
        madden_categorical_columns=madden_categorical_columns,
        game_features=GameFeaturesConfig(
            weather="parsed", officials="included", include=("week",)
        ),
        slot_shapes=("flat",),
    )


def _tiny_madden() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"madden_id": "2024-00001", "Overall Rating": "85", "Archetype": "QB_Field_General", "matched": "1"},
            {"madden_id": "2024-00002", "Overall Rating": "72", "Archetype": "WR_Deep_Threat",   "matched": "1"},
            {"madden_id": "2024-00003", "Overall Rating": "68.0", "Archetype": "RB_Power",       "matched": "0"},  # null-filled
        ]
    )


def test_canonical_slots_count_and_order():
    assert len(CANONICAL_SLOTS) == 44
    assert CANONICAL_SLOTS[0] == "HomeOff01"
    assert CANONICAL_SLOTS[10] == "HomeOff11"
    assert CANONICAL_SLOTS[11] == "HomeDef01"
    assert CANONICAL_SLOTS[22] == "AwayOff01"
    assert CANONICAL_SLOTS[33] == "AwayDef01"
    assert CANONICAL_SLOTS[-1] == "AwayDef11"


def test_resolve_returns_columns_and_matched_flag():
    lookup = build_madden_lookup(_tiny_madden())
    features, matched = resolve_madden_features("2024-00001", lookup, _tiny_config())
    assert features == {"Overall Rating": 85.0, "Archetype": "QB_Field_General"}
    assert matched == 1
    assert isinstance(features["Overall Rating"], float)


def test_resolve_unmatched_row_still_returns_filled_values():
    lookup = build_madden_lookup(_tiny_madden())
    features, matched = resolve_madden_features("2024-00003", lookup, _tiny_config())
    assert features["Overall Rating"] == 68.0
    assert features["Archetype"] == "RB_Power"
    assert matched == 0


def test_numeric_pass_through_raises_on_non_numeric():
    df = pd.DataFrame([{
        "madden_id": "2024-00001",
        "Overall Rating": "85",
        "Height": "6'5",   # non-numeric, not declared categorical
        "matched": "1",
    }])
    lookup = build_madden_lookup(df)
    config = _tiny_config(
        madden_columns=("Overall Rating", "Height"),
        madden_categorical_columns=(),
    )
    with pytest.raises(ValueError, match="non-numeric value in Madden column 'Height'"):
        resolve_madden_features("2024-00001", lookup, config)


def test_missing_madden_id_raises():
    lookup = build_madden_lookup(_tiny_madden())
    with pytest.raises(KeyError, match="2024-99999"):
        resolve_madden_features("2024-99999", lookup, _tiny_config())


def test_categorical_empty_value_becomes_none():
    df = pd.DataFrame([{
        "madden_id": "2024-00001",
        "Overall Rating": "85",
        "Archetype": "",  # empty string
        "matched": "1",
    }])
    lookup = build_madden_lookup(df)
    features, _ = resolve_madden_features("2024-00001", lookup, _tiny_config())
    assert features["Archetype"] is None


def _tiny_box_scores_row(game_id: str = "g1") -> dict:
    row: dict = {"GameId": game_id}
    for slot in CANONICAL_SLOTS:
        row[f"{slot}_Position"] = "QB" if slot.endswith("01") else "RB"
        row[f"{slot}_ID"] = f"2024-{int(slot[-2:]):05d}"
        row[f"{slot}_Name"] = f"Player {slot}"
    return row


def test_iter_starters_yields_44_per_game():
    df = pd.DataFrame([_tiny_box_scores_row("g1"), _tiny_box_scores_row("g2")])
    starters = list(iter_starters(df))
    assert len(starters) == 88
    # First starter is HomeOff01 of g1; 45th is HomeOff01 of g2 (after 44 of g1).
    assert starters[0] == Starter(
        game_id="g1", slot="HomeOff01", side="Home", unit="Off",
        box_score_position="QB", madden_id="2024-00001", box_score_name="Player HomeOff01",
    )
    assert starters[44].game_id == "g2"
    assert starters[44].slot == "HomeOff01"


def test_iter_starters_slot_order_matches_canonical():
    df = pd.DataFrame([_tiny_box_scores_row("g1")])
    starters = list(iter_starters(df))
    observed_slots = [s.slot for s in starters]
    assert observed_slots == list(CANONICAL_SLOTS)


def test_iter_starters_side_unit_decomposition():
    df = pd.DataFrame([_tiny_box_scores_row("g1")])
    starters = list(iter_starters(df))
    # HomeOff* → side=Home, unit=Off
    assert starters[0].side == "Home" and starters[0].unit == "Off"
    # HomeDef01 is index 11
    assert starters[11].side == "Home" and starters[11].unit == "Def"
    # AwayOff01 is index 22
    assert starters[22].side == "Away" and starters[22].unit == "Off"
    # AwayDef01 is index 33
    assert starters[33].side == "Away" and starters[33].unit == "Def"
