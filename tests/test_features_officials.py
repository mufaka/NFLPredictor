"""Tests for the officials feature group (§3.6 / FE-OFF-01..03)."""

from __future__ import annotations

import pathlib

import pandas as pd

from nflpredictor.features.officials import (
    CANONICAL_OFFICIAL_ROLES,
    OFFICIAL_COLUMN_NAMES,
    assemble_officials,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BOX_SCORES = REPO_ROOT / "Data" / "processed" / "box_scores_2024.csv"


def _row_with_officials(game_id: str, roles_to_names: dict[str, str]) -> dict:
    """Build a synthetic box-score-shaped row with the seven Official slots filled.

    Roles not provided are left blank — simulates a game whose officials list
    is missing one of the canonical roles.
    """
    row: dict = {"GameId": game_id}
    items = list(roles_to_names.items())
    for i in range(1, 8):
        role, name = items[i - 1] if i - 1 < len(items) else ("", "")
        row[f"Official{i:02d}_Role"] = role
        row[f"Official{i:02d}_Name"] = name
    return row


def test_canonical_column_names_in_pinned_order():
    assert OFFICIAL_COLUMN_NAMES == (
        "official_referee",
        "official_umpire",
        "official_down_judge",
        "official_line_judge",
        "official_back_judge",
        "official_side_judge",
        "official_field_judge",
    )


def test_assemble_shuffled_roles_routes_correctly():
    df = pd.DataFrame([_row_with_officials("g1", {
        "Side Judge":  "Alice",
        "Referee":     "Bob",
        "Field Judge": "Carol",
        "Down Judge":  "Dave",
        "Umpire":      "Eve",
        "Line Judge":  "Frank",
        "Back Judge":  "Grace",
    })])
    out = assemble_officials(df)
    assert list(out.columns) == ["GameId", *OFFICIAL_COLUMN_NAMES]
    assert out.loc[0, "official_referee"] == "Bob"
    assert out.loc[0, "official_umpire"] == "Eve"
    assert out.loc[0, "official_down_judge"] == "Dave"
    assert out.loc[0, "official_line_judge"] == "Frank"
    assert out.loc[0, "official_back_judge"] == "Grace"
    assert out.loc[0, "official_side_judge"] == "Alice"
    assert out.loc[0, "official_field_judge"] == "Carol"


def test_missing_role_leaves_none():
    # Only six roles provided; Field Judge missing.
    df = pd.DataFrame([_row_with_officials("g1", {
        "Referee":    "Bob",
        "Umpire":     "Eve",
        "Down Judge": "Dave",
        "Line Judge": "Frank",
        "Back Judge": "Grace",
        "Side Judge": "Alice",
    })])
    out = assemble_officials(df)
    assert out.loc[0, "official_field_judge"] is None


def test_unknown_role_is_dropped():
    df = pd.DataFrame([_row_with_officials("g1", {
        "Referee":             "Bob",
        "Umpire":              "Eve",
        "Down Judge":          "Dave",
        "Line Judge":          "Frank",
        "Back Judge":          "Grace",
        "Side Judge":          "Alice",
        "Replay Official":     "Henry",  # not in canonical set
    })])
    out = assemble_officials(df)
    # All seven canonical columns populated except field_judge.
    assert out.loc[0, "official_field_judge"] is None
    assert out.loc[0, "official_referee"] == "Bob"


def test_real_data_smoke():
    df = pd.read_csv(BOX_SCORES, dtype=str, keep_default_na=False)
    out = assemble_officials(df)
    assert len(out) == len(df)
    # Real 2024 data fills every canonical role in every game.
    for col in OFFICIAL_COLUMN_NAMES:
        non_null = out[col].notna() & (out[col] != "")
        assert non_null.sum() == len(out), f"{col} not populated for every game"


def test_canonical_roles_count():
    assert len(CANONICAL_OFFICIAL_ROLES) == 7
