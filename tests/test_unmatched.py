from __future__ import annotations

import pandas as pd
import pytest

from nflpredictor.databuild.matching import MatchResult, Starter
from nflpredictor.databuild.unmatched import (
    CATEGORICAL_COLUMNS,
    MATCHED_COLUMN,
    NUMERIC_COLUMNS,
    UnmatchedPlayer,
    append_unmatched_rows,
    collect_unmatched_starters,
    compute_fill_values,
    fill_unmatched_rows,
)


def _matched_pair(
    game_id: str, name: str, team_code: str, position: str, tier: int
) -> tuple[Starter, MatchResult]:
    starter = Starter(
        game_id=game_id,
        slot_column="HomeOff01",
        name=name,
        team_code=team_code,
        position=position,
        box_score_id="",
    )
    result = MatchResult(
        madden_id=None if tier == 0 else f"2024-X{tier}",
        tier=tier,
        note_fragment="(test)",
        position_mismatch=None,
    )
    return starter, result


def test_collect_dedupes_same_player_across_games():
    pairs = [
        _matched_pair("g1", "John Doe", "kan", "QB", tier=0),
        _matched_pair("g2", "John Doe", "kan", "QB", tier=0),
        _matched_pair("g3", "John Doe", "kan", "QB", tier=0),
    ]
    unmatched = collect_unmatched_starters(pairs)
    assert len(unmatched) == 1
    assert unmatched[0].first_game_id_seen == "g1"


def test_collect_ignores_matched_starters():
    pairs = [
        _matched_pair("g1", "Real Player", "kan", "QB", tier=2),
        _matched_pair("g1", "Ghost", "kan", "WR", tier=0),
    ]
    unmatched = collect_unmatched_starters(pairs)
    assert [u.name for u in unmatched] == ["Ghost"]


def _tiny_madden_with_ids() -> pd.DataFrame:
    rows = [
        {
            "madden_id": "2024-00001",
            "Team": "Chiefs",
            "Position": "QB",
            "Full Name": "Patrick Mahomes",
            "Overall Rating": "97",
            "Jersey Number": "15",
            "Speed": "80",
            "Acceleration": "82",
            "Strength": "80",
            "Agility": "85",
            "Awareness": "99",
            "Catching": "60",
            "Carrying": "60",
            "Throw Power": "97",
            "Kick Power": "55",
            "Kick Accuracy": "55",
            "Run Block": "30",
            "Pass Block": "40",
            "Tackle": "30",
            "Break Tackle": "55",
            "Jumping": "70",
            "Kick Return": "30",
            "Injury": "90",
            "Stamina": "92",
            "Toughness": "92",
            "Trucking": "55",
            "Change Of Direction": "85",
            "Ball Carrier Vision": "75",
            "Stiff Arm": "40",
            "Spin Move": "55",
            "Juke Move": "60",
            "Impact Blocking": "30",
            "Run Block Power": "30",
            "Run Block Finesse": "30",
            "Pass Block Power": "30",
            "Pass Block Finesse": "30",
            "Lead Block": "30",
            "Break Sack": "75",
            "Throw Under Pressure": "95",
            "Power Moves": "30",
            "Finesse Moves": "30",
            "Block Shedding": "30",
            "Pursuit": "40",
            "Play Recognition": "85",
            "Man Coverage": "30",
            "Zone Coverage": "30",
            "Spectacular Catch": "55",
            "Catch In Traffic": "55",
            "Short Route Running": "55",
            "Medium Route Running": "55",
            "Deep Route Running": "55",
            "Hit Power": "30",
            "Press": "30",
            "Release": "55",
            "Throw Accuracy Short": "94",
            "Throw Accuracy Mid": "94",
            "Throw Accuracy Deep": "92",
            "Play Action": "95",
            "Throw On The Run": "98",
            "Height": "75",
            "Weight": "230",
            "Age": "28",
            "Birthdate": "34000",
            "Years Pro": "7",
            "Running Style": "Default Stride Loose",
            "Archetype": "QB_Improviser",
            "College": "Texas Tech",
            "Total Salary": "60000000",
            "Signing Bonus": "10000000",
            "Player Handness": "Right",
        },
        {
            "madden_id": "2024-00002",
            "Team": "Chiefs",
            "Position": "TE",
            "Full Name": "Travis Kelce",
            "Overall Rating": "89",
            "Jersey Number": "87",
            "Speed": "82",
            "Acceleration": "84",
            "Strength": "75",
            "Agility": "82",
            "Awareness": "95",
            "Catching": "95",
            "Carrying": "70",
            "Throw Power": "30",
            "Kick Power": "30",
            "Kick Accuracy": "30",
            "Run Block": "60",
            "Pass Block": "50",
            "Tackle": "30",
            "Break Tackle": "70",
            "Jumping": "82",
            "Kick Return": "30",
            "Injury": "85",
            "Stamina": "88",
            "Toughness": "88",
            "Trucking": "65",
            "Change Of Direction": "82",
            "Ball Carrier Vision": "70",
            "Stiff Arm": "70",
            "Spin Move": "60",
            "Juke Move": "70",
            "Impact Blocking": "60",
            "Run Block Power": "60",
            "Run Block Finesse": "55",
            "Pass Block Power": "50",
            "Pass Block Finesse": "50",
            "Lead Block": "60",
            "Break Sack": "30",
            "Throw Under Pressure": "30",
            "Power Moves": "30",
            "Finesse Moves": "30",
            "Block Shedding": "30",
            "Pursuit": "30",
            "Play Recognition": "75",
            "Man Coverage": "30",
            "Zone Coverage": "30",
            "Spectacular Catch": "85",
            "Catch In Traffic": "92",
            "Short Route Running": "90",
            "Medium Route Running": "92",
            "Deep Route Running": "85",
            "Hit Power": "30",
            "Press": "30",
            "Release": "85",
            "Throw Accuracy Short": "30",
            "Throw Accuracy Mid": "30",
            "Throw Accuracy Deep": "30",
            "Play Action": "55",
            "Throw On The Run": "55",
            "Height": "77",
            "Weight": "260",
            "Age": "34",
            "Birthdate": "32100",
            "Years Pro": "11",
            "Running Style": "Default Stride Loose",
            "Archetype": "TE_Vertical",
            "College": "Cincinnati",
            "Total Salary": "14000000",
            "Signing Bonus": "5000000",
            "Player Handness": "Right",
        },
    ]
    return pd.DataFrame(rows)


def test_append_unmatched_rows_assigns_ids_and_sets_matched():
    base = _tiny_madden_with_ids()
    unmatched = [
        UnmatchedPlayer(
            name="John Doe",
            team_code="buf",
            position="QB",
            normalized_name="john doe",
            first_game_id_seen="g1",
        ),
    ]
    appended, assignments = append_unmatched_rows(base, unmatched)
    assert len(appended) == 3
    assert appended.iloc[2]["madden_id"] == "2024-00003"
    assert appended.iloc[2]["Team"] == "Bills"
    assert appended.iloc[2]["Position"] == "QB"
    assert appended.iloc[2]["Full Name"] == "John Doe"
    assert appended.iloc[2][MATCHED_COLUMN] == "0"
    assert (appended.iloc[:2][MATCHED_COLUMN] == "1").all()
    assert assignments == {("john doe", "buf"): "2024-00003"}


def test_append_unmatched_sort_order_matches_db_id_03():
    base = _tiny_madden_with_ids()
    # Insertion order intentionally NOT the sort order.
    unmatched = [
        UnmatchedPlayer("Zed", "kan", "WR", "zed", "g2"),
        UnmatchedPlayer("Alpha", "buf", "QB", "alpha", "g3"),
        UnmatchedPlayer("Beta", "buf", "QB", "alpha", "g1"),   # same norm name, earlier game
    ]
    appended, assignments = append_unmatched_rows(base, unmatched)
    appended_rows = appended.iloc[2:]
    # Expect sort: (team_code asc, normalized_name asc, first_game_id_seen asc)
    assert list(appended_rows["Full Name"]) == ["Beta", "Alpha", "Zed"]
    assert list(appended_rows["madden_id"]) == [
        "2024-00003", "2024-00004", "2024-00005",
    ]


def test_compute_and_fill_values():
    base = _tiny_madden_with_ids()
    unmatched = [
        UnmatchedPlayer("John Doe", "buf", "QB", "john doe", "g1"),
    ]
    appended, _ = append_unmatched_rows(base, unmatched)
    fill = compute_fill_values(appended)
    # Mean Overall Rating across matched=1 rows (97, 89) = 93.0
    assert fill["Overall Rating"] == "93.0000"
    # Categorical mode on a 2-row table with ties uses lex-smallest.
    # Both Archetype values differ; lex-smallest wins.
    assert fill["Archetype"] in {"QB_Improviser", "TE_Vertical"}
    # No null fill for `matched` itself or for the identity columns.
    assert MATCHED_COLUMN not in fill

    filled = fill_unmatched_rows(appended, fill)
    unmatched_row = filled[filled[MATCHED_COLUMN] == "0"].iloc[0]
    # No null cell on a matched=0 row (DB-OUT-13 precondition).
    assert not (unmatched_row == "").any()
    # Identity columns preserved.
    assert unmatched_row["Team"] == "Bills"
    assert unmatched_row["Full Name"] == "John Doe"
    # Sanity: numeric mean applied to a known column.
    assert unmatched_row["Overall Rating"] == "93.0000"


def test_fill_does_not_touch_matched_one_rows():
    """DB-FILL-04: nulls on matched=1 rows must remain null."""
    base = _tiny_madden_with_ids()
    # Empty an arbitrary cell on a matched=1 row before fill.
    base.loc[0, "College"] = ""
    base[MATCHED_COLUMN] = "1"
    unmatched = [UnmatchedPlayer("John Doe", "buf", "QB", "john doe", "g1")]
    appended, _ = append_unmatched_rows(base, unmatched)
    fill = compute_fill_values(appended)
    filled = fill_unmatched_rows(appended, fill)
    # matched=1 row 0 still has empty College.
    assert filled.iloc[0]["College"] == ""
    # matched=0 row got the categorical fill.
    assert filled[filled[MATCHED_COLUMN] == "0"].iloc[0]["College"] != ""


def test_mode_lex_tiebreak():
    base = _tiny_madden_with_ids()
    # Force a tie on Archetype across matched=1 rows.
    base.loc[0, "Archetype"] = "ZZZ"
    base.loc[1, "Archetype"] = "AAA"
    base[MATCHED_COLUMN] = "1"
    fill = compute_fill_values(base.assign(**{MATCHED_COLUMN: "1"}))
    assert fill["Archetype"] == "AAA"


def test_numeric_categorical_partition_is_complete():
    overlap = set(NUMERIC_COLUMNS) & set(CATEGORICAL_COLUMNS)
    assert not overlap, f"column appears in both buckets: {overlap}"
