from __future__ import annotations

import pandas as pd

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


def _madden_row(madden_id: str, team: str, position: str, fullname: str,
                overall: int, archetype: str) -> dict[str, str]:
    """Build one assigned-ID Madden row in the multi-year (56-column) schema."""
    row: dict[str, str] = {
        "madden_id": madden_id,
        "team": team,
        "season": "2024",
        "fullname": fullname,
        "position": position,
    }
    for col in NUMERIC_COLUMNS:
        row[col] = "55"
    row["overallrating"] = str(overall)
    for col in CATEGORICAL_COLUMNS:
        row[col] = "common"
    row["archetype"] = archetype
    return row


def _tiny_madden_with_ids() -> pd.DataFrame:
    return pd.DataFrame([
        _madden_row("2024-00001", "KC", "QB", "Patrick Mahomes", 97, "QB_Improviser"),
        _madden_row("2024-00002", "KC", "TE", "Travis Kelce", 89, "TE_Vertical"),
    ])


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
    appended, assignments = append_unmatched_rows(base, unmatched, 2024)
    assert len(appended) == 3
    assert appended.iloc[2]["madden_id"] == "2024-00003"
    assert appended.iloc[2]["team"] == "BUF"
    assert appended.iloc[2]["season"] == "2024"
    assert appended.iloc[2]["position"] == "QB"
    assert appended.iloc[2]["fullname"] == "John Doe"
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
    appended, assignments = append_unmatched_rows(base, unmatched, 2024)
    appended_rows = appended.iloc[2:]
    # Expect sort: (team_code asc, normalized_name asc, first_game_id_seen asc)
    assert list(appended_rows["fullname"]) == ["Beta", "Alpha", "Zed"]
    assert list(appended_rows["madden_id"]) == [
        "2024-00003", "2024-00004", "2024-00005",
    ]


def test_compute_and_fill_values():
    base = _tiny_madden_with_ids()
    unmatched = [
        UnmatchedPlayer("John Doe", "buf", "QB", "john doe", "g1"),
    ]
    appended, _ = append_unmatched_rows(base, unmatched, 2024)
    fill = compute_fill_values(appended)
    # Mean overallrating across matched=1 rows (97, 89) = 93.0
    assert fill["overallrating"] == "93.0000"
    # Categorical mode: distinct archetypes → lex-smallest wins.
    assert fill["archetype"] in {"QB_Improviser", "TE_Vertical"}
    # No null fill for `matched` itself.
    assert MATCHED_COLUMN not in fill

    filled = fill_unmatched_rows(appended, fill)
    unmatched_row = filled[filled[MATCHED_COLUMN] == "0"].iloc[0]
    # No null cell on a matched=0 row (DB-OUT-13 precondition).
    assert not (unmatched_row == "").any()
    # Identity columns preserved.
    assert unmatched_row["team"] == "BUF"
    assert unmatched_row["fullname"] == "John Doe"
    # Sanity: numeric mean applied to a known column.
    assert unmatched_row["overallrating"] == "93.0000"


def test_fill_does_not_touch_matched_one_rows():
    """DB-FILL-04: nulls on matched=1 rows must remain null."""
    base = _tiny_madden_with_ids()
    # Empty an arbitrary categorical cell on a matched=1 row before fill.
    base.loc[0, "runningstyle"] = ""
    base[MATCHED_COLUMN] = "1"
    unmatched = [UnmatchedPlayer("John Doe", "buf", "QB", "john doe", "g1")]
    appended, _ = append_unmatched_rows(base, unmatched, 2024)
    fill = compute_fill_values(appended)
    filled = fill_unmatched_rows(appended, fill)
    # matched=1 row 0 still has empty runningstyle.
    assert filled.iloc[0]["runningstyle"] == ""
    # matched=0 row got the categorical fill.
    assert filled[filled[MATCHED_COLUMN] == "0"].iloc[0]["runningstyle"] != ""


def test_mode_lex_tiebreak():
    base = _tiny_madden_with_ids()
    # Force a tie on archetype across matched=1 rows.
    base.loc[0, "archetype"] = "ZZZ"
    base.loc[1, "archetype"] = "AAA"
    base[MATCHED_COLUMN] = "1"
    fill = compute_fill_values(base)
    assert fill["archetype"] == "AAA"


def test_numeric_categorical_partition_is_complete():
    overlap = set(NUMERIC_COLUMNS) & set(CATEGORICAL_COLUMNS)
    assert not overlap, f"column appears in both buckets: {overlap}"
