from __future__ import annotations

import pytest

from nflpredictor.databuild.matching import (
    MaddenRow,
    Starter,
    build_match_indexes,
    match_starter,
    tier1_override,
    tier2_team_and_name,
    tier3_name_leaguewide,
    tier4_fuzzy,
)
from nflpredictor.databuild.normalization import normalize_name
from nflpredictor.databuild.overrides import Override, build_override_index


def _row(madden_id: str, team: str, position: str, full_name: str) -> MaddenRow:
    return MaddenRow(
        madden_id=madden_id,
        team=team,
        position=position,
        full_name=full_name,
        normalized_name=normalize_name(full_name),
    )


@pytest.fixture
def fixture_indexes():
    rows = [
        _row("2024-00001", "Chiefs",  "QB", "Patrick Mahomes"),
        _row("2024-00002", "Chiefs",  "TE", "Travis Kelce"),
        _row("2024-00003", "Ravens",  "QB", "Lamar Jackson"),
        _row("2024-00004", "Ravens",  "RB", "Derrick Henry"),
        # Duplicate normalized name across teams: tier 3 must NOT match this.
        _row("2024-00005", "49ers",   "WR", "Michael Wilson"),
        _row("2024-00006", "Cardinals","WR", "Michael Wilson"),
        # Lone league-wide unique name -- good Tier 3 candidate (different team).
        _row("2024-00007", "Bears",   "WR", "DJ Moore"),
        # Fuzzy-match target: misspelling support inside Chiefs.
        _row("2024-00008", "Chiefs",  "WR", "Justin Watson"),
    ]
    return build_match_indexes(rows)


@pytest.fixture
def empty_override_index():
    return build_override_index([], {f"2024-0000{i}" for i in range(1, 9)})


def test_tier1_wins_over_tier2(fixture_indexes):
    # Even though tier 2 would resolve Mahomes to 2024-00001, an override
    # pointing him at Travis Kelce's row must take precedence.
    override = Override(
        box_score_name="Patrick Mahomes",
        box_score_team_code="kan",
        box_score_id=None,
        madden_id="2024-00002",
        reason="forced for test",
    )
    override_index = build_override_index([override], set(fixture_indexes.rows_by_madden_id))
    starter = Starter(
        game_id="g1", slot_column="HomeOff01",
        name="Patrick Mahomes", team_code="kan",
        position="QB", box_score_id="MahoPa00",
    )
    result = match_starter(starter, fixture_indexes, override_index)
    assert result.tier == 1
    assert result.madden_id == "2024-00002"
    assert "manual override" in result.note_fragment
    assert "forced for test" in result.note_fragment


def test_tier1_records_no_reason_when_blank(fixture_indexes):
    override = Override("Patrick Mahomes", "kan", None, "2024-00001", "")
    idx = build_override_index([override], set(fixture_indexes.rows_by_madden_id))
    starter = Starter("g1", "HomeOff01", "Patrick Mahomes", "kan", "QB", "")
    result = tier1_override(starter, idx, fixture_indexes)
    assert result is not None
    assert "(no reason)" in result.note_fragment


def test_tier2_hits_mahomes(fixture_indexes, empty_override_index):
    starter = Starter("g1", "HomeOff01", "Patrick Mahomes", "kan", "QB", "MahoPa00")
    result = tier2_team_and_name(starter, fixture_indexes)
    assert result is not None
    assert result.tier == 2
    assert result.madden_id == "2024-00001"
    assert result.position_mismatch is None


def test_tier3_unique_leaguewide(fixture_indexes):
    # "DJ Moore" appears once league-wide on Bears; box-score team says Panthers
    # (stale due to off-season trade). Tier 2 fails (no Bears row for DJ Moore
    # on Panthers); Tier 3 should resolve uniquely.
    starter = Starter("g1", "HomeOff03", "DJ Moore", "car", "WR", "MoorDJ00")
    result = tier3_name_leaguewide(starter, fixture_indexes)
    assert result is not None
    assert result.tier == 3
    assert result.madden_id == "2024-00007"


def test_tier3_fails_when_multiple_leaguewide(fixture_indexes):
    # "Michael Wilson" exists on two teams; tier 3 must abstain.
    starter = Starter("g1", "AwayOff09", "Michael Wilson", "sfo", "WR", "")
    assert tier3_name_leaguewide(starter, fixture_indexes) is None


def test_tier4_succeeds_above_threshold_with_clear_margin(fixture_indexes):
    # "Justin Watson" present in Chiefs roster; box score writes "Justyn Watson".
    starter = Starter("g1", "HomeOff04", "Justyn Watson", "kan", "WR", "")
    result = tier4_fuzzy(starter, fixture_indexes)
    assert result is not None
    assert result.tier == 4
    assert result.madden_id == "2024-00008"
    assert "score=" in result.note_fragment


def test_tier4_fails_when_runner_up_too_close(fixture_indexes):
    # Inject two near-twin names on the same team so the top-2 are within margin.
    rows = list(fixture_indexes.rows_by_madden_id.values()) + [
        _row("2024-09001", "Chiefs", "WR", "John Smith"),
        _row("2024-09002", "Chiefs", "WR", "Jon Smith"),
    ]
    idx = build_match_indexes(rows)
    starter = Starter("g1", "HomeOff05", "Johnathan Smith", "kan", "WR", "")
    assert tier4_fuzzy(starter, idx) is None


def test_tier4_fails_below_threshold(fixture_indexes):
    starter = Starter("g1", "HomeOff06", "Zzz Qqq", "kan", "WR", "")
    assert tier4_fuzzy(starter, fixture_indexes) is None


def test_all_tiers_fail_returns_unmatched(fixture_indexes, empty_override_index):
    starter = Starter("g1", "AwayDef11", "Imaginary Person", "buf", "CB", "")
    result = match_starter(starter, fixture_indexes, empty_override_index)
    assert result.tier == 0
    assert result.madden_id is None
    assert result.note_fragment == "unmatched: appended with null-fill"


def test_position_mismatch_recorded(fixture_indexes, empty_override_index):
    # Mahomes is QB; lying says he's a TE on the box score. Tier 2 still hits
    # by team+name; position mismatch must surface.
    starter = Starter("g1", "HomeOff01", "Patrick Mahomes", "kan", "TE", "MahoPa00")
    result = match_starter(starter, fixture_indexes, empty_override_index)
    assert result.tier == 2
    assert result.position_mismatch == ("TE", "QB")


def test_position_mismatch_absent_for_compatible(fixture_indexes, empty_override_index):
    starter = Starter("g1", "HomeOff01", "Patrick Mahomes", "kan", "QB", "MahoPa00")
    result = match_starter(starter, fixture_indexes, empty_override_index)
    assert result.position_mismatch is None
