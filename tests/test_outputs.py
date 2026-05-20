from __future__ import annotations

import pandas as pd

from nflpredictor.databuild.matching import MatchResult, Starter
from nflpredictor.databuild.outputs import (
    MappingRecord,
    build_mapping_records,
    rewrite_box_score_ids,
    write_mapping,
)


def _starter(slot: str, game_id: str, name: str, team_code: str, box_id: str) -> Starter:
    return Starter(
        game_id=game_id, slot_column=slot, name=name, team_code=team_code,
        position="QB", box_score_id=box_id,
    )


def test_build_mapping_records_dedupes_by_pair():
    s1 = _starter("HomeOff01", "g1", "Patrick Mahomes", "kan", "MahoPa00")
    s2 = _starter("HomeOff01", "g2", "Patrick Mahomes", "kan", "MahoPa00")
    r2 = MatchResult("2024-00709", 2, "tier2: deterministic team+name match", None)
    r3 = MatchResult("2024-00709", 3, "tier3: deterministic name match league-wide", None)
    records = build_mapping_records([(s1, r2), (s2, r3)], "2024")
    assert len(records) == 1
    # DB-MAP-05: prefer later tier (3 over 2).
    assert "tier3" in records[0].note
    assert records[0].season == "2024"


def test_build_mapping_records_appends_position_mismatch_suffix():
    s = _starter("HomeOff01", "g1", "Player", "kan", "PlayPl00")
    r = MatchResult("2024-00001", 2, "tier2: deterministic team+name match",
                    position_mismatch=("DB", "WR"))
    records = build_mapping_records([(s, r)], "2024")
    assert records[0].note.endswith("; position mismatch box=DB madden=WR")


def test_build_mapping_records_appends_blank_id_suffix():
    s = _starter("HomeOff01", "g1", "Player", "kan", "")  # blank box_score_id
    r = MatchResult("2024-00001", 3, "tier3: deterministic name match league-wide", None)
    records = build_mapping_records([(s, r)], "2024")
    assert records[0].box_score_id == ""
    assert records[0].note.endswith("; blank source _id; resolved by name")


def test_rewrite_box_score_ids_replaces_per_slot_ids():
    df = pd.DataFrame([
        {
            "season": "2024", "GameId": "g1",
            "HomeTeamCode": "kan", "AwayTeamCode": "buf",
            "HomeOff01_ID": "MahoPa00", "HomeOff02_ID": "KelcTr00",
            "AwayOff01_ID": "AlleJo00", "AwayOff02_ID": "DiggSt00",
        },
    ])
    mapping = {
        ("g1", "HomeOff01_ID"): "2024-00001",
        ("g1", "HomeOff02_ID"): "2024-00002",
        ("g1", "AwayOff01_ID"): "2024-00003",
        ("g1", "AwayOff02_ID"): "2024-00004",
    }
    out = rewrite_box_score_ids(df, mapping)
    assert out.iloc[0]["HomeOff01_ID"] == "2024-00001"
    assert out.iloc[0]["HomeOff02_ID"] == "2024-00002"
    assert out.iloc[0]["AwayOff01_ID"] == "2024-00003"
    assert out.iloc[0]["AwayOff02_ID"] == "2024-00004"


def test_write_mapping_is_sorted(tmp_path):
    path = tmp_path / "mapping.csv"
    records = [
        MappingRecord("2024", "Bbbb", "2024-00002", "x"),
        MappingRecord("2024", "Aaaa", "2024-00001", "y"),
        MappingRecord("2024", "Aaaa", "2024-00002", "z"),
    ]
    write_mapping(records, path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(df.columns) == ["season", "box_score_id", "madden_id", "note"]
    # Sorted (madden_id, box_score_id) asc
    assert list(zip(df["madden_id"], df["box_score_id"])) == [
        ("2024-00001", "Aaaa"),
        ("2024-00002", "Aaaa"),
        ("2024-00002", "Bbbb"),
    ]
