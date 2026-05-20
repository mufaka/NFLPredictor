import pathlib

import pytest

from nflpredictor.databuild.overrides import (
    OVERRIDES_HEADER,
    Override,
    build_override_index,
    load_overrides,
)


def _write_overrides(path: pathlib.Path, lines: list[str]) -> None:
    path.write_text(
        ",".join(OVERRIDES_HEADER) + "\n" + "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


def test_header_only_stub_returns_empty(tmp_path: pathlib.Path):
    path = tmp_path / "overrides.csv"
    _write_overrides(path, [])
    assert load_overrides(path) == []


def test_real_stub_returns_empty():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    overrides = load_overrides(repo_root / "Data" / "raw" / "player_overrides.csv")
    assert overrides == []


def test_load_overrides_parses_rows(tmp_path: pathlib.Path):
    path = tmp_path / "overrides.csv"
    _write_overrides(
        path,
        [
            "2024,Patrick Mahomes,kan,MahoPa00,2024-00001,roster swap",
            "2023,AJ Brown,phi,,2023-00002,name spelling",
        ],
    )
    rows = load_overrides(path)
    assert rows == [
        Override(
            season="2024",
            box_score_name="Patrick Mahomes",
            box_score_team_code="kan",
            box_score_id="MahoPa00",
            madden_id="2024-00001",
            reason="roster swap",
        ),
        Override(
            season="2023",
            box_score_name="AJ Brown",
            box_score_team_code="phi",
            box_score_id=None,
            madden_id="2023-00002",
            reason="name spelling",
        ),
    ]


def test_header_mismatch_raises(tmp_path: pathlib.Path):
    path = tmp_path / "overrides.csv"
    path.write_text("a,b,c\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header mismatch"):
        load_overrides(path)


def test_unknown_madden_id_raises():
    overrides = [
        Override("2024", "Patrick Mahomes", "kan", None, "2024-99999", "typo")
    ]
    assigned = {"2024-00001", "2024-00002"}
    with pytest.raises(ValueError, match="DB-OVR-04"):
        build_override_index(overrides, assigned)


def test_ambiguous_overrides_by_id_raises():
    overrides = [
        Override("2024", "Patrick Mahomes", "kan", "MahoPa00", "2024-00001", "a"),
        Override("2024", "Pat Mahomes",     "kan", "MahoPa00", "2024-00002", "b"),
    ]
    with pytest.raises(ValueError, match="DB-OVR-05"):
        build_override_index(overrides, {"2024-00001", "2024-00002"})


def test_ambiguous_overrides_by_name_and_team_raises():
    overrides = [
        Override("2024", "Patrick Mahomes", "kan", None, "2024-00001", "a"),
        Override("2024", "patrick  mahomes", "kan", None, "2024-00002", "b"),
    ]
    with pytest.raises(ValueError, match="DB-OVR-05"):
        build_override_index(overrides, {"2024-00001", "2024-00002"})


def test_lookup_by_box_score_id():
    overrides = [
        Override("2024", "Patrick Mahomes", "kan", "MahoPa00", "2024-00001", "swap")
    ]
    idx = build_override_index(overrides, {"2024-00001"})
    hit = idx.lookup(normalized_name="patrick mahomes", team_code="kan", box_score_id="MahoPa00")
    assert hit is not None and hit.madden_id == "2024-00001"


def test_lookup_by_name_and_team_when_id_blank():
    overrides = [
        Override("2024", "Patrick Mahomes", "kan", None, "2024-00001", "blank id")
    ]
    idx = build_override_index(overrides, {"2024-00001"})
    hit = idx.lookup(normalized_name="patrick mahomes", team_code="kan", box_score_id="")
    assert hit is not None and hit.madden_id == "2024-00001"
    miss = idx.lookup(normalized_name="patrick mahomes", team_code="phi", box_score_id="")
    assert miss is None
