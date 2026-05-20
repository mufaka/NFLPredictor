import pandas as pd
import pytest

from nflpredictor.databuild.ids import (
    assign_raw_madden_ids,
    format_madden_id,
    next_madden_id,
)


def test_format_madden_id():
    assert format_madden_id(2024, 1) == "2024-00001"
    assert format_madden_id(2024, 12345) == "2024-12345"
    assert format_madden_id(2020, 99999) == "2020-99999"


def test_next_madden_id():
    assert next_madden_id("2024-00001") == "2024-00002"
    assert next_madden_id("2024-00099") == "2024-00100"
    assert next_madden_id("2020-12344") == "2020-12345"


def test_next_madden_id_rejects_malformed():
    with pytest.raises(ValueError):
        next_madden_id("not-an-id")
    with pytest.raises(ValueError):
        next_madden_id("2024-123")


@pytest.fixture
def tiny_madden() -> pd.DataFrame:
    rows = [
        {"team": "KC",  "position": "QB", "fullname": "Patrick Mahomes", "jerseynumber": "15"},
        {"team": "KC",  "position": "TE", "fullname": "Travis Kelce",    "jerseynumber": "87"},
        {"team": "BAL", "position": "QB", "fullname": "Lamar Jackson",   "jerseynumber": "8"},
        {"team": "BAL", "position": "RB", "fullname": "Derrick Henry",   "jerseynumber": "22"},
        {"team": "SF",  "position": "QB", "fullname": "Brock Purdy",     "jerseynumber": "13"},
    ]
    return pd.DataFrame(rows)


def test_assign_raw_madden_ids_sequence(tiny_madden: pd.DataFrame):
    result = assign_raw_madden_ids(tiny_madden, 2024)
    # Sort order: team asc, position asc, fullname asc, jerseynumber asc.
    expected = [
        ("2024-00001", "BAL", "QB", "Lamar Jackson"),
        ("2024-00002", "BAL", "RB", "Derrick Henry"),
        ("2024-00003", "KC",  "QB", "Patrick Mahomes"),
        ("2024-00004", "KC",  "TE", "Travis Kelce"),
        ("2024-00005", "SF",  "QB", "Brock Purdy"),
    ]
    got = list(zip(result["madden_id"], result["team"], result["position"], result["fullname"]))
    assert got == expected


def test_madden_id_is_first_column(tiny_madden: pd.DataFrame):
    result = assign_raw_madden_ids(tiny_madden, 2024)
    assert result.columns[0] == "madden_id"


def test_season_prefixes_the_id(tiny_madden: pd.DataFrame):
    result = assign_raw_madden_ids(tiny_madden, 2021)
    assert result["madden_id"].iloc[0] == "2021-00001"
    assert all(mid.startswith("2021-") for mid in result["madden_id"])


def test_assign_raw_madden_ids_is_deterministic(tiny_madden: pd.DataFrame):
    first = assign_raw_madden_ids(tiny_madden.copy(), 2024)
    second = assign_raw_madden_ids(tiny_madden.copy(), 2024)
    pd.testing.assert_frame_equal(first, second)


def test_assign_raw_madden_ids_independent_of_input_order(tiny_madden: pd.DataFrame):
    shuffled = tiny_madden.iloc[::-1].reset_index(drop=True)
    a = assign_raw_madden_ids(tiny_madden, 2024)
    b = assign_raw_madden_ids(shuffled, 2024)
    pd.testing.assert_frame_equal(a, b)
