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
    assert format_madden_id(2024, 99999) == "2024-99999"


def test_next_madden_id():
    assert next_madden_id("2024-00001") == "2024-00002"
    assert next_madden_id("2024-00099") == "2024-00100"
    assert next_madden_id("2024-12344") == "2024-12345"


def test_next_madden_id_rejects_malformed():
    with pytest.raises(ValueError):
        next_madden_id("not-an-id")
    with pytest.raises(ValueError):
        next_madden_id("2024-123")


@pytest.fixture
def tiny_madden() -> pd.DataFrame:
    rows = [
        {"Team": "Chiefs",  "Position": "QB", "Full Name": "Patrick Mahomes", "Jersey Number": "15"},
        {"Team": "Chiefs",  "Position": "TE", "Full Name": "Travis Kelce",    "Jersey Number": "87"},
        {"Team": "Ravens",  "Position": "QB", "Full Name": "Lamar Jackson",   "Jersey Number": "8"},
        {"Team": "Ravens",  "Position": "RB", "Full Name": "Derrick Henry",   "Jersey Number": "22"},
        {"Team": "49ers",   "Position": "QB", "Full Name": "Brock Purdy",     "Jersey Number": "13"},
    ]
    return pd.DataFrame(rows)


def test_assign_raw_madden_ids_sequence(tiny_madden: pd.DataFrame):
    result = assign_raw_madden_ids(tiny_madden)
    # Sort order: Team asc, Position asc, Full Name asc, Jersey Number asc.
    expected = [
        ("2024-00001", "49ers",  "QB", "Brock Purdy"),
        ("2024-00002", "Chiefs", "QB", "Patrick Mahomes"),
        ("2024-00003", "Chiefs", "TE", "Travis Kelce"),
        ("2024-00004", "Ravens", "QB", "Lamar Jackson"),
        ("2024-00005", "Ravens", "RB", "Derrick Henry"),
    ]
    got = list(zip(result["madden_id"], result["Team"], result["Position"], result["Full Name"]))
    assert got == expected


def test_madden_id_is_first_column(tiny_madden: pd.DataFrame):
    result = assign_raw_madden_ids(tiny_madden)
    assert result.columns[0] == "madden_id"


def test_assign_raw_madden_ids_is_deterministic(tiny_madden: pd.DataFrame):
    first = assign_raw_madden_ids(tiny_madden.copy())
    second = assign_raw_madden_ids(tiny_madden.copy())
    pd.testing.assert_frame_equal(first, second)


def test_assign_raw_madden_ids_independent_of_input_order(tiny_madden: pd.DataFrame):
    shuffled = tiny_madden.iloc[::-1].reset_index(drop=True)
    a = assign_raw_madden_ids(tiny_madden)
    b = assign_raw_madden_ids(shuffled)
    pd.testing.assert_frame_equal(a, b)
