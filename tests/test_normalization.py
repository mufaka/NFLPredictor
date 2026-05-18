import pytest

from nflpredictor.databuild.normalization import (
    NORMALIZATION_VERSION,
    normalize_name,
)


def test_version_is_v1():
    assert NORMALIZATION_VERSION == "v1"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Patrick Mahomes", "patrick mahomes"),
        ("Odafe Oweh", "odafe oweh"),
        ("Roquan Smith", "roquan smith"),
        ("Lamar Jackson", "lamar jackson"),
        ("A.J. Brown", "a.j. brown"),
        ("D'Andre Swift", "d'andre swift"),
        ("DJ Moore", "dj moore"),
        ("Patrick Mahomes II", "patrick mahomes"),
        ("Marvin Harrison Jr.", "marvin harrison"),
        ("Marvin Harrison Jr", "marvin harrison"),
        ("Robert Griffin III", "robert griffin"),
        ("Mel Renfro IV", "mel renfro"),
        ("José Ramírez", "jose ramirez"),
        ("  Travis  Kelce  ", "travis kelce"),
    ],
)
def test_normalize_name(raw: str, expected: str):
    assert normalize_name(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "Patrick Mahomes",
        "Marvin Harrison Jr.",
        "  Travis  Kelce  ",
        "José Ramírez",
        "Patrick Mahomes II",
    ],
)
def test_idempotence(raw: str):
    once = normalize_name(raw)
    twice = normalize_name(once)
    assert once == twice
