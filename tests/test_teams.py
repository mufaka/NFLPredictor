import pytest

from nflpredictor.databuild.teams import (
    PFR_CODE_TO_MADDEN_ABBREV,
    madden_abbrev_to_pfr_code,
    normalize_team_code,
)


def test_table_is_exactly_thirty_two_teams():
    assert len(PFR_CODE_TO_MADDEN_ABBREV) == 32
    assert len(set(PFR_CODE_TO_MADDEN_ABBREV.values())) == 32


@pytest.mark.parametrize("pfr_code", sorted(PFR_CODE_TO_MADDEN_ABBREV))
def test_round_trip(pfr_code: str):
    abbrev = normalize_team_code(pfr_code)
    assert madden_abbrev_to_pfr_code(abbrev) == pfr_code


def test_unknown_pfr_code_raises():
    with pytest.raises(KeyError):
        normalize_team_code("zzz")


def test_unknown_abbrev_raises():
    with pytest.raises(KeyError):
        madden_abbrev_to_pfr_code("ZZZ")
