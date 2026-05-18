import pytest

from nflpredictor.databuild.teams import (
    PFR_CODE_TO_MADDEN_NICKNAME,
    madden_nickname_to_pfr_code,
    normalize_team_code,
)


def test_table_is_exactly_thirty_two_teams():
    assert len(PFR_CODE_TO_MADDEN_NICKNAME) == 32
    assert len(set(PFR_CODE_TO_MADDEN_NICKNAME.values())) == 32


@pytest.mark.parametrize("pfr_code", sorted(PFR_CODE_TO_MADDEN_NICKNAME))
def test_round_trip(pfr_code: str):
    nickname = normalize_team_code(pfr_code)
    assert madden_nickname_to_pfr_code(nickname) == pfr_code


def test_unknown_pfr_code_raises():
    with pytest.raises(KeyError):
        normalize_team_code("zzz")


def test_unknown_nickname_raises():
    with pytest.raises(KeyError):
        madden_nickname_to_pfr_code("Aliens")
