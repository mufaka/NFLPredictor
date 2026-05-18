"""PFR three-letter team code <-> Madden franchise nickname mapping.

The box-score CSV uses Pro-Football-Reference codes (e.g. ``kan``, ``rav``).
The Madden ratings CSV uses franchise nicknames (e.g. ``Chiefs``, ``Ravens``).
These two tables are the canonical bridge between them.
"""

from __future__ import annotations


PFR_CODE_TO_MADDEN_NICKNAME: dict[str, str] = {
    "atl": "Falcons",
    "buf": "Bills",
    "car": "Panthers",
    "chi": "Bears",
    "cin": "Bengals",
    "cle": "Browns",
    "clt": "Colts",
    "crd": "Cardinals",
    "dal": "Cowboys",
    "den": "Broncos",
    "det": "Lions",
    "gnb": "Packers",
    "htx": "Texans",
    "jax": "Jaguars",
    "kan": "Chiefs",
    "mia": "Dolphins",
    "min": "Vikings",
    "nor": "Saints",
    "nwe": "Patriots",
    "nyg": "Giants",
    "nyj": "Jets",
    "oti": "Titans",
    "phi": "Eagles",
    "pit": "Steelers",
    "rai": "Raiders",
    "ram": "Rams",
    "rav": "Ravens",
    "sdg": "Chargers",
    "sea": "Seahawks",
    "sfo": "49ers",
    "tam": "Buccaneers",
    "was": "Commanders",
}


_MADDEN_NICKNAME_TO_PFR_CODE: dict[str, str] = {
    nickname: code for code, nickname in PFR_CODE_TO_MADDEN_NICKNAME.items()
}


def normalize_team_code(pfr_code: str) -> str:
    """Return the Madden nickname for a PFR code. Raises ``KeyError`` for unknowns."""
    try:
        return PFR_CODE_TO_MADDEN_NICKNAME[pfr_code]
    except KeyError as exc:
        raise KeyError(
            f"Unknown PFR team code {pfr_code!r}; "
            f"known codes: {sorted(PFR_CODE_TO_MADDEN_NICKNAME)}"
        ) from exc


def madden_nickname_to_pfr_code(nickname: str) -> str:
    """Reverse lookup: Madden nickname -> PFR code. Useful for diagnostics."""
    try:
        return _MADDEN_NICKNAME_TO_PFR_CODE[nickname]
    except KeyError as exc:
        raise KeyError(
            f"Unknown Madden nickname {nickname!r}; "
            f"known nicknames: {sorted(_MADDEN_NICKNAME_TO_PFR_CODE)}"
        ) from exc
