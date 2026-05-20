"""PFR three-letter team code <-> Madden team abbreviation mapping.

The box-score CSVs use Pro-Football-Reference codes (e.g. ``kan``, ``rav``).
The Madden ratings CSVs use modern NFL team abbreviations (e.g. ``KC``,
``BAL``). PFR codes are franchise-stable across all six seasons, so this
single table is the canonical bridge for every year.
"""

from __future__ import annotations


PFR_CODE_TO_MADDEN_ABBREV: dict[str, str] = {
    "atl": "ATL",
    "buf": "BUF",
    "car": "CAR",
    "chi": "CHI",
    "cin": "CIN",
    "cle": "CLE",
    "clt": "IND",
    "crd": "ARI",
    "dal": "DAL",
    "den": "DEN",
    "det": "DET",
    "gnb": "GB",
    "htx": "HOU",
    "jax": "JAX",
    "kan": "KC",
    "mia": "MIA",
    "min": "MIN",
    "nor": "NO",
    "nwe": "NE",
    "nyg": "NYG",
    "nyj": "NYJ",
    "oti": "TEN",
    "phi": "PHI",
    "pit": "PIT",
    "rai": "LV",
    "ram": "LAR",
    "rav": "BAL",
    "sdg": "LAC",
    "sea": "SEA",
    "sfo": "SF",
    "tam": "TB",
    "was": "WAS",
}


_MADDEN_ABBREV_TO_PFR_CODE: dict[str, str] = {
    abbrev: code for code, abbrev in PFR_CODE_TO_MADDEN_ABBREV.items()
}


def normalize_team_code(pfr_code: str) -> str:
    """Return the Madden abbreviation for a PFR code. Raises ``KeyError`` for unknowns."""
    try:
        return PFR_CODE_TO_MADDEN_ABBREV[pfr_code]
    except KeyError as exc:
        raise KeyError(
            f"Unknown PFR team code {pfr_code!r}; "
            f"known codes: {sorted(PFR_CODE_TO_MADDEN_ABBREV)}"
        ) from exc


def madden_abbrev_to_pfr_code(abbrev: str) -> str:
    """Reverse lookup: Madden abbreviation -> PFR code. Useful for diagnostics."""
    try:
        return _MADDEN_ABBREV_TO_PFR_CODE[abbrev]
    except KeyError as exc:
        raise KeyError(
            f"Unknown Madden abbreviation {abbrev!r}; "
            f"known abbreviations: {sorted(_MADDEN_ABBREV_TO_PFR_CODE)}"
        ) from exc
