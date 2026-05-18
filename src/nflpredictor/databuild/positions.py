"""Box-score <-> Madden position equivalence map (DB-POS-01..04).

Box-score `_Position` values are coarser than Madden positions (e.g.,
box-score uses ``OL`` where Madden splits the line into ``LT``, ``LG``,
``C``, ``RG``, ``RT``). :func:`positions_compatible` returns True when
the two refer to the same player role under this documented mapping;
otherwise the caller logs a position-mismatch warning per DB-POS-02.
"""

from __future__ import annotations


POSITION_EQUIVALENCES: dict[str, set[str]] = {
    "OL": {"C", "LG", "RG", "LT", "RT"},
    "T": {"LT", "RT"},
    "OT": {"LT", "RT"},
    "G": {"LG", "RG"},
    "OG": {"LG", "RG"},
    "DL": {"LE", "RE", "DT"},
    "DE": {"LE", "RE"},
    "NT": {"DT"},
    "DB": {"CB", "FS", "SS"},
    "S": {"FS", "SS"},
    "LB": {"LOLB", "ROLB", "MLB"},
    "OLB": {"LOLB", "ROLB"},
    "RB": {"HB", "FB"},
}


def positions_compatible(box_score_position: str, madden_position: str) -> bool:
    """Return True iff the two positions refer to the same role."""
    if box_score_position == madden_position:
        return True
    return madden_position in POSITION_EQUIVALENCES.get(box_score_position, set())
