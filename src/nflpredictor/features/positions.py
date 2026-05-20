"""Canonical position taxonomy for B-pos (§4.3)."""

from __future__ import annotations


# Single box-score position label → canonical bucket (§4.3). Dual labels
# such as ``C/G`` or ``FB/RB`` are resolved by :func:`bucket_for_position`
# on the first ``/``-delimited token, so only single labels appear here.
# Covers every first-token label observed across box_scores_2020-2025.
POSITION_BUCKETS: dict[str, str] = {
    # QB
    "QB": "QB",
    # RB (running back family — includes fullback and the HB label)
    "RB": "RB",
    "HB": "RB",
    "FB": "RB",
    # WR
    "WR": "WR",
    # TE
    "TE": "TE",
    # OL (offensive line — tackles, guards, center, generic)
    "OL": "OL",
    "T":  "OL",
    "OT": "OL",
    "G":  "OL",
    "OG": "OL",
    "C":  "OL",
    # DL (defensive line — tackles, ends, nose tackle, generic)
    "DL": "DL",
    "DT": "DL",
    "NT": "DL",
    "DE": "DL",
    # LB (linebackers — middle, outside, inside, generic)
    "LB":  "LB",
    "MLB": "LB",
    "OLB": "LB",
    "ILB": "LB",
    # DB (defensive backs — corners, safeties, generic)
    "DB": "DB",
    "CB": "DB",
    "S":  "DB",
    "SAF": "DB",
    "FS": "DB",
    "SS": "DB",
}


# Per-side capacity for each bucket (§4.3). Total per side: 1+2+4+3+5+5+4+5 = 29
# slots (15 offensive + 14 defensive), so a row of B-pos carries 58 player
# slots (29 home + 29 away) before the present/matched flags.
BUCKET_CAPACITY: dict[str, int] = {
    "QB": 1,
    "RB": 2,
    "WR": 4,
    "TE": 3,
    "OL": 5,
    "DL": 5,
    "LB": 4,
    "DB": 5,
}


# Canonical per-side slot list in taxonomy order. Per FE-POS-04 the column
# names follow ``{side}{bucket}{index}_...`` where ``{side}`` is ``Home`` or
# ``Away`` and ``{index}`` is the within-bucket ordinal (1-based).
def _per_side_slots() -> tuple[str, ...]:
    slots: list[str] = []
    # Order pinned by §4.3's table walk.
    for bucket in ("QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"):
        for idx in range(1, BUCKET_CAPACITY[bucket] + 1):
            slots.append(f"{bucket}{idx}")
    return tuple(slots)


CANONICAL_BPOS_SLOTS_PER_SIDE: tuple[str, ...] = _per_side_slots()


def bucket_for_position(box_score_position: str) -> str:
    """Return the canonical bucket for a box-score position label (§4.3).

    Dual labels (``C/G``, ``FB/RB``, ``WR/RS``, …) are resolved on their
    first ``/``-delimited token — the player's primary position. Raises
    ``KeyError`` with a clear message for unmapped labels (FE-POS-03); new
    labels require a code change here and a ``normalization_version`` bump.
    """
    primary = box_score_position.split("/", 1)[0]
    try:
        return POSITION_BUCKETS[primary]
    except KeyError as exc:
        raise KeyError(
            f"box-score position label {box_score_position!r} (primary "
            f"token {primary!r}) is not in the canonical taxonomy; add a "
            f"mapping in positions.POSITION_BUCKETS and bump normalization_version"
        ) from exc
