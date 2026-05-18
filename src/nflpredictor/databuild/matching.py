"""Four-tier starter-to-Madden matching pipeline (DB-MATCH-01..07)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz

from .normalization import normalize_name
from .overrides import OverrideIndex
from .positions import positions_compatible
from .teams import normalize_team_code


TIER4_THRESHOLD = 0.85
TIER4_MARGIN = 0.10


@dataclass(frozen=True)
class Starter:
    """One starter slot from a box-score row (game_id, slot, identity)."""

    game_id: str
    slot_column: str  # e.g. "HomeOff01"; ``_Position``/``_Name``/``_ID`` are appended elsewhere.
    name: str
    team_code: str
    position: str
    box_score_id: str  # raw PFR id; may be empty


@dataclass(frozen=True)
class MaddenRow:
    """A light view over a Madden row used by the matcher."""

    madden_id: str
    team: str          # Madden nickname (e.g. "Chiefs")
    position: str
    full_name: str
    normalized_name: str


@dataclass(frozen=True)
class MatchResult:
    """The outcome of running a single starter through the four tiers."""

    madden_id: Optional[str]
    tier: int  # 0 = unmatched, 1..4 = the winning tier
    note_fragment: str
    position_mismatch: Optional[tuple[str, str]] = None  # (box, madden) when incompatible


@dataclass(frozen=True)
class MatchIndexes:
    """Pre-built indexes over the assigned Madden table."""

    rows_by_madden_id: dict[str, MaddenRow]
    rows_by_team_and_normname: dict[tuple[str, str], list[MaddenRow]]
    rows_by_normname: dict[str, list[MaddenRow]]
    rows_by_team: dict[str, list[MaddenRow]]


def build_match_indexes(rows: list[MaddenRow]) -> MatchIndexes:
    """Construct the lookup tables consumed by the tier functions."""
    by_id: dict[str, MaddenRow] = {}
    by_team_name: dict[tuple[str, str], list[MaddenRow]] = {}
    by_name: dict[str, list[MaddenRow]] = {}
    by_team: dict[str, list[MaddenRow]] = {}
    for row in rows:
        by_id[row.madden_id] = row
        by_team_name.setdefault((row.team, row.normalized_name), []).append(row)
        by_name.setdefault(row.normalized_name, []).append(row)
        by_team.setdefault(row.team, []).append(row)
    return MatchIndexes(
        rows_by_madden_id=by_id,
        rows_by_team_and_normname=by_team_name,
        rows_by_normname=by_name,
        rows_by_team=by_team,
    )


def _position_check(starter: Starter, row: MaddenRow) -> Optional[tuple[str, str]]:
    """Return ``(box, madden)`` when the positions are incompatible."""
    if positions_compatible(starter.position, row.position):
        return None
    return (starter.position, row.position)


def tier1_override(
    starter: Starter,
    override_index: OverrideIndex,
    indexes: MatchIndexes,
) -> Optional[MatchResult]:
    """DB-MATCH-02: manual override wins absolutely when one applies."""
    override = override_index.lookup(
        normalized_name=normalize_name(starter.name),
        team_code=starter.team_code,
        box_score_id=starter.box_score_id,
    )
    if override is None:
        return None
    row = indexes.rows_by_madden_id.get(override.madden_id)
    pos_mismatch = _position_check(starter, row) if row is not None else None
    reason = override.reason or "no reason"
    return MatchResult(
        madden_id=override.madden_id,
        tier=1,
        note_fragment=f"tier1: manual override ({reason})",
        position_mismatch=pos_mismatch,
    )


def tier2_team_and_name(
    starter: Starter, indexes: MatchIndexes
) -> Optional[MatchResult]:
    """DB-MATCH-03: unique exact normalized-name match within the team."""
    try:
        team = normalize_team_code(starter.team_code)
    except KeyError:
        return None
    key = (team, normalize_name(starter.name))
    candidates = indexes.rows_by_team_and_normname.get(key, [])
    if len(candidates) != 1:
        return None
    row = candidates[0]
    return MatchResult(
        madden_id=row.madden_id,
        tier=2,
        note_fragment="tier2: deterministic team+name match",
        position_mismatch=_position_check(starter, row),
    )


def tier3_name_leaguewide(
    starter: Starter, indexes: MatchIndexes
) -> Optional[MatchResult]:
    """DB-MATCH-04: exact name match league-wide, only if unique."""
    candidates = indexes.rows_by_normname.get(normalize_name(starter.name), [])
    if len(candidates) != 1:
        return None
    row = candidates[0]
    return MatchResult(
        madden_id=row.madden_id,
        tier=3,
        note_fragment="tier3: deterministic name match league-wide",
        position_mismatch=_position_check(starter, row),
    )


def tier4_fuzzy(
    starter: Starter,
    indexes: MatchIndexes,
    threshold: float = TIER4_THRESHOLD,
    margin: float = TIER4_MARGIN,
) -> Optional[MatchResult]:
    """DB-MATCH-05: token-set fuzzy match within the team."""
    try:
        team = normalize_team_code(starter.team_code)
    except KeyError:
        return None
    candidates = indexes.rows_by_team.get(team, [])
    if not candidates:
        return None
    target = normalize_name(starter.name)
    scored: list[tuple[float, MaddenRow]] = []
    for row in candidates:
        score = fuzz.token_set_ratio(target, row.normalized_name) / 100.0
        scored.append((score, row))
    # DB-MATCH-05: top score must clear threshold AND beat runner-up by margin.
    scored.sort(key=lambda x: (-x[0], x[1].madden_id))
    top_score, top_row = scored[0]
    runner_up_score = scored[1][0] if len(scored) > 1 else 0.0
    if top_score < threshold:
        return None
    if (top_score - runner_up_score) < margin:
        return None
    return MatchResult(
        madden_id=top_row.madden_id,
        tier=4,
        note_fragment=f"tier4: fuzzy match score={top_score:.2f}",
        position_mismatch=_position_check(starter, top_row),
    )


def match_starter(
    starter: Starter,
    indexes: MatchIndexes,
    override_index: OverrideIndex,
) -> MatchResult:
    """Run the four tiers in order; return the first hit or an unmatched result."""
    for tier_fn in (
        lambda s: tier1_override(s, override_index, indexes),
        lambda s: tier2_team_and_name(s, indexes),
        lambda s: tier3_name_leaguewide(s, indexes),
        lambda s: tier4_fuzzy(s, indexes),
    ):
        result = tier_fn(starter)
        if result is not None:
            return result
    return MatchResult(
        madden_id=None,
        tier=0,
        note_fragment="unmatched: appended with null-fill",
        position_mismatch=None,
    )
