"""Manual override loading and indexing (DB-OVR-01..05).

Overrides are season-scoped: each row carries a ``season`` and applies
only to that season's games. The pipeline filters the loaded overrides
to one season before building a per-season :class:`OverrideIndex`.
"""

from __future__ import annotations

import csv
import pathlib
from dataclasses import dataclass
from typing import Optional

from .normalization import normalize_name


OVERRIDES_HEADER: tuple[str, ...] = (
    "season",
    "box_score_name",
    "box_score_team_code",
    "box_score_id",
    "madden_id",
    "reason",
)


@dataclass(frozen=True)
class Override:
    """One row from `player_overrides.csv` (DB-OVR-01..03)."""

    season: str
    box_score_name: str
    box_score_team_code: str
    box_score_id: Optional[str]
    madden_id: str
    reason: str


@dataclass(frozen=True)
class OverrideIndex:
    """Lookup helper built from a list of overrides for a single season."""

    by_box_score_id: dict[str, Override]
    by_name_and_team: dict[tuple[str, str], Override]

    def lookup(
        self,
        *,
        normalized_name: str,
        team_code: str,
        box_score_id: str,
    ) -> Optional[Override]:
        """Return the matching override or None."""
        if box_score_id:
            hit = self.by_box_score_id.get(box_score_id)
            if hit is not None:
                return hit
        return self.by_name_and_team.get((normalized_name, team_code))


def load_overrides(path: pathlib.Path) -> list[Override]:
    """Parse `player_overrides.csv`. Header-only stub returns ``[]``."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = tuple(next(reader))
        except StopIteration:
            raise ValueError(
                f"{path} is empty; expected header "
                f"{','.join(OVERRIDES_HEADER)}"
            )
        if header != OVERRIDES_HEADER:
            raise ValueError(
                f"{path} header mismatch: got {header}, "
                f"expected {OVERRIDES_HEADER}"
            )
        overrides: list[Override] = []
        for row_num, row in enumerate(reader, start=2):
            if not row or all(cell == "" for cell in row):
                continue
            if len(row) != len(OVERRIDES_HEADER):
                raise ValueError(
                    f"{path}:{row_num} expected "
                    f"{len(OVERRIDES_HEADER)} fields, got {len(row)}"
                )
            season, name, team_code, box_id, madden_id, reason = row
            overrides.append(
                Override(
                    season=season,
                    box_score_name=name,
                    box_score_team_code=team_code,
                    box_score_id=box_id or None,
                    madden_id=madden_id,
                    reason=reason,
                )
            )
        return overrides


def build_override_index(
    overrides: list[Override],
    assigned_madden_ids: set[str],
) -> OverrideIndex:
    """Validate and index one season's overrides for fast lookup.

    ``overrides`` is expected to be pre-filtered to a single season.
    Each override is keyed by ``box_score_id`` (when present) and by
    ``(normalized_name, team_code)``. Raises ``ValueError`` if an
    override references an unknown ``madden_id`` (DB-OVR-04) or if two
    overrides match the same starter (DB-OVR-05).
    """
    by_box_score_id: dict[str, Override] = {}
    by_name_and_team: dict[tuple[str, str], Override] = {}
    for override in overrides:
        if override.madden_id not in assigned_madden_ids:
            raise ValueError(
                f"Override for {override.box_score_name!r} "
                f"({override.box_score_team_code}, season {override.season}) "
                f"targets unknown madden_id {override.madden_id!r}; "
                "overrides may only reference already-assigned IDs (DB-OVR-04)"
            )
        if override.box_score_id is not None:
            existing = by_box_score_id.get(override.box_score_id)
            if existing is not None and existing != override:
                raise ValueError(
                    "Ambiguous overrides: two rows target "
                    f"box_score_id={override.box_score_id!r} "
                    f"in season {override.season} (DB-OVR-05)"
                )
            by_box_score_id[override.box_score_id] = override
        key = (normalize_name(override.box_score_name), override.box_score_team_code)
        existing_kv = by_name_and_team.get(key)
        if existing_kv is not None and existing_kv != override:
            raise ValueError(
                "Ambiguous overrides: two rows target "
                f"(normalized_name={key[0]!r}, team_code={key[1]!r}) "
                f"in season {override.season} (DB-OVR-05)"
            )
        by_name_and_team[key] = override
    return OverrideIndex(
        by_box_score_id=by_box_score_id,
        by_name_and_team=by_name_and_team,
    )
