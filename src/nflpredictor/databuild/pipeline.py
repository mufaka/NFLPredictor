"""Top-level orchestration of the Phase 1 data build (multi-year, 2020-2025)."""

from __future__ import annotations

import collections
import pathlib
import sys
from typing import Iterator

import pandas as pd

from .ids import assign_raw_madden_ids
from .matching import (
    MaddenRow,
    MatchResult,
    Starter,
    build_match_indexes,
    match_starter,
)
from .manifest import build_manifest, write_manifest
from .normalization import normalize_name
from .outputs import (
    MappingRecord,
    build_mapping_records,
    rewrite_box_score_ids,
    write_box_scores,
    write_madden,
    write_mapping,
)
from .overrides import build_override_index, load_overrides
from .unmatched import (
    append_unmatched_rows,
    collect_unmatched_starters,
    compute_fill_values,
    fill_unmatched_rows,
)


# The canonical season set for the real build. The build actually processes
# whatever ``box_scores_<YYYY>.csv`` files are present in the raw directory
# (see :func:`discover_seasons`), so a 2026 season needs no code change and
# the tiny test fixture can ship a two-season subset.
SEASONS: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025)


def discover_seasons(raw_dir: pathlib.Path) -> list[int]:
    """Return the sorted seasons that have a ``box_scores_<YYYY>.csv`` in ``raw_dir``."""
    seasons: list[int] = []
    for path in raw_dir.glob("box_scores_*.csv"):
        suffix = path.stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            seasons.append(int(suffix))
    if not seasons:
        raise ValueError(
            f"no box_scores_<YYYY>.csv files found in {raw_dir}"
        )
    return sorted(seasons)


EXPECTED_BOX_SCORES_HEADER: tuple[str, ...] = (
    "GameId",
    "GameDate",
    "DayOfWeek",
    "StartTime",
    "HomeTeam",
    "AwayTeam",
    "HomeTeamCode",
    "AwayTeamCode",
    "HomeScore",
    "AwayScore",
    "HomeCoach",
    "AwayCoach",
    "Stadium",
    "Attendance",
    "Duration",
    "Roof",
    "Surface",
    "Weather",
    "HomeOff01_Position", "HomeOff01_Name", "HomeOff01_ID",
    "HomeOff02_Position", "HomeOff02_Name", "HomeOff02_ID",
    "HomeOff03_Position", "HomeOff03_Name", "HomeOff03_ID",
    "HomeOff04_Position", "HomeOff04_Name", "HomeOff04_ID",
    "HomeOff05_Position", "HomeOff05_Name", "HomeOff05_ID",
    "HomeOff06_Position", "HomeOff06_Name", "HomeOff06_ID",
    "HomeOff07_Position", "HomeOff07_Name", "HomeOff07_ID",
    "HomeOff08_Position", "HomeOff08_Name", "HomeOff08_ID",
    "HomeOff09_Position", "HomeOff09_Name", "HomeOff09_ID",
    "HomeOff10_Position", "HomeOff10_Name", "HomeOff10_ID",
    "HomeOff11_Position", "HomeOff11_Name", "HomeOff11_ID",
    "HomeDef01_Position", "HomeDef01_Name", "HomeDef01_ID",
    "HomeDef02_Position", "HomeDef02_Name", "HomeDef02_ID",
    "HomeDef03_Position", "HomeDef03_Name", "HomeDef03_ID",
    "HomeDef04_Position", "HomeDef04_Name", "HomeDef04_ID",
    "HomeDef05_Position", "HomeDef05_Name", "HomeDef05_ID",
    "HomeDef06_Position", "HomeDef06_Name", "HomeDef06_ID",
    "HomeDef07_Position", "HomeDef07_Name", "HomeDef07_ID",
    "HomeDef08_Position", "HomeDef08_Name", "HomeDef08_ID",
    "HomeDef09_Position", "HomeDef09_Name", "HomeDef09_ID",
    "HomeDef10_Position", "HomeDef10_Name", "HomeDef10_ID",
    "HomeDef11_Position", "HomeDef11_Name", "HomeDef11_ID",
    "AwayOff01_Position", "AwayOff01_Name", "AwayOff01_ID",
    "AwayOff02_Position", "AwayOff02_Name", "AwayOff02_ID",
    "AwayOff03_Position", "AwayOff03_Name", "AwayOff03_ID",
    "AwayOff04_Position", "AwayOff04_Name", "AwayOff04_ID",
    "AwayOff05_Position", "AwayOff05_Name", "AwayOff05_ID",
    "AwayOff06_Position", "AwayOff06_Name", "AwayOff06_ID",
    "AwayOff07_Position", "AwayOff07_Name", "AwayOff07_ID",
    "AwayOff08_Position", "AwayOff08_Name", "AwayOff08_ID",
    "AwayOff09_Position", "AwayOff09_Name", "AwayOff09_ID",
    "AwayOff10_Position", "AwayOff10_Name", "AwayOff10_ID",
    "AwayOff11_Position", "AwayOff11_Name", "AwayOff11_ID",
    "AwayDef01_Position", "AwayDef01_Name", "AwayDef01_ID",
    "AwayDef02_Position", "AwayDef02_Name", "AwayDef02_ID",
    "AwayDef03_Position", "AwayDef03_Name", "AwayDef03_ID",
    "AwayDef04_Position", "AwayDef04_Name", "AwayDef04_ID",
    "AwayDef05_Position", "AwayDef05_Name", "AwayDef05_ID",
    "AwayDef06_Position", "AwayDef06_Name", "AwayDef06_ID",
    "AwayDef07_Position", "AwayDef07_Name", "AwayDef07_ID",
    "AwayDef08_Position", "AwayDef08_Name", "AwayDef08_ID",
    "AwayDef09_Position", "AwayDef09_Name", "AwayDef09_ID",
    "AwayDef10_Position", "AwayDef10_Name", "AwayDef10_ID",
    "AwayDef11_Position", "AwayDef11_Name", "AwayDef11_ID",
    "Official01_Role", "Official01_Name",
    "Official02_Role", "Official02_Name",
    "Official03_Role", "Official03_Name",
    "Official04_Role", "Official04_Name",
    "Official05_Role", "Official05_Name",
    "Official06_Role", "Official06_Name",
    "Official07_Role", "Official07_Name",
)


# The raw Madden file's 56-column header. The leading ``madden_id`` is the
# source's own (non-unique, non-PFR) identifier; it is dropped on read
# (DB-IN-06) and replaced by the build's assigned ``madden_id``.
EXPECTED_MADDEN_HEADER: tuple[str, ...] = (
    "madden_id",
    "team",
    "season",
    "fullname",
    "high_pos_group",
    "position_group",
    "position",
    "overallrating",
    "agility",
    "acceleration",
    "speed",
    "stamina",
    "strength",
    "toughness",
    "injury",
    "awareness",
    "jumping",
    "trucking",
    "archetype",
    "runningstyle",
    "changeofdirection",
    "playrecognition",
    "throwpower",
    "throwaccuracyshort",
    "throwaccuracymid",
    "throwaccuracydeep",
    "playaction",
    "throwonrun",
    "carrying",
    "ballcarriervision",
    "stiffarm",
    "spinmove",
    "jukemove",
    "catching",
    "shortrouterunning",
    "midrouterunning",
    "deeprouterunning",
    "spectacularcatch",
    "catchintraffic",
    "release",
    "runblocking",
    "passblocking",
    "impactblocking",
    "mancoverage",
    "zonecoverage",
    "tackle",
    "hitpower",
    "press",
    "pursuit",
    "kickaccuracy",
    "kickpower",
    "return",
    "jerseynumber",
    "yearspro",
    "age",
    "birthdate",
)


SOURCE_MADDEN_ID_COLUMN = "madden_id"


def _format_header_diff(
    actual: tuple[str, ...], expected: tuple[str, ...]
) -> str:
    """Build a human-readable header diff for schema-mismatch errors."""
    if actual == expected:
        return "<headers match>"
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    parts = [
        f"actual length={len(actual)}, expected length={len(expected)}",
    ]
    if missing:
        parts.append(f"missing columns: {missing}")
    if extra:
        parts.append(f"unexpected columns: {extra}")
    if not missing and not extra:
        parts.append("column order differs")
    return "; ".join(parts)


def load_raw_box_scores(path: pathlib.Path) -> pd.DataFrame:
    """Load one season's box-scores CSV and assert its header matches the spec.

    Pandas' parser transparently normalizes both ``LF`` and ``CRLF`` line
    endings on read (DB-IN-05), so no explicit handling is required here.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    actual = tuple(df.columns)
    if actual != EXPECTED_BOX_SCORES_HEADER:
        raise ValueError(
            f"box_scores header mismatch in {path.name}: "
            + _format_header_diff(actual, EXPECTED_BOX_SCORES_HEADER)
        )
    return df


def load_raw_madden(path: pathlib.Path) -> pd.DataFrame:
    """Load one season's Madden CSV; validate header and drop the source ``madden_id``.

    The build assigns its own ``madden_id`` (DB-ID-01), so the raw file's
    own column of that name is dropped on read (DB-IN-06).
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    actual = tuple(df.columns)
    if actual != EXPECTED_MADDEN_HEADER:
        raise ValueError(
            f"Madden header mismatch in {path.name}: "
            + _format_header_diff(actual, EXPECTED_MADDEN_HEADER)
        )
    return df.drop(columns=[SOURCE_MADDEN_ID_COLUMN])


SLOT_PREFIXES: tuple[str, ...] = tuple(
    f"{side}{nn:02d}"
    for side in ("HomeOff", "HomeDef", "AwayOff", "AwayDef")
    for nn in range(1, 12)
)


def iter_starters(box_scores_df: pd.DataFrame) -> Iterator[Starter]:
    """Yield one :class:`Starter` per per-game per-slot lineup cell.

    Iterates games in GameId-sorted order so the resulting sequence is
    deterministic and the first-seen game for any unmatched player is
    chronological (DB-ID-03).
    """
    sorted_df = box_scores_df.sort_values("GameId", kind="stable")
    for _, row in sorted_df.iterrows():
        game_id = row["GameId"]
        home_code = row["HomeTeamCode"]
        away_code = row["AwayTeamCode"]
        for prefix in SLOT_PREFIXES:
            team_code = home_code if prefix.startswith("Home") else away_code
            yield Starter(
                game_id=game_id,
                slot_column=prefix,
                name=row[f"{prefix}_Name"],
                team_code=team_code,
                position=row[f"{prefix}_Position"],
                box_score_id=row[f"{prefix}_ID"],
            )


def _build_madden_rows_from_df(madden_df: pd.DataFrame) -> list[MaddenRow]:
    """Project an assigned-ID Madden DataFrame down to lightweight match rows."""
    return [
        MaddenRow(
            madden_id=r["madden_id"],
            team=r["team"],
            position=r["position"],
            full_name=r["fullname"],
            normalized_name=normalize_name(r["fullname"]),
        )
        for _, r in madden_df.iterrows()
    ]


def _resolve_unmatched_ids(
    pairs: list[tuple[Starter, MatchResult]],
    assignments: dict[tuple[str, str], str],
) -> list[tuple[Starter, MatchResult]]:
    """Fill in ``madden_id`` on tier-0 results using the unmatched assignments."""
    resolved: list[tuple[Starter, MatchResult]] = []
    for starter, result in pairs:
        if result.madden_id is not None:
            resolved.append((starter, result))
            continue
        key = (normalize_name(starter.name), starter.team_code)
        try:
            madden_id = assignments[key]
        except KeyError as exc:
            raise RuntimeError(
                f"Unmatched starter {starter.name!r} ({starter.team_code}) "
                "has no appended row; this indicates a bug in collect_unmatched_starters"
            ) from exc
        resolved.append(
            (
                starter,
                MatchResult(
                    madden_id=madden_id,
                    tier=result.tier,
                    note_fragment=result.note_fragment,
                    position_mismatch=result.position_mismatch,
                ),
            )
        )
    return resolved


def _log_position_mismatch(starter: Starter, result: MatchResult) -> None:
    if result.position_mismatch is None:
        return
    box_pos, madden_pos = result.position_mismatch
    print(
        f"WARN position mismatch: game={starter.game_id} slot={starter.slot_column} "
        f"name={starter.name!r} team={starter.team_code} "
        f"box={box_pos} madden={madden_pos}",
        file=sys.stderr,
    )


def _counts(
    *,
    raw_madden_rows: int,
    unmatched_appended_rows: int,
    box_score_games: int,
    resolved: list[tuple[Starter, MatchResult]],
) -> dict[str, int]:
    tier_counter: collections.Counter[int] = collections.Counter()
    position_mismatches = 0
    for _, result in resolved:
        tier_counter[result.tier] += 1
        if result.position_mismatch is not None:
            position_mismatches += 1
    return {
        "raw_madden_rows": raw_madden_rows,
        "unmatched_appended_rows": unmatched_appended_rows,
        "total_madden_rows_processed": raw_madden_rows + unmatched_appended_rows,
        "box_score_games": box_score_games,
        "total_starter_slots": len(resolved),
        "tier1_matches": tier_counter[1],
        "tier2_matches": tier_counter[2],
        "tier3_matches": tier_counter[3],
        "tier4_matches": tier_counter[4],
        "unmatched_players_unique": unmatched_appended_rows,
        "position_mismatches_logged": position_mismatches,
    }


def _sum_counts(by_season: dict[str, dict[str, int]]) -> dict[str, int]:
    """Sum each per-season count key into a project-wide total."""
    total: collections.Counter[str] = collections.Counter()
    for season_counts in by_season.values():
        for key, value in season_counts.items():
            total[key] += value
    return dict(total)


def _build_one_season(
    season: int,
    box_scores_df: pd.DataFrame,
    raw_madden_df: pd.DataFrame,
    season_overrides: list,
) -> tuple[pd.DataFrame, pd.DataFrame, list[MappingRecord], dict[str, int]]:
    """Run the full match/append/null-fill pipeline for a single season."""
    raw_madden_count = len(raw_madden_df)

    madden_with_ids = assign_raw_madden_ids(raw_madden_df, season)

    override_index = build_override_index(
        season_overrides, set(madden_with_ids["madden_id"])
    )

    match_indexes = build_match_indexes(_build_madden_rows_from_df(madden_with_ids))

    box_with_season = box_scores_df.copy()
    box_with_season.insert(0, "season", str(season))

    pairs: list[tuple[Starter, MatchResult]] = []
    for starter in iter_starters(box_with_season):
        result = match_starter(starter, match_indexes, override_index)
        _log_position_mismatch(starter, result)
        pairs.append((starter, result))

    unmatched_players = collect_unmatched_starters(pairs)
    augmented_madden, assignments = append_unmatched_rows(
        madden_with_ids, unmatched_players, season
    )

    resolved = _resolve_unmatched_ids(pairs, assignments)

    fill_values = compute_fill_values(augmented_madden)
    filled_madden = fill_unmatched_rows(augmented_madden, fill_values)

    slot_to_madden_id: dict[tuple[str, str], str] = {
        (s.game_id, f"{s.slot_column}_ID"): r.madden_id  # type: ignore[dict-item]
        for s, r in resolved
    }
    processed_box_scores = rewrite_box_score_ids(box_with_season, slot_to_madden_id)

    mapping_records = build_mapping_records(resolved, str(season))

    counts = _counts(
        raw_madden_rows=raw_madden_count,
        unmatched_appended_rows=len(unmatched_players),
        box_score_games=len(box_scores_df),
        resolved=resolved,
    )
    return filled_madden, processed_box_scores, mapping_records, counts


def run_build(raw_dir: pathlib.Path, processed_dir: pathlib.Path) -> None:
    """Run the full multi-year data build end-to-end."""
    overrides = load_overrides(raw_dir / "player_overrides.csv")
    seasons = discover_seasons(raw_dir)

    per_season_madden: list[pd.DataFrame] = []
    per_season_box: list[pd.DataFrame] = []
    all_mapping_records: list[MappingRecord] = []
    by_season_counts: dict[str, dict[str, int]] = {}

    for season in seasons:
        box_df = load_raw_box_scores(raw_dir / f"box_scores_{season}.csv")
        raw_madden_df = load_raw_madden(raw_dir / f"madden_{season}.csv")
        season_overrides = [o for o in overrides if o.season == str(season)]

        madden_df, box_out, records, counts = _build_one_season(
            season, box_df, raw_madden_df, season_overrides
        )
        per_season_madden.append(madden_df)
        per_season_box.append(box_out)
        all_mapping_records.extend(records)
        by_season_counts[str(season)] = counts

    madden_all = pd.concat(per_season_madden, ignore_index=True)
    box_scores_all = pd.concat(per_season_box, ignore_index=True)

    madden_path = processed_dir / "madden_all.csv"
    box_scores_path = processed_dir / "box_scores_all.csv"
    mapping_path = processed_dir / "player_id_mapping.csv"
    manifest_path = processed_dir / "build_manifest.json"

    write_madden(madden_all, madden_path)
    write_box_scores(box_scores_all, box_scores_path)
    write_mapping(all_mapping_records, mapping_path)

    raw_inputs: dict[str, pathlib.Path] = {}
    for season in seasons:
        raw_inputs[f"Data/raw/box_scores_{season}.csv"] = (
            raw_dir / f"box_scores_{season}.csv"
        )
        raw_inputs[f"Data/raw/madden_{season}.csv"] = (
            raw_dir / f"madden_{season}.csv"
        )
    raw_inputs["Data/raw/player_overrides.csv"] = raw_dir / "player_overrides.csv"

    counts = {
        "total": _sum_counts(by_season_counts),
        "by_season": by_season_counts,
    }
    manifest = build_manifest(
        raw_inputs=raw_inputs,
        outputs={
            "Data/processed/madden_all.csv": madden_path,
            "Data/processed/box_scores_all.csv": box_scores_path,
            "Data/processed/player_id_mapping.csv": mapping_path,
        },
        counts=counts,
        repo_dir=raw_dir.parent.parent,
    )
    write_manifest(manifest, manifest_path)

    total = counts["total"]
    print(
        "multi-year build complete ({n} seasons): "
        "tier1: {tier1_matches}, tier2: {tier2_matches}, "
        "tier3: {tier3_matches}, tier4: {tier4_matches}, "
        "unmatched: {unmatched_players_unique}, "
        "position mismatches: {position_mismatches_logged}".format(
            n=len(seasons), **total
        ),
        file=sys.stderr,
    )
