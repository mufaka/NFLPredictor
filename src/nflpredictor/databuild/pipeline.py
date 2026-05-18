"""Top-level orchestration of the Phase 1 data build."""

from __future__ import annotations

import collections
import pathlib
import sys
from typing import Iterable, Iterator

import pandas as pd

from .ids import assign_raw_madden_ids
from .matching import (
    MaddenRow,
    MatchIndexes,
    MatchResult,
    Starter,
    build_match_indexes,
    match_starter,
)
from .manifest import build_manifest, write_manifest
from .normalization import normalize_name
from .outputs import (
    build_mapping_records,
    rewrite_box_score_ids,
    write_box_scores,
    write_madden,
    write_mapping,
)
from .overrides import build_override_index, load_overrides
from .unmatched import (
    MATCHED_COLUMN,
    UnmatchedPlayer,
    append_unmatched_rows,
    collect_unmatched_starters,
    compute_fill_values,
    fill_unmatched_rows,
)


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


EXPECTED_MADDEN_HEADER: tuple[str, ...] = (
    "Team",
    "Position",
    "Full Name",
    "Overall Rating",
    "Jersey Number",
    "Speed",
    "Acceleration",
    "Strength",
    "Agility",
    "Awareness",
    "Catching",
    "Carrying",
    "Throw Power",
    "Kick Power",
    "Kick Accuracy",
    "Run Block",
    "Pass Block",
    "Tackle",
    "Break Tackle",
    "Jumping",
    "Kick Return",
    "Injury",
    "Stamina",
    "Toughness",
    "Trucking",
    "Change Of Direction",
    "Ball Carrier Vision",
    "Stiff Arm",
    "Spin Move",
    "Juke Move",
    "Impact Blocking",
    "Run Block Power",
    "Run Block Finesse",
    "Pass Block Power",
    "Pass Block Finesse",
    "Lead Block",
    "Break Sack",
    "Throw Under Pressure",
    "Power Moves",
    "Finesse Moves",
    "Block Shedding",
    "Pursuit",
    "Play Recognition",
    "Man Coverage",
    "Zone Coverage",
    "Spectacular Catch",
    "Catch In Traffic",
    "Short Route Running",
    "Medium Route Running",
    "Deep Route Running",
    "Hit Power",
    "Press",
    "Release",
    "Throw Accuracy Short",
    "Throw Accuracy Mid",
    "Throw Accuracy Deep",
    "Play Action",
    "Throw On The Run",
    "Height",
    "Weight",
    "Age",
    "Birthdate",
    "Years Pro",
    "Running Style",
    "Archetype",
    "College",
    "Total Salary",
    "Signing Bonus",
    "Player Handness",
)


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
    """Load ``box_scores_2024.csv`` and assert its header matches the spec."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    actual = tuple(df.columns)
    if actual != EXPECTED_BOX_SCORES_HEADER:
        raise ValueError(
            "box_scores header mismatch: "
            + _format_header_diff(actual, EXPECTED_BOX_SCORES_HEADER)
        )
    return df


def load_raw_madden(path: pathlib.Path) -> pd.DataFrame:
    """Load the Madden ratings CSV, stripping whitespace from headers (DB-IN-05)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    actual = tuple(df.columns)
    if actual != EXPECTED_MADDEN_HEADER:
        raise ValueError(
            "Madden header mismatch: "
            + _format_header_diff(actual, EXPECTED_MADDEN_HEADER)
        )
    return df


def load_raw_inputs(
    raw_dir: pathlib.Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper returning ``(box_scores_df, madden_df)``."""
    box_scores = load_raw_box_scores(raw_dir / "box_scores_2024.csv")
    madden = load_raw_madden(raw_dir / "maddennfl24fullplayerratings.csv")
    return box_scores, madden


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
            team=r["Team"],
            position=r["Position"],
            full_name=r["Full Name"],
            normalized_name=normalize_name(r["Full Name"]),
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


def run_build(raw_dir: pathlib.Path, processed_dir: pathlib.Path) -> None:
    """Run the full data build end-to-end."""
    box_scores_df, raw_madden_df = load_raw_inputs(raw_dir)
    raw_madden_count = len(raw_madden_df)

    madden_with_ids = assign_raw_madden_ids(raw_madden_df)

    overrides = load_overrides(raw_dir / "player_overrides.csv")
    override_index = build_override_index(
        overrides, set(madden_with_ids["madden_id"])
    )

    match_indexes = build_match_indexes(_build_madden_rows_from_df(madden_with_ids))

    pairs: list[tuple[Starter, MatchResult]] = []
    for starter in iter_starters(box_scores_df):
        result = match_starter(starter, match_indexes, override_index)
        _log_position_mismatch(starter, result)
        pairs.append((starter, result))

    unmatched_players = collect_unmatched_starters(pairs)
    augmented_madden, assignments = append_unmatched_rows(
        madden_with_ids, unmatched_players
    )

    resolved = _resolve_unmatched_ids(pairs, assignments)

    fill_values = compute_fill_values(augmented_madden)
    filled_madden = fill_unmatched_rows(augmented_madden, fill_values)

    slot_to_madden_id: dict[tuple[str, str], str] = {
        (s.game_id, f"{s.slot_column}_ID"): r.madden_id  # type: ignore[arg-type]
        for s, r in resolved
    }
    processed_box_scores = rewrite_box_score_ids(box_scores_df, slot_to_madden_id)

    madden_path = processed_dir / "madden_2024.csv"
    box_scores_path = processed_dir / "box_scores_2024.csv"
    mapping_path = processed_dir / "player_id_mapping.csv"
    manifest_path = processed_dir / "build_manifest.json"

    write_madden(filled_madden, madden_path)
    write_box_scores(processed_box_scores, box_scores_path)
    write_mapping(build_mapping_records(resolved), mapping_path)

    counts = _counts(
        raw_madden_rows=raw_madden_count,
        unmatched_appended_rows=len(unmatched_players),
        box_score_games=len(box_scores_df),
        resolved=resolved,
    )
    manifest = build_manifest(
        raw_inputs={
            "Data/raw/box_scores_2024.csv": raw_dir / "box_scores_2024.csv",
            "Data/raw/maddennfl24fullplayerratings.csv":
                raw_dir / "maddennfl24fullplayerratings.csv",
            "Data/raw/player_overrides.csv": raw_dir / "player_overrides.csv",
        },
        outputs={
            "Data/processed/madden_2024.csv": madden_path,
            "Data/processed/box_scores_2024.csv": box_scores_path,
            "Data/processed/player_id_mapping.csv": mapping_path,
        },
        counts=counts,
        repo_dir=raw_dir.parent.parent,
    )
    write_manifest(manifest, manifest_path)

    print(
        "tier1: {tier1_matches}, tier2: {tier2_matches}, "
        "tier3: {tier3_matches}, tier4: {tier4_matches}, "
        "unmatched: {unmatched_players_unique}, "
        "position mismatches: {position_mismatches_logged}".format(**counts),
        file=sys.stderr,
    )
