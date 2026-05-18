"""Top-level orchestration of the Phase 1 data build.

Phase 3 scope: raw input loaders and the canonical header constants used
to detect upstream schema drift. Subsequent phases extend this module
with matching, unmatched handling, and output emission.
"""

from __future__ import annotations

import pathlib

import pandas as pd


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
