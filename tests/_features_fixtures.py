"""Shared synthetic-data builders for Phase 6/7 tests.

These helpers fabricate the minimum shape — a full 44-slot box-score row plus
a matching Madden frame keyed by ``madden_id`` — that the assemblers walk.
Building one of these by hand in every test would drown the assertions, so
they live here.
"""

from __future__ import annotations

import pandas as pd

from nflpredictor.features.config import FeatureConfig, GameFeaturesConfig
from nflpredictor.features.vocab import Vocabulary, build_vocabulary


DEFAULT_OFF_POSITIONS: tuple[str, ...] = (
    "QB", "RB", "RB", "WR", "WR", "WR", "WR", "TE", "OL", "OL", "OL",
)
DEFAULT_DEF_POSITIONS: tuple[str, ...] = (
    "DL", "DL", "DL", "DL", "LB", "LB", "LB", "DB", "DB", "DB", "DB",
)


def default_config(
    madden_columns: tuple[str, ...] = ("Overall Rating", "Archetype"),
    madden_categorical_columns: tuple[str, ...] = ("Archetype",),
    slot_shapes: tuple[str, ...] = ("flat", "pos"),
) -> FeatureConfig:
    return FeatureConfig(
        normalization_version="v1",
        madden_columns=madden_columns,
        madden_categorical_columns=madden_categorical_columns,
        game_features=GameFeaturesConfig(
            weather="parsed", officials="included", include=("week",)
        ),
        slot_shapes=slot_shapes,
    )


def build_box_scores(
    games: list[dict[str, str]] | None = None,
    *,
    n_games: int = 1,
) -> pd.DataFrame:
    """Build a synthetic box-scores DataFrame.

    Each game has all 44 slot columns filled with a default offensive /
    defensive starter pattern. Pass ``games`` to override per-slot positions
    in specific games — keys like ``"HomeOff03_Position": "WR"``.
    """
    if games is None:
        games = [{} for _ in range(n_games)]
    rows: list[dict[str, str]] = []
    madden_seq = 1
    for game_idx, overrides in enumerate(games):
        row: dict[str, str] = {"GameId": f"g{game_idx + 1}"}
        # Game-level scalars used by some tests (weather + officials phases).
        row.update({
            "GameDate": "2024-09-08",
            "DayOfWeek": "Sunday",
            "StartTime": "1:00pm",
            "Stadium": f"Stadium {game_idx + 1}",
            "Roof": "outdoors",
            "Surface": "grass",
            "Weather": "67 degrees, relative humidity 50%, wind 5 mph",
            "HomeTeam": "Home Team",
            "AwayTeam": "Away Team",
            "HomeTeamCode": "hom",
            "AwayTeamCode": "awy",
            "HomeCoach": "Home Coach",
            "AwayCoach": "Away Coach",
            "HomeScore": str(20 + game_idx),
            "AwayScore": str(17 + game_idx),
        })
        for i, role in enumerate((
            "Referee", "Umpire", "Down Judge", "Line Judge",
            "Back Judge", "Side Judge", "Field Judge",
        ), start=1):
            row[f"Official{i:02d}_Role"] = role
            row[f"Official{i:02d}_Name"] = f"Official {role}"

        for side in ("Home", "Away"):
            for unit, defaults in (
                ("Off", DEFAULT_OFF_POSITIONS),
                ("Def", DEFAULT_DEF_POSITIONS),
            ):
                for i, default_pos in enumerate(defaults):
                    slot = f"{side}{unit}{i + 1:02d}"
                    pos_key = f"{slot}_Position"
                    position = overrides.get(pos_key, default_pos)
                    row[f"{slot}_Position"] = position
                    row[f"{slot}_ID"] = f"2024-{madden_seq:05d}"
                    row[f"{slot}_Name"] = f"Player {slot} g{game_idx + 1}"
                    madden_seq += 1
        rows.append(row)
    return pd.DataFrame(rows)


def build_madden(
    box_scores_df: pd.DataFrame,
    *,
    archetypes_by_position: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a Madden frame populating every ``madden_id`` referenced in box scores."""
    archetypes_by_position = archetypes_by_position or {}
    rows: list[dict[str, str]] = []
    for row in box_scores_df.itertuples(index=False):
        for side in ("Home", "Away"):
            for unit in ("Off", "Def"):
                for i in range(1, 12):
                    slot = f"{side}{unit}{i:02d}"
                    position = str(getattr(row, f"{slot}_Position"))
                    madden_id = str(getattr(row, f"{slot}_ID"))
                    archetype = archetypes_by_position.get(
                        position, f"{position}_Default"
                    )
                    # Encode the box-score slot in the rating so spot-checks
                    # can derive the expected value without book-keeping.
                    rating = 50 + (i * 2) + (10 if unit == "Off" else 0)
                    rows.append({
                        "madden_id": madden_id,
                        "Team": "TestTeam",
                        "Position": position,
                        "Full Name": f"Player {slot} {row.GameId}",
                        "Overall Rating": str(rating),
                        "Archetype": archetype,
                        "matched": "1",
                    })
    return pd.DataFrame(rows)


def build_vocab_from(
    box_scores_df: pd.DataFrame, madden_df: pd.DataFrame
) -> Vocabulary:
    """Build a Vocabulary covering every categorical value in the synthetic frames."""
    positions = set()
    for row in box_scores_df.itertuples(index=False):
        for side in ("Home", "Away"):
            for unit in ("Off", "Def"):
                for i in range(1, 12):
                    positions.add(str(getattr(row, f"{side}{unit}{i:02d}_Position")))
    archetypes = set(madden_df["Archetype"].unique())
    return build_vocabulary({
        "positions": positions,
        "Archetype": archetypes,
    })
