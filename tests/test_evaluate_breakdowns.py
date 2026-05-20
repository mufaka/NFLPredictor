"""EV-TEST-03: per-dimension breakdown aggregation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nflpredictor.evaluate.breakdowns import (
    ALL_SPECS,
    BY_HOME_AWAY,
    BY_ROOF,
    BY_SURFACE,
    BY_TEAM,
    BY_WEEK,
    WEEK_UNIVERSE,
    aggregate_breakdown,
    build_breakdown_table,
)


# Six-game synthetic joined fixture.
# Teams as integer codes (vocab indices). Two teams (0=atl, 1=buf) play each other
# repeatedly so they accumulate enough cell weight to verify per-team metrics.
# A third team (2=car) plays away in one game and home in another.
# Weeks: spread across 1, 2, 3 so empty-week behavior can be tested at week=4.
# Surfaces: 0=grass, 1=fieldturf — one game on surface 2 (a_turf) is absent
# from the vocab universe at index 2 only when by_surface excludes; tested.
SIX_GAMES = pd.DataFrame({
    "GameId": ["g1", "g2", "g3", "g4", "g5", "g6"],
    "week":   [1, 1, 2, 2, 3, 3],
    "home_team_code": [0, 1, 0, 2, 1, 2],
    "away_team_code": [1, 0, 2, 0, 2, 1],
    "surface": [0, 0, 1, 0, 1, 0],
    "roof":    [0, 1, 0, 0, 1, 0],
    "slice":   ["val"] * 6,
    "pred_home": [21.0, 28.0, 14.0, 31.0, 13.0, 28.0],
    "pred_away": [17.0, 20.0, 21.0, 10.0, 13.0, 21.0],
    "true_home": [24.0, 21.0, 24.0, 14.0, 17.0, 27.0],
    "true_away": [17.0, 27.0, 21.0, 10.0, 17.0, 21.0],
})


VOCAB = {
    "team_codes": ["atl", "buf", "car", "den"],   # 4 teams; "den" never appears (empty cell test)
    "surface":    ["grass", "fieldturf", "a_turf"],  # "a_turf" appears in no game
    "roof":       ["dome", "outdoors"],
}


def _team_row(df: pd.DataFrame, team: str, role: str, slice_name: str = "val") -> pd.Series:
    sub = df[
        (df["team_code"] == team)
        & (df["home_or_away"] == role)
        & (df["slice"] == slice_name)
    ]
    assert len(sub) == 1, f"expected 1 row for {team}/{role}/{slice_name}, got {len(sub)}"
    return sub.iloc[0]


# ---------------------------------------------------------------------------
# by_team
# ---------------------------------------------------------------------------


def test_by_team_doubles_each_game_per_home_away_role() -> None:
    df = aggregate_breakdown("combo", SIX_GAMES, "season_holdout", BY_TEAM, VOCAB)
    # 4 teams × 2 roles × 2 slices (val + empty-test) = 16 rows.
    assert len(df) == 16
    assert set(df["team_code"]) == {"atl", "buf", "car", "den"}
    assert set(df["home_or_away"]) == {"home", "away"}
    assert set(df["slice"]) == {"val", "test"}
    # n_games per (team, role) on the val slice = team's appearances on that side.
    assert int(_team_row(df, "atl", "home")["n_games"]) == 2   # g1, g3
    assert int(_team_row(df, "atl", "away")["n_games"]) == 2   # g2, g4
    assert int(_team_row(df, "buf", "home")["n_games"]) == 2   # g2, g5
    assert int(_team_row(df, "buf", "away")["n_games"]) == 2   # g1, g6
    assert int(_team_row(df, "car", "home")["n_games"]) == 2   # g4, g6
    assert int(_team_row(df, "car", "away")["n_games"]) == 2   # g3, g5


def test_by_team_empty_team_emits_null_metrics() -> None:
    df = aggregate_breakdown("combo", SIX_GAMES, "season_holdout", BY_TEAM, VOCAB)
    # "den" never plays — every (slice, role) row should have n_games=0 and null metrics.
    # Pandas coerces None -> NaN inside float metric columns; both representations
    # round-trip to parquet null.
    den_home = _team_row(df, "den", "home")
    den_away = _team_row(df, "den", "away")
    assert int(den_home["n_games"]) == 0
    assert int(den_away["n_games"]) == 0
    assert pd.isna(den_home["mae"])
    assert pd.isna(den_away["mae"])
    assert pd.isna(den_home["wl_accuracy"])


def test_by_team_metrics_match_hand_computation() -> None:
    df = aggregate_breakdown("combo", SIX_GAMES, "season_holdout", BY_TEAM, VOCAB)
    # atl home plays g1 + g3:
    # g1: pred(21,17), true(24,17) → home_err=3, away_err=0 → mae=(3+0)/2=1.5
    # g3: pred(14,21), true(24,21) → home_err=10, away_err=0 → mae=(10+0)/2=5.0
    # cell mae = mean( (3+0)/1 * 2 ... ) actually headline_mae over cell:
    #   mean(home_err + away_err) / 2 = mean(3+0, 10+0) / 2 = (3+10)/2/2 = 13/4 = 3.25
    # Wait: mean(home_err+away_err) = mean(3, 10) = 6.5; /2 = 3.25.
    atl_home = _team_row(df, "atl", "home")
    assert float(atl_home["mae"]) == pytest.approx(3.25)
    # mae_home = mean(home_err) = (3+10)/2 = 6.5
    assert float(atl_home["mae_home"]) == pytest.approx(6.5)


# ---------------------------------------------------------------------------
# by_week
# ---------------------------------------------------------------------------


def test_by_week_emits_full_universe_with_empty_weeks() -> None:
    df = aggregate_breakdown("combo", SIX_GAMES, "season_holdout", BY_WEEK, VOCAB)
    # 18 weeks × 2 slices (val + empty-test) = 36 rows.
    assert len(df) == 2 * len(WEEK_UNIVERSE)
    assert set(df["week"]) == set(WEEK_UNIVERSE)
    # Weeks 1..3 (val slice) have 2 games each; weeks 4..18 have 0; test slice empty.
    week_1 = df[(df["week"] == 1) & (df["slice"] == "val")].iloc[0]
    week_4 = df[(df["week"] == 4) & (df["slice"] == "val")].iloc[0]
    assert int(week_1["n_games"]) == 2
    assert int(week_4["n_games"]) == 0
    assert pd.isna(week_4["mae"])
    assert isinstance(float(week_1["mae"]), float)


def test_by_week_metrics_match_hand_computation() -> None:
    df = aggregate_breakdown("combo", SIX_GAMES, "season_holdout", BY_WEEK, VOCAB)
    # Week 1 val: g1 + g2
    # g1: home_err=3, away_err=0 → contribute 3, 0
    # g2: home_err=|28-21|=7, away_err=|20-27|=7 → contribute 7, 7
    # mean(home_err+away_err)/2 = (3+14)/2/2 = 17/4 = 4.25
    week_1 = df[(df["week"] == 1) & (df["slice"] == "val")].iloc[0]
    assert float(week_1["mae"]) == pytest.approx(17.0 / 4.0)
    # mae_home = mean(3, 7) = 5.0
    assert float(week_1["mae_home"]) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# by_home_away
# ---------------------------------------------------------------------------


def test_by_home_away_two_rows_per_cell() -> None:
    df = aggregate_breakdown("combo", SIX_GAMES, "season_holdout", BY_HOME_AWAY, VOCAB)
    # 2 roles × 2 slices = 4 rows (test slice empty).
    assert len(df) == 4
    val_rows = df[df["slice"] == "val"]
    assert set(val_rows["home_or_away"]) == {"home", "away"}
    home = val_rows[val_rows["home_or_away"] == "home"].iloc[0]
    away = val_rows[val_rows["home_or_away"] == "away"].iloc[0]
    assert int(home["n_games"]) == 6
    assert int(away["n_games"]) == 6
    assert float(home["mae_home"]) == pytest.approx(float(away["mae_home"]))
    assert float(home["mae_away"]) == pytest.approx(float(away["mae_away"]))


def test_by_home_away_mae_home_reflects_home_side_residuals_only() -> None:
    """Plan §3.4 assertion: mae_home in the home row is the home-side MAE."""
    df = aggregate_breakdown("combo", SIX_GAMES, "season_holdout", BY_HOME_AWAY, VOCAB)
    val = df[df["slice"] == "val"]
    home_row = val[val["home_or_away"] == "home"].iloc[0]
    # Hand: home errors = |21-24|, |28-21|, |14-24|, |31-14|, |13-17|, |28-27|
    #                   = 3, 7, 10, 17, 4, 1 → mean = 42/6 = 7.0
    assert float(home_row["mae_home"]) == pytest.approx(7.0)
    away_row = val[val["home_or_away"] == "away"].iloc[0]
    # Away errors: |17-17|, |20-27|, |21-21|, |10-10|, |13-17|, |21-21|
    #            = 0, 7, 0, 0, 4, 0 → mean = 11/6
    assert float(away_row["mae_away"]) == pytest.approx(11.0 / 6.0)


# ---------------------------------------------------------------------------
# by_surface and by_roof
# ---------------------------------------------------------------------------


def test_by_surface_emits_vocab_universe_with_label_column() -> None:
    df = aggregate_breakdown("combo", SIX_GAMES, "season_holdout", BY_SURFACE, VOCAB)
    # 3 surfaces × 2 slices = 6 rows.
    assert len(df) == 2 * len(VOCAB["surface"])
    val = df[df["slice"] == "val"]
    grass = val[val["surface_code"] == 0].iloc[0]
    assert grass["surface_label"] == "grass"
    # Games with surface=0: g1, g2, g4, g6 → n=4
    assert int(grass["n_games"]) == 4
    # surface "a_turf" absent → null cell
    a_turf = val[val["surface_code"] == 2].iloc[0]
    assert a_turf["surface_label"] == "a_turf"
    assert int(a_turf["n_games"]) == 0
    assert pd.isna(a_turf["mae"])


def test_by_roof_emits_vocab_universe_with_label_column() -> None:
    df = aggregate_breakdown("combo", SIX_GAMES, "season_holdout", BY_ROOF, VOCAB)
    # 2 roofs × 2 slices = 4 rows.
    assert len(df) == 2 * len(VOCAB["roof"])
    val = df[df["slice"] == "val"]
    dome = val[val["roof_code"] == 0].iloc[0]
    assert dome["roof_label"] == "dome"
    # Games with roof=0: g1, g3, g4, g6 → n=4
    assert int(dome["n_games"]) == 4


def test_by_surface_excludes_null_sentinel_rows() -> None:
    games = SIX_GAMES.copy()
    games.loc[games.index[0], "surface"] = -1   # NULL sentinel
    df = aggregate_breakdown("combo", games, "season_holdout", BY_SURFACE, VOCAB)
    # No row for surface_code = -1 should appear.
    assert -1 not in set(df["surface_code"])


# ---------------------------------------------------------------------------
# Stacking across combinations: build_breakdown_table sort order
# ---------------------------------------------------------------------------


def test_build_breakdown_table_sorts_lexicographically() -> None:
    joined_by_combo = {
        "rung2_linear__flat__season_holdout": SIX_GAMES,
        "rung0_mean__none__season_holdout": SIX_GAMES,
    }
    ordered = sorted(joined_by_combo.keys())  # lexicographic
    pairs = [(c, "season_holdout") for c in ordered]
    df = build_breakdown_table(BY_WEEK, pairs, joined_by_combo, VOCAB)
    # Combination order — first combination must come first.
    first_block = df["combination_id"].iloc[0]
    assert first_block == "rung0_mean__none__season_holdout"
    # Within a combination + slice, week sorts ascending.
    val_weeks = list(
        df[
            (df["combination_id"] == "rung0_mean__none__season_holdout")
            & (df["slice"] == "val")
        ]["week"]
    )
    assert val_weeks == sorted(val_weeks)
    # And slice ordering is lexicographic per EV-OUT-02: "test" < "val".
    slice_rows = df[df["combination_id"] == "rung0_mean__none__season_holdout"]
    slice_seq = list(slice_rows["slice"].unique())
    assert slice_seq == ["test", "val"]


def test_all_specs_registered() -> None:
    assert tuple(s.name for s in ALL_SPECS) == (
        "by_team", "by_week", "by_home_away", "by_surface", "by_roof",
    )


def test_aggregate_breakdown_s3_per_fold_and_pooled() -> None:
    games = SIX_GAMES.drop(columns=["slice"]).assign(fold_index=[0, 0, 0, 1, 1, 1])
    df = aggregate_breakdown("combo", games, "loso_cv", BY_WEEK, VOCAB)
    # Slices: fold_0, fold_1, pooled — 3 slices × 18 weeks = 54 rows.
    assert set(df["slice"]) == {"fold_0", "fold_1", "pooled"}
    assert len(df) == 3 * len(WEEK_UNIVERSE)
    # Pooled n_games for week 1 must equal fold_0 + fold_1 contributions.
    pooled_w1 = df[(df["slice"] == "pooled") & (df["week"] == 1)].iloc[0]
    fold_0_w1 = df[(df["slice"] == "fold_0") & (df["week"] == 1)].iloc[0]
    fold_1_w1 = df[(df["slice"] == "fold_1") & (df["week"] == 1)].iloc[0]
    assert int(pooled_w1["n_games"]) == int(fold_0_w1["n_games"]) + int(fold_1_w1["n_games"])
