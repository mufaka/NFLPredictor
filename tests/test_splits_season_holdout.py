"""Tests for the season_holdout single-fold partition (SP-TEST-02)."""

from __future__ import annotations

import pandas as pd
import pytest

from nflpredictor.splits.config import SplitsConfig
from nflpredictor.splits.season_holdout import assign_season_holdout


V2_CONFIG = SplitsConfig(
    splits_version="v2",
    strategies=("season_holdout", "loso_cv"),
    train_seasons=(2020, 2021, 2022, 2023),
    val_season=2024,
    test_season=2025,
)

SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)


def _make_universe(per_season: int = 3) -> pd.DataFrame:
    rows = []
    for season in SEASONS:
        for i in range(per_season):
            rows.append({"GameId": f"{season}_g{i}", "season": season})
    return pd.DataFrame(rows)


def test_partition_covers_every_game_exactly_once():
    universe = _make_universe(per_season=3)
    parts = assign_season_holdout(universe, V2_CONFIG)
    train, val, test = parts["train"], parts["val"], parts["test"]
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    assert set(train) | set(val) | set(test) == set(universe["GameId"])
    assert len(train) + len(val) + len(test) == len(universe)


def test_season_assignments_match_roles():
    universe = _make_universe(per_season=2)
    parts = assign_season_holdout(universe, V2_CONFIG)
    season_of = dict(zip(universe["GameId"], universe["season"]))
    for gid in parts["train"]:
        assert season_of[gid] in (2020, 2021, 2022, 2023)
    for gid in parts["val"]:
        assert season_of[gid] == 2024
    for gid in parts["test"]:
        assert season_of[gid] == 2025


def test_each_list_is_lexicographically_sorted():
    universe = _make_universe(per_season=4).sample(frac=1, random_state=7)
    universe = universe.reset_index(drop=True)
    parts = assign_season_holdout(universe, V2_CONFIG)
    for role in ("train", "val", "test"):
        assert parts[role] == sorted(parts[role])


def test_per_role_counts_match():
    universe = _make_universe(per_season=2)
    parts = assign_season_holdout(universe, V2_CONFIG)
    assert len(parts["train"]) == 4 * 2  # four train seasons
    assert len(parts["val"]) == 2
    assert len(parts["test"]) == 2


def test_function_is_idempotent():
    universe = _make_universe(per_season=3)
    assert assign_season_holdout(universe, V2_CONFIG) == assign_season_holdout(
        universe, V2_CONFIG
    )


def test_input_dataframe_is_not_mutated():
    universe = _make_universe(per_season=2)
    before = universe.copy(deep=True)
    assign_season_holdout(universe, V2_CONFIG)
    pd.testing.assert_frame_equal(universe, before)
