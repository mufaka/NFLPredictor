"""Tests for leave-one-season-out CV folds (SP-TEST-03)."""

from __future__ import annotations

import pandas as pd
import pytest

from nflpredictor.splits.config import SplitsConfig
from nflpredictor.splits.loso_cv import Fold, build_loso_cv
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


def test_loso_test_list_equals_season_holdout_test_list():
    universe = _make_universe(per_season=3)
    sh = assign_season_holdout(universe, V2_CONFIG)
    loso = build_loso_cv(universe, V2_CONFIG, sh["test"])
    assert loso["test"] == sh["test"]


def test_one_fold_per_rotation_pool_season():
    universe = _make_universe(per_season=2)
    sh = assign_season_holdout(universe, V2_CONFIG)
    loso = build_loso_cv(universe, V2_CONFIG, sh["test"])
    folds = loso["folds"]
    # Rotation pool = train_seasons ∪ {val_season} = 5 seasons.
    assert len(folds) == 5
    assert [f.val_season for f in folds] == [2020, 2021, 2022, 2023, 2024]
    assert [f.fold_index for f in folds] == [0, 1, 2, 3, 4]


def test_each_fold_val_is_exactly_one_season():
    universe = _make_universe(per_season=3)
    sh = assign_season_holdout(universe, V2_CONFIG)
    loso = build_loso_cv(universe, V2_CONFIG, sh["test"])
    season_of = dict(zip(universe["GameId"], universe["season"]))
    for fold in loso["folds"]:
        val_seasons = {season_of[g] for g in fold.val}
        assert val_seasons == {fold.val_season}


def test_fold_disjoint_invariants_and_test_never_in_fold():
    universe = _make_universe(per_season=3)
    sh = assign_season_holdout(universe, V2_CONFIG)
    loso = build_loso_cv(universe, V2_CONFIG, sh["test"])
    test_set = set(loso["test"])
    for fold in loso["folds"]:
        train_set, val_set = set(fold.train), set(fold.val)
        assert train_set.isdisjoint(val_set)
        assert (train_set | val_set).isdisjoint(test_set)


def test_fold_union_covers_every_non_test_game():
    universe = _make_universe(per_season=3)
    sh = assign_season_holdout(universe, V2_CONFIG)
    loso = build_loso_cv(universe, V2_CONFIG, sh["test"])
    test_set = set(loso["test"])
    non_test = set(universe["GameId"]) - test_set
    fold0 = loso["folds"][0]
    assert set(fold0.train) | set(fold0.val) == non_test


def test_per_fold_lists_are_sorted_ascending():
    universe = _make_universe(per_season=4).sample(frac=1, random_state=11)
    universe = universe.reset_index(drop=True)
    sh = assign_season_holdout(universe, V2_CONFIG)
    loso = build_loso_cv(universe, V2_CONFIG, sh["test"])
    for fold in loso["folds"]:
        assert list(fold.train) == sorted(fold.train)
        assert list(fold.val) == sorted(fold.val)


def test_fold_is_frozen_dataclass():
    fold = Fold(fold_index=0, val_season=2020, train=("a",), val=("b",))
    with pytest.raises((AttributeError, Exception)):
        fold.fold_index = 1  # type: ignore[misc]
