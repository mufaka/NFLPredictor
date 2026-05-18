"""Tests for S3 expanding-window CV folds (SP-TEST-03)."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from nflpredictor.splits.config import SplitsConfig
from nflpredictor.splits.s1 import assign_s1
from nflpredictor.splits.s3 import Fold, build_s3


V1_CONFIG = SplitsConfig(
    splits_version="v1",
    strategies=("S1", "S3"),
    train_weeks=(1, 12),
    val_weeks=(13, 15),
    test_weeks=(16, 18),
    s3_k_start=6,
)


def _make_universe(per_week: int = 3) -> pd.DataFrame:
    rows = []
    for week in range(1, 19):
        for i in range(per_week):
            rows.append({"GameId": f"w{week:02d}_g{i}", "week": week})
    return pd.DataFrame(rows)


def test_s3_test_list_equals_s1_test_list():
    universe = _make_universe(per_week=3)
    s1 = assign_s1(universe, V1_CONFIG)
    s3 = build_s3(universe, V1_CONFIG, s1["test"])
    assert s3["test"] == s1["test"]


def test_v1_defaults_yield_nine_folds_with_expected_k_values():
    universe = _make_universe(per_week=2)
    s1 = assign_s1(universe, V1_CONFIG)
    s3 = build_s3(universe, V1_CONFIG, s1["test"])
    folds = s3["folds"]
    assert len(folds) == 9
    assert [f.k for f in folds] == list(range(6, 15))
    # Val weeks should run 7..15 inclusive.
    week_of = dict(zip(universe["GameId"], universe["week"]))
    for fold in folds:
        val_weeks_seen = {week_of[g] for g in fold.val}
        assert val_weeks_seen == {fold.k + 1}


def test_fold_disjoint_invariants():
    universe = _make_universe(per_week=3)
    s1 = assign_s1(universe, V1_CONFIG)
    s3 = build_s3(universe, V1_CONFIG, s1["test"])
    test_set = set(s3["test"])
    for fold in s3["folds"]:
        train_set = set(fold.train)
        val_set = set(fold.val)
        assert train_set.isdisjoint(val_set)
        assert (train_set | val_set).isdisjoint(test_set)


def test_per_fold_lists_are_sorted_ascending():
    universe = _make_universe(per_week=4)
    universe = universe.sample(frac=1, random_state=11).reset_index(drop=True)
    s1 = assign_s1(universe, V1_CONFIG)
    s3 = build_s3(universe, V1_CONFIG, s1["test"])
    for fold in s3["folds"]:
        assert list(fold.train) == sorted(fold.train)
        assert list(fold.val) == sorted(fold.val)


def test_fold_index_is_ascending_and_dense():
    universe = _make_universe(per_week=2)
    s1 = assign_s1(universe, V1_CONFIG)
    s3 = build_s3(universe, V1_CONFIG, s1["test"])
    assert [f.fold_index for f in s3["folds"]] == list(range(len(s3["folds"])))


def test_different_k_start_yields_expected_fold_count():
    cfg = replace(V1_CONFIG, s3_k_start=4)
    universe = _make_universe(per_week=2)
    s1 = assign_s1(universe, cfg)
    s3 = build_s3(universe, cfg, s1["test"])
    # k runs 4..14 -> 11 folds; val weeks run 5..15.
    assert len(s3["folds"]) == 11
    assert [f.k for f in s3["folds"]] == list(range(4, 15))


def test_train_set_grows_monotonically_across_folds():
    universe = _make_universe(per_week=3)
    s1 = assign_s1(universe, V1_CONFIG)
    s3 = build_s3(universe, V1_CONFIG, s1["test"])
    prev_train: set[str] = set()
    for fold in s3["folds"]:
        cur = set(fold.train)
        assert prev_train.issubset(cur)
        prev_train = cur


def test_build_s3_raises_when_k_start_missing():
    cfg = replace(V1_CONFIG, strategies=("S1",), s3_k_start=None)
    universe = _make_universe(per_week=2)
    with pytest.raises(ValueError, match="no s3_k_start"):
        build_s3(universe, cfg, [])


def test_fold_is_frozen_dataclass():
    fold = Fold(fold_index=0, k=6, train=("a",), val=("b",))
    with pytest.raises((AttributeError, Exception)):
        fold.fold_index = 1  # type: ignore[misc]
