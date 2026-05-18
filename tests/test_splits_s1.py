"""Tests for the S1 single-fold partition (SP-TEST-02)."""

from __future__ import annotations

import pandas as pd
import pytest

from nflpredictor.splits.config import SplitsConfig
from nflpredictor.splits.s1 import assign_s1


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


def test_partition_covers_every_game_exactly_once():
    universe = _make_universe(per_week=3)
    parts = assign_s1(universe, V1_CONFIG)

    train, val, test = parts["train"], parts["val"], parts["test"]
    assert set(train).isdisjoint(val)
    assert set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)
    union = set(train) | set(val) | set(test)
    assert union == set(universe["GameId"])
    assert len(train) + len(val) + len(test) == len(universe)


def test_week_assignments_match_boundaries():
    universe = _make_universe(per_week=2)
    parts = assign_s1(universe, V1_CONFIG)
    week_of = dict(zip(universe["GameId"], universe["week"]))

    for gid in parts["train"]:
        assert 1 <= week_of[gid] <= 12
    for gid in parts["val"]:
        assert 13 <= week_of[gid] <= 15
    for gid in parts["test"]:
        assert 16 <= week_of[gid] <= 18


def test_each_list_is_lexicographically_sorted():
    universe = _make_universe(per_week=4)
    # Shuffle row order to confirm the function sorts regardless of input order.
    universe = universe.sample(frac=1, random_state=7).reset_index(drop=True)
    parts = assign_s1(universe, V1_CONFIG)
    for role in ("train", "val", "test"):
        assert parts[role] == sorted(parts[role])


def test_out_of_config_week_raises():
    universe = _make_universe(per_week=2)
    universe = pd.concat(
        [universe, pd.DataFrame([{"GameId": "x_19", "week": 19}])],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="weeks outside the configured"):
        assign_s1(universe, V1_CONFIG)


def test_function_is_idempotent():
    universe = _make_universe(per_week=3)
    parts_a = assign_s1(universe, V1_CONFIG)
    parts_b = assign_s1(universe, V1_CONFIG)
    assert parts_a == parts_b


def test_input_dataframe_is_not_mutated():
    universe = _make_universe(per_week=2)
    before = universe.copy(deep=True)
    assign_s1(universe, V1_CONFIG)
    pd.testing.assert_frame_equal(universe, before)


def test_per_week_counts_match():
    universe = _make_universe(per_week=2)
    parts = assign_s1(universe, V1_CONFIG)
    assert len(parts["train"]) == 12 * 2
    assert len(parts["val"]) == 3 * 2
    assert len(parts["test"]) == 3 * 2
