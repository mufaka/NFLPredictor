"""Partial TR-TEST-03: trivial rungs are correct and deterministic."""

from __future__ import annotations

import pandas as pd

from nflpredictor.train.predict import (
    run_trivial_combo_holdout,
    run_trivial_combo_cv,
)
from nflpredictor.train.trivial import (
    AWAY_LABEL,
    HOME_LABEL,
    predict_mean,
    predict_team_mean,
)


def _train_fixture() -> pd.DataFrame:
    """Six rows with three distinct teams per side; per-team means are easy to verify by hand."""
    return pd.DataFrame({
        "GameId": [f"G{i}" for i in range(6)],
        "home_team_code": [1, 1, 2, 2, 3, 3],
        "away_team_code": [4, 5, 4, 5, 4, 5],
        HOME_LABEL: [20.0, 24.0, 30.0, 10.0, 14.0, 28.0],
        AWAY_LABEL: [17.0, 21.0, 14.0, 28.0, 24.0, 17.0],
    })


# --------------------------------- rung 0 ---------------------------------


def test_predict_mean_returns_train_means_for_every_row() -> None:
    train = _train_fixture()
    targets = ["X1", "X2", "X3"]
    out = predict_mean(train, targets)
    assert list(out["GameId"]) == targets
    expected_home = train[HOME_LABEL].mean()
    expected_away = train[AWAY_LABEL].mean()
    assert (out["pred_home"] == expected_home).all()
    assert (out["pred_away"] == expected_away).all()


def test_predict_mean_is_idempotent() -> None:
    train = _train_fixture()
    a = predict_mean(train, ["X1", "X2"])
    b = predict_mean(train, ["X1", "X2"])
    pd.testing.assert_frame_equal(a, b)


# --------------------------------- rung 1 ---------------------------------


def test_predict_team_mean_per_team_lookup() -> None:
    train = _train_fixture()
    # Home team 1 is in train rows 0 (20.0) and 1 (24.0) → home mean = 22.0
    # Away team 4 is in train rows 0 (17.0), 2 (14.0), 4 (24.0) → away mean = 18.333…
    target = pd.DataFrame({
        "GameId": ["T1"],
        "home_team_code": [1],
        "away_team_code": [4],
    })
    out = predict_team_mean(train, target)
    assert out.loc[0, "pred_home"] == 22.0
    assert abs(out.loc[0, "pred_away"] - (17.0 + 14.0 + 24.0) / 3) < 1e-12


def test_predict_team_mean_global_fallback_for_unseen_team() -> None:
    train = _train_fixture()
    target = pd.DataFrame({
        "GameId": ["T-unseen"],
        "home_team_code": [99],   # never appeared in train as home
        "away_team_code": [99],   # never appeared in train as away
    })
    out = predict_team_mean(train, target)
    assert out.loc[0, "pred_home"] == train[HOME_LABEL].mean()
    assert out.loc[0, "pred_away"] == train[AWAY_LABEL].mean()


def test_predict_team_mean_partial_fallback() -> None:
    """One side has a seen team, the other doesn't — each side falls back independently."""
    train = _train_fixture()
    target = pd.DataFrame({
        "GameId": ["T-mixed"],
        "home_team_code": [1],       # seen
        "away_team_code": [99],      # unseen
    })
    out = predict_team_mean(train, target)
    assert out.loc[0, "pred_home"] == 22.0
    assert out.loc[0, "pred_away"] == train[AWAY_LABEL].mean()


def test_predict_team_mean_is_idempotent() -> None:
    train = _train_fixture()
    target = pd.DataFrame({
        "GameId": ["T1", "T2"],
        "home_team_code": [1, 2],
        "away_team_code": [4, 5],
    })
    a = predict_team_mean(train, target)
    b = predict_team_mean(train, target)
    pd.testing.assert_frame_equal(a, b)


# ----------------------- run_trivial_combo dispatchers ----------------------


def _full_fixture() -> tuple[pd.DataFrame, dict]:
    """A small (GameId, team, label) frame + a hand-built splits artifact."""
    df = pd.DataFrame({
        "GameId": ["G1", "G2", "G3", "G4", "G5", "G6"],
        "home_team_code": [1, 2, 1, 3, 2, 1],
        "away_team_code": [4, 5, 5, 4, 4, 5],
        HOME_LABEL: [21.0, 28.0, 24.0, 10.0, 14.0, 30.0],
        AWAY_LABEL: [17.0, 14.0, 20.0, 24.0, 27.0, 10.0],
    })
    splits = {
        "splits_version": "vTest",
        "season_holdout": {
            "train": ["G1", "G2", "G3", "G4"],
            "val":   ["G5"],
            "test":  ["G6"],
        },
        "loso_cv": {
            "test": ["G6"],
            "folds": [
                {"fold_index": 0, "val_season": 2020, "train": ["G1", "G2"], "val": ["G3"]},
                {"fold_index": 1, "val_season": 2021, "train": ["G1", "G2", "G3"], "val": ["G4"]},
                {"fold_index": 2, "val_season": 2022, "train": ["G1", "G2", "G3", "G4"], "val": ["G5"]},
            ],
        },
    }
    return df, splits


def test_run_trivial_combo_holdout_covers_val_and_test_slices() -> None:
    df, splits = _full_fixture()
    out = run_trivial_combo_holdout("mean", df, splits)
    assert list(out.columns) == ["slice", "GameId", "pred_home", "pred_away"]
    assert sorted(out["slice"].unique().tolist()) == ["test", "val"]
    val_ids = out[out["slice"] == "val"]["GameId"].tolist()
    test_ids = out[out["slice"] == "test"]["GameId"].tolist()
    assert val_ids == splits["season_holdout"]["val"]
    assert test_ids == splits["season_holdout"]["test"]


def test_run_trivial_combo_cv_one_row_per_fold_val_game() -> None:
    df, splits = _full_fixture()
    out = run_trivial_combo_cv("team_mean", df, splits)
    assert list(out.columns) == ["fold_index", "GameId", "pred_home", "pred_away"]
    # 3 folds × 1 val game each = 3 rows.
    assert len(out) == 3
    assert out["fold_index"].tolist() == [0, 1, 2]
    assert out["GameId"].tolist() == ["G3", "G4", "G5"]


def test_run_trivial_combo_holdout_uses_train_only_for_means() -> None:
    """rung 0 mean predictions equal the holdout train mean, not the full-frame mean."""
    df, splits = _full_fixture()
    out = run_trivial_combo_holdout("mean", df, splits)
    train_df = df[df["GameId"].isin(splits["season_holdout"]["train"])]
    expected_home = train_df[HOME_LABEL].mean()
    expected_away = train_df[AWAY_LABEL].mean()
    assert all(abs(p - expected_home) < 1e-12 for p in out["pred_home"])
    assert all(abs(p - expected_away) < 1e-12 for p in out["pred_away"])


def test_run_trivial_combos_are_byte_idempotent() -> None:
    """TR-RUNG-05: closed-form rungs produce byte-identical output across runs."""
    df, splits = _full_fixture()
    h_a = run_trivial_combo_holdout("team_mean", df, splits)
    h_b = run_trivial_combo_holdout("team_mean", df, splits)
    c_a = run_trivial_combo_cv("mean", df, splits)
    c_b = run_trivial_combo_cv("mean", df, splits)
    pd.testing.assert_frame_equal(h_a, h_b)
    pd.testing.assert_frame_equal(c_a, c_b)
