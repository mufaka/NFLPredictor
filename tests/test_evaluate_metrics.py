"""EV-TEST-02: metric primitives + cell aggregation match hand computation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nflpredictor.evaluate.metrics import (
    METRIC_KEYS,
    Cell,
    cell_metrics,
    compute_all_metrics,
    compute_cell_metrics,
    compute_mae_headline,
    compute_mae_per_side,
    compute_rmse_per_side,
    compute_spread_mae,
    compute_total_mae,
    compute_wl_accuracy,
    iter_cells,
    join_predictions_with_labels,
)


# Six-game fixture, hand-computed expectations.
# Game | pred_h | pred_a | true_h | true_a | home_err | away_err | spread_err | total_err | pred_win | true_win
#  1   |  24    |  17    |  21    |  17    |   3      |   0      |   3        |   3        |  H      |  H
#  2   |  20    |  20    |  27    |  20    |   7      |   0      |   7        |   7        |  tie    |  H
#  3   |  14    |  21    |  24    |  21    |  10      |   0      |  10        |  10        |  A      |  H
#  4   |  31    |  10    |  14    |  10    |  17      |   0      |  17        |  17        |  H      |  H
#  5   |  13    |  13    |  17    |  17    |   4      |   4      |   0        |   8        |  tie    |  tie
#  6   |  28    |  21    |  27    |  21    |   1      |   0      |   1        |   1        |  H      |  H
PRED_H = np.array([24.0, 20.0, 14.0, 31.0, 13.0, 28.0])
PRED_A = np.array([17.0, 20.0, 21.0, 10.0, 13.0, 21.0])
TRUE_H = np.array([21.0, 27.0, 24.0, 14.0, 17.0, 27.0])
TRUE_A = np.array([17.0, 20.0, 21.0, 10.0, 17.0, 21.0])


def test_mae_headline_matches_hand_computation() -> None:
    # mean(|home_err| + |away_err|) / 2
    home_err = np.abs(PRED_H - TRUE_H)
    away_err = np.abs(PRED_A - TRUE_A)
    expected = float(np.mean(home_err + away_err) / 2.0)
    assert compute_mae_headline(PRED_H, PRED_A, TRUE_H, TRUE_A) == pytest.approx(expected)
    # Sanity-check arithmetic: (3+0+7+0+10+0+17+0+4+4+1+0)/12 = 46/12
    assert expected == pytest.approx(46.0 / 12.0)


def test_mae_per_side_matches_hand_computation() -> None:
    mae_home, mae_away = compute_mae_per_side(PRED_H, PRED_A, TRUE_H, TRUE_A)
    # mean(home_err) = (3+7+10+17+4+1)/6 = 42/6 = 7.0
    # mean(away_err) = (0+0+0+0+4+0)/6 = 4/6
    assert mae_home == pytest.approx(7.0)
    assert mae_away == pytest.approx(4.0 / 6.0)


def test_rmse_per_side_matches_hand_computation() -> None:
    rmse_home, rmse_away = compute_rmse_per_side(PRED_H, PRED_A, TRUE_H, TRUE_A)
    # sqrt(mean((home_err)^2)) = sqrt((9+49+100+289+16+1)/6) = sqrt(464/6)
    assert rmse_home == pytest.approx(math.sqrt(464.0 / 6.0))
    # sqrt((0+0+0+0+16+0)/6) = sqrt(16/6)
    assert rmse_away == pytest.approx(math.sqrt(16.0 / 6.0))


def test_wl_accuracy_with_tie_rules() -> None:
    # Game 1: H == H → correct
    # Game 2: pred=tie, actual=H → incorrect
    # Game 3: pred=A, actual=H → incorrect
    # Game 4: H == H → correct
    # Game 5: pred=tie, actual=tie → correct
    # Game 6: H == H → correct
    # → 4/6 correct
    acc = compute_wl_accuracy(PRED_H, PRED_A, TRUE_H, TRUE_A)
    assert acc == pytest.approx(4.0 / 6.0)


def test_spread_mae_matches_hand_computation() -> None:
    # |spread_err| sequence: 3, 7, 10, 17, 0, 1 → 38/6
    assert compute_spread_mae(PRED_H, PRED_A, TRUE_H, TRUE_A) == pytest.approx(38.0 / 6.0)


def test_total_mae_matches_hand_computation() -> None:
    # |total_err| sequence: 3, 7, 10, 17, 8, 1 → 46/6
    assert compute_total_mae(PRED_H, PRED_A, TRUE_H, TRUE_A) == pytest.approx(46.0 / 6.0)


def test_all_metrics_keys_match_canonical_set() -> None:
    result = compute_all_metrics(PRED_H, PRED_A, TRUE_H, TRUE_A)
    assert set(result.keys()) == set(METRIC_KEYS)
    for value in result.values():
        assert isinstance(value, float)


def test_compute_all_metrics_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_all_metrics(
            np.array([]), np.array([]), np.array([]), np.array([])
        )


# EV-MET-04 tie cases: explicit individual checks.


def test_wl_actual_tie_with_pred_tie_counts_correct() -> None:
    acc = compute_wl_accuracy(
        np.array([20.0]), np.array([20.0]),
        np.array([17.0]), np.array([17.0]),
    )
    assert acc == 1.0


def test_wl_actual_tie_with_pred_non_tie_counts_wrong() -> None:
    acc = compute_wl_accuracy(
        np.array([28.0]), np.array([17.0]),
        np.array([21.0]), np.array([21.0]),
    )
    assert acc == 0.0


def test_wl_pred_tie_with_actual_non_tie_counts_wrong() -> None:
    acc = compute_wl_accuracy(
        np.array([20.0]), np.array([20.0]),
        np.array([28.0]), np.array([17.0]),
    )
    assert acc == 0.0


def test_wl_both_pred_and_actual_winners_match() -> None:
    acc = compute_wl_accuracy(
        np.array([28.0, 17.0]), np.array([21.0, 24.0]),
        np.array([28.0, 17.0]), np.array([21.0, 24.0]),
    )
    assert acc == 1.0


# EV-MET-09 empty cell.


def test_cell_metrics_empty_returns_nulls_and_zero_count() -> None:
    out = compute_cell_metrics(np.array([]), np.array([]), np.array([]), np.array([]))
    assert out["n_games"] == 0
    for key in METRIC_KEYS:
        assert key in out
        assert out[key] is None


def test_cell_metrics_nonempty_returns_floats() -> None:
    out = compute_cell_metrics(PRED_H, PRED_A, TRUE_H, TRUE_A)
    assert out["n_games"] == 6
    for key in METRIC_KEYS:
        assert isinstance(out[key], float)
    assert out["mae"] == pytest.approx(46.0 / 12.0)


# --------------------------------------------------------------------------
# join_predictions_with_labels
# --------------------------------------------------------------------------


def _make_features_flat(game_ids: list[str], home: list[float], away: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GameId": game_ids,
            "home_score": home,
            "away_score": away,
            "week": [1] * len(game_ids),
        }
    )


def test_join_predictions_with_labels_attaches_truths() -> None:
    preds = pd.DataFrame({
        "GameId": ["g1", "g2", "g3"],
        "slice": ["val", "val", "test"],
        "pred_home": [20.0, 30.0, 10.0],
        "pred_away": [17.0, 21.0, 24.0],
    })
    flat = _make_features_flat(
        ["g1", "g2", "g3", "g4"],
        [21.0, 28.0, 17.0, 14.0],
        [17.0, 24.0, 24.0, 10.0],
    )
    joined = join_predictions_with_labels(preds, flat)
    assert list(joined["GameId"]) == ["g1", "g2", "g3"]
    assert list(joined["true_home"]) == [21.0, 28.0, 17.0]
    assert list(joined["true_away"]) == [17.0, 24.0, 24.0]


def test_join_predictions_raises_on_unjoined_game() -> None:
    preds = pd.DataFrame({
        "GameId": ["g1", "gZ"],
        "slice": ["val", "val"],
        "pred_home": [20.0, 30.0],
        "pred_away": [17.0, 21.0],
    })
    flat = _make_features_flat(["g1"], [21.0], [17.0])
    with pytest.raises(ValueError, match="unjoined GameIds"):
        join_predictions_with_labels(preds, flat)


# --------------------------------------------------------------------------
# iter_cells
# --------------------------------------------------------------------------


def _make_s1_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "GameId": ["g1", "g2", "g3", "g4"],
        "slice": ["val", "val", "test", "test"],
        "pred_home": [20.0, 28.0, 24.0, 31.0],
        "pred_away": [17.0, 21.0, 20.0, 28.0],
        "true_home": [21.0, 28.0, 24.0, 28.0],
        "true_away": [17.0, 21.0, 20.0, 31.0],
    })


def _make_s3_frame_3folds_with_overlap() -> pd.DataFrame:
    # fold_0: g1, g2 (2 games)
    # fold_1: g3 (1 game)
    # fold_2: g2 (1 game — overlaps fold_0)
    return pd.DataFrame({
        "GameId": ["g1", "g2", "g3", "g2"],
        "fold_index": [0, 0, 1, 2],
        "pred_home": [20.0, 28.0, 24.0, 28.0],
        "pred_away": [17.0, 21.0, 20.0, 21.0],
        "true_home": [21.0, 28.0, 24.0, 28.0],
        "true_away": [17.0, 21.0, 20.0, 21.0],
    })


def test_iter_cells_s1_yields_val_then_test_in_order() -> None:
    frame = _make_s1_frame()
    cells = list(iter_cells(frame, "S1"))
    assert [c.slice_name for c in cells] == ["val", "test"]
    assert len(cells[0].games) == 2  # val
    assert len(cells[1].games) == 2  # test


def test_iter_cells_s3_yields_per_fold_then_pooled_with_repeats() -> None:
    frame = _make_s3_frame_3folds_with_overlap()
    cells = list(iter_cells(frame, "S3"))
    assert [c.slice_name for c in cells] == ["fold_0", "fold_1", "fold_2", "pooled"]
    assert [len(c.games) for c in cells] == [2, 1, 1, 4]
    pooled = cell_metrics(cells[-1])
    assert pooled["n_games"] == 4  # includes the repeated g2 in fold_2


def test_iter_cells_unknown_strategy_raises() -> None:
    frame = _make_s1_frame()
    with pytest.raises(ValueError, match="unknown strategy"):
        list(iter_cells(frame, "S2"))


def test_cell_metrics_dispatches_to_cell_arrays() -> None:
    frame = pd.DataFrame({
        "GameId": ["g1"],
        "pred_home": [24.0],
        "pred_away": [17.0],
        "true_home": [21.0],
        "true_away": [17.0],
    })
    out = cell_metrics(Cell(slice_name="val", games=frame))
    assert out["n_games"] == 1
    assert out["mae"] == pytest.approx(1.5)  # (3 + 0) / 2
