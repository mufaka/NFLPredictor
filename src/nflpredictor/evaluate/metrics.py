"""Metric primitives + per-cell aggregation (§3.4, EV-MET-01..09, EV-COMB-02..05).

All primitives operate on four parallel 1-D arrays of per-game home/away
predictions and labels. ``compute_cell_metrics`` adds the empty-cell handling
(EV-MET-09): when no games contribute, every metric value is ``None`` and
``n_games`` is ``0``. The ``iter_cells`` helper produces the canonical slice
sequence per ``EV-COMB-05``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd


# Canonical key set returned by compute_all_metrics; useful for tests/writers.
METRIC_KEYS: tuple[str, ...] = (
    "mae",
    "mae_home",
    "mae_away",
    "rmse_home",
    "rmse_away",
    "wl_accuracy",
    "spread_mae",
    "total_mae",
)


# ---------------------------------------------------------------------------
# Metric primitives (EV-MET-01..06)
# ---------------------------------------------------------------------------


def compute_mae_headline(
    pred_home: np.ndarray, pred_away: np.ndarray,
    true_home: np.ndarray, true_away: np.ndarray,
) -> float:
    """``mean(|pred_home - true_home| + |pred_away - true_away|) / 2`` (EV-MET-01).

    Numerically identical to Phase 4's TR-MAN-04 formula.
    """
    home_err = np.abs(pred_home - true_home)
    away_err = np.abs(pred_away - true_away)
    return float(np.mean(home_err + away_err) / 2.0)


def compute_mae_per_side(
    pred_home: np.ndarray, pred_away: np.ndarray,
    true_home: np.ndarray, true_away: np.ndarray,
) -> tuple[float, float]:
    """Returns ``(mae_home, mae_away)`` (EV-MET-02)."""
    mae_home = float(np.mean(np.abs(pred_home - true_home)))
    mae_away = float(np.mean(np.abs(pred_away - true_away)))
    return mae_home, mae_away


def compute_rmse_per_side(
    pred_home: np.ndarray, pred_away: np.ndarray,
    true_home: np.ndarray, true_away: np.ndarray,
) -> tuple[float, float]:
    """Returns ``(rmse_home, rmse_away)`` (EV-MET-03)."""
    rmse_home = float(np.sqrt(np.mean((pred_home - true_home) ** 2)))
    rmse_away = float(np.sqrt(np.mean((pred_away - true_away) ** 2)))
    return rmse_home, rmse_away


def compute_wl_accuracy(
    pred_home: np.ndarray, pred_away: np.ndarray,
    true_home: np.ndarray, true_away: np.ndarray,
) -> float:
    """W/L accuracy with explicit tie handling (EV-MET-04).

    Predicted winner is ``sign(pred_home - pred_away)``; actual is
    ``sign(true_home - true_away)``. A predicted tie counts correct only if
    the actual is also a tie; an actual tie counts correct only if the
    prediction is also a tie. Otherwise: agreement of signs.
    """
    pred_diff = pred_home - pred_away
    true_diff = true_home - true_away
    pred_sign = np.sign(pred_diff)
    true_sign = np.sign(true_diff)
    correct = pred_sign == true_sign
    return float(np.mean(correct))


def compute_spread_mae(
    pred_home: np.ndarray, pred_away: np.ndarray,
    true_home: np.ndarray, true_away: np.ndarray,
) -> float:
    """``mean(|(pred_home - pred_away) - (true_home - true_away)|)`` (EV-MET-05)."""
    return float(
        np.mean(np.abs((pred_home - pred_away) - (true_home - true_away)))
    )


def compute_total_mae(
    pred_home: np.ndarray, pred_away: np.ndarray,
    true_home: np.ndarray, true_away: np.ndarray,
) -> float:
    """``mean(|(pred_home + pred_away) - (true_home + true_away)|)`` (EV-MET-06)."""
    return float(
        np.mean(np.abs((pred_home + pred_away) - (true_home + true_away)))
    )


def compute_all_metrics(
    pred_home: np.ndarray, pred_away: np.ndarray,
    true_home: np.ndarray, true_away: np.ndarray,
) -> dict[str, float]:
    """Return the canonical 8-key metric dict; raises on empty input."""
    if len(pred_home) == 0:
        raise ValueError("compute_all_metrics called with empty arrays")
    mae_home, mae_away = compute_mae_per_side(pred_home, pred_away, true_home, true_away)
    rmse_home, rmse_away = compute_rmse_per_side(
        pred_home, pred_away, true_home, true_away
    )
    return {
        "mae": compute_mae_headline(pred_home, pred_away, true_home, true_away),
        "mae_home": mae_home,
        "mae_away": mae_away,
        "rmse_home": rmse_home,
        "rmse_away": rmse_away,
        "wl_accuracy": compute_wl_accuracy(pred_home, pred_away, true_home, true_away),
        "spread_mae": compute_spread_mae(pred_home, pred_away, true_home, true_away),
        "total_mae": compute_total_mae(pred_home, pred_away, true_home, true_away),
    }


def compute_cell_metrics(
    pred_home: np.ndarray, pred_away: np.ndarray,
    true_home: np.ndarray, true_away: np.ndarray,
) -> dict[str, Any]:
    """Return ``{n_games, *METRIC_KEYS}`` with EV-MET-09 empty-cell handling.

    When ``n_games == 0``, every metric value is ``None`` (carried straight
    through to JSON ``null`` / parquet null); the dict still contains every key.
    """
    n_games = int(len(pred_home))
    if n_games == 0:
        out: dict[str, Any] = {"n_games": 0}
        for key in METRIC_KEYS:
            out[key] = None
        return out
    metrics = compute_all_metrics(pred_home, pred_away, true_home, true_away)
    out = {"n_games": n_games}
    out.update(metrics)
    return out


# ---------------------------------------------------------------------------
# Predictions / labels joiner
# ---------------------------------------------------------------------------


def join_predictions_with_labels(
    predictions: pd.DataFrame, features_flat: pd.DataFrame
) -> pd.DataFrame:
    """Left-join predictions on ``features_flat`` to attach labels + breakdown columns.

    The output frame carries every prediction column plus ``true_home`` /
    ``true_away`` plus every other column the caller supplied in
    ``features_flat`` (week / surface / roof / team codes). The breakdown
    aggregators downstream read those columns to partition cells, so the
    joined frame is the single source of truth for both metrics and
    breakdown grouping. Raises on any unjoined prediction GameId.
    """
    if "home_score" not in features_flat.columns or "away_score" not in features_flat.columns:
        raise ValueError("features_flat must carry home_score + away_score columns")
    renamed = features_flat.rename(
        columns={"home_score": "true_home", "away_score": "true_away"}
    )
    out = predictions.merge(renamed, how="left", on="GameId")
    null_mask = out["true_home"].isna() | out["true_away"].isna()
    if null_mask.any():
        missing = sorted(out.loc[null_mask, "GameId"].astype(str).unique())
        raise ValueError(
            f"join_predictions_with_labels: unjoined GameIds {missing[:5]} "
            f"(of {len(missing)}) — features_flat is missing labels"
        )
    return out


# ---------------------------------------------------------------------------
# Slice iteration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One ``(combination, slice)`` cell ready for metric computation.

    ``games`` is a frame with at least ``GameId``, ``pred_home``, ``pred_away``,
    ``true_home``, ``true_away`` columns; row order is undefined.
    """

    slice_name: str
    games: pd.DataFrame


def _slice_arrays(games: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        games["pred_home"].to_numpy(dtype=np.float64, copy=False),
        games["pred_away"].to_numpy(dtype=np.float64, copy=False),
        games["true_home"].to_numpy(dtype=np.float64, copy=False),
        games["true_away"].to_numpy(dtype=np.float64, copy=False),
    )


def iter_cells(
    predictions_with_labels: pd.DataFrame, strategy: str
) -> Iterator[Cell]:
    """Yield per-slice cells in the canonical EV-COMB-05 order.

    S1 → ``["val", "test"]``
    S3 → ``["fold_0", ..., "fold_<n-1>", "pooled"]``

    For S3, fold ordering is by ascending ``fold_index``; the pooled cell
    concatenates every fold's rows (duplicates preserved per EV-MET-07).
    Empty slices still yield a Cell with a 0-row frame so EV-MET-09 fires.
    """
    if strategy == "S1":
        for slice_name in ("val", "test"):
            mask = predictions_with_labels["slice"] == slice_name
            yield Cell(slice_name=slice_name, games=predictions_with_labels.loc[mask])
        return
    if strategy != "S3":
        raise ValueError(f"unknown strategy {strategy!r}; expected 'S1' or 'S3'")

    fold_indices = sorted(
        int(x) for x in predictions_with_labels["fold_index"].unique()
    )
    for i in fold_indices:
        mask = predictions_with_labels["fold_index"] == i
        yield Cell(
            slice_name=f"fold_{i}",
            games=predictions_with_labels.loc[mask],
        )
    # Pooled — concat all fold rows (duplicates preserved per EV-MET-07).
    # Since each row already belongs to exactly one fold_index in S3, the
    # "concatenation" is just every row; if a GameId happens to appear in
    # multiple folds it counts twice, as the spec requires.
    yield Cell(slice_name="pooled", games=predictions_with_labels)


def cell_metrics(cell: Cell) -> dict[str, Any]:
    """Convenience: compute_cell_metrics(...) over a Cell's arrays."""
    arrays = _slice_arrays(cell.games)
    return compute_cell_metrics(*arrays)


# ---------------------------------------------------------------------------
# Headline aggregator (§4.2)
# ---------------------------------------------------------------------------


def build_headline_for_combination(
    joined_predictions: pd.DataFrame, strategy: str
) -> dict[str, Any]:
    """Per-combination headline block: ``{slice_name: {n_games, *metrics}, ...}``.

    Slice key insertion order follows EV-COMB-05 (S1: val→test; S3: fold_0→...→pooled).
    Actual on-disk JSON ordering is determined by ``json.dump(sort_keys=True)`` per
    EV-OUT-01, which sorts keys alphabetically. The in-memory ordering preserved here
    is informational only.
    """
    return {cell.slice_name: cell_metrics(cell) for cell in iter_cells(joined_predictions, strategy)}


def build_headline_metrics(
    ordered_combinations: list[tuple[str, str]],
    joined_by_combo: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Top-level structure: ``{"combinations": {<combo>: <per-combo-block>, ...}}``.

    ``ordered_combinations`` is a list of ``(combination_id, strategy)`` pairs;
    callers (the pipeline) supply them in lexicographic order. The on-disk
    output ordering is finalized by ``json.dump(sort_keys=True)``.
    """
    combinations_block: dict[str, Any] = {
        combination_id: build_headline_for_combination(joined_by_combo[combination_id], strategy)
        for combination_id, strategy in ordered_combinations
    }
    return {"combinations": combinations_block}
