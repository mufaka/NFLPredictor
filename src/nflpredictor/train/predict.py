"""Strategy dispatch for trivial-rung predictions (TR-STRAT-01..04).

Learned-rung dispatch lands in a later plan phase; this module handles rungs
0 and 1 against an S1 or S3 split.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .trivial import predict_mean, predict_team_mean


def _select(df: pd.DataFrame, game_ids: list[str]) -> pd.DataFrame:
    """Index ``df`` by the supplied GameIds, preserving the GameId order from the split."""
    if not game_ids:
        return df.iloc[0:0]
    indexed = df.set_index("GameId", drop=False)
    return indexed.loc[game_ids].reset_index(drop=True)


def _predict_trivial(rung_id: str, train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    if rung_id == "mean":
        return predict_mean(train_df, target_df["GameId"].astype(str).tolist())
    if rung_id == "team_mean":
        return predict_team_mean(train_df, target_df)
    raise ValueError(f"_predict_trivial called for non-trivial rung {rung_id!r}")


def run_trivial_combo_s1(
    rung_id: str,
    feature_df: pd.DataFrame,
    splits: dict[str, Any],
) -> pd.DataFrame:
    """Train on S1.train, predict on S1.val and S1.test, return long frame with ``slice`` (TR-STRAT-01, TR-STRAT-04)."""
    s1 = splits["S1"]
    train_df = _select(feature_df, s1["train"])
    val_df = _select(feature_df, s1["val"])
    test_df = _select(feature_df, s1["test"])

    val_preds = _predict_trivial(rung_id, train_df, val_df).assign(slice="val")
    test_preds = _predict_trivial(rung_id, train_df, test_df).assign(slice="test")
    return pd.concat([val_preds, test_preds], ignore_index=True)[
        ["slice", "GameId", "pred_home", "pred_away"]
    ]


def run_trivial_combo_s3(
    rung_id: str,
    feature_df: pd.DataFrame,
    splits: dict[str, Any],
) -> pd.DataFrame:
    """Train per fold on fold.train, predict on fold.val, return long frame with ``fold_index`` (TR-STRAT-02, TR-STRAT-03, TR-STRAT-04)."""
    folds = splits["S3"]["folds"]
    chunks: list[pd.DataFrame] = []
    for fold in folds:
        fold_idx = int(fold["fold_index"])
        train_df = _select(feature_df, fold["train"])
        val_df = _select(feature_df, fold["val"])
        preds = _predict_trivial(rung_id, train_df, val_df).assign(fold_index=fold_idx)
        chunks.append(preds)
    if not chunks:
        return pd.DataFrame(columns=["fold_index", "GameId", "pred_home", "pred_away"])
    return pd.concat(chunks, ignore_index=True)[
        ["fold_index", "GameId", "pred_home", "pred_away"]
    ]
