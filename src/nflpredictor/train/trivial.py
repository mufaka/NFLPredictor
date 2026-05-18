"""Trivial-rung predictors (TR-RUNG-01, TR-RUNG-02, TR-RUNG-05).

Pure pandas: no PyTorch optimizer, no RNG. Predictions are closed-form
functions of the training labels and (for rung 1) the home/away team codes.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


HOME_LABEL: str = "home_score"
AWAY_LABEL: str = "away_score"

# Team-code columns produced by Phase 2 (integer codes into the team_codes vocab).
HOME_TEAM_COLUMN: str = "home_team_code"
AWAY_TEAM_COLUMN: str = "away_team_code"


def predict_mean(
    train_df: pd.DataFrame,
    target_game_ids: Iterable[str],
) -> pd.DataFrame:
    """Rung 0 — every prediction is ``(mean(train home), mean(train away))`` (TR-RUNG-01)."""
    home_mean = float(train_df[HOME_LABEL].mean())
    away_mean = float(train_df[AWAY_LABEL].mean())
    gids = list(target_game_ids)
    return pd.DataFrame({
        "GameId": gids,
        "pred_home": [home_mean] * len(gids),
        "pred_away": [away_mean] * len(gids),
    })


def predict_team_mean(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    home_col: str = HOME_TEAM_COLUMN,
    away_col: str = AWAY_TEAM_COLUMN,
) -> pd.DataFrame:
    """Rung 1 — per-team conditional means with global-mean fallback (TR-RUNG-02).

    For each row in ``target_df``: ``pred_home`` is the training mean of
    ``home_score`` conditioned on the home team's code; ``pred_away`` likewise
    for the away team. Teams unseen in ``train_df`` fall back to the
    corresponding global mean (rung 0's prediction).
    """
    home_global = float(train_df[HOME_LABEL].mean())
    away_global = float(train_df[AWAY_LABEL].mean())
    home_by_team = (
        train_df.groupby(home_col)[HOME_LABEL].mean().to_dict()
    )
    away_by_team = (
        train_df.groupby(away_col)[AWAY_LABEL].mean().to_dict()
    )

    pred_home = [
        float(home_by_team.get(code, home_global))
        for code in target_df[home_col].tolist()
    ]
    pred_away = [
        float(away_by_team.get(code, away_global))
        for code in target_df[away_col].tolist()
    ]
    return pd.DataFrame({
        "GameId": target_df["GameId"].astype(str).tolist(),
        "pred_home": pred_home,
        "pred_away": pred_away,
    })
