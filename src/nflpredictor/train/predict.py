"""Strategy dispatch for trivial- and learned-rung predictions (TR-STRAT-01..04)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from .config import LinearHyperparams, MlpHyperparams
from .encoders import (
    ColumnClassification,
    FeatureEncoder,
)
from .models import build_model
from .train_loop import (
    ResolvedDevice,
    TrainingResult,
    predict_with,
    prepare_tensors,
    seed_all,
    train_learned_rung,
)
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


# --------------------------------------------------------------------------
# Learned-rung dispatch
# --------------------------------------------------------------------------


EncoderFactory = Callable[[], FeatureEncoder]
"""Zero-arg factory that returns a freshly-initialized FeatureEncoder.

The pipeline pre-bakes ``classification`` / ``vocab`` / ``embedding_dims`` into
the factory; each (combination, fold) gets a fresh encoder so embedding
weights start from the seeded init every time.
"""


@dataclass
class LearnedComboResult:
    """A learned-rung combo's predictions plus per-fit training summary."""

    predictions: pd.DataFrame              # cols: S1 → (slice,GameId,...); S3 → (fold_index,GameId,...)
    train_results: list[TrainingResult]    # length 1 for S1, len(folds) for S3


def _hyperparams_for(rung_id: str, linear: LinearHyperparams, mlp: MlpHyperparams):
    """Return the (lr, batch_size, max_epochs, early_stop_patience) tuple for ``rung_id``."""
    if rung_id == "linear":
        return linear.lr, linear.batch_size, linear.max_epochs, linear.early_stop_patience
    if rung_id == "mlp":
        return mlp.lr, mlp.batch_size, mlp.max_epochs, mlp.early_stop_patience
    raise ValueError(f"_hyperparams_for called for non-learned rung {rung_id!r}")


def _train_and_predict(
    rung_id: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    extra_predict_df: pd.DataFrame | None,
    *,
    classification: ColumnClassification,
    encoder_factory: EncoderFactory,
    linear_hp: LinearHyperparams,
    mlp_hp: MlpHyperparams,
    seed: int,
    device: ResolvedDevice,
    val_labels_lookup: dict[str, tuple[float, float]] | None = None,
) -> tuple[TrainingResult, pd.DataFrame, pd.DataFrame | None]:
    """One full train-then-predict cycle for a learned rung.

    Returns ``(training_result, val_predictions, extra_predictions)``. The
    extra frame is ``None`` unless ``extra_predict_df`` was given (S1's test
    slice — S3 never predicts test).
    """
    seed_all(seed, device)  # TR-TRAIN-02
    encoder = encoder_factory()
    model = build_model(rung_id, encoder, mlp_hp)
    lr, batch_size, max_epochs, early_stop_patience = _hyperparams_for(
        rung_id, linear_hp, mlp_hp
    )

    train_batch = prepare_tensors(train_df, classification, device)
    val_batch = prepare_tensors(val_df, classification, device)

    result = train_learned_rung(
        model,
        train_batch,
        val_batch,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stop_patience=early_stop_patience,
        seed=seed,
        device=device,
        val_labels_lookup=val_labels_lookup,
    )

    val_preds = predict_with(model, val_batch, state_dict=result.best_state_dict)
    extra_preds: pd.DataFrame | None = None
    if extra_predict_df is not None:
        extra_batch = prepare_tensors(extra_predict_df, classification, device)
        extra_preds = predict_with(model, extra_batch)  # model already has best params loaded

    # Drop trained tensors before returning so memory churn stays low (TR-TRAIN-07).
    del model, encoder, train_batch, val_batch
    return result, val_preds, extra_preds


def run_learned_combo_s1(
    rung_id: str,
    feature_df: pd.DataFrame,
    splits: dict[str, Any],
    *,
    classification: ColumnClassification,
    encoder_factory: EncoderFactory,
    linear_hp: LinearHyperparams,
    mlp_hp: MlpHyperparams,
    seed: int,
    device: ResolvedDevice,
    labels_lookup: dict[str, tuple[float, float]] | None = None,
) -> LearnedComboResult:
    """Train on S1.train, predict on S1.val and S1.test (TR-STRAT-01)."""
    s1 = splits["S1"]
    train_df = _select(feature_df, s1["train"])
    val_df = _select(feature_df, s1["val"])
    test_df = _select(feature_df, s1["test"])

    result, val_preds, test_preds = _train_and_predict(
        rung_id,
        train_df,
        val_df,
        test_df,
        classification=classification,
        encoder_factory=encoder_factory,
        linear_hp=linear_hp,
        mlp_hp=mlp_hp,
        seed=seed,
        device=device,
        val_labels_lookup=labels_lookup,
    )
    assert test_preds is not None
    val_long = val_preds.assign(slice="val")
    test_long = test_preds.assign(slice="test")
    predictions = pd.concat([val_long, test_long], ignore_index=True)[
        ["slice", "GameId", "pred_home", "pred_away"]
    ]
    return LearnedComboResult(predictions=predictions, train_results=[result])


def run_learned_combo_s3(
    rung_id: str,
    feature_df: pd.DataFrame,
    splits: dict[str, Any],
    *,
    classification: ColumnClassification,
    encoder_factory: EncoderFactory,
    linear_hp: LinearHyperparams,
    mlp_hp: MlpHyperparams,
    seed: int,
    device: ResolvedDevice,
    labels_lookup: dict[str, tuple[float, float]] | None = None,
) -> LearnedComboResult:
    """Train per fold on fold.train, predict on fold.val (TR-STRAT-02, TR-STRAT-03).

    Each fold reseeds via :func:`seed_all` so per-fold predictions are
    individually byte-stable independent of fold order.
    """
    folds = splits["S3"]["folds"]
    results: list[TrainingResult] = []
    chunks: list[pd.DataFrame] = []
    for fold in folds:
        fold_idx = int(fold["fold_index"])
        train_df = _select(feature_df, fold["train"])
        val_df = _select(feature_df, fold["val"])
        result, val_preds, _ = _train_and_predict(
            rung_id,
            train_df,
            val_df,
            None,
            classification=classification,
            encoder_factory=encoder_factory,
            linear_hp=linear_hp,
            mlp_hp=mlp_hp,
            seed=seed,
            device=device,
            val_labels_lookup=labels_lookup,
        )
        results.append(result)
        chunks.append(val_preds.assign(fold_index=fold_idx))
    if not chunks:
        predictions = pd.DataFrame(
            columns=["fold_index", "GameId", "pred_home", "pred_away"]
        )
    else:
        predictions = pd.concat(chunks, ignore_index=True)[
            ["fold_index", "GameId", "pred_home", "pred_away"]
        ]
    return LearnedComboResult(predictions=predictions, train_results=results)
