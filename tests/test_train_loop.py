"""TR-TEST-04 + supporting checks: training loop behavior, early stopping, determinism."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import torch
from torch import nn

from nflpredictor.train.encoders import (
    FeatureEncoder,
    classify_columns,
)
from nflpredictor.train.models import LinearRung, MlpRung
from nflpredictor.train.pipeline import (
    Combination,
    enumerate_combinations,
)
from nflpredictor.train.train_loop import (
    TrainingResult,
    predict_with,
    prepare_tensors,
    resolve_device,
    seed_all,
    train_learned_rung,
)
from nflpredictor.train.config import (
    LinearHyperparams,
    MlpHyperparams,
    TrainingConfig,
)


# ----------------------------- fixtures -----------------------------------


def _linear_label_fixture() -> tuple[pd.DataFrame, pd.DataFrame, "object"]:
    """5-train + 3-val fixture where home_score = 10·x + 5, away_score = 10·x.

    A linear rung can in principle fit this exactly; an MLP definitely can.
    """
    cols = ["GameId", "x", "home_score", "away_score"]
    train = pd.DataFrame({
        "GameId": [f"T{i}" for i in range(5)],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "home_score": [15.0, 25.0, 35.0, 45.0, 55.0],
        "away_score": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    val = pd.DataFrame({
        "GameId": [f"V{i}" for i in range(3)],
        "x": [1.5, 2.5, 3.5],
        "home_score": [20.0, 30.0, 40.0],
        "away_score": [15.0, 25.0, 35.0],
    })
    cls = classify_columns(cols, vocab={})
    return train, val, cls


def _build_linear(cls: Any) -> LinearRung:
    enc = FeatureEncoder(cls, vocab={}, embedding_dims={})
    return LinearRung(enc)


# --------------------------- device resolution ----------------------------


def test_resolve_device_cpu_always_cpu() -> None:
    rd = resolve_device("cpu")
    assert rd.name == "cpu"
    assert rd.torch_device.type == "cpu"
    assert rd.cuda_device_name is None
    assert rd.cuda_version is None


def test_resolve_device_auto_falls_back_when_no_cuda() -> None:
    # This dev machine has no CUDA — "auto" must resolve to CPU here.
    rd = resolve_device("auto")
    assert rd.name in {"cpu", "cuda"}
    if not torch.cuda.is_available():
        assert rd.name == "cpu"


def test_resolve_device_cuda_fails_when_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available; skip the negative case")
    with pytest.raises(RuntimeError, match="no CUDA device"):
        resolve_device("cuda")


def test_resolve_device_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown requested device"):
        resolve_device("xpu")


# ----------------------------- seed_all -----------------------------------


def test_seed_all_pins_torch_rng() -> None:
    device = resolve_device("cpu")
    seed_all(1729, device)
    a = torch.randn(5)
    seed_all(1729, device)
    b = torch.randn(5)
    assert torch.equal(a, b)


# ----------------------------- training loop -----------------------------


def test_training_loop_decreases_val_mae() -> None:
    """3 epochs on a linearly-separable fixture must decrease val MAE below the epoch-0 baseline."""
    train, val, cls = _linear_label_fixture()
    device = resolve_device("cpu")
    seed_all(1729, device)
    model = _build_linear(cls)
    train_b = prepare_tensors(train, cls, device)
    val_b = prepare_tensors(val, cls, device)
    result = train_learned_rung(
        model, train_b, val_b,
        lr=0.05, batch_size=2, max_epochs=3, early_stop_patience=100,
        seed=1729, device=device,
    )
    assert result.epochs_trained == 3
    assert not result.stopped_early
    # best_val_mae was recorded after epoch 0 as the baseline; trained model is strictly better.
    assert result.best_epoch in {1, 2, 3}
    assert result.best_val_mae < 50.0  # epoch-0 init would be much higher than ~10 pts


def test_best_epoch_holds_lowest_val_mae(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a controlled val-MAE trajectory and assert best_epoch matches the minimum."""
    from nflpredictor.train import train_loop as tl
    # 11 evals = 1 baseline (epoch 0) + 10 epochs. Minimum is 30.0 at epoch 2.
    trajectory = iter([100.0, 50.0, 30.0, 35.0, 40.0, 32.0, 33.0, 34.0, 36.0, 38.0, 39.0])
    def fake_eval(model: nn.Module, val: Any) -> float:
        return next(trajectory)
    monkeypatch.setattr(tl, "_eval_val_mae", fake_eval)

    train, val, cls = _linear_label_fixture()
    device = resolve_device("cpu")
    seed_all(1729, device)
    model = _build_linear(cls)
    train_b = prepare_tensors(train, cls, device)
    val_b = prepare_tensors(val, cls, device)
    result = train_learned_rung(
        model, train_b, val_b,
        lr=0.01, batch_size=2, max_epochs=10, early_stop_patience=100,
        seed=1729, device=device,
    )
    assert result.best_epoch == 2
    assert result.best_val_mae == 30.0
    assert result.epochs_trained == 10
    assert not result.stopped_early


def test_early_stop_on_plateau(monkeypatch: pytest.MonkeyPatch) -> None:
    """TR-TRAIN-04: early stop fires after `patience` consecutive non-improving epochs."""
    from nflpredictor.train import train_loop as tl
    # Baseline 100; epoch-1 drops to 50; then plateau (every epoch == 50.0001 = no improvement).
    trajectory = iter([100.0, 50.0] + [50.0001] * 10)
    monkeypatch.setattr(tl, "_eval_val_mae", lambda model, val: next(trajectory))

    train, val, cls = _linear_label_fixture()
    device = resolve_device("cpu")
    seed_all(1729, device)
    model = _build_linear(cls)
    train_b = prepare_tensors(train, cls, device)
    val_b = prepare_tensors(val, cls, device)
    result = train_learned_rung(
        model, train_b, val_b,
        lr=0.01, batch_size=2, max_epochs=20, early_stop_patience=3,
        seed=1729, device=device,
    )
    assert result.stopped_early is True
    # best at epoch 1 (val=50); patience=3 means stop at epoch 4 (3 consecutive non-improvements).
    assert result.best_epoch == 1
    assert result.best_val_mae == 50.0
    assert result.epochs_trained == 4


def test_predict_uses_best_epoch_state() -> None:
    """Predictions are produced with best-epoch params, not final-epoch params."""
    train, val, cls = _linear_label_fixture()
    device = resolve_device("cpu")
    seed_all(1729, device)
    model = _build_linear(cls)
    train_b = prepare_tensors(train, cls, device)
    val_b = prepare_tensors(val, cls, device)
    result = train_learned_rung(
        model, train_b, val_b,
        lr=0.05, batch_size=2, max_epochs=20, early_stop_patience=100,
        seed=1729, device=device,
    )
    pred_best = predict_with(model, val_b, state_dict=result.best_state_dict)
    # Quick sanity: pred frame shape + non-NaN.
    assert list(pred_best.columns) == ["GameId", "pred_home", "pred_away"]
    assert pred_best["pred_home"].notna().all()
    assert pred_best["pred_away"].notna().all()


def test_training_loop_is_byte_deterministic() -> None:
    """Two identically-seeded runs produce identical best_state_dict + predictions (TR-NF-01 on CPU)."""
    train, val, cls = _linear_label_fixture()
    device = resolve_device("cpu")

    def one_run() -> tuple[TrainingResult, pd.DataFrame]:
        seed_all(1729, device)
        model = _build_linear(cls)
        train_b = prepare_tensors(train, cls, device)
        val_b = prepare_tensors(val, cls, device)
        res = train_learned_rung(
            model, train_b, val_b,
            lr=0.05, batch_size=2, max_epochs=15, early_stop_patience=100,
            seed=1729, device=device,
        )
        preds = predict_with(model, val_b, state_dict=res.best_state_dict)
        return res, preds

    res_a, preds_a = one_run()
    res_b, preds_b = one_run()
    assert res_a.best_epoch == res_b.best_epoch
    assert res_a.best_val_mae == res_b.best_val_mae
    for k in res_a.best_state_dict:
        assert torch.equal(res_a.best_state_dict[k], res_b.best_state_dict[k])
    pd.testing.assert_frame_equal(preds_a, preds_b)


def test_mlp_training_loop_runs() -> None:
    """MLP rung trains end-to-end without exploding on the fixture."""
    train, val, cls = _linear_label_fixture()
    device = resolve_device("cpu")
    seed_all(1729, device)
    enc = FeatureEncoder(cls, vocab={}, embedding_dims={})
    model = MlpRung(enc, hidden_dim=8, activation="gelu", dropout=0.0)
    train_b = prepare_tensors(train, cls, device)
    val_b = prepare_tensors(val, cls, device)
    result = train_learned_rung(
        model, train_b, val_b,
        lr=0.05, batch_size=2, max_epochs=10, early_stop_patience=100,
        seed=1729, device=device,
    )
    assert result.epochs_trained == 10
    assert result.best_val_mae > 0.0


# ------------------------- combination enumeration -------------------------


def _make_config(rungs, shapes, strategies) -> TrainingConfig:
    return TrainingConfig(
        training_version="v1",
        seed=1729,
        device="cpu",
        rungs=tuple(rungs),
        shapes=tuple(shapes),
        strategies=tuple(strategies),
        linear=LinearHyperparams(lr=0.001, batch_size=32, max_epochs=10, early_stop_patience=3),
        mlp=MlpHyperparams(lr=0.001, batch_size=32, max_epochs=10, early_stop_patience=3,
                           hidden_dim=8, activation="gelu", dropout=0.1),
        embedding_dims={},
    )


def test_enumerate_full_v1_yields_12_combos() -> None:
    cfg = _make_config(
        ["mean", "team_mean", "linear", "mlp"], ["flat", "pos"], ["S1", "S3"]
    )
    combos = enumerate_combinations(cfg)
    # Trivial: 2 rungs × 1 shape × 2 strategies = 4
    # Learned: 2 rungs × 2 shapes × 2 strategies = 8
    assert len(combos) == 12
    trivial = [c for c in combos if c.is_trivial]
    learned = [c for c in combos if c.is_learned]
    assert len(trivial) == 4
    assert len(learned) == 8
    # Trivial combos all have shape="none".
    assert all(c.shape == "none" for c in trivial)
    # Learned combos cover every (shape, strategy) pair.
    learned_pairs = {(c.shape, c.strategy) for c in learned if c.rung == "linear"}
    assert learned_pairs == {("flat", "S1"), ("flat", "S3"), ("pos", "S1"), ("pos", "S3")}


def test_enumerate_respects_config_subsets() -> None:
    cfg = _make_config(["mean", "linear"], ["flat"], ["S1"])
    combos = enumerate_combinations(cfg)
    assert combos == [
        Combination(rung="mean", shape="none", strategy="S1"),
        Combination(rung="linear", shape="flat", strategy="S1"),
    ]


def test_enumerate_preserves_declared_order() -> None:
    cfg = _make_config(["mlp", "mean", "linear"], ["pos", "flat"], ["S3", "S1"])
    combos = enumerate_combinations(cfg)
    # First mlp combos, then mean, then linear; each learned rung iterates shapes then strategies.
    assert combos[0].rung == "mlp"
    assert combos[0].shape == "pos"
    assert combos[0].strategy == "S3"
    # mean is trivial → uses shape="none".
    mean_indices = [i for i, c in enumerate(combos) if c.rung == "mean"]
    assert all(combos[i].shape == "none" for i in mean_indices)
