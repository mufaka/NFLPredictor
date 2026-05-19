"""Device resolution, seeding, and the learned-rung training loop (TR-TRAIN-01..07).

Determinism rules:
- ``resolve_device`` honors the config: ``"cpu"`` always picks CPU; ``"cuda"``
  fails fast if no CUDA device is available; ``"auto"`` picks CUDA when
  available, else CPU.
- ``seed_all`` is invoked at the start of every learned-rung training run. It
  pins ``torch``, ``numpy``, and Python ``random``, sets ``PYTHONHASHSEED``,
  and calls ``torch.use_deterministic_algorithms(True)``. On CUDA it also
  seeds ``torch.cuda`` and assumes ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` was set
  earlier in the process (the pipeline entry point sets it before any CUDA
  allocation).
- ``train_learned_rung`` uses ``Adam + L1Loss``, shuffles training indices via
  a ``torch.Generator`` seeded from ``seed + epoch`` (deterministic per epoch),
  evaluates val MAE after every epoch, and tracks ``best_epoch`` / best
  ``state_dict``. Early stopping triggers when val MAE hasn't improved over
  ``early_stop_patience`` epochs. Predictions then use the best-epoch params.
"""

from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from .encoders import (
    ColumnClassification,
    FeatureEncoder,
    GAME_ID_COLUMN,
    prepare_batch,
)


# --------------------------------------------------------------------------
# Device resolution
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedDevice:
    """The device the training loop will use, plus identifying metadata for the manifest."""

    name: str                         # "cpu" or "cuda"
    torch_device: torch.device
    cuda_device_name: Optional[str]   # e.g. "NVIDIA RTX 4090"; None on CPU
    cuda_version: Optional[str]       # e.g. "12.4"; None on CPU


def resolve_device(requested: str) -> ResolvedDevice:
    """Resolve ``config.device`` to a concrete torch device (TR-CFG-10).

    - ``"cpu"``  → always CPU.
    - ``"cuda"`` → CUDA when available, else raise.
    - ``"auto"`` → CUDA when available, else CPU.
    """
    if requested == "cpu":
        return ResolvedDevice(
            name="cpu",
            torch_device=torch.device("cpu"),
            cuda_device_name=None,
            cuda_version=None,
        )
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "training_config.device='cuda' but no CUDA device is available; "
                "switch to 'auto' or 'cpu', or install a CUDA-enabled torch wheel"
            )
        return _build_cuda_resolved()
    if requested == "auto":
        if torch.cuda.is_available():
            return _build_cuda_resolved()
        return ResolvedDevice(
            name="cpu",
            torch_device=torch.device("cpu"),
            cuda_device_name=None,
            cuda_version=None,
        )
    raise ValueError(
        f"unknown requested device {requested!r}; expected 'auto' | 'cpu' | 'cuda'"
    )


def _build_cuda_resolved() -> ResolvedDevice:
    return ResolvedDevice(
        name="cuda",
        torch_device=torch.device("cuda"),
        cuda_device_name=torch.cuda.get_device_name(0),
        cuda_version=torch.version.cuda,
    )


# --------------------------------------------------------------------------
# Seeding (TR-TRAIN-02)
# --------------------------------------------------------------------------


def seed_all(seed: int, device: ResolvedDevice) -> None:
    """Pin every RNG to ``seed`` for the resolved device.

    The pipeline entry point is responsible for setting ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
    and ``PYTHONHASHSEED`` in the environment BEFORE any CUDA allocation or any
    hashed-structure iteration — Python only reads ``PYTHONHASHSEED`` at
    interpreter startup, so a mid-process write here cannot re-seed string
    hashing. We still set it for documentation/observability of intent;
    output determinism does not depend on it because every collection that
    crosses an output boundary is sorted explicitly.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if device.name == "cuda":
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Tensor batching
# --------------------------------------------------------------------------


@dataclass
class TensorBatch:
    """All of one slice (train, val, or test) preloaded into device memory.

    The full slice fits comfortably for 272 games × ~1300 features, so we
    avoid per-epoch dataloader machinery and slice indices directly.
    """

    numeric: torch.Tensor
    low_card: dict[str, torch.Tensor]
    high_card: dict[str, torch.Tensor]
    labels: torch.Tensor
    game_ids: tuple[str, ...]

    @property
    def n(self) -> int:
        return self.labels.shape[0]


def prepare_tensors(
    df: pd.DataFrame,
    classification: ColumnClassification,
    device: ResolvedDevice,
) -> TensorBatch:
    """Materialize a TensorBatch from a slice DataFrame."""
    out = prepare_batch(df, classification, device.torch_device)
    return TensorBatch(
        numeric=out["numeric"],         # type: ignore[arg-type]
        low_card=out["low_card"],       # type: ignore[arg-type]
        high_card=out["high_card"],     # type: ignore[arg-type]
        labels=out["labels"],           # type: ignore[arg-type]
        game_ids=out["game_ids"],       # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# Training loop (TR-TRAIN-01..07, TR-MODEL-02..04)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochRecord:
    """One row's worth of per-epoch loss-curve data (DD-LC-01, DD-LC-06).

    ``train_loss`` is the mean of per-batch L1 losses across the epoch's
    training batches; ``val_loss`` is the L1 loss over the full val slice
    using the running model state at end-of-epoch; ``val_mae`` is the
    Phase 5 headline formula evaluated on the same running model state.
    For the v1 trainer's single-val-batch eval path, ``val_loss == val_mae``
    numerically; the two columns are kept separate so the schema survives a
    future multi-batch eval rewrite without a parquet break.
    """

    epoch: int
    train_loss: float
    val_loss: float
    val_mae: float


@dataclass
class TrainingResult:
    """Per-(rung, shape, strategy) or per-fold training outcome."""

    best_epoch: int                 # 1-indexed; 0 if training never improved past init
    best_val_mae: float
    epochs_trained: int
    stopped_early: bool
    best_state_dict: dict[str, torch.Tensor]
    loss_curve: tuple[EpochRecord, ...] = ()  # DD-LC-01: empty when capture is disabled.


def _slice_batch(batch: TensorBatch, idx: torch.Tensor) -> tuple[
    torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor
]:
    """Index every tensor in ``batch`` by ``idx`` (used for minibatch construction)."""
    return (
        batch.numeric[idx],
        {k: v[idx] for k, v in batch.low_card.items()},
        {k: v[idx] for k, v in batch.high_card.items()},
        batch.labels[idx],
    )


def _eval_val(model: nn.Module, val: TensorBatch) -> tuple[torch.Tensor, float]:
    """Forward + tensor-side MAE in one pass. Returns ``(pred, tensor_mae)``.

    The tensor MAE is per TR-MAN-04 — a mean over the full ``(N, 2)`` absolute
    error matrix — but interpreted against ``val.labels``, which is
    column-ordered by ``classification.labels`` (alphabetical sort, so
    ``("away_score", "home_score")``). The manifest's
    ``training_summaries.val_mae`` field uses a different per-prediction
    formula — see :func:`_val_mae_parquet_style` — so when surfacing
    per-epoch ``val_mae`` for Phase 6's loss-curve sidecar we recompute under
    that formula.
    """
    model.eval()
    with torch.no_grad():
        pred = model(val.numeric, val.low_card, val.high_card)
        abs_err = (pred - val.labels).abs()
    return pred, float(abs_err.mean().item())


def _eval_val_mae(model: nn.Module, val: TensorBatch) -> float:
    """Convenience wrapper around :func:`_eval_val` returning only the tensor MAE."""
    _, mae = _eval_val(model, val)
    return mae


def _val_mae_parquet_style(
    pred: torch.Tensor,
    game_ids: tuple[str, ...],
    labels_lookup: dict[str, tuple[float, float]],
) -> float:
    """Recompute val MAE under the manifest's per-prediction Python formula.

    Mirrors ``pipeline._val_mae_from_predictions`` so the per-epoch ``val_mae``
    row at ``epoch == best_epoch`` matches ``training_summaries.val_mae`` to
    machine precision (DD-LC-07). ``labels_lookup`` is ``GameId → (home_score,
    away_score)`` as the pipeline builds it.
    """
    pred_np = pred.detach().cpu().numpy()
    abs_errs: list[float] = []
    for i, gid in enumerate(game_ids):
        true_h, true_a = labels_lookup[gid]
        abs_errs.append(abs(float(pred_np[i, 0]) - true_h))
        abs_errs.append(abs(float(pred_np[i, 1]) - true_a))
    return float(sum(abs_errs) / len(abs_errs)) if abs_errs else float("nan")


def _snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Deep-copy model parameters so the model can keep training without overwriting the snapshot."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def train_learned_rung(
    model: nn.Module,
    train: TensorBatch,
    val: TensorBatch,
    *,
    lr: float,
    batch_size: int,
    max_epochs: int,
    early_stop_patience: int,
    seed: int,
    device: ResolvedDevice,
    capture_loss_curve: bool = True,
    val_labels_lookup: Optional[dict[str, tuple[float, float]]] = None,
) -> TrainingResult:
    """Train ``model`` with Adam + L1Loss, early-stop on val MAE plateau (TR-TRAIN-01..05).

    When ``capture_loss_curve`` is true (the default), the loop records one
    :class:`EpochRecord` per completed epoch on ``TrainingResult.loss_curve``.
    The capture is read-only: it consumes the per-batch ``loss`` value that
    the optimizer already produces and reuses the forward pass ``_eval_val``
    already runs; it draws no RNG, takes no optimizer step, and runs no
    additional gradient computation (DD-LC-08).

    ``val_labels_lookup`` — ``GameId → (home_score, away_score)`` — is the
    same dict the pipeline uses to compute the manifest's
    ``training_summaries.val_mae``. When provided, per-epoch ``val_mae`` is
    recomputed under that Python-side formula so the row at
    ``epoch == best_epoch`` matches the manifest to machine precision
    (DD-LC-07). When omitted (e.g., the capture-is-read-only verification
    test, DD-TEST-06), the loop falls back to the tensor-side MAE — fine
    because that test does not compare values, only trajectory.
    """
    model.to(device.torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss(reduction="mean")  # TR-MODEL-02

    n_train = train.n
    loss_curve: list[EpochRecord] = []

    # Capture epoch-0 val MAE so an unlucky init still gets a baseline best.
    _, best_val_mae = _eval_val(model, val)
    best_epoch = 0
    best_state = _snapshot_state(model)
    epochs_since_best = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        gen = torch.Generator(device="cpu").manual_seed(seed + epoch)  # TR-TRAIN-03
        perm = torch.randperm(n_train, generator=gen)
        if device.name != "cpu":
            perm = perm.to(device.torch_device)

        batch_losses: list[float] = []
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            num, low, high, y = _slice_batch(train, idx)
            optimizer.zero_grad()
            pred = model(num, low, high)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            if capture_loss_curve:
                batch_losses.append(float(loss.detach().item()))

        # One forward pass at end of epoch — used for early stopping AND, when
        # capturing, for the loss-curve row. Reusing the same pred keeps capture
        # read-only with respect to optimizer state (DD-LC-08).
        val_pred, val_tensor_mae = _eval_val(model, val)
        val_mae = val_tensor_mae
        if capture_loss_curve:
            train_loss = (
                float(sum(batch_losses) / len(batch_losses))
                if batch_losses else float("nan")
            )
            if val_labels_lookup is not None:
                val_mae_record = _val_mae_parquet_style(
                    val_pred, val.game_ids, val_labels_lookup,
                )
            else:
                val_mae_record = val_tensor_mae
            loss_curve.append(
                EpochRecord(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_tensor_mae,
                    val_mae=val_mae_record,
                )
            )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = _snapshot_state(model)
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= early_stop_patience:
                return TrainingResult(
                    best_epoch=best_epoch,
                    best_val_mae=best_val_mae,
                    epochs_trained=epoch,
                    stopped_early=True,
                    best_state_dict=best_state,
                    loss_curve=tuple(loss_curve),
                )

    return TrainingResult(
        best_epoch=best_epoch,
        best_val_mae=best_val_mae,
        epochs_trained=max_epochs,
        stopped_early=False,
        best_state_dict=best_state,
        loss_curve=tuple(loss_curve),
    )


# --------------------------------------------------------------------------
# Prediction helpers (TR-TRAIN-05, TR-TRAIN-07)
# --------------------------------------------------------------------------


def predict_with(
    model: nn.Module,
    batch: TensorBatch,
    state_dict: Optional[dict[str, torch.Tensor]] = None,
) -> pd.DataFrame:
    """Run ``model`` over ``batch`` and return a ``(GameId, pred_home, pred_away)`` frame.

    Pass ``state_dict`` to use a specific snapshot (typically the best-epoch
    params); omit to use the model's current weights.
    """
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        pred = model(batch.numeric, batch.low_card, batch.high_card).cpu().numpy()
    return pd.DataFrame({
        GAME_ID_COLUMN: list(batch.game_ids),
        "pred_home": pred[:, 0].astype("float64"),
        "pred_away": pred[:, 1].astype("float64"),
    })
