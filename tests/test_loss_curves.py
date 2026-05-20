"""Schema, sort-order, integrity, and capture-neutrality tests for the
per-epoch loss-curve sidecar parquet introduced by Phase 6 (DD-LC-01..08).

The tests run against the same synthetic 36-game train fixture used by
``tests/test_train_integration.py``. Determinism tests for the parquet
itself live in ``tests/test_train_determinism.py``.
"""

from __future__ import annotations

import json
import math
import pathlib
import shutil

import pyarrow.parquet as pq
import pytest

from nflpredictor.train.outputs import (
    PREDICTIONS_DIRNAME,
    TRAINING_LOSS_CURVES_BASENAME,
)
from nflpredictor.train.pipeline import (
    TRAINING_MANIFEST_BASENAME,
    run_training_build,
)
from nflpredictor.train.sources import (
    PHASE2_FEATURES_FLAT_BASENAME,
    PHASE2_FEATURES_POS_BASENAME,
    PHASE2_MANIFEST_BASENAME,
    PHASE2_VOCAB_BASENAME,
    PHASE3_MANIFEST_BASENAME,
    PHASE3_SPLITS_BASENAME,
)
from nflpredictor.train.train_loop import (
    ResolvedDevice,
    prepare_tensors,
    seed_all,
    train_learned_rung,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "train"
FIXTURE_RAW = FIXTURE_DIR / "raw"
FIXTURE_RAW_PHASE2 = FIXTURE_DIR / "raw_phase2"
FIXTURE_RAW_PHASE3 = FIXTURE_DIR / "raw_phase3"
FIXTURE_EXPECTED = FIXTURE_DIR / "expected"


_FIXTURE_INPUTS = (
    (FIXTURE_RAW_PHASE2, PHASE2_FEATURES_FLAT_BASENAME),
    (FIXTURE_RAW_PHASE2, PHASE2_FEATURES_POS_BASENAME),
    (FIXTURE_RAW_PHASE2, PHASE2_VOCAB_BASENAME),
    (FIXTURE_RAW_PHASE2, PHASE2_MANIFEST_BASENAME),
    (FIXTURE_RAW_PHASE3, PHASE3_SPLITS_BASENAME),
    (FIXTURE_RAW_PHASE3, PHASE3_MANIFEST_BASENAME),
)


def _stage_processed(target: pathlib.Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for src_dir, name in _FIXTURE_INPUTS:
        shutil.copy2(src_dir / name, target / name)


@pytest.fixture(scope="module")
def fixture_outputs(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """One training build per pytest session; all loss-curve tests reuse it."""
    if not FIXTURE_EXPECTED.exists():
        pytest.skip(
            "tests/fixtures/train/expected/ is missing — "
            "regenerate via `python -m tests.fixtures.train._regenerate`"
        )
    tmp = tmp_path_factory.mktemp("loss_curves")
    processed = tmp / "Data" / "processed"
    _stage_processed(processed)
    run_training_build(FIXTURE_RAW, processed, repo_dir=tmp)
    return processed


# --------------------------------------------------------------------------
# DD-TEST-03: schema
# --------------------------------------------------------------------------


def test_loss_curves_schema_matches_spec(fixture_outputs: pathlib.Path) -> None:
    """DD-TEST-03 / Spec §4.1: columns and dtypes are exactly as specified."""
    schema = pq.read_schema(fixture_outputs / TRAINING_LOSS_CURVES_BASENAME)
    assert [f.name for f in schema] == [
        "combination_id", "fold", "epoch", "train_loss", "val_loss", "val_mae",
    ]
    types = {f.name: str(f.type) for f in schema}
    assert types["combination_id"] == "string"
    assert types["fold"] == "int32"
    assert types["epoch"] == "int32"
    assert types["train_loss"] == "double"
    assert types["val_loss"] == "double"
    assert types["val_mae"] == "double"


# --------------------------------------------------------------------------
# DD-TEST-04: sort order
# --------------------------------------------------------------------------


def test_loss_curves_sorted_by_combination_fold_epoch(
    fixture_outputs: pathlib.Path,
) -> None:
    """DD-TEST-04 / DD-LC-02: rows are sorted by (combination_id, fold, epoch)."""
    df = pq.read_table(fixture_outputs / TRAINING_LOSS_CURVES_BASENAME).to_pandas()
    keys = list(zip(df["combination_id"], df["fold"].astype(int), df["epoch"].astype(int)))
    assert keys == sorted(keys), "loss-curve rows are not lexicographically sorted"


def test_only_learned_rungs_contribute_rows(
    fixture_outputs: pathlib.Path,
) -> None:
    """DD-LC-01: trivial rungs (rung0_mean, rung1_team_mean) emit no rows."""
    df = pq.read_table(fixture_outputs / TRAINING_LOSS_CURVES_BASENAME).to_pandas()
    combos = set(df["combination_id"].unique())
    assert not any(c.startswith("rung0_") or c.startswith("rung1_") for c in combos), (
        f"trivial rungs unexpectedly emitted loss-curve rows: {sorted(combos)}"
    )


# --------------------------------------------------------------------------
# DD-TEST-02 / DD-LC-07: integrity — best-epoch val_mae matches training_summaries
# --------------------------------------------------------------------------


def _loso_fold_keys(splits_path: pathlib.Path) -> dict[int, int]:
    """Map loso_cv fold_index → val_season (the loss-curve ``fold`` value).

    Returns ``{}`` when the splits artifact carries no loso_cv strategy — the
    default config trains season_holdout only, where every curve has fold 0.
    """
    splits = json.loads(splits_path.read_text())
    loso = splits.get("loso_cv")
    if not loso:
        return {}
    return {int(f["fold_index"]): int(f["val_season"]) for f in loso["folds"]}


def test_best_epoch_val_mae_matches_training_summaries(
    fixture_outputs: pathlib.Path,
) -> None:
    """DD-LC-07 / DD-TEST-02: for each learned (combo, fold) the loss-curve row at
    ``epoch == best_epoch`` matches the corresponding ``training_summaries`` val MAE
    to ~1e-12 relative tolerance.

    Note: the spec speaks of the "final-epoch row"; with early stopping enabled
    in the trainer the meaningful invariant is "the row at best_epoch" (the
    parameters that produced the manifest's val MAE). The training_summaries
    value comes from Python-side mean over per-prediction abs errors; the
    parquet's val_mae comes from torch's tensor-side mean — the two sums can
    differ by a few ulps, hence the relative tolerance.
    """
    manifest = json.loads(
        (fixture_outputs / TRAINING_MANIFEST_BASENAME).read_text()
    )
    df = pq.read_table(fixture_outputs / TRAINING_LOSS_CURVES_BASENAME).to_pandas()
    summaries = manifest["training_summaries"]
    fold_index_to_k = _loso_fold_keys(fixture_outputs / PHASE3_SPLITS_BASENAME)

    checked = 0
    for combo_id, summary in summaries.items():
        # Trivial rungs have no loss curve rows.
        sub_combo = df[df["combination_id"] == combo_id]
        if sub_combo.empty:
            continue

        if "per_fold" in summary:  # S3
            for fold_entry in summary["per_fold"]:
                fold_index = int(fold_entry["fold_index"])
                k = fold_index_to_k[fold_index]
                best_epoch = int(fold_entry["best_epoch"])
                expected_val_mae = float(fold_entry["val_mae"])
                row = sub_combo[
                    (sub_combo["fold"] == k) & (sub_combo["epoch"] == best_epoch)
                ]
                assert len(row) == 1, (
                    f"missing best-epoch row for {combo_id} fold(k)={k} "
                    f"best_epoch={best_epoch}"
                )
                got = float(row["val_mae"].iloc[0])
                assert math.isclose(got, expected_val_mae, rel_tol=1e-12, abs_tol=1e-12), (
                    f"{combo_id} fold(k)={k} best_epoch={best_epoch}: "
                    f"parquet val_mae={got} vs summary val_mae={expected_val_mae}"
                )
                checked += 1
        else:  # S1
            best_epoch = int(summary["best_epoch"])
            expected_val_mae = float(summary["val_mae"])
            row = sub_combo[
                (sub_combo["fold"] == 0) & (sub_combo["epoch"] == best_epoch)
            ]
            assert len(row) == 1, (
                f"missing best-epoch row for {combo_id} fold=0 best_epoch={best_epoch}"
            )
            got = float(row["val_mae"].iloc[0])
            assert math.isclose(got, expected_val_mae, rel_tol=1e-12, abs_tol=1e-12), (
                f"{combo_id}: parquet val_mae={got} vs summary val_mae={expected_val_mae}"
            )
            checked += 1

    assert checked > 0, "no learned (combo, fold) entries were checked"


# --------------------------------------------------------------------------
# DD-LC-01 / epoch coverage
# --------------------------------------------------------------------------


def test_epoch_coverage_matches_epochs_trained(
    fixture_outputs: pathlib.Path,
) -> None:
    """Each (combo, fold) curve runs from epoch=1 up to ``epochs_trained``.

    Not a formal DD-* requirement but covers the contract DD-LC-01 implies:
    "per epoch" means one row per completed epoch, and the highest epoch
    recorded equals the manifest's ``epochs_trained`` for that (combo, fold).
    """
    manifest = json.loads(
        (fixture_outputs / TRAINING_MANIFEST_BASENAME).read_text()
    )
    df = pq.read_table(fixture_outputs / TRAINING_LOSS_CURVES_BASENAME).to_pandas()
    fold_index_to_k = _loso_fold_keys(fixture_outputs / PHASE3_SPLITS_BASENAME)

    for combo_id, summary in manifest["training_summaries"].items():
        sub = df[df["combination_id"] == combo_id]
        if sub.empty:
            continue

        if "per_fold" in summary:
            for fold_entry in summary["per_fold"]:
                fold_index = int(fold_entry["fold_index"])
                k = fold_index_to_k[fold_index]
                epochs_trained = int(fold_entry["epochs_trained"])
                epochs = sorted(
                    int(e) for e in sub[sub["fold"] == k]["epoch"].tolist()
                )
                assert epochs == list(range(1, epochs_trained + 1)), (
                    f"{combo_id} fold(k)={k}: epoch coverage {epochs} != "
                    f"range(1, {epochs_trained + 1})"
                )
        else:
            epochs_trained = int(summary["epochs_trained"])
            epochs = sorted(
                int(e) for e in sub[sub["fold"] == 0]["epoch"].tolist()
            )
            assert epochs == list(range(1, epochs_trained + 1)), (
                f"{combo_id}: epoch coverage {epochs} != range(1, {epochs_trained + 1})"
            )


# --------------------------------------------------------------------------
# DD-TEST-06 / DD-LC-08: capture is read-only with respect to optimization
# --------------------------------------------------------------------------


def test_capture_off_matches_capture_on_predictions() -> None:
    """DD-LC-08 / DD-TEST-06: enabling per-epoch capture leaves the trained
    model bit-identical.

    Calls ``train_learned_rung`` twice on a tiny synthetic problem — once
    with ``capture_loss_curve=True`` (the pipeline default) and once with
    ``capture_loss_curve=False`` — and asserts the ``best_state_dict`` and
    the per-epoch best val MAE match exactly. This isolates the capture
    code path from the rest of the build; running the entire pipeline twice
    is unnecessary given the capture is a pure observation of values the
    optimizer already produces.
    """
    import torch
    from torch import nn

    from nflpredictor.train.train_loop import TensorBatch

    device = ResolvedDevice(
        name="cpu",
        torch_device=torch.device("cpu"),
        cuda_device_name=None,
        cuda_version=None,
    )

    def _make_problem() -> tuple[TensorBatch, TensorBatch]:
        seed_all(42, device)
        gen = torch.Generator(device="cpu").manual_seed(42)
        n_train, n_val, n_features = 32, 16, 8
        X_train = torch.randn(n_train, n_features, generator=gen)
        y_train = torch.randn(n_train, 2, generator=gen)
        X_val = torch.randn(n_val, n_features, generator=gen)
        y_val = torch.randn(n_val, 2, generator=gen)

        def _wrap(X: torch.Tensor, y: torch.Tensor, prefix: str) -> TensorBatch:
            return TensorBatch(
                numeric=X,
                low_card={},
                high_card={},
                labels=y,
                game_ids=tuple(f"{prefix}{i:02d}" for i in range(X.shape[0])),
            )

        return _wrap(X_train, y_train, "T"), _wrap(X_val, y_val, "V")

    class _ToyModel(nn.Module):
        def __init__(self, n_features: int) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 2)

        def forward(
            self,
            numeric: torch.Tensor,
            low_card: dict[str, torch.Tensor],
            high_card: dict[str, torch.Tensor],
        ) -> torch.Tensor:
            return self.linear(numeric)

    def _run(capture: bool):
        train_batch, val_batch = _make_problem()
        seed_all(1729, device)
        model = _ToyModel(train_batch.numeric.shape[1])
        return train_learned_rung(
            model,
            train_batch,
            val_batch,
            lr=1e-3,
            batch_size=8,
            max_epochs=8,
            early_stop_patience=100,
            seed=1729,
            device=device,
            capture_loss_curve=capture,
        )

    result_on = _run(capture=True)
    result_off = _run(capture=False)

    # Optimization outcomes match bit-for-bit.
    assert result_on.best_epoch == result_off.best_epoch
    assert result_on.epochs_trained == result_off.epochs_trained
    assert result_on.stopped_early == result_off.stopped_early
    assert result_on.best_val_mae == result_off.best_val_mae
    assert set(result_on.best_state_dict.keys()) == set(result_off.best_state_dict.keys())
    for key in result_on.best_state_dict:
        a = result_on.best_state_dict[key]
        b = result_off.best_state_dict[key]
        assert torch.equal(a, b), f"best_state_dict[{key!r}] diverged between capture on/off"

    # And the capture surface is honored: rows only exist when requested.
    assert len(result_on.loss_curve) == result_on.epochs_trained
    assert result_off.loss_curve == ()
