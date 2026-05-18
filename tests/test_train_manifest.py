"""TR-MAN-* manifest construction and TR-TEST-10 (no test_mae anywhere)."""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

from nflpredictor.train.config import (
    LinearHyperparams,
    MlpHyperparams,
    TrainingConfig,
)
from nflpredictor.train.manifest import (
    build_training_manifest,
    build_training_summary_s1,
    build_training_summary_s3,
    write_training_manifest,
)
from nflpredictor.train.train_loop import ResolvedDevice, TrainingResult


def _resolved_cpu() -> ResolvedDevice:
    return ResolvedDevice(
        name="cpu",
        torch_device=torch.device("cpu"),
        cuda_device_name=None,
        cuda_version=None,
    )


def _resolved_cuda() -> ResolvedDevice:
    return ResolvedDevice(
        name="cuda",
        torch_device=torch.device("cuda"),
        cuda_device_name="NVIDIA RTX 4090",
        cuda_version="12.4",
    )


def _config() -> TrainingConfig:
    return TrainingConfig(
        training_version="v1",
        seed=1729,
        device="auto",
        rungs=("mean",),
        shapes=("flat",),
        strategies=("S1",),
        linear=LinearHyperparams(lr=0.001, batch_size=32, max_epochs=10, early_stop_patience=3),
        mlp=MlpHyperparams(lr=0.001, batch_size=32, max_epochs=10, early_stop_patience=3,
                           hidden_dim=8, activation="gelu", dropout=0.1),
        embedding_dims={"Archetype": 16},
    )


def _learned_result() -> TrainingResult:
    return TrainingResult(
        best_epoch=42,
        best_val_mae=8.3,
        epochs_trained=62,
        stopped_early=True,
        best_state_dict={},
    )


# ---------------------------- per-combination summaries ---------------------


def test_summary_s1_for_trivial_rung() -> None:
    s = build_training_summary_s1(None, val_mae=7.99)
    assert s == {
        "val_mae": 7.99,
        "epochs_trained": None,
        "best_epoch": None,
        "stopped_early": None,
    }


def test_summary_s1_for_learned_rung() -> None:
    s = build_training_summary_s1(_learned_result(), val_mae=8.30)
    assert s["val_mae"] == 8.30
    assert s["epochs_trained"] == 62
    assert s["best_epoch"] == 42
    assert s["stopped_early"] is True


def test_summary_s3_shape_and_mean() -> None:
    results: list = [_learned_result(), _learned_result(), None]
    maes = [9.0, 8.0, 7.5]
    s = build_training_summary_s3(results, maes)
    assert s["fold_count"] == 3
    assert s["mean_val_mae"] == pytest.approx((9.0 + 8.0 + 7.5) / 3)
    assert [f["fold_index"] for f in s["per_fold"]] == [0, 1, 2]
    assert s["per_fold"][2]["epochs_trained"] is None  # trivial
    assert s["per_fold"][0]["epochs_trained"] == 62    # learned


def test_summary_s3_empty_when_no_folds() -> None:
    s = build_training_summary_s3([], [])
    assert s["fold_count"] == 0
    assert s["per_fold"] == []


# ------------------------------- full manifest ------------------------------


def test_build_training_manifest_has_all_required_keys() -> None:
    """TR-MAN-01: every required top-level key is present."""
    m = build_training_manifest(
        config=_config(),
        resolved_device=_resolved_cpu(),
        torch_version="2.12.0+cpu",
        phase2_source_sha256={"Data/processed/features_flat_2024.parquet": "aaa"},
        phase3_source_sha256={"Data/processed/splits_2024.json": "bbb"},
        output_sha256={"predictions/rung0_mean__none__s1.parquet": "ccc"},
        training_config_sha256="ddd",
        phase2_manifest_git_commit="commit-phase2",
        phase3_manifest_git_commit="commit-phase3",
        training_summaries={"rung0_mean__none__s1": build_training_summary_s1(None, 7.9)},
        repo_dir=pathlib.Path(__file__).parent,  # any real dir; git_commit may resolve or be None
    )
    required_keys = {
        "build_timestamp_utc", "training_version", "seed", "torch_version",
        "device_requested", "device_resolved", "cuda_device_name", "cuda_version",
        "training_config_sha256", "phase2_source_sha256", "phase3_source_sha256",
        "output_sha256", "git_commit", "phase2_manifest_git_commit",
        "phase3_manifest_git_commit", "training_summaries",
    }
    assert set(m.keys()) >= required_keys


def test_cpu_resolved_records_null_cuda_fields() -> None:
    m = build_training_manifest(
        config=_config(),
        resolved_device=_resolved_cpu(),
        torch_version="2.12.0+cpu",
        phase2_source_sha256={}, phase3_source_sha256={}, output_sha256={},
        training_config_sha256="d",
        phase2_manifest_git_commit=None, phase3_manifest_git_commit=None,
        training_summaries={}, repo_dir=pathlib.Path("/nonexistent"),
    )
    assert m["device_resolved"] == "cpu"
    assert m["cuda_device_name"] is None
    assert m["cuda_version"] is None


def test_cuda_resolved_records_cuda_fields() -> None:
    m = build_training_manifest(
        config=_config(),
        resolved_device=_resolved_cuda(),
        torch_version="2.12.0+cu124",
        phase2_source_sha256={}, phase3_source_sha256={}, output_sha256={},
        training_config_sha256="d",
        phase2_manifest_git_commit=None, phase3_manifest_git_commit=None,
        training_summaries={}, repo_dir=pathlib.Path("/nonexistent"),
    )
    assert m["device_resolved"] == "cuda"
    assert m["cuda_device_name"] == "NVIDIA RTX 4090"
    assert m["cuda_version"] == "12.4"


# ------------------------ TR-TEST-10: no test_mae anywhere ------------------


def test_no_test_mae_key_anywhere_in_summaries() -> None:
    """TR-TEST-10: enforce TR-MAN-03 across both S1 and S3 summary shapes."""
    s1 = build_training_summary_s1(_learned_result(), val_mae=8.3)
    s3 = build_training_summary_s3([_learned_result(), None], [9.0, 7.0])
    summaries = {"x_s1": s1, "x_s3": s3}
    m = build_training_manifest(
        config=_config(),
        resolved_device=_resolved_cpu(),
        torch_version="2.12.0+cpu",
        phase2_source_sha256={}, phase3_source_sha256={}, output_sha256={},
        training_config_sha256="d",
        phase2_manifest_git_commit=None, phase3_manifest_git_commit=None,
        training_summaries=summaries,
        repo_dir=pathlib.Path(__file__).parent,  # any real dir; git_commit may resolve or be None
    )
    serialized = json.dumps(m)
    assert "test_mae" not in serialized, "TR-MAN-03 violated: manifest contains test_mae"


# ---------------------------- write_training_manifest -----------------------


def test_write_training_manifest_sorted_keys_and_trailing_newline(tmp_path: pathlib.Path) -> None:
    m = {"zebra": 1, "alpha": 2, "nested": {"y": 1, "x": 2}}
    out = tmp_path / "manifest.json"
    write_training_manifest(m, out)
    text = out.read_text()
    assert text.endswith("\n")
    loaded = json.loads(text)
    # Keys sorted at every level.
    assert list(loaded.keys()) == sorted(loaded.keys())
    assert list(loaded["nested"].keys()) == sorted(loaded["nested"].keys())


def test_write_training_manifest_byte_identical_across_runs(tmp_path: pathlib.Path) -> None:
    m = {"a": 1, "b": [3, 1, 2], "c": {"y": 1, "x": 2}}
    p1 = tmp_path / "m1.json"
    p2 = tmp_path / "m2.json"
    write_training_manifest(m, p1)
    write_training_manifest(m, p2)
    assert p1.read_bytes() == p2.read_bytes()
