"""EV-MAN-01/02/03/05 manifest construction tests."""

from __future__ import annotations

import json
import pathlib

import pytest

from nflpredictor.evaluate.manifest import (
    build_evaluation_manifest,
    build_evaluation_summary,
    write_evaluation_manifest,
)


REQUIRED_MANIFEST_KEYS: set[str] = {
    "build_timestamp_utc",
    "evaluation_version",
    "headline_metric",
    "evaluation_config_sha256",
    "phase2_source_sha256",
    "phase3_source_sha256",
    "phase4_source_sha256",
    "output_sha256",
    "git_commit",
    "phase4_manifest_git_commit",
    "matplotlib_version",
    "numpy_version",
    "pyarrow_version",
    "combination_ids",
    "evaluation_summary",
}


def _headline() -> dict:
    return {
        "combinations": {
            "rung0_mean__none__season_holdout": {
                "val": {"n_games": 3, "mae": 5.0},
                "test": {"n_games": 2, "mae": 6.0},
            },
            "rung2_linear__flat__loso_cv": {
                "fold_0": {"n_games": 2, "mae": 4.0},
                "fold_1": {"n_games": 2, "mae": 3.5},
                "pooled": {"n_games": 4, "mae": 3.75},
            },
        }
    }


# ---------------------------------------------------------------------------
# build_evaluation_summary (EV-MAN-02)
# ---------------------------------------------------------------------------


def test_evaluation_summary_s1_shape() -> None:
    summary = build_evaluation_summary(_headline(), "mae")
    s1 = summary["rung0_mean__none__season_holdout"]
    assert set(s1.keys()) == {"val", "test"}
    assert s1["val"] == 5.0
    assert s1["test"] == 6.0


def test_evaluation_summary_s3_shape() -> None:
    summary = build_evaluation_summary(_headline(), "mae")
    s3 = summary["rung2_linear__flat__loso_cv"]
    assert set(s3.keys()) == {"pooled", "mean_per_fold", "per_fold"}
    assert s3["pooled"] == 3.75
    assert s3["per_fold"] == [4.0, 3.5]
    assert s3["mean_per_fold"] == pytest.approx((4.0 + 3.5) / 2.0)


def test_evaluation_summary_handles_null_cells() -> None:
    """Empty cells (n_games=0) show up as None values in the summary."""
    headline = {
        "combinations": {
            "rung0_mean__none__season_holdout": {
                "val": {"n_games": 0, "mae": None},
                "test": {"n_games": 0, "mae": None},
            },
        }
    }
    summary = build_evaluation_summary(headline, "mae")
    s1 = summary["rung0_mean__none__season_holdout"]
    assert s1["val"] is None
    assert s1["test"] is None


def test_evaluation_summary_s3_all_null_mean_is_null() -> None:
    headline = {
        "combinations": {
            "x__none__loso_cv": {
                "fold_0": {"n_games": 0, "mae": None},
                "fold_1": {"n_games": 0, "mae": None},
                "pooled": {"n_games": 0, "mae": None},
            },
        }
    }
    summary = build_evaluation_summary(headline, "mae")
    s3 = summary["x__none__loso_cv"]
    assert s3["pooled"] is None
    assert s3["mean_per_fold"] is None
    assert s3["per_fold"] == [None, None]


def test_evaluation_summary_unrecognized_combo_id_rejected() -> None:
    bad = {"combinations": {"weird_combo_no_suffix": {"val": {"mae": 1.0}}}}
    with pytest.raises(ValueError, match="combination id"):
        build_evaluation_summary(bad, "mae")


# ---------------------------------------------------------------------------
# build_evaluation_manifest (EV-MAN-01)
# ---------------------------------------------------------------------------


def test_manifest_has_all_required_keys(tmp_path: pathlib.Path) -> None:
    summary = build_evaluation_summary(_headline(), "mae")
    manifest = build_evaluation_manifest(
        evaluation_version="v1",
        headline_metric="mae",
        evaluation_config_sha256="abc",
        phase2_source_sha256={"x": "1"},
        phase3_source_sha256={"y": "2"},
        phase4_source_sha256={"z": "3"},
        output_sha256={"out": "4"},
        phase4_manifest_git_commit="deadbeef",
        combination_ids=["rung2_linear__flat__loso_cv", "rung0_mean__none__season_holdout"],
        evaluation_summary=summary,
        matplotlib_version="3.10.9",
        numpy_version="2.0.0",
        pyarrow_version="24.0.0",
        repo_dir=tmp_path,
    )
    assert set(manifest.keys()) == REQUIRED_MANIFEST_KEYS
    # combination_ids re-sorted lexicographically.
    assert manifest["combination_ids"] == [
        "rung0_mean__none__season_holdout",
        "rung2_linear__flat__loso_cv",
    ]
    # Hash maps are sorted.
    assert list(manifest["phase2_source_sha256"].keys()) == ["x"]


def test_manifest_round_trip_byte_identical(tmp_path: pathlib.Path) -> None:
    """Two writes with the same dict produce identical bytes (EV-MAN-05)."""
    summary = build_evaluation_summary(_headline(), "mae")
    manifest = build_evaluation_manifest(
        evaluation_version="v1",
        headline_metric="mae",
        evaluation_config_sha256="abc",
        phase2_source_sha256={"x": "1"},
        phase3_source_sha256={"y": "2"},
        phase4_source_sha256={"z": "3"},
        output_sha256={"out": "4"},
        phase4_manifest_git_commit=None,
        combination_ids=["c1"],
        evaluation_summary=summary,
        matplotlib_version="3.10.9",
        numpy_version="2.0.0",
        pyarrow_version="24.0.0",
        repo_dir=tmp_path,
    )
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_evaluation_manifest(manifest, a)
    write_evaluation_manifest(manifest, b)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes().endswith(b"\n")
    loaded = json.loads(a.read_text(encoding="utf-8"))
    assert set(loaded.keys()) == REQUIRED_MANIFEST_KEYS


def test_manifest_timestamp_is_iso_8601_utc(tmp_path: pathlib.Path) -> None:
    summary = build_evaluation_summary(_headline(), "mae")
    manifest = build_evaluation_manifest(
        evaluation_version="v1",
        headline_metric="mae",
        evaluation_config_sha256="abc",
        phase2_source_sha256={},
        phase3_source_sha256={},
        phase4_source_sha256={},
        output_sha256={},
        phase4_manifest_git_commit=None,
        combination_ids=[],
        evaluation_summary=summary,
        matplotlib_version="3.10.9",
        numpy_version="2.0.0",
        pyarrow_version="24.0.0",
        repo_dir=tmp_path,
    )
    ts = manifest["build_timestamp_utc"]
    # Format: YYYY-MM-DDTHH:MM:SSZ
    assert ts.endswith("Z")
    assert len(ts) == 20
    assert ts[4] == "-" and ts[7] == "-" and ts[10] == "T" and ts[13] == ":"
