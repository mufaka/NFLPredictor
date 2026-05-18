"""TR-TEST-08: pinned identity assertions against real Phase 2/3 outputs.

These tests skip when the training build hasn't been run against real data
(``Data/processed/predictions/`` is empty). The full default-config run on
CPU takes ~30-60 minutes, so it's expected to land from the CUDA training
machine, not from CI on a CPU dev machine.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from nflpredictor.train.outputs import PREDICTIONS_DIRNAME, combination_filename
from nflpredictor.train.pipeline import TRAINING_MANIFEST_BASENAME
from nflpredictor.train.sources import load_splits_artifact


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL_PROCESSED = REPO_ROOT / "Data" / "processed"
REAL_PREDICTIONS_DIR = REAL_PROCESSED / PREDICTIONS_DIRNAME
REAL_MANIFEST = REAL_PROCESSED / TRAINING_MANIFEST_BASENAME


EXPECTED_FILES: tuple[tuple[str, str, str], ...] = tuple(
    (rung, shape, strategy)
    for rung, shape in (
        ("mean", "none"),
        ("team_mean", "none"),
        ("linear", "flat"),
        ("linear", "pos"),
        ("mlp", "flat"),
        ("mlp", "pos"),
    )
    for strategy in ("S1", "S3")
)


def _require_real_run() -> None:
    if not REAL_PREDICTIONS_DIR.exists() or not REAL_MANIFEST.exists():
        pytest.skip(
            "Real-data training run hasn't been performed yet. "
            "Run `python -m nflpredictor.train` (best on the CUDA training machine — "
            "the full default config takes ~30-60 minutes on CPU)."
        )


def test_all_twelve_expected_parquets_exist() -> None:
    """TR-TEST-08(a): all 12 combination parquets are present."""
    _require_real_run()
    for rung, shape, strategy in EXPECTED_FILES:
        name = combination_filename(rung, shape, strategy)
        assert (REAL_PREDICTIONS_DIR / name).exists(), f"missing prediction parquet: {name}"


def test_s1_parquets_cover_exact_val_and_test_gameid_sets() -> None:
    """TR-TEST-08(b): each S1 parquet covers exactly S1.val + S1.test."""
    _require_real_run()
    import pyarrow.parquet as pq

    splits = load_splits_artifact(REAL_PROCESSED)
    s1_val = set(splits["S1"]["val"])
    s1_test = set(splits["S1"]["test"])

    for rung, shape, strategy in EXPECTED_FILES:
        if strategy != "S1":
            continue
        path = REAL_PREDICTIONS_DIR / combination_filename(rung, shape, strategy)
        df = pq.read_table(path).to_pandas()
        val_gids = set(df[df["slice"] == "val"]["GameId"].astype(str))
        test_gids = set(df[df["slice"] == "test"]["GameId"].astype(str))
        assert val_gids == s1_val, f"{path.name}: val GameId set != S1.val"
        assert test_gids == s1_test, f"{path.name}: test GameId set != S1.test"


def test_s3_parquets_cover_exact_per_fold_val_gameid_sets() -> None:
    """TR-TEST-08(c): each S3 parquet covers exactly the per-fold val sets across all 9 folds."""
    _require_real_run()
    import pyarrow.parquet as pq

    splits = load_splits_artifact(REAL_PROCESSED)
    folds = splits["S3"]["folds"]
    assert len(folds) == 9, "expected 9 S3 folds for the v1 boundaries"

    for rung, shape, strategy in EXPECTED_FILES:
        if strategy != "S3":
            continue
        path = REAL_PREDICTIONS_DIR / combination_filename(rung, shape, strategy)
        df = pq.read_table(path).to_pandas()
        for fold in folds:
            fold_idx = int(fold["fold_index"])
            expected = set(fold["val"])
            actual = set(df[df["fold_index"] == fold_idx]["GameId"].astype(str))
            assert actual == expected, (
                f"{path.name}: fold {fold_idx} GameId set != splits S3 fold val"
            )


def test_manifest_carries_val_mae_for_every_combination() -> None:
    """TR-TEST-08(d) + TR-TEST-10: each combo has a val_mae; no test_mae anywhere."""
    _require_real_run()
    m = json.loads(REAL_MANIFEST.read_text())
    summaries = m["training_summaries"]
    expected_keys = {
        f"rung0_mean__none__{s.lower()}" for s in ("S1", "S3")
    } | {
        f"rung1_team_mean__none__{s.lower()}" for s in ("S1", "S3")
    } | {
        f"rung2_linear__{sh}__{s.lower()}" for sh in ("flat", "pos") for s in ("S1", "S3")
    } | {
        f"rung3_mlp__{sh}__{s.lower()}" for sh in ("flat", "pos") for s in ("S1", "S3")
    }
    assert set(summaries.keys()) >= expected_keys
    for key, summary in summaries.items():
        if "fold_count" in summary:
            for fold in summary["per_fold"]:
                assert "val_mae" in fold, f"{key}/per_fold[{fold['fold_index']}] missing val_mae"
        else:
            assert "val_mae" in summary, f"{key} missing val_mae"
    assert "test_mae" not in json.dumps(m), "TR-MAN-03 violated"
