"""TR-TEST-08: pinned identity assertions against real Phase 2/3 outputs.

These tests skip when the training build hasn't been run against real data
(``Data/processed/predictions/`` is empty). The default-config run trains
the ``season_holdout`` strategy only (6 prediction parquets); enabling
``loso_cv`` adds the per-fold parquets at ≈6× cost.
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


# The default config's combinations: season_holdout strategy only.
EXPECTED_FILES: tuple[tuple[str, str, str], ...] = tuple(
    (rung, shape, "season_holdout")
    for rung, shape in (
        ("mean", "none"),
        ("team_mean", "none"),
        ("linear", "flat"),
        ("linear", "pos"),
        ("mlp", "flat"),
        ("mlp", "pos"),
    )
)


def _require_real_run() -> None:
    if not REAL_PREDICTIONS_DIR.exists() or not REAL_MANIFEST.exists():
        pytest.skip(
            "Real-data training run hasn't been performed yet. "
            "Run `python -m nflpredictor.train`."
        )


def test_all_expected_parquets_exist() -> None:
    """TR-TEST-08(a): all default-config combination parquets are present."""
    _require_real_run()
    for rung, shape, strategy in EXPECTED_FILES:
        name = combination_filename(rung, shape, strategy)
        assert (REAL_PREDICTIONS_DIR / name).exists(), f"missing prediction parquet: {name}"


def test_holdout_parquets_cover_exact_val_and_test_gameid_sets() -> None:
    """TR-TEST-08(b): each season_holdout parquet covers exactly val + test."""
    _require_real_run()
    import pyarrow.parquet as pq

    splits = load_splits_artifact(REAL_PROCESSED)
    sh_val = set(splits["season_holdout"]["val"])
    sh_test = set(splits["season_holdout"]["test"])

    for rung, shape, strategy in EXPECTED_FILES:
        path = REAL_PREDICTIONS_DIR / combination_filename(rung, shape, strategy)
        df = pq.read_table(path).to_pandas()
        val_gids = set(df[df["slice"] == "val"]["GameId"].astype(str))
        test_gids = set(df[df["slice"] == "test"]["GameId"].astype(str))
        assert val_gids == sh_val, f"{path.name}: val GameId set != season_holdout.val"
        assert test_gids == sh_test, f"{path.name}: test GameId set != season_holdout.test"


def test_manifest_carries_val_mae_for_every_combination() -> None:
    """TR-TEST-08(d) + TR-TEST-10: each combo has a val_mae; no test_mae anywhere."""
    _require_real_run()
    m = json.loads(REAL_MANIFEST.read_text())
    summaries = m["training_summaries"]
    expected_keys = {
        "rung0_mean__none__season_holdout",
        "rung1_team_mean__none__season_holdout",
    } | {
        f"rung2_linear__{sh}__season_holdout" for sh in ("flat", "pos")
    } | {
        f"rung3_mlp__{sh}__season_holdout" for sh in ("flat", "pos")
    }
    assert set(summaries.keys()) >= expected_keys
    for key, summary in summaries.items():
        if "fold_count" in summary:
            for fold in summary["per_fold"]:
                assert "val_mae" in fold, f"{key}/per_fold missing val_mae"
        else:
            assert "val_mae" in summary, f"{key} missing val_mae"
    assert "test_mae" not in json.dumps(m), "TR-MAN-03 violated"
