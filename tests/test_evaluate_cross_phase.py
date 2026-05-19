"""EV-TEST-08: cross-phase agreement on the headline MAE formula.

Against the synthetic Phase 5 fixture (which carries real Phase 4
training_summaries with per-combo val_mae values), compute Phase 5's
headline MAE for every S1 val cell and every S3 per-fold cell and assert
numerical agreement with Phase 4's recorded value. Prevents EV-MET-01 and
TR-MAN-04 from silently drifting apart.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from nflpredictor.evaluate.metrics import (
    build_headline_for_combination,
    join_predictions_with_labels,
)
from nflpredictor.evaluate.sources import (
    PHASE4_MANIFEST_BASENAME,
    PHASE4_PREDICTIONS_DIRNAME,
    enumerate_combinations,
    load_features_flat,
    load_prediction_parquet,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "evaluate"

# Phase 4's training stored learned-rung val_mae in float32-derived float64.
# Phase 5 recomputes from the same float64 prediction parquet bytes, so the
# computations differ at most by associativity-of-summation: a few ulps. A
# 1e-6 relative tolerance is well within the float32 precision of the
# input predictions and catches any formula-level drift.
TOLERANCE = dict(rel=1e-6, abs=1e-9)


def _load_phase4_manifest() -> dict:
    return json.loads(
        (FIXTURE / "raw_phase4" / PHASE4_MANIFEST_BASENAME).read_text()
    )


def _staged_processed_dir() -> pathlib.Path:
    """Path layout matching what the fixture _regenerate.py produces.

    The cross-phase test only needs the Phase 2 features (for labels) and the
    Phase 4 prediction parquets — no full eval build required.
    """
    return FIXTURE / "raw_phase2"


def _load_features_flat_from_fixture():
    return load_features_flat(_staged_processed_dir())


def test_cross_phase_s1_val_mae_matches_training_summary() -> None:
    phase4_manifest = _load_phase4_manifest()
    training_summaries = phase4_manifest["training_summaries"]
    features_flat = _load_features_flat_from_fixture()

    # enumerate_combinations needs the parquet paths; point it at the
    # raw_phase4/ tree under the fixture.
    combinations = enumerate_combinations(
        phase4_manifest, FIXTURE / "raw_phase4"
    )

    for key in combinations:
        if key.strategy != "S1":
            continue
        preds = load_prediction_parquet(key)
        joined = join_predictions_with_labels(preds, features_flat)
        headline = build_headline_for_combination(joined, "S1")
        ev_val_mae = headline["val"]["mae"]
        tr_val_mae = training_summaries[key.combination_id]["val_mae"]
        assert ev_val_mae == pytest.approx(tr_val_mae, **TOLERANCE), (
            f"{key.combination_id}: Phase 5 val MAE {ev_val_mae!r} disagrees with "
            f"Phase 4 training_summaries.val_mae {tr_val_mae!r}"
        )


def test_cross_phase_s3_per_fold_val_mae_matches_training_summary() -> None:
    phase4_manifest = _load_phase4_manifest()
    training_summaries = phase4_manifest["training_summaries"]
    features_flat = _load_features_flat_from_fixture()

    combinations = enumerate_combinations(
        phase4_manifest, FIXTURE / "raw_phase4"
    )

    for key in combinations:
        if key.strategy != "S3":
            continue
        preds = load_prediction_parquet(key)
        joined = join_predictions_with_labels(preds, features_flat)
        headline = build_headline_for_combination(joined, "S3")
        tr_per_fold = training_summaries[key.combination_id]["per_fold"]
        for fold in tr_per_fold:
            i = int(fold["fold_index"])
            tr_val_mae = fold["val_mae"]
            ev_val_mae = headline[f"fold_{i}"]["mae"]
            assert ev_val_mae == pytest.approx(tr_val_mae, **TOLERANCE), (
                f"{key.combination_id} fold {i}: Phase 5 val MAE {ev_val_mae!r} "
                f"disagrees with Phase 4 per_fold[{i}].val_mae {tr_val_mae!r}"
            )
