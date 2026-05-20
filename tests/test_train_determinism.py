"""TR-TEST-07: byte-determinism across two successive runs of the training build."""

from __future__ import annotations

import json
import pathlib
import shutil

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


def _run_once(tmp_path: pathlib.Path) -> pathlib.Path:
    processed = tmp_path / "Data" / "processed"
    _stage_processed(processed)
    run_training_build(FIXTURE_RAW, processed, repo_dir=tmp_path)
    return processed


def test_two_runs_produce_byte_identical_outputs(tmp_path: pathlib.Path) -> None:
    """TR-NF-01 / TR-TEST-07: same inputs + same device + same wheel → identical bytes."""
    if not FIXTURE_EXPECTED.exists():
        pytest.skip("Phase 4 fixture missing; regenerate first.")

    run_a = _run_once(tmp_path / "a")
    run_b = _run_once(tmp_path / "b")

    a_preds = sorted((run_a / PREDICTIONS_DIRNAME).glob("*.parquet"))
    b_preds = sorted((run_b / PREDICTIONS_DIRNAME).glob("*.parquet"))
    assert [p.name for p in a_preds] == [p.name for p in b_preds]
    # Default config trains season_holdout only → 6 prediction parquets.
    assert len(a_preds) == 6

    for pa, pb in zip(a_preds, b_preds):
        assert pa.read_bytes() == pb.read_bytes(), (
            f"prediction {pa.name} differs between two runs"
        )

    # DD-TEST-01 / DD-NF-01: the Phase 6 loss-curve sidecar parquet is also
    # byte-deterministic across runs under the same per-device contract.
    loss_a = (run_a / TRAINING_LOSS_CURVES_BASENAME).read_bytes()
    loss_b = (run_b / TRAINING_LOSS_CURVES_BASENAME).read_bytes()
    assert loss_a == loss_b, "training_loss_curves.parquet differs between two runs"

    manifest_a = json.loads((run_a / TRAINING_MANIFEST_BASENAME).read_text())
    manifest_b = json.loads((run_b / TRAINING_MANIFEST_BASENAME).read_text())
    # Timestamps differ by definition; blank them.
    manifest_a["build_timestamp_utc"] = "X"
    manifest_b["build_timestamp_utc"] = "X"
    assert manifest_a == manifest_b
