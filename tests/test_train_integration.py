"""TR-TEST-06: integration test against the synthetic Phase 4 fixture."""

from __future__ import annotations

import json
import pathlib
import shutil

import pytest

import hashlib

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


def _run_against_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    processed = tmp_path / "Data" / "processed"
    _stage_processed(processed)
    run_training_build(FIXTURE_RAW, processed, repo_dir=tmp_path)
    return processed


@pytest.fixture(scope="module")
def fixture_outputs(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """One run per pytest session; downstream tests reuse the outputs."""
    if not FIXTURE_EXPECTED.exists():
        pytest.skip(
            "tests/fixtures/train/expected/ is missing — "
            "regenerate via `python -m tests.fixtures.train._regenerate`"
        )
    tmp = tmp_path_factory.mktemp("train_integration")
    return _run_against_fixture(tmp)


def test_every_expected_prediction_parquet_is_byte_identical(
    fixture_outputs: pathlib.Path,
) -> None:
    """TR-TEST-06: each of the 12 prediction parquets matches the checked-in expected file."""
    expected_dir = FIXTURE_EXPECTED / PREDICTIONS_DIRNAME
    actual_dir = fixture_outputs / PREDICTIONS_DIRNAME
    expected_files = sorted(p.name for p in expected_dir.glob("*.parquet"))
    actual_files = sorted(p.name for p in actual_dir.glob("*.parquet"))
    assert expected_files == actual_files
    for name in expected_files:
        assert (actual_dir / name).read_bytes() == (expected_dir / name).read_bytes(), (
            f"prediction parquet {name} diverged from the checked-in expected fixture"
        )


def test_training_manifest_matches_expected(fixture_outputs: pathlib.Path) -> None:
    """TR-TEST-06: manifest matches expected after blanking timestamp + git_commit."""
    actual = json.loads((fixture_outputs / TRAINING_MANIFEST_BASENAME).read_text())
    expected = json.loads((FIXTURE_EXPECTED / TRAINING_MANIFEST_BASENAME).read_text())
    actual["build_timestamp_utc"] = "FIXTURE_TIMESTAMP"
    actual["git_commit"] = "FIXTURE_GIT_COMMIT"
    assert actual == expected


def test_loss_curves_parquet_is_byte_identical(
    fixture_outputs: pathlib.Path,
) -> None:
    """DD-LC-05 / DD-NF-01: the loss-curve sidecar matches the checked-in fixture."""
    actual = fixture_outputs / TRAINING_LOSS_CURVES_BASENAME
    expected = FIXTURE_EXPECTED / TRAINING_LOSS_CURVES_BASENAME
    assert actual.read_bytes() == expected.read_bytes(), (
        "training_loss_curves.parquet diverged from the checked-in expected fixture"
    )


def test_manifest_records_loss_curves_sha_and_version_bump(
    fixture_outputs: pathlib.Path,
) -> None:
    """DD-LC-03 / DD-LC-04 / DD-BWD-05(d,e): manifest tracks the new artifact.

    Verifies:
      (a) ``output_sha256["training_loss_curves.parquet"]`` is present and
          matches the on-disk file's SHA-256.
      (b) ``training_version`` was bumped — the fixture's recorded value is
          something other than the pre-amendment ``"v1"``. The exact bumped
          value is read from the fixture so this test does not pin the
          version string itself (DD-LC-04 gives Phase 4 author discretion).
    """
    manifest = json.loads(
        (fixture_outputs / TRAINING_MANIFEST_BASENAME).read_text()
    )

    output_sha256 = manifest["output_sha256"]
    assert TRAINING_LOSS_CURVES_BASENAME in output_sha256, (
        f"manifest output_sha256 is missing the new key "
        f"{TRAINING_LOSS_CURVES_BASENAME!r}; got {sorted(output_sha256.keys())}"
    )
    on_disk = hashlib.sha256(
        (fixture_outputs / TRAINING_LOSS_CURVES_BASENAME).read_bytes()
    ).hexdigest()
    assert output_sha256[TRAINING_LOSS_CURVES_BASENAME] == on_disk

    # DD-LC-04: training_version differs from the pre-amendment "v1".
    assert manifest["training_version"] != "v1", (
        f"training_version was not bumped after the Phase 6 amendment "
        f"(still {manifest['training_version']!r})"
    )
