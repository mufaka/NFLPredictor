"""Integration test against the synthetic Phase 3 fixture (SP-TEST-04)."""

from __future__ import annotations

import json
import pathlib
import shutil

from nflpredictor.splits.pipeline import (
    PHASE2_FEATURES_FLAT_BASENAME,
    PHASE2_MANIFEST_BASENAME,
    SPLITS_ARTIFACT_BASENAME,
    SPLITS_MANIFEST_BASENAME,
    run_split_build,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "splits"
FIXTURE_RAW = FIXTURE_DIR / "raw"
FIXTURE_RAW_PHASE2 = FIXTURE_DIR / "raw_phase2"
FIXTURE_EXPECTED = FIXTURE_DIR / "expected"


def _seed_processed(target: pathlib.Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in (PHASE2_FEATURES_FLAT_BASENAME, PHASE2_MANIFEST_BASENAME):
        shutil.copy2(FIXTURE_RAW_PHASE2 / name, target / name)


def _run_against_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    processed = tmp_path / "Data" / "processed"
    _seed_processed(processed)
    run_split_build(FIXTURE_RAW, processed, repo_dir=tmp_path)
    return processed


def test_fixture_splits_artifact_byte_equal(tmp_path):
    processed = _run_against_fixture(tmp_path)
    assert (
        (processed / SPLITS_ARTIFACT_BASENAME).read_bytes()
        == (FIXTURE_EXPECTED / SPLITS_ARTIFACT_BASENAME).read_bytes()
    )


def test_fixture_manifest_structure_matches(tmp_path):
    """Manifest matches expected after blanking timestamp and git_commit."""
    processed = _run_against_fixture(tmp_path)
    actual = json.loads((processed / SPLITS_MANIFEST_BASENAME).read_text())
    expected = json.loads((FIXTURE_EXPECTED / SPLITS_MANIFEST_BASENAME).read_text())
    # Blank the build-run-specific fields.
    actual["build_timestamp_utc"] = "FIXTURE_TIMESTAMP"
    actual["git_commit"] = "FIXTURE_GIT_COMMIT"
    assert actual == expected
