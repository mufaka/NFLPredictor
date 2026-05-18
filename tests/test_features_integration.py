"""Integration test against the synthetic Phase 2 fixture (FE-TEST-05, FE-TEST-06)."""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil

import pytest

from nflpredictor.features.pipeline import (
    FEATURE_MANIFEST_BASENAME,
    FEATURE_VOCAB_BASENAME,
    FLAT_PARQUET_BASENAME,
    POS_PARQUET_BASENAME,
    PHASE1_BOX_SCORES_BASENAME,
    PHASE1_MADDEN_BASENAME,
    PHASE1_MANIFEST_BASENAME,
    run_feature_build,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "features"
FIXTURE_RAW = FIXTURE_DIR / "raw"
FIXTURE_RAW_PHASE1 = FIXTURE_DIR / "raw_phase1"
FIXTURE_EXPECTED = FIXTURE_DIR / "expected"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_processed(target: pathlib.Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        PHASE1_BOX_SCORES_BASENAME,
        PHASE1_MADDEN_BASENAME,
        PHASE1_MANIFEST_BASENAME,
    ):
        shutil.copy2(FIXTURE_RAW_PHASE1 / name, target / name)


def _run_against_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    processed = tmp_path / "Data" / "processed"
    _seed_processed(processed)
    run_feature_build(FIXTURE_RAW, processed, repo_dir=tmp_path)
    return processed


def test_fixture_flat_parquet_byte_equal(tmp_path):
    processed = _run_against_fixture(tmp_path)
    assert (
        (processed / FLAT_PARQUET_BASENAME).read_bytes()
        == (FIXTURE_EXPECTED / FLAT_PARQUET_BASENAME).read_bytes()
    )


def test_fixture_pos_parquet_byte_equal(tmp_path):
    processed = _run_against_fixture(tmp_path)
    assert (
        (processed / POS_PARQUET_BASENAME).read_bytes()
        == (FIXTURE_EXPECTED / POS_PARQUET_BASENAME).read_bytes()
    )


def test_fixture_vocab_byte_equal(tmp_path):
    processed = _run_against_fixture(tmp_path)
    assert (
        (processed / FEATURE_VOCAB_BASENAME).read_bytes()
        == (FIXTURE_EXPECTED / FEATURE_VOCAB_BASENAME).read_bytes()
    )


def test_fixture_manifest_structure_matches(tmp_path):
    """Manifest matches expected after blanking the timestamp (FE-NF-04)."""
    processed = _run_against_fixture(tmp_path)
    actual = json.loads((processed / FEATURE_MANIFEST_BASENAME).read_text())
    expected = json.loads((FIXTURE_EXPECTED / FEATURE_MANIFEST_BASENAME).read_text())
    actual["build_timestamp_utc"] = "FIXTURE_TIMESTAMP"
    # The fixture was generated under a specific git commit; the test's run
    # may have a different one. Blank both git_commit fields before compare.
    actual["git_commit"] = "FIXTURE_GIT_COMMIT"
    expected["git_commit"] = "FIXTURE_GIT_COMMIT"
    assert actual == expected


def test_fixture_byte_identical_rerun(tmp_path):
    """FE-TEST-06: two runs against identical inputs → byte-identical outputs."""
    processed_a = tmp_path / "a"
    processed_b = tmp_path / "b"
    _seed_processed(processed_a)
    _seed_processed(processed_b)

    run_feature_build(FIXTURE_RAW, processed_a, repo_dir=tmp_path)
    run_feature_build(FIXTURE_RAW, processed_b, repo_dir=tmp_path)

    for name in (FLAT_PARQUET_BASENAME, POS_PARQUET_BASENAME, FEATURE_VOCAB_BASENAME):
        assert _sha256(processed_a / name) == _sha256(processed_b / name), name

    # Manifest minus timestamp must match across runs.
    m1 = json.loads((processed_a / FEATURE_MANIFEST_BASENAME).read_text())
    m2 = json.loads((processed_b / FEATURE_MANIFEST_BASENAME).read_text())
    m1.pop("build_timestamp_utc")
    m2.pop("build_timestamp_utc")
    assert m1 == m2
