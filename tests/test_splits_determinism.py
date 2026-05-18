"""Two consecutive split builds against identical inputs must agree byte-for-byte (SP-TEST-05)."""

from __future__ import annotations

import hashlib
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


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_processed(target: pathlib.Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in (PHASE2_FEATURES_FLAT_BASENAME, PHASE2_MANIFEST_BASENAME):
        shutil.copy2(FIXTURE_RAW_PHASE2 / name, target / name)


def test_fixture_byte_identical_rerun(tmp_path):
    processed_a = tmp_path / "a"
    processed_b = tmp_path / "b"
    _seed_processed(processed_a)
    _seed_processed(processed_b)

    run_split_build(FIXTURE_RAW, processed_a, repo_dir=tmp_path)
    run_split_build(FIXTURE_RAW, processed_b, repo_dir=tmp_path)

    assert (
        _sha256(processed_a / SPLITS_ARTIFACT_BASENAME)
        == _sha256(processed_b / SPLITS_ARTIFACT_BASENAME)
    )

    # Manifest minus the timestamp must match.
    m1 = json.loads((processed_a / SPLITS_MANIFEST_BASENAME).read_text())
    m2 = json.loads((processed_b / SPLITS_MANIFEST_BASENAME).read_text())
    m1.pop("build_timestamp_utc")
    m2.pop("build_timestamp_utc")
    assert m1 == m2
