"""Tests for the Phase 1 source-hash gate (FE-TEST-08)."""

from __future__ import annotations

import pathlib
import shutil

import pytest

from nflpredictor.features.pipeline import (
    PHASE1_BOX_SCORES_BASENAME,
    PHASE1_MADDEN_BASENAME,
    PHASE1_MANIFEST_BASENAME,
    Phase1OutputMismatchError,
    verify_phase1_outputs,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "Data" / "processed"


def _copy_real_phase1(target: pathlib.Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for basename in (
        PHASE1_MADDEN_BASENAME,
        PHASE1_BOX_SCORES_BASENAME,
        PHASE1_MANIFEST_BASENAME,
    ):
        shutil.copy2(PROCESSED_DIR / basename, target / basename)


def test_verify_succeeds_against_real_phase1_outputs(tmp_path):
    _copy_real_phase1(tmp_path)
    manifest = verify_phase1_outputs(tmp_path)
    assert manifest["normalization_version"] == "v2"
    assert f"Data/processed/{PHASE1_MADDEN_BASENAME}" in manifest["output_sha256"]


def test_verify_raises_on_tampered_madden(tmp_path):
    _copy_real_phase1(tmp_path)
    with (tmp_path / PHASE1_MADDEN_BASENAME).open("ab") as f:
        f.write(b"\n")
    with pytest.raises(Phase1OutputMismatchError, match=PHASE1_MADDEN_BASENAME):
        verify_phase1_outputs(tmp_path)


def test_verify_raises_on_tampered_box_scores(tmp_path):
    _copy_real_phase1(tmp_path)
    with (tmp_path / PHASE1_BOX_SCORES_BASENAME).open("ab") as f:
        f.write(b"\n")
    with pytest.raises(Phase1OutputMismatchError, match=PHASE1_BOX_SCORES_BASENAME):
        verify_phase1_outputs(tmp_path)


def test_verify_raises_when_manifest_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Phase 1 manifest"):
        verify_phase1_outputs(tmp_path)


def test_verify_raises_when_output_missing(tmp_path):
    _copy_real_phase1(tmp_path)
    (tmp_path / PHASE1_MADDEN_BASENAME).unlink()
    with pytest.raises(FileNotFoundError, match=PHASE1_MADDEN_BASENAME):
        verify_phase1_outputs(tmp_path)
