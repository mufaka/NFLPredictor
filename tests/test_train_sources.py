"""TR-TEST-09: source-hash gates fail fast on tampered upstream artifacts."""

from __future__ import annotations

import json
import pathlib
import shutil

import pytest

from nflpredictor.train.sources import (
    PHASE2_FEATURES_FLAT_BASENAME,
    PHASE2_FEATURES_POS_BASENAME,
    PHASE2_MANIFEST_BASENAME,
    PHASE2_VOCAB_BASENAME,
    PHASE3_MANIFEST_BASENAME,
    PHASE3_SPLITS_BASENAME,
    Phase2OutputMismatchError,
    Phase3OutputMismatchError,
    StrategyUnavailableError,
    high_card_vocab_keys,
    load_splits_artifact,
    load_vocab,
    verify_phase2_outputs,
    verify_phase3_outputs,
    verify_strategy_availability,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL_PROCESSED = REPO_ROOT / "Data" / "processed"


def _stage_processed(tmp_path: pathlib.Path) -> pathlib.Path:
    """Copy the on-disk Phase 2 + Phase 3 outputs into ``tmp_path`` so tests can mutate them."""
    dest = tmp_path / "processed"
    dest.mkdir()
    for name in (
        PHASE2_FEATURES_FLAT_BASENAME,
        PHASE2_FEATURES_POS_BASENAME,
        PHASE2_VOCAB_BASENAME,
        PHASE2_MANIFEST_BASENAME,
        PHASE3_SPLITS_BASENAME,
        PHASE3_MANIFEST_BASENAME,
    ):
        shutil.copy2(REAL_PROCESSED / name, dest / name)
    return dest


def _tamper_bytes(path: pathlib.Path) -> None:
    """Flip a byte in the middle of the file so its SHA changes but it stays parseable enough."""
    data = bytearray(path.read_bytes())
    idx = len(data) // 2
    data[idx] = data[idx] ^ 0x01
    path.write_bytes(bytes(data))


# (a) happy path — real artifacts pass both gates
def test_real_artifacts_pass_gates() -> None:
    phase2_manifest = verify_phase2_outputs(REAL_PROCESSED)
    phase3_manifest = verify_phase3_outputs(REAL_PROCESSED)
    assert "output_sha256" in phase2_manifest
    assert "output_sha256" in phase3_manifest


# (b) tampered features_flat parquet → fail-fast
def test_tampered_features_flat_fails(tmp_path: pathlib.Path) -> None:
    staged = _stage_processed(tmp_path)
    _tamper_bytes(staged / PHASE2_FEATURES_FLAT_BASENAME)
    with pytest.raises(Phase2OutputMismatchError, match=PHASE2_FEATURES_FLAT_BASENAME):
        verify_phase2_outputs(staged)


# (c) tampered features_pos parquet → fail-fast
def test_tampered_features_pos_fails(tmp_path: pathlib.Path) -> None:
    staged = _stage_processed(tmp_path)
    _tamper_bytes(staged / PHASE2_FEATURES_POS_BASENAME)
    with pytest.raises(Phase2OutputMismatchError, match=PHASE2_FEATURES_POS_BASENAME):
        verify_phase2_outputs(staged)


# (d) tampered feature_vocab.json → fail-fast
def test_tampered_feature_vocab_fails(tmp_path: pathlib.Path) -> None:
    staged = _stage_processed(tmp_path)
    _tamper_bytes(staged / PHASE2_VOCAB_BASENAME)
    with pytest.raises(Phase2OutputMismatchError, match=PHASE2_VOCAB_BASENAME):
        verify_phase2_outputs(staged)


# (e) tampered splits_2024.json → fail-fast
def test_tampered_splits_fails(tmp_path: pathlib.Path) -> None:
    staged = _stage_processed(tmp_path)
    _tamper_bytes(staged / PHASE3_SPLITS_BASENAME)
    with pytest.raises(Phase3OutputMismatchError, match=PHASE3_SPLITS_BASENAME):
        verify_phase3_outputs(staged)


# (f) missing Phase 2 manifest → fail-fast
def test_missing_phase2_manifest_fails(tmp_path: pathlib.Path) -> None:
    staged = _stage_processed(tmp_path)
    (staged / PHASE2_MANIFEST_BASENAME).unlink()
    with pytest.raises(FileNotFoundError, match="Phase 2 manifest"):
        verify_phase2_outputs(staged)


def test_missing_phase3_manifest_fails(tmp_path: pathlib.Path) -> None:
    staged = _stage_processed(tmp_path)
    (staged / PHASE3_MANIFEST_BASENAME).unlink()
    with pytest.raises(FileNotFoundError, match="Phase 3 manifest"):
        verify_phase3_outputs(staged)


def test_missing_features_flat_fails(tmp_path: pathlib.Path) -> None:
    staged = _stage_processed(tmp_path)
    (staged / PHASE2_FEATURES_FLAT_BASENAME).unlink()
    with pytest.raises(FileNotFoundError, match=PHASE2_FEATURES_FLAT_BASENAME):
        verify_phase2_outputs(staged)


def test_phase2_manifest_missing_output_sha256_block(tmp_path: pathlib.Path) -> None:
    staged = _stage_processed(tmp_path)
    manifest_path = staged / PHASE2_MANIFEST_BASENAME
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("output_sha256")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(Phase2OutputMismatchError, match="output_sha256"):
        verify_phase2_outputs(staged)


# (g) strategy requested but absent from splits_2024.json → fail-fast
def test_strategy_unavailable_fails() -> None:
    splits = load_splits_artifact(REAL_PROCESSED)
    # Pretend the user asked for a fictional strategy "S5" alongside S1.
    with pytest.raises(StrategyUnavailableError, match=r"S5"):
        verify_strategy_availability(splits, ("S1", "S5"))


def test_strategy_available_passes() -> None:
    splits = load_splits_artifact(REAL_PROCESSED)
    verify_strategy_availability(splits, ("S1", "S3"))


# Helper: vocab classification matches the spec's TR-CAT-02 boundary at 8.
def test_high_card_vocab_keys_boundary() -> None:
    vocab = load_vocab(REAL_PROCESSED)
    keys = high_card_vocab_keys(vocab)
    # Real-data fixture: 6 high-card keys.
    assert keys == frozenset({
        "Archetype", "coaches", "officials", "positions", "stadium", "team_codes",
    })
    # The three low-card keys (≤ 8): roof, surface, day_of_week.
    assert keys.isdisjoint({"roof", "surface", "day_of_week"})
