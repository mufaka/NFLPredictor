"""EV-TEST-10 (partial): upstream source-hash gates for Phase 2, Phase 3, Phase 4."""

from __future__ import annotations

import json
import pathlib
import shutil

import pytest

import pandas as pd

from nflpredictor.databuild.manifest import compute_sha256
from nflpredictor.evaluate.sources import (
    PHASE2_FEATURES_FLAT_BASENAME,
    PHASE2_FEATURES_POS_BASENAME,
    PHASE2_MANIFEST_BASENAME,
    PHASE2_VOCAB_BASENAME,
    PHASE3_MANIFEST_BASENAME,
    PHASE3_SPLITS_BASENAME,
    PHASE4_MANIFEST_BASENAME,
    PHASE4_PREDICTIONS_DIRNAME,
    CombinationKey,
    LabelParityError,
    Phase2OutputMismatchError,
    Phase3OutputMismatchError,
    Phase4OutputMismatchError,
    PredictionCoverageError,
    assert_label_parity,
    enumerate_combinations,
    load_features_flat,
    load_features_pos,
    load_splits,
    load_vocab,
    validate_prediction_coverage,
    verify_phase2_outputs,
    verify_phase3_outputs,
    verify_phase4_outputs,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL_PROCESSED = REPO_ROOT / "Data" / "processed"
REAL_PREDICTIONS_DIR = REAL_PROCESSED / PHASE4_PREDICTIONS_DIRNAME


def _stage_phase2_phase3(tmp_path: pathlib.Path) -> pathlib.Path:
    """Stage Phase 2 + Phase 3 outputs into a tmp dir so tests can mutate them."""
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
    data = bytearray(path.read_bytes())
    idx = len(data) // 2
    data[idx] = data[idx] ^ 0x01
    path.write_bytes(bytes(data))


# ---------------------------------------------------------------------------
# Phase 2 hash gate
# ---------------------------------------------------------------------------


def test_real_phase2_gate_passes() -> None:
    manifest = verify_phase2_outputs(REAL_PROCESSED)
    assert "output_sha256" in manifest


def test_tampered_features_flat_fails(tmp_path: pathlib.Path) -> None:
    staged = _stage_phase2_phase3(tmp_path)
    _tamper_bytes(staged / PHASE2_FEATURES_FLAT_BASENAME)
    with pytest.raises(Phase2OutputMismatchError, match=PHASE2_FEATURES_FLAT_BASENAME):
        verify_phase2_outputs(staged)


def test_missing_phase2_manifest_fails(tmp_path: pathlib.Path) -> None:
    staged = _stage_phase2_phase3(tmp_path)
    (staged / PHASE2_MANIFEST_BASENAME).unlink()
    with pytest.raises(FileNotFoundError, match="Phase 2 manifest"):
        verify_phase2_outputs(staged)


# ---------------------------------------------------------------------------
# Phase 3 hash gate
# ---------------------------------------------------------------------------


def test_real_phase3_gate_passes() -> None:
    manifest = verify_phase3_outputs(REAL_PROCESSED)
    assert "output_sha256" in manifest


def test_tampered_splits_fails(tmp_path: pathlib.Path) -> None:
    staged = _stage_phase2_phase3(tmp_path)
    _tamper_bytes(staged / PHASE3_SPLITS_BASENAME)
    with pytest.raises(Phase3OutputMismatchError, match=PHASE3_SPLITS_BASENAME):
        verify_phase3_outputs(staged)


def test_missing_phase3_manifest_fails(tmp_path: pathlib.Path) -> None:
    staged = _stage_phase2_phase3(tmp_path)
    (staged / PHASE3_MANIFEST_BASENAME).unlink()
    with pytest.raises(FileNotFoundError, match="Phase 3 manifest"):
        verify_phase3_outputs(staged)


# ---------------------------------------------------------------------------
# Phase 4 hash gate
# ---------------------------------------------------------------------------


def _synthesize_phase4(processed: pathlib.Path) -> dict[str, pathlib.Path]:
    """Create a minimal Phase 4 layout (2 fake prediction parquets + manifest).

    The "parquets" don't need to be valid parquet for the hash-gate test —
    verify_phase4_outputs only SHAs the bytes against the manifest. Returns
    the {manifest_key → on_disk_path} pairs the test can mutate.
    """
    preds_dir = processed / PHASE4_PREDICTIONS_DIRNAME
    preds_dir.mkdir(parents=True, exist_ok=True)
    fake_a = preds_dir / "rung0_mean__none__s1.parquet"
    fake_b = preds_dir / "rung2_linear__flat__s3.parquet"
    fake_a.write_bytes(b"\x00fake-s1-bytes\n")
    fake_b.write_bytes(b"\x00fake-s3-bytes\n")

    output_sha256 = {
        "predictions/rung0_mean__none__s1.parquet": compute_sha256(fake_a),
        "predictions/rung2_linear__flat__s3.parquet": compute_sha256(fake_b),
    }
    manifest = {
        "build_timestamp_utc": "2026-05-18T00:00:00Z",
        "training_version": "v1",
        "output_sha256": output_sha256,
    }
    (processed / PHASE4_MANIFEST_BASENAME).write_text(json.dumps(manifest))
    return {
        "predictions/rung0_mean__none__s1.parquet": fake_a,
        "predictions/rung2_linear__flat__s3.parquet": fake_b,
    }


def test_real_phase4_gate_passes_or_skips() -> None:
    if not REAL_PREDICTIONS_DIR.exists() or not any(REAL_PREDICTIONS_DIR.glob("*.parquet")):
        pytest.skip(
            "Data/processed/predictions/ is empty; the real Phase 4 run is deferred "
            "to the CUDA training machine."
        )
    manifest = verify_phase4_outputs(REAL_PROCESSED)
    assert "output_sha256" in manifest


def test_synthetic_phase4_gate_passes(tmp_path: pathlib.Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    _synthesize_phase4(processed)
    manifest = verify_phase4_outputs(processed)
    assert "output_sha256" in manifest


def test_tampered_phase4_parquet_fails(tmp_path: pathlib.Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    files = _synthesize_phase4(processed)
    target = files["predictions/rung2_linear__flat__s3.parquet"]
    _tamper_bytes(target)
    with pytest.raises(
        Phase4OutputMismatchError, match="rung2_linear__flat__s3.parquet"
    ):
        verify_phase4_outputs(processed)


def test_missing_phase4_manifest_fails(tmp_path: pathlib.Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    with pytest.raises(FileNotFoundError, match="Phase 4 manifest"):
        verify_phase4_outputs(processed)


def test_phase4_manifest_missing_output_sha256_block(tmp_path: pathlib.Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / PHASE4_MANIFEST_BASENAME).write_text(json.dumps({"training_version": "v1"}))
    with pytest.raises(Phase4OutputMismatchError, match="output_sha256"):
        verify_phase4_outputs(processed)


def test_phase4_manifest_no_predictions_block(tmp_path: pathlib.Path) -> None:
    """An output_sha256 with no 'predictions/*' keys is a structural error."""
    processed = tmp_path / "processed"
    processed.mkdir()
    manifest = {
        "training_version": "v1",
        "output_sha256": {"not_predictions/foo.json": "deadbeef"},
    }
    (processed / PHASE4_MANIFEST_BASENAME).write_text(json.dumps(manifest))
    with pytest.raises(Phase4OutputMismatchError, match="prediction parquets"):
        verify_phase4_outputs(processed)


def test_phase4_missing_listed_parquet_fails(tmp_path: pathlib.Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    files = _synthesize_phase4(processed)
    files["predictions/rung0_mean__none__s1.parquet"].unlink()
    with pytest.raises(FileNotFoundError, match="rung0_mean__none__s1.parquet"):
        verify_phase4_outputs(processed)


# ---------------------------------------------------------------------------
# Phase 2 / Phase 3 source loaders + label parity
# ---------------------------------------------------------------------------


def test_load_features_flat_returns_sorted_frame() -> None:
    df = load_features_flat(REAL_PROCESSED)
    assert "GameId" in df.columns
    assert "home_score" in df.columns
    assert "away_score" in df.columns
    assert list(df["GameId"]) == sorted(df["GameId"])
    assert len(df) == 272


def test_load_features_pos_returns_sorted_frame() -> None:
    df = load_features_pos(REAL_PROCESSED)
    assert len(df) == 272
    assert list(df["GameId"]) == sorted(df["GameId"])


def test_load_vocab_returns_dict() -> None:
    vocab = load_vocab(REAL_PROCESSED)
    assert "team_codes" in vocab
    assert "surface" in vocab
    assert "roof" in vocab
    assert isinstance(vocab["team_codes"], list)
    assert len(vocab["team_codes"]) == 32


def test_load_splits_returns_dict() -> None:
    splits = load_splits(REAL_PROCESSED)
    assert "S1" in splits
    assert "S3" in splits


def test_label_parity_accepts_real_pair() -> None:
    flat = load_features_flat(REAL_PROCESSED)
    pos = load_features_pos(REAL_PROCESSED)
    assert_label_parity(flat, pos)  # no raise


def test_label_parity_rejects_synthetic_mismatch() -> None:
    flat = pd.DataFrame(
        {"GameId": ["g1", "g2"], "home_score": [10.0, 20.0], "away_score": [7.0, 14.0]}
    )
    pos = pd.DataFrame(
        {"GameId": ["g1", "g2"], "home_score": [10.0, 21.0], "away_score": [7.0, 14.0]}
    )
    with pytest.raises(LabelParityError, match="home_score"):
        assert_label_parity(flat, pos)


def test_label_parity_rejects_different_gameid_sets() -> None:
    flat = pd.DataFrame(
        {"GameId": ["g1", "g2"], "home_score": [10.0, 20.0], "away_score": [7.0, 14.0]}
    )
    pos = pd.DataFrame(
        {"GameId": ["g1", "gZ"], "home_score": [10.0, 20.0], "away_score": [7.0, 14.0]}
    )
    with pytest.raises(LabelParityError, match="GameId"):
        assert_label_parity(flat, pos)


# ---------------------------------------------------------------------------
# enumerate_combinations
# ---------------------------------------------------------------------------


def test_enumerate_combinations_returns_lexicographic_keys(tmp_path: pathlib.Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    _synthesize_phase4(processed)
    manifest = verify_phase4_outputs(processed)
    combos = enumerate_combinations(manifest, processed)
    assert [c.combination_id for c in combos] == [
        "rung0_mean__none__s1",
        "rung2_linear__flat__s3",
    ]
    assert [c.strategy for c in combos] == ["S1", "S3"]
    assert combos[0].parquet_path.name == "rung0_mean__none__s1.parquet"


def test_enumerate_combinations_rejects_unknown_strategy_suffix(
    tmp_path: pathlib.Path,
) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / PHASE4_PREDICTIONS_DIRNAME).mkdir()
    bad = processed / PHASE4_PREDICTIONS_DIRNAME / "rung0_mean__none__s9.parquet"
    bad.write_bytes(b"x")
    manifest = {
        "output_sha256": {
            "predictions/rung0_mean__none__s9.parquet": compute_sha256(bad),
        }
    }
    with pytest.raises(Phase4OutputMismatchError, match="__s1 or __s3"):
        enumerate_combinations(manifest, processed)


# ---------------------------------------------------------------------------
# validate_prediction_coverage
# ---------------------------------------------------------------------------


def _key(combo_id: str, strategy: str) -> CombinationKey:
    return CombinationKey(
        combination_id=combo_id, strategy=strategy, parquet_path=pathlib.Path("/dev/null")
    )


def test_validate_prediction_coverage_s1_happy() -> None:
    splits = {"S1": {"val": ["g1", "g2"], "test": ["g3"]}, "S3": {"folds": []}}
    preds = pd.DataFrame({
        "GameId": ["g1", "g2", "g3"],
        "slice": ["val", "val", "test"],
        "pred_home": [1.0, 2.0, 3.0],
        "pred_away": [1.0, 2.0, 3.0],
    })
    validate_prediction_coverage(preds, _key("c", "S1"), splits)  # no raise


def test_validate_prediction_coverage_s1_missing_game_fails() -> None:
    splits = {"S1": {"val": ["g1", "g2"], "test": ["g3"]}}
    preds = pd.DataFrame({
        "GameId": ["g1", "g3"],
        "slice": ["val", "test"],
        "pred_home": [1.0, 3.0],
        "pred_away": [1.0, 3.0],
    })
    with pytest.raises(PredictionCoverageError, match="slice='val'"):
        validate_prediction_coverage(preds, _key("c", "S1"), splits)


def test_validate_prediction_coverage_s3_happy() -> None:
    splits = {
        "S1": {},
        "S3": {
            "folds": [
                {"fold_index": 0, "val": ["g1", "g2"]},
                {"fold_index": 1, "val": ["g3"]},
            ]
        },
    }
    preds = pd.DataFrame({
        "GameId": ["g1", "g2", "g3"],
        "fold_index": [0, 0, 1],
        "pred_home": [1.0, 2.0, 3.0],
        "pred_away": [1.0, 2.0, 3.0],
    })
    validate_prediction_coverage(preds, _key("c", "S3"), splits)


def test_validate_prediction_coverage_s3_extra_fold_fails() -> None:
    splits = {
        "S3": {
            "folds": [{"fold_index": 0, "val": ["g1"]}],
        }
    }
    preds = pd.DataFrame({
        "GameId": ["g1", "g2"],
        "fold_index": [0, 5],
        "pred_home": [1.0, 2.0],
        "pred_away": [1.0, 2.0],
    })
    with pytest.raises(PredictionCoverageError, match="unknown fold_index"):
        validate_prediction_coverage(preds, _key("c", "S3"), splits)


def test_validate_prediction_coverage_s3_fold_diverges_fails() -> None:
    splits = {
        "S3": {
            "folds": [{"fold_index": 0, "val": ["g1", "g2"]}],
        }
    }
    preds = pd.DataFrame({
        "GameId": ["g1"],
        "fold_index": [0],
        "pred_home": [1.0],
        "pred_away": [1.0],
    })
    with pytest.raises(PredictionCoverageError, match="fold_index=0"):
        validate_prediction_coverage(preds, _key("c", "S3"), splits)
