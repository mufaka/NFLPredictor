"""End-to-end feature build smoke against real Phase 1 outputs."""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil

import pandas as pd
import pyarrow.parquet as pq
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
REAL_RAW = REPO_ROOT / "Data" / "raw"
REAL_PROCESSED = REPO_ROOT / "Data" / "processed"


def _copy_phase1_to(target: pathlib.Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for basename in (
        PHASE1_BOX_SCORES_BASENAME,
        PHASE1_MADDEN_BASENAME,
        PHASE1_MANIFEST_BASENAME,
    ):
        shutil.copy2(REAL_PROCESSED / basename, target / basename)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def real_build(tmp_path_factory) -> pathlib.Path:
    """Run the real feature build once into a shared temp directory."""
    target = tmp_path_factory.mktemp("real_features")
    _copy_phase1_to(target)
    run_feature_build(REAL_RAW, target, repo_dir=REPO_ROOT)
    return target


def test_flat_parquet_shape(real_build):
    pf = pq.read_table(real_build / FLAT_PARQUET_BASENAME).to_pandas()
    assert len(pf) == 272
    # v1 default: 1 GameId + 12 game-level + 4 weather + 7 officials
    # + 88 slot_madden + 44 slot_position + 44 slot_matched + 2 labels = 202.
    assert pf.shape[1] == 202
    assert list(pf.columns[:5]) == [
        "GameId", "week", "day_of_week", "start_hour", "stadium",
    ]
    assert list(pf.columns[-2:]) == ["home_score", "away_score"]


def test_pos_parquet_shape(real_build):
    pf = pq.read_table(real_build / POS_PARQUET_BASENAME).to_pandas()
    assert len(pf) == 272
    # v1 default: 1 + 12 + 4 + 7 + 116 (58 × 2) + 58 present + 58 matched + 2 = 258.
    assert pf.shape[1] == 258


def test_vocab_keys_match_spec(real_build):
    vocab = json.loads((real_build / FEATURE_VOCAB_BASENAME).read_text())
    # FE-VOC-03 keys for v1 default.
    expected_keys = {
        "Archetype",
        "day_of_week",
        "stadium",
        "roof",
        "surface",
        "team_codes",
        "coaches",
        "officials",
        "positions",
    }
    assert set(vocab.keys()) == expected_keys
    # Vocab values are lexicographically sorted (FE-VOC-04).
    for key, values in vocab.items():
        assert values == sorted(values), f"{key} not sorted"


def test_manifest_structure(real_build):
    manifest = json.loads((real_build / FEATURE_MANIFEST_BASENAME).read_text())
    required = {
        "build_timestamp_utc", "normalization_version", "feature_config_sha256",
        "phase1_source_sha256", "output_sha256", "git_commit",
        "phase1_manifest_git_commit", "column_counts", "vocab_sizes",
    }
    assert set(manifest.keys()) == required
    assert manifest["normalization_version"] == "v1"
    # phase1_source_sha256 references the two consumed Phase 1 outputs.
    assert set(manifest["phase1_source_sha256"].keys()) == {
        f"Data/processed/{PHASE1_BOX_SCORES_BASENAME}",
        f"Data/processed/{PHASE1_MADDEN_BASENAME}",
    }
    # output_sha256 covers both parquets + vocab.
    assert set(manifest["output_sha256"].keys()) == {
        f"Data/processed/{FLAT_PARQUET_BASENAME}",
        f"Data/processed/{POS_PARQUET_BASENAME}",
        f"Data/processed/{FEATURE_VOCAB_BASENAME}",
    }
    # column_counts breakdown matches FE-MAN-01 section keys.
    flat_counts = manifest["column_counts"][FLAT_PARQUET_BASENAME]
    assert flat_counts["total"] == 202
    assert flat_counts["game_id"] == 1
    assert flat_counts["game_level"] == 12
    assert flat_counts["weather"] == 4
    assert flat_counts["officials"] == 7
    assert flat_counts["slot_madden"] == 88
    assert flat_counts["slot_position"] == 44
    assert flat_counts["slot_matched"] == 44
    assert flat_counts["labels"] == 2

    pos_counts = manifest["column_counts"][POS_PARQUET_BASENAME]
    assert pos_counts["total"] == 258
    assert pos_counts["slot_madden"] == 116
    assert pos_counts["slot_present"] == 58
    assert pos_counts["slot_matched"] == 58


def test_mahomes_pinned_identity(real_build):
    """Spot-check: the Chiefs/Ravens opener resolves Mahomes to a 99 overall rating."""
    flat = pq.read_table(real_build / FLAT_PARQUET_BASENAME).to_pandas()
    opener = flat[flat["GameId"] == "202409050kan"].iloc[0]
    # Mahomes is HomeOff01; per Madden 24 he's the #1 QB at 99 overall.
    assert opener["HomeOff01_madden_overall_rating"] == 99.0
    # Scores from the raw box score: KC 27, BAL 20.
    assert opener["home_score"] == 27.0
    assert opener["away_score"] == 20.0


def test_byte_identical_rerun(tmp_path):
    """Two independent runs produce byte-identical parquet + vocab (FE-NF-01)."""
    _copy_phase1_to(tmp_path)
    run_feature_build(REAL_RAW, tmp_path, repo_dir=REPO_ROOT)
    h1 = {
        name: _sha256(tmp_path / name)
        for name in (FLAT_PARQUET_BASENAME, POS_PARQUET_BASENAME, FEATURE_VOCAB_BASENAME)
    }
    run_feature_build(REAL_RAW, tmp_path, repo_dir=REPO_ROOT)
    h2 = {
        name: _sha256(tmp_path / name)
        for name in (FLAT_PARQUET_BASENAME, POS_PARQUET_BASENAME, FEATURE_VOCAB_BASENAME)
    }
    assert h1 == h2


def test_pos_handles_real_bucket_overflow_without_raising(real_build, capsys):
    """The real 2024 data has ~18 OL/LB overflow games — they must drop quietly, not raise."""
    # The fixture finished without exception; we just confirm the parquet has all 272 rows.
    pf = pq.read_table(real_build / POS_PARQUET_BASENAME).to_pandas()
    assert len(pf) == 272
