"""End-to-end feature build smoke against real Phase 1 outputs."""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil

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

# The combined six-season game universe.
N_GAMES = 1622


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
    assert len(pf) == N_GAMES
    # default config: 2 identifiers + 12 game-level + 4 weather + 7 officials
    # + 88 slot_madden + 44 slot_position + 44 slot_matched + 2 labels = 203.
    assert pf.shape[1] == 203
    assert list(pf.columns[:5]) == [
        "GameId", "season", "week", "day_of_week", "start_hour",
    ]
    assert list(pf.columns[-2:]) == ["home_score", "away_score"]


def test_pos_parquet_shape(real_build):
    pf = pq.read_table(real_build / POS_PARQUET_BASENAME).to_pandas()
    assert len(pf) == N_GAMES
    # default config: 2 + 12 + 4 + 7 + 116 (58 × 2) + 58 present + 58 matched + 2 = 259.
    assert pf.shape[1] == 259


def test_vocab_keys_match_spec(real_build):
    payload = json.loads((real_build / FEATURE_VOCAB_BASENAME).read_text())
    assert set(payload.keys()) == {"vocab_version", "entries", "column_vocab_keys"}
    assert payload["vocab_version"] == "v3"
    entries = payload["entries"]
    expected_keys = {
        "archetype",
        "day_of_week",
        "stadium",
        "roof",
        "surface",
        "team_codes",
        "coaches",
        "officials",
        "positions",
    }
    assert set(entries.keys()) == expected_keys
    for key, values in entries.items():
        assert values == sorted(values), f"{key} not sorted"
    for col, vk in payload["column_vocab_keys"].items():
        assert vk in expected_keys, f"{col!r} routes to unknown vocab key {vk!r}"


def test_manifest_structure(real_build):
    manifest = json.loads((real_build / FEATURE_MANIFEST_BASENAME).read_text())
    required = {
        "build_timestamp_utc", "normalization_version", "feature_config_sha256",
        "phase1_source_sha256", "output_sha256", "git_commit",
        "phase1_manifest_git_commit", "column_counts", "vocab_sizes", "row_count",
    }
    assert set(manifest.keys()) == required
    assert manifest["normalization_version"] == "v2"
    assert manifest["row_count"] == N_GAMES
    assert set(manifest["phase1_source_sha256"].keys()) == {
        f"Data/processed/{PHASE1_BOX_SCORES_BASENAME}",
        f"Data/processed/{PHASE1_MADDEN_BASENAME}",
    }
    assert set(manifest["output_sha256"].keys()) == {
        f"Data/processed/{FLAT_PARQUET_BASENAME}",
        f"Data/processed/{POS_PARQUET_BASENAME}",
        f"Data/processed/{FEATURE_VOCAB_BASENAME}",
    }
    flat_counts = manifest["column_counts"][FLAT_PARQUET_BASENAME]
    assert flat_counts["total"] == 203
    assert flat_counts["identifiers"] == 2
    assert flat_counts["game_level"] == 12
    assert flat_counts["weather"] == 4
    assert flat_counts["officials"] == 7
    assert flat_counts["slot_madden"] == 88
    assert flat_counts["slot_position"] == 44
    assert flat_counts["slot_matched"] == 44
    assert flat_counts["labels"] == 2

    pos_counts = manifest["column_counts"][POS_PARQUET_BASENAME]
    assert pos_counts["total"] == 259
    assert pos_counts["slot_madden"] == 116
    assert pos_counts["slot_present"] == 58
    assert pos_counts["slot_matched"] == 58


def test_mahomes_pinned_identity_flat(real_build):
    """The Chiefs/Ravens 2024 opener resolves Mahomes to a 99 overall in B-flat."""
    flat = pq.read_table(real_build / FLAT_PARQUET_BASENAME).to_pandas()
    opener = flat[flat["GameId"] == "202409050kan"].iloc[0]
    assert opener["season"] == 2024
    assert opener["HomeOff01_madden_overallrating"] == 99.0
    assert opener["HomeOff01_matched"] == 1
    assert opener["home_score"] == 27.0
    assert opener["away_score"] == 20.0


def test_mahomes_pinned_identity_pos(real_build):
    """FE-TEST-07: Mahomes also shows up at HomeQB1 in B-pos with the same rating."""
    pos = pq.read_table(real_build / POS_PARQUET_BASENAME).to_pandas()
    opener = pos[pos["GameId"] == "202409050kan"].iloc[0]
    assert opener["HomeQB1_present"] == 1
    assert opener["HomeQB1_madden_overallrating"] == 99.0
    assert opener["HomeQB1_matched"] == 1


def test_vocab_decode_round_trip(real_build):
    """Integer codes in the parquet round-trip back to strings via the vocab."""
    flat = pq.read_table(real_build / FLAT_PARQUET_BASENAME).to_pandas()
    entries = json.loads((real_build / FEATURE_VOCAB_BASENAME).read_text())["entries"]
    opener = flat[flat["GameId"] == "202409050kan"].iloc[0]
    home_code = int(opener["home_team_code"])
    assert entries["team_codes"][home_code] == "kan"
    dow_code = int(opener["day_of_week"])
    assert entries["day_of_week"][dow_code] == "Thursday"


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


def test_pos_handles_real_bucket_overflow_without_raising(real_build):
    """The real data has OL/TE overflow games — they must drop quietly, not raise."""
    pf = pq.read_table(real_build / POS_PARQUET_BASENAME).to_pandas()
    assert len(pf) == N_GAMES
