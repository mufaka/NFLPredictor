"""Pinned-identity test against the real Phase 2 outputs (SP-TEST-06).

Also exercises SP-TEST-05's determinism property on the real dataset
(companion to ``test_splits_determinism.py``'s fixture-based test).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil

from nflpredictor.splits.pipeline import (
    PHASE2_FEATURES_FLAT_BASENAME,
    PHASE2_MANIFEST_BASENAME,
    SPLITS_ARTIFACT_BASENAME,
    SPLITS_CONFIG_BASENAME,
    SPLITS_MANIFEST_BASENAME,
    run_split_build,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL_RAW = REPO_ROOT / "Data" / "raw"
REAL_PROCESSED = REPO_ROOT / "Data" / "processed"
N_GAMES = 1622


def _seed_real_inputs(processed: pathlib.Path, raw: pathlib.Path) -> None:
    processed.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    for name in (PHASE2_FEATURES_FLAT_BASENAME, PHASE2_MANIFEST_BASENAME):
        shutil.copy2(REAL_PROCESSED / name, processed / name)
    shutil.copy2(REAL_RAW / SPLITS_CONFIG_BASENAME, raw / SPLITS_CONFIG_BASENAME)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(tmp_path: pathlib.Path) -> pathlib.Path:
    processed = tmp_path / "Data" / "processed"
    raw = tmp_path / "Data" / "raw"
    _seed_real_inputs(processed, raw)
    run_split_build(raw, processed, repo_dir=REPO_ROOT)
    return processed


def test_real_data_season_holdout_partition_sizes(tmp_path):
    processed = _run(tmp_path)
    artifact = json.loads((processed / SPLITS_ARTIFACT_BASENAME).read_text())

    sh = artifact["season_holdout"]
    train, val, test = sh["train"], sh["val"], sh["test"]

    # Pinned counts for the v2 default assignment (train 2020-2023 /
    # val 2024 / test 2025) on the real combined data.
    assert len(train) == 1077
    assert len(val) == 272
    assert len(test) == 273
    assert len(train) + len(val) + len(test) == N_GAMES

    train_set, val_set, test_set = set(train), set(val), set(test)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(train_set | val_set | test_set) == N_GAMES


def _season_of(game_id: str) -> int:
    """NFL season of a GameId — month >= 8 maps to the calendar year, else year-1."""
    year, month = int(game_id[:4]), int(game_id[4:6])
    return year if month >= 8 else year - 1


def test_real_data_test_season_never_pooled(tmp_path):
    """The 2025 test season must not leak into train or val (SP-TEST-06)."""
    processed = _run(tmp_path)
    artifact = json.loads((processed / SPLITS_ARTIFACT_BASENAME).read_text())
    sh = artifact["season_holdout"]
    for gid in sh["train"] + sh["val"]:
        assert _season_of(gid) != 2025, (
            f"{gid} from the test season leaked into train/val"
        )


def test_real_data_byte_identical_rerun(tmp_path):
    """Two real-data runs against identical inputs → byte-identical outputs."""
    processed_a, raw_a = tmp_path / "a", tmp_path / "raw_a"
    processed_b, raw_b = tmp_path / "b", tmp_path / "raw_b"
    _seed_real_inputs(processed_a, raw_a)
    _seed_real_inputs(processed_b, raw_b)

    run_split_build(raw_a, processed_a, repo_dir=REPO_ROOT)
    run_split_build(raw_b, processed_b, repo_dir=REPO_ROOT)

    assert (
        _sha256(processed_a / SPLITS_ARTIFACT_BASENAME)
        == _sha256(processed_b / SPLITS_ARTIFACT_BASENAME)
    )
    m1 = json.loads((processed_a / SPLITS_MANIFEST_BASENAME).read_text())
    m2 = json.loads((processed_b / SPLITS_MANIFEST_BASENAME).read_text())
    m1.pop("build_timestamp_utc")
    m2.pop("build_timestamp_utc")
    assert m1 == m2


def test_real_data_manifest_has_required_keys(tmp_path):
    processed = _run(tmp_path)
    manifest = json.loads((processed / SPLITS_MANIFEST_BASENAME).read_text())
    expected_keys = {
        "build_timestamp_utc",
        "splits_version",
        "splits_config_sha256",
        "season_assignment",
        "phase2_source_sha256",
        "output_sha256",
        "git_commit",
        "phase2_manifest_git_commit",
        "strategy_summaries",
    }
    assert expected_keys.issubset(manifest.keys())
    assert f"Data/processed/{SPLITS_ARTIFACT_BASENAME}" in manifest["output_sha256"]
    assert manifest["season_assignment"]["test_season"] == 2025
