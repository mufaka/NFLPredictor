"""Pinned-identity test against the real Phase 2 outputs (SP-TEST-06).

Also exercises SP-TEST-05's determinism property on the real 272-game
dataset (companion to ``test_splits_determinism.py``'s fixture-based test).
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


def test_real_data_s1_partition_sizes(tmp_path):
    processed = _run(tmp_path)
    artifact = json.loads((processed / SPLITS_ARTIFACT_BASENAME).read_text())

    s1 = artifact["S1"]
    train, val, test = s1["train"], s1["val"], s1["test"]

    # Pinned counts for the v1 default boundaries on the 2024 season.
    assert len(train) == 179
    assert len(val) == 45
    assert len(test) == 48
    assert len(train) + len(val) + len(test) == 272

    # The three role lists partition the universe.
    train_set, val_set, test_set = set(train), set(val), set(test)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(train_set | val_set | test_set) == 272


def test_real_data_s3_structure(tmp_path):
    processed = _run(tmp_path)
    artifact = json.loads((processed / SPLITS_ARTIFACT_BASENAME).read_text())
    s1 = artifact["S1"]
    s3 = artifact["S3"]

    assert s3["test"] == s1["test"]
    assert len(s3["folds"]) == 9
    assert [f["k"] for f in s3["folds"]] == list(range(6, 15))
    assert [f["fold_index"] for f in s3["folds"]] == list(range(9))


def test_real_data_byte_identical_rerun(tmp_path):
    """Two real-data runs against identical inputs → byte-identical outputs."""
    processed_a = tmp_path / "a"
    raw_a = tmp_path / "raw_a"
    processed_b = tmp_path / "b"
    raw_b = tmp_path / "raw_b"
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
        "phase2_source_sha256",
        "output_sha256",
        "git_commit",
        "phase2_manifest_git_commit",
        "strategy_summaries",
    }
    assert expected_keys.issubset(manifest.keys())
    assert (
        f"Data/processed/{SPLITS_ARTIFACT_BASENAME}"
        in manifest["output_sha256"]
    )
