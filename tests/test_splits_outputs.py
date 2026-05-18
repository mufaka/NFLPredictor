"""Sort-order test for every GameId list in the artifact (SP-TEST-08).

Also asserts artifact-level structural invariants that depend on the
deterministic writer in ``outputs.py``.
"""

from __future__ import annotations

import json
import pathlib
import shutil

from nflpredictor.splits.pipeline import (
    PHASE2_FEATURES_FLAT_BASENAME,
    PHASE2_MANIFEST_BASENAME,
    SPLITS_ARTIFACT_BASENAME,
    run_split_build,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "splits"
FIXTURE_RAW = FIXTURE_DIR / "raw"
FIXTURE_RAW_PHASE2 = FIXTURE_DIR / "raw_phase2"


def _seed_processed(target: pathlib.Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in (PHASE2_FEATURES_FLAT_BASENAME, PHASE2_MANIFEST_BASENAME):
        shutil.copy2(FIXTURE_RAW_PHASE2 / name, target / name)


def _load_artifact(tmp_path: pathlib.Path) -> dict:
    processed = tmp_path / "Data" / "processed"
    _seed_processed(processed)
    run_split_build(FIXTURE_RAW, processed, repo_dir=tmp_path)
    return json.loads((processed / SPLITS_ARTIFACT_BASENAME).read_text())


def test_every_gameid_list_is_sorted(tmp_path):
    artifact = _load_artifact(tmp_path)

    for role in ("train", "val", "test"):
        ids = artifact["S1"][role]
        assert ids == sorted(ids), f"S1.{role} not sorted"

    assert artifact["S3"]["test"] == sorted(artifact["S3"]["test"]), "S3.test not sorted"

    for fold in artifact["S3"]["folds"]:
        assert fold["train"] == sorted(fold["train"]), (
            f"S3.folds[{fold['fold_index']}].train not sorted"
        )
        assert fold["val"] == sorted(fold["val"]), (
            f"S3.folds[{fold['fold_index']}].val not sorted"
        )


def test_artifact_key_order_is_pinned(tmp_path):
    """Top-level and per-strategy key order matches §4.2."""
    artifact = _load_artifact(tmp_path)
    assert list(artifact.keys()) == ["splits_version", "S1", "S3"]
    assert list(artifact["S1"].keys()) == ["train", "val", "test"]
    assert list(artifact["S3"].keys()) == ["test", "folds"]
    for fold in artifact["S3"]["folds"]:
        assert list(fold.keys()) == ["fold_index", "k", "train", "val"]


def test_artifact_ends_with_trailing_newline(tmp_path):
    processed = tmp_path / "Data" / "processed"
    _seed_processed(processed)
    run_split_build(FIXTURE_RAW, processed, repo_dir=tmp_path)
    raw = (processed / SPLITS_ARTIFACT_BASENAME).read_bytes()
    assert raw.endswith(b"\n")
