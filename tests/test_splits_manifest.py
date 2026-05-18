"""Unit tests for splits manifest construction (SP-MAN-01..06)."""

from __future__ import annotations

import datetime as _dt
import json
import pathlib

from nflpredictor.splits.config import SplitsConfig
from nflpredictor.splits.manifest import (
    build_splits_manifest,
    build_strategy_summaries,
    utc_timestamp,
    write_splits_manifest,
)
from nflpredictor.splits.s3 import Fold


V1_CONFIG = SplitsConfig(
    splits_version="v1",
    strategies=("S1", "S3"),
    train_weeks=(1, 12),
    val_weeks=(13, 15),
    test_weeks=(16, 18),
    s3_k_start=6,
)


def test_utc_timestamp_format():
    ts = utc_timestamp(_dt.datetime(2026, 5, 18, 12, 34, 56, tzinfo=_dt.timezone.utc))
    assert ts == "2026-05-18T12:34:56Z"


def test_build_strategy_summaries_s1_only():
    s1 = {"train": ["a", "b"], "val": ["c"], "test": ["d", "e", "f"]}
    out = build_strategy_summaries(s1, None)
    assert out == {"S1": {"train_n": 2, "val_n": 1, "test_n": 3}}


def test_build_strategy_summaries_s3_includes_folds():
    s3 = {
        "test": ["t1", "t2"],
        "folds": [
            Fold(fold_index=0, k=6, train=("a", "b"), val=("c",)),
            Fold(fold_index=1, k=7, train=("a", "b", "c"), val=("d",)),
        ],
    }
    out = build_strategy_summaries(None, s3)
    assert out == {
        "S3": {
            "test_n": 2,
            "fold_count": 2,
            "folds": [
                {"fold_index": 0, "k": 6, "train_n": 2, "val_n": 1},
                {"fold_index": 1, "k": 7, "train_n": 3, "val_n": 1},
            ],
        }
    }


def test_build_splits_manifest_keys_present(tmp_path):
    manifest = build_splits_manifest(
        config=V1_CONFIG,
        phase2_source_sha256={"Data/processed/features_flat_2024.parquet": "deadbeef"},
        output_sha256={"Data/processed/splits_2024.json": "cafebabe"},
        phase2_manifest_git_commit="abc123",
        strategy_summaries={"S1": {"train_n": 1, "val_n": 1, "test_n": 1}},
        splits_config_sha256="0badc0de",
        repo_dir=tmp_path,
        now=_dt.datetime(2026, 5, 18, 0, 0, 0, tzinfo=_dt.timezone.utc),
    )
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
    assert expected_keys == set(manifest.keys())
    assert manifest["splits_version"] == "v1"
    assert manifest["phase2_manifest_git_commit"] == "abc123"
    assert manifest["splits_config_sha256"] == "0badc0de"


def test_write_splits_manifest_sorts_keys_and_ends_with_newline(tmp_path):
    manifest = {"b": 1, "a": {"y": 2, "x": 1}}
    out = tmp_path / "m.json"
    write_splits_manifest(manifest, out)
    raw = out.read_bytes()
    assert raw.endswith(b"\n")
    # Decoded keys should round-trip; sort_keys=True at every level.
    decoded = json.loads(raw)
    assert decoded == manifest
    # Inline sort order check.
    text = raw.decode("utf-8")
    assert text.index('"a"') < text.index('"b"')
    assert text.index('"x"') < text.index('"y"')
