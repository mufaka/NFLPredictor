"""Unit tests for splits manifest construction (SP-MAN-01..06)."""

from __future__ import annotations

import datetime as _dt
import json

from nflpredictor.splits.config import SplitsConfig
from nflpredictor.splits.loso_cv import Fold
from nflpredictor.splits.manifest import (
    build_splits_manifest,
    build_strategy_summaries,
    utc_timestamp,
    write_splits_manifest,
)


V2_CONFIG = SplitsConfig(
    splits_version="v2",
    strategies=("season_holdout", "loso_cv"),
    train_seasons=(2020, 2021, 2022, 2023),
    val_season=2024,
    test_season=2025,
)


def test_utc_timestamp_format():
    ts = utc_timestamp(_dt.datetime(2026, 5, 18, 12, 34, 56, tzinfo=_dt.timezone.utc))
    assert ts == "2026-05-18T12:34:56Z"


def test_build_strategy_summaries_season_holdout_only():
    sh = {"train": ["a", "b"], "val": ["c"], "test": ["d", "e", "f"]}
    out = build_strategy_summaries(sh, None)
    assert out == {"season_holdout": {"train_n": 2, "val_n": 1, "test_n": 3}}


def test_build_strategy_summaries_loso_cv_includes_folds():
    loso = {
        "test": ["t1", "t2"],
        "folds": [
            Fold(fold_index=0, val_season=2020, train=("a", "b"), val=("c",)),
            Fold(fold_index=1, val_season=2021, train=("a", "b", "c"), val=("d",)),
        ],
    }
    out = build_strategy_summaries(None, loso)
    assert out == {
        "loso_cv": {
            "test_n": 2,
            "fold_count": 2,
            "folds": [
                {"fold_index": 0, "val_season": 2020, "train_n": 2, "val_n": 1},
                {"fold_index": 1, "val_season": 2021, "train_n": 3, "val_n": 1},
            ],
        }
    }


def test_build_splits_manifest_keys_present(tmp_path):
    manifest = build_splits_manifest(
        config=V2_CONFIG,
        phase2_source_sha256={"Data/processed/features_flat_all.parquet": "deadbeef"},
        output_sha256={"Data/processed/splits_all.json": "cafebabe"},
        phase2_manifest_git_commit="abc123",
        strategy_summaries={"season_holdout": {"train_n": 1, "val_n": 1, "test_n": 1}},
        splits_config_sha256="0badc0de",
        repo_dir=tmp_path,
        now=_dt.datetime(2026, 5, 18, 0, 0, 0, tzinfo=_dt.timezone.utc),
    )
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
    assert expected_keys == set(manifest.keys())
    assert manifest["splits_version"] == "v2"
    assert manifest["phase2_manifest_git_commit"] == "abc123"
    assert manifest["splits_config_sha256"] == "0badc0de"
    assert manifest["season_assignment"] == {
        "train_seasons": [2020, 2021, 2022, 2023],
        "val_season": 2024,
        "test_season": 2025,
    }


def test_write_splits_manifest_sorts_keys_and_ends_with_newline(tmp_path):
    manifest = {"b": 1, "a": {"y": 2, "x": 1}}
    out = tmp_path / "m.json"
    write_splits_manifest(manifest, out)
    raw = out.read_bytes()
    assert raw.endswith(b"\n")
    decoded = json.loads(raw)
    assert decoded == manifest
    text = raw.decode("utf-8")
    assert text.index('"a"') < text.index('"b"')
    assert text.index('"x"') < text.index('"y"')
