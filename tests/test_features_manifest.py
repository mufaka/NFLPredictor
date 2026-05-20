"""Tests for the feature manifest builder (§3.12 / FE-MAN-01..05)."""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import pytest

from nflpredictor.features.manifest import (
    build_feature_manifest,
    utc_timestamp,
    write_feature_manifest,
)
from nflpredictor.features.vocab import build_vocabulary


REQUIRED_KEYS = {
    "build_timestamp_utc",
    "normalization_version",
    "feature_config_sha256",
    "phase1_source_sha256",
    "output_sha256",
    "git_commit",
    "phase1_manifest_git_commit",
    "column_counts",
    "vocab_sizes",
    "row_count",
}


def _write_bytes(tmp_path: pathlib.Path, name: str, body: bytes) -> pathlib.Path:
    path = tmp_path / name
    path.write_bytes(body)
    return path


def test_utc_timestamp_format():
    ts = utc_timestamp(dt.datetime(2026, 5, 18, 12, 30, 45, tzinfo=dt.timezone.utc))
    assert ts == "2026-05-18T12:30:45Z"


def test_manifest_has_all_required_keys(tmp_path):
    config = _write_bytes(tmp_path, "feature_config.yaml", b"some: yaml")
    madden = _write_bytes(tmp_path, "madden_2024.csv", b"madden,csv,content")
    box = _write_bytes(tmp_path, "box_scores_2024.csv", b"box,scores,csv")
    flat = _write_bytes(tmp_path, "features_flat_2024.parquet", b"\x00\x01parquet")
    vocab_file = _write_bytes(tmp_path, "feature_vocab.json", b"{}\n")

    vocab = build_vocabulary({"colors": ["red", "blue"]})

    manifest = build_feature_manifest(
        config_path=config,
        normalization_version="v1",
        phase1_outputs={
            "Data/processed/madden_2024.csv": madden,
            "Data/processed/box_scores_2024.csv": box,
        },
        phase1_manifest={"git_commit": "deadbeef"},
        feature_outputs={
            "Data/processed/features_flat_2024.parquet": flat,
            "Data/processed/feature_vocab.json": vocab_file,
        },
        column_counts={"features_flat_all.parquet": {"total": 100, "identifiers": 2}},
        row_count=1622,
        vocab=vocab,
        repo_dir=tmp_path,
    )

    assert set(manifest.keys()) == REQUIRED_KEYS
    assert manifest["normalization_version"] == "v1"
    assert manifest["row_count"] == 1622
    assert manifest["phase1_manifest_git_commit"] == "deadbeef"
    assert manifest["vocab_sizes"] == {"colors": 2}
    assert manifest["column_counts"]["features_flat_all.parquet"]["total"] == 100

    # SHAs computed for every input/output.
    assert set(manifest["phase1_source_sha256"].keys()) == {
        "Data/processed/madden_2024.csv",
        "Data/processed/box_scores_2024.csv",
    }
    assert set(manifest["output_sha256"].keys()) == {
        "Data/processed/features_flat_2024.parquet",
        "Data/processed/feature_vocab.json",
    }
    # All SHA values are hex strings of length 64.
    for v in manifest["phase1_source_sha256"].values():
        assert isinstance(v, str) and len(v) == 64
    for v in manifest["output_sha256"].values():
        assert isinstance(v, str) and len(v) == 64


def test_manifest_phase1_git_commit_passes_through_none(tmp_path):
    config = _write_bytes(tmp_path, "feature_config.yaml", b"")
    madden = _write_bytes(tmp_path, "madden_2024.csv", b"")
    box = _write_bytes(tmp_path, "box_scores_2024.csv", b"")

    manifest = build_feature_manifest(
        config_path=config,
        normalization_version="v1",
        phase1_outputs={
            "Data/processed/madden_2024.csv": madden,
            "Data/processed/box_scores_2024.csv": box,
        },
        phase1_manifest={"git_commit": None},
        feature_outputs={},
        column_counts={},
        row_count=0,
        vocab=build_vocabulary({}),
        repo_dir=tmp_path,
    )
    assert manifest["phase1_manifest_git_commit"] is None


def test_write_manifest_round_trip(tmp_path):
    payload = {
        "b_key": "after",
        "a_key": "first",
        "nested": {"z": 1, "a": 2},
    }
    path = tmp_path / "feature_manifest.json"
    write_feature_manifest(payload, path)

    content = path.read_text(encoding="utf-8")
    assert content.endswith("\n")
    # Sorted keys at the top level: 'a_key' must appear before 'b_key'.
    assert content.index("a_key") < content.index("b_key")
    # And in nested dicts.
    assert content.index('"a": 2') < content.index('"z": 1')

    loaded = json.loads(content)
    assert loaded == payload
