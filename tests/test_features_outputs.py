"""Tests for parquet + vocab sidecar writers (§3.10, §3.11)."""

from __future__ import annotations

import json

import pandas as pd
import pyarrow.parquet as pq

from nflpredictor.features.outputs import (
    PARQUET_WRITE_OPTIONS,
    write_parquet,
    write_vocab,
)
from nflpredictor.features.vocab import build_vocabulary


def test_parquet_write_options_pinned():
    # Defensive: any change to the deterministic options must be deliberate.
    assert PARQUET_WRITE_OPTIONS["compression"] == "snappy"
    assert PARQUET_WRITE_OPTIONS["use_dictionary"] is False
    assert PARQUET_WRITE_OPTIONS["write_statistics"] is False
    assert PARQUET_WRITE_OPTIONS["data_page_version"] == "1.0"
    assert PARQUET_WRITE_OPTIONS["version"] == "2.6"


def test_write_parquet_round_trip(tmp_path):
    df = pd.DataFrame({
        "GameId": ["g2", "g1", "g3"],
        "value": [1.0, 2.0, 3.0],
        "flag": pd.array([0, 1, 0], dtype="int32"),
    })
    path = tmp_path / "test.parquet"
    write_parquet(df, path)
    assert path.exists()

    loaded = pq.read_table(path).to_pandas()
    assert list(loaded.columns) == ["GameId", "value", "flag"]
    assert loaded["value"].dtype == "float64"
    assert loaded["flag"].dtype == "int32"
    # Row order is preserved (no auto-sort).
    assert loaded["GameId"].tolist() == ["g2", "g1", "g3"]


def test_write_parquet_single_row_group(tmp_path):
    df = pd.DataFrame({"x": list(range(100))})
    path = tmp_path / "test.parquet"
    write_parquet(df, path)
    pf = pq.ParquetFile(path)
    assert pf.num_row_groups == 1


def test_write_parquet_byte_identical_on_rerun(tmp_path):
    df = pd.DataFrame({
        "GameId": ["g1", "g2"],
        "value": [10.5, 20.5],
        "code": pd.array([0, 1], dtype="int32"),
    })
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    write_parquet(df, p1)
    write_parquet(df, p2)
    assert p1.read_bytes() == p2.read_bytes()


def test_write_vocab_sorted_keys_and_trailing_newline(tmp_path):
    vocab = build_vocabulary({
        "colors": ["red", "blue"],
        "animals": ["cat", "dog"],
    })
    path = tmp_path / "vocab.json"
    write_vocab(vocab, path)
    content = path.read_text(encoding="utf-8")
    # Trailing newline (FE-VOC-05).
    assert content.endswith("\n")
    # Sorted keys: 'animals' before 'colors'.
    assert content.index("animals") < content.index("colors")

    payload = json.loads(content)
    assert payload == {"animals": ["cat", "dog"], "colors": ["blue", "red"]}


def test_write_vocab_byte_identical_on_rerun(tmp_path):
    vocab = build_vocabulary({"colors": ["red", "blue", "green"]})
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    write_vocab(vocab, p1)
    write_vocab(vocab, p2)
    assert p1.read_bytes() == p2.read_bytes()
