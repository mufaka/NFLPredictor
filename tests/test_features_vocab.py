"""Tests for the vocabulary builder + integer encoder (FE-VOC-01..06, FE-TEST-10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nflpredictor.features.vocab import (
    NULL_SENTINEL,
    Vocabulary,
    build_vocabulary,
    encode_column,
)


def test_entries_sorted_lexicographically():
    vocab = build_vocabulary({"colors": ["red", "blue", "green"]})
    assert vocab.entries == {"colors": ("blue", "green", "red")}
    assert vocab.code("colors", "blue") == 0
    assert vocab.code("colors", "green") == 1
    assert vocab.code("colors", "red") == 2


def test_duplicates_collapsed_and_nulls_dropped():
    vocab = build_vocabulary({
        "colors": ["red", "red", "blue", "", None, "blue"],
    })
    assert vocab.entries == {"colors": ("blue", "red")}
    assert vocab.size("colors") == 2


def test_null_sentinel_for_missing_value():
    vocab = build_vocabulary({"colors": ["red", "blue"]})
    assert vocab.code("colors", None) == NULL_SENTINEL
    assert vocab.code("colors", "") == NULL_SENTINEL
    assert NULL_SENTINEL == -1


def test_unknown_value_raises():
    vocab = build_vocabulary({"colors": ["red", "blue"]})
    with pytest.raises(KeyError, match="purple"):
        vocab.code("colors", "purple")


def test_unknown_key_raises():
    vocab = build_vocabulary({"colors": ["red"]})
    with pytest.raises(KeyError):
        vocab.code("animals", "cat")


def test_decode_round_trip():
    vocab = build_vocabulary({"colors": ["red", "blue", "green"]})
    for value in ("red", "blue", "green"):
        code = vocab.code("colors", value)
        assert vocab.decode("colors", code) == value
    assert vocab.decode("colors", NULL_SENTINEL) is None


def test_build_is_deterministic_across_calls():
    # Same observations, possibly in different orders, must produce identical entries.
    a = build_vocabulary({"colors": ["red", "blue", "green"]})
    b = build_vocabulary({"colors": ["green", "red", "blue"]})
    assert a.entries == b.entries


def test_multiple_keys_sorted_independently():
    vocab = build_vocabulary({
        "z_letters": ["c", "a", "b"],
        "a_numbers": ["3", "1", "2"],
    })
    assert vocab.entries["z_letters"] == ("a", "b", "c")
    assert vocab.entries["a_numbers"] == ("1", "2", "3")


def test_sizes_excludes_sentinel():
    vocab = build_vocabulary({"colors": ["red", "blue", None, ""]})
    assert vocab.sizes() == {"colors": 2}


def test_encode_column_basic():
    vocab = build_vocabulary({"colors": ["red", "blue", "green"]})
    s = pd.Series(["red", None, "blue", "", "green"])
    encoded = encode_column(s, "colors", vocab)
    assert encoded.dtype == np.int32
    assert encoded.tolist() == [2, NULL_SENTINEL, 0, NULL_SENTINEL, 1]


def test_encode_column_handles_nan():
    vocab = build_vocabulary({"colors": ["red", "blue"]})
    s = pd.Series(["red", float("nan"), "blue"])
    encoded = encode_column(s, "colors", vocab)
    assert encoded.tolist() == [1, NULL_SENTINEL, 0]


def test_encode_column_raises_on_unknown():
    vocab = build_vocabulary({"colors": ["red", "blue"]})
    s = pd.Series(["red", "purple"])
    with pytest.raises(KeyError, match="purple"):
        encode_column(s, "colors", vocab)


def test_empty_observations_produce_empty_entry():
    vocab = build_vocabulary({"empty_key": []})
    assert vocab.entries == {"empty_key": ()}
    assert vocab.size("empty_key") == 0
    assert vocab.code("empty_key", None) == NULL_SENTINEL
