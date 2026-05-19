"""Parquet + JSON writers for the feature artifacts (§3.10, §3.11)."""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .vocab import Vocabulary


# Deterministic parquet write options (FE-OUT-01, FE-NF-08).
#
# - Snappy compression: pinned codec; deterministic given the same input.
# - use_dictionary=False: forces plain encoding so dictionary-page byte layout
#   never depends on observed value frequency.
# - write_statistics=False: parquet statistics include per-page min/max/null
#   counts that aren't strictly determined by the rows (they depend on how the
#   writer chunks pages). Disabling them keeps bytes deterministic.
# - data_page_version='1.0': pin the page format version.
PARQUET_WRITE_OPTIONS: dict[str, object] = {
    "compression": "snappy",
    "use_dictionary": False,
    "write_statistics": False,
    "data_page_version": "1.0",
    "version": "2.6",
}


def write_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    """Write ``df`` to a parquet file using the pinned options.

    Forces a single row group equal to the total row count so block boundaries
    don't drift with pandas chunking heuristics (FE-OUT-01).
    """
    table = pa.Table.from_pandas(df, preserve_index=False).combine_chunks()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        row_group_size=max(len(df), 1),
        **PARQUET_WRITE_OPTIONS,
    )


def write_vocab(
    vocab: Vocabulary,
    column_vocab_keys: dict[str, str],
    vocab_version: str,
    path: pathlib.Path,
) -> None:
    """Write the vocabulary sidecar (FE-VOC-05).

    Payload schema (``vocab_version: "v2"``):

    - ``vocab_version`` — schema tag.
    - ``entries`` — ``{vocab_key: [value, ...]}`` (sorted lexicographically per key).
    - ``column_vocab_keys`` — ``{column_name: vocab_key}`` map covering every
      categorical column across both flat and pos shapes. Phase 4's encoder
      reads this directly instead of pattern-matching column names.

    ``newline="\\n"`` keeps the file byte-identical across OSes
    (Windows text-mode would otherwise translate ``\\n`` to ``\\r\\n``).
    """
    payload = {
        "vocab_version": vocab_version,
        "entries": {key: list(values) for key, values in vocab.entries.items()},
        "column_vocab_keys": dict(column_vocab_keys),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, sort_keys=True, indent=2)
        f.write("\n")
