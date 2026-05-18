"""Vocabulary builder for integer-coded categorical columns (§3.11)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd


# Sentinel returned by Vocabulary.code for missing values; never assigned to a
# vocabulary entry (FE-VOC-02).
NULL_SENTINEL: int = -1


@dataclass(frozen=True)
class Vocabulary:
    """Integer-code domain for every categorical column (FE-VOC-01..04).

    ``entries`` maps a vocab key to a lexicographically-sorted list of distinct
    string values. The integer code for a value is its 0-based index in that
    list. A vocab key may be shared by multiple columns (per FE-VOC-03 the
    ``team_codes`` key backs both ``home_team_code`` and ``away_team_code``).
    """

    entries: Mapping[str, tuple[str, ...]]

    def code(self, key: str, value: str | None) -> int:
        """Return the integer code for ``value`` under ``key``.

        Returns ``NULL_SENTINEL`` (-1) for ``None`` or empty-string inputs.
        Raises ``KeyError`` for unknown keys or values — the vocabulary is
        fully rebuilt on every run (FE-VOC-06), so an unknown value at encode
        time always indicates a builder bug.
        """
        if value is None or value == "":
            return NULL_SENTINEL
        domain = self.entries[key]
        try:
            return domain.index(value)
        except ValueError as exc:
            raise KeyError(
                f"value {value!r} is not in vocabulary[{key!r}]"
            ) from exc

    def decode(self, key: str, code: int) -> str | None:
        """Inverse of :meth:`code`; ``-1`` decodes to ``None``."""
        if code == NULL_SENTINEL:
            return None
        return self.entries[key][code]

    def size(self, key: str) -> int:
        """Number of distinct values under ``key`` (excludes the null sentinel)."""
        return len(self.entries[key])

    def sizes(self) -> dict[str, int]:
        """Per-key sizes — convenience for manifest ``vocab_sizes``."""
        return {key: len(values) for key, values in self.entries.items()}


def build_vocabulary(observations: Mapping[str, Iterable[str]]) -> Vocabulary:
    """Build a :class:`Vocabulary` from observed values per key (FE-VOC-04, FE-VOC-06).

    ``observations[key]`` is any iterable of strings observed for that key.
    Duplicates are collapsed; ``None`` and empty strings are dropped (the null
    sentinel is never a vocabulary entry). Within each key the surviving values
    are sorted lexicographically and the integer code is the 0-based position.

    The vocabulary is rebuilt from scratch on every call; no prior state is
    carried over (FE-VOC-06).
    """
    entries: dict[str, tuple[str, ...]] = {}
    for key in sorted(observations.keys()):
        distinct = {v for v in observations[key] if v is not None and v != ""}
        entries[key] = tuple(sorted(distinct))
    return Vocabulary(entries=entries)


def encode_column(values: pd.Series, key: str, vocab: Vocabulary) -> pd.Series:
    """Encode a pandas Series of strings to an ``int32`` series of codes.

    ``None``, ``NaN``, and empty-string cells map to ``NULL_SENTINEL`` (-1).
    Any other value missing from ``vocab.entries[key]`` raises ``KeyError``.
    """
    def _encode(v: object) -> int:
        if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
            return NULL_SENTINEL
        return vocab.code(key, v)  # type: ignore[arg-type]

    return values.map(_encode).astype("int32")
