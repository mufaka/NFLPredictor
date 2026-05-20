"""Deterministic player-name normalization (DB-NORM-01..05).

Normalization runs only when comparing names; the original raw string is
preserved everywhere it is stored. Bumping :data:`NORMALIZATION_VERSION`
invalidates any cached match results.
"""

from __future__ import annotations

import re
import unicodedata


NORMALIZATION_VERSION = "v2"

_SUFFIX_PATTERN = re.compile(
    r"\s+(?:jr|sr|ii|iii|iv)\.?$",
    re.IGNORECASE,
)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_name(raw: str) -> str:
    """Normalize a player name for cross-source comparison.

    Steps (DB-NORM-03):
    1. Strip trailing suffix tokens (``Jr.``, ``Sr.``, ``II``, ``III``, ``IV``;
       case-insensitive; trailing period optional).
    2. Unicode-fold via NFKD and drop combining marks (accents -> ASCII).
    3. Collapse internal whitespace runs to a single space.
    4. Trim leading/trailing whitespace.
    5. Lowercase.

    The original input is never mutated; only the returned string is used
    for comparison. The raw form remains the canonical display value.
    """
    name = _SUFFIX_PATTERN.sub("", raw)
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    collapsed = _WHITESPACE_PATTERN.sub(" ", ascii_only)
    return collapsed.strip().lower()
