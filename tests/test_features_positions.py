"""Tests for the canonical position taxonomy (§4.3 / FE-TEST-03)."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from nflpredictor.features.positions import (
    BUCKET_CAPACITY,
    CANONICAL_BPOS_SLOTS_PER_SIDE,
    POSITION_BUCKETS,
    bucket_for_position,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BOX_SCORES = REPO_ROOT / "Data" / "processed" / "box_scores_all.csv"


def test_every_real_position_label_maps_to_a_bucket():
    """FE-TEST-03: every box-score position observed across 2020-2025 must map."""
    df = pd.read_csv(BOX_SCORES, dtype=str, keep_default_na=False)
    observed: set[str] = set()
    for slot_kind in ("HomeOff", "HomeDef", "AwayOff", "AwayDef"):
        for i in range(1, 12):
            col = f"{slot_kind}{i:02d}_Position"
            observed.update(df[col].unique())
    observed.discard("")  # drop blank cells if any
    for label in sorted(observed):
        bucket = bucket_for_position(label)  # must not raise
        assert bucket in BUCKET_CAPACITY, f"{label!r} → unknown bucket {bucket!r}"


def test_unmapped_label_raises():
    with pytest.raises(KeyError, match="canonical taxonomy"):
        bucket_for_position("XYZ")


def test_capacity_per_side_sums_to_29():
    assert sum(BUCKET_CAPACITY.values()) == 29


def test_canonical_slots_per_side_count():
    # Per side: 1 + 2 + 4 + 3 + 5 + 5 + 4 + 5 = 29 slots.
    assert len(CANONICAL_BPOS_SLOTS_PER_SIDE) == 29


def test_canonical_slots_per_side_order():
    # Spot-check the first few and last few entries to confirm taxonomy order.
    assert CANONICAL_BPOS_SLOTS_PER_SIDE[0] == "QB1"
    assert CANONICAL_BPOS_SLOTS_PER_SIDE[1] == "RB1"
    assert CANONICAL_BPOS_SLOTS_PER_SIDE[2] == "RB2"
    assert CANONICAL_BPOS_SLOTS_PER_SIDE[3] == "WR1"
    assert CANONICAL_BPOS_SLOTS_PER_SIDE[6] == "WR4"
    assert CANONICAL_BPOS_SLOTS_PER_SIDE[7] == "TE1"
    assert CANONICAL_BPOS_SLOTS_PER_SIDE[-1] == "DB5"


def test_specific_bucket_mappings():
    # Sanity checks for tricky cases, including dual labels and the new
    # multi-season single labels (HB, ILB, SAF).
    assert bucket_for_position("FB") == "RB"
    assert bucket_for_position("HB") == "RB"
    assert bucket_for_position("OT") == "OL"
    assert bucket_for_position("OG") == "OL"
    assert bucket_for_position("C") == "OL"
    assert bucket_for_position("NT") == "DL"
    assert bucket_for_position("DE") == "DL"
    assert bucket_for_position("MLB") == "LB"
    assert bucket_for_position("ILB") == "LB"
    assert bucket_for_position("FS") == "DB"
    assert bucket_for_position("SS") == "DB"
    assert bucket_for_position("SAF") == "DB"
    # Dual labels resolve on the first /-delimited token.
    assert bucket_for_position("C/G") == "OL"
    assert bucket_for_position("FB/RB") == "RB"
    assert bucket_for_position("WR/RS") == "WR"
    assert bucket_for_position("DE/LB") == "DL"


def test_position_buckets_count():
    # Regression alarm if someone adds or removes a single-label mapping.
    assert len(POSITION_BUCKETS) == 26
