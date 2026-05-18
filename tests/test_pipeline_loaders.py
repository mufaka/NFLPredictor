"""Lightweight tests for the raw-input loaders (DB-IN-01, DB-IN-03..05)."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from nflpredictor.databuild.pipeline import (
    EXPECTED_BOX_SCORES_HEADER,
    EXPECTED_MADDEN_HEADER,
    load_raw_box_scores,
    load_raw_inputs,
    load_raw_madden,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "Data" / "raw"


def test_expected_header_shapes():
    assert len(EXPECTED_BOX_SCORES_HEADER) == 164
    assert len(EXPECTED_MADDEN_HEADER) == 69


def test_load_raw_box_scores_real():
    df = load_raw_box_scores(RAW_DIR / "box_scores_2024.csv")
    assert len(df) == 272
    assert tuple(df.columns) == EXPECTED_BOX_SCORES_HEADER


def test_load_raw_madden_real_strips_header_whitespace():
    df = load_raw_madden(RAW_DIR / "maddennfl24fullplayerratings.csv")
    assert len(df) == 2368
    assert tuple(df.columns) == EXPECTED_MADDEN_HEADER
    assert "Total Salary" in df.columns
    assert " Total Salary " not in df.columns


def test_load_raw_inputs_returns_pair():
    box_scores, madden = load_raw_inputs(RAW_DIR)
    assert len(box_scores) == 272
    assert len(madden) == 2368


def test_box_scores_header_mismatch_raises(tmp_path: pathlib.Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("col1,col2\na,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="box_scores header mismatch"):
        load_raw_box_scores(bad)


def test_madden_header_mismatch_raises(tmp_path: pathlib.Path):
    bad = tmp_path / "bad.csv"
    bad.write_text("col1,col2\na,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Madden header mismatch"):
        load_raw_madden(bad)


def test_real_madden_assigns_2368_ids():
    """End-to-end smoke: load real Madden, assign IDs, confirm count and range."""
    from nflpredictor.databuild.ids import assign_raw_madden_ids

    madden = load_raw_madden(RAW_DIR / "maddennfl24fullplayerratings.csv")
    with_ids = assign_raw_madden_ids(madden)
    assert with_ids.iloc[0]["madden_id"] == "2024-00001"
    assert with_ids.iloc[-1]["madden_id"] == "2024-02368"
    assert with_ids["madden_id"].is_unique
