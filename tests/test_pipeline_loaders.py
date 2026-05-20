"""Lightweight tests for the raw-input loaders (DB-IN-01, DB-IN-03..06)."""

from __future__ import annotations

import pathlib

import pytest

from nflpredictor.databuild.pipeline import (
    EXPECTED_BOX_SCORES_HEADER,
    EXPECTED_MADDEN_HEADER,
    discover_seasons,
    load_raw_box_scores,
    load_raw_madden,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "Data" / "raw"


def test_expected_header_shapes():
    assert len(EXPECTED_BOX_SCORES_HEADER) == 164
    # The raw Madden header is 56 columns including the source ``madden_id``.
    assert len(EXPECTED_MADDEN_HEADER) == 56
    assert EXPECTED_MADDEN_HEADER[0] == "madden_id"


def test_discover_seasons_finds_all_six():
    assert discover_seasons(RAW_DIR) == [2020, 2021, 2022, 2023, 2024, 2025]


def test_load_raw_box_scores_real():
    df = load_raw_box_scores(RAW_DIR / "box_scores_2024.csv")
    assert len(df) == 272
    assert tuple(df.columns) == EXPECTED_BOX_SCORES_HEADER


def test_load_raw_box_scores_handles_lf_and_crlf():
    # box_scores_2024.csv uses CRLF; the other seasons use LF (DB-IN-05).
    crlf = load_raw_box_scores(RAW_DIR / "box_scores_2024.csv")
    lf = load_raw_box_scores(RAW_DIR / "box_scores_2025.csv")
    assert tuple(crlf.columns) == EXPECTED_BOX_SCORES_HEADER
    assert tuple(lf.columns) == EXPECTED_BOX_SCORES_HEADER


def test_load_raw_madden_drops_source_madden_id():
    df = load_raw_madden(RAW_DIR / "madden_2024.csv")
    assert len(df) == 2316
    # The source madden_id column is dropped on read (DB-IN-06).
    assert "madden_id" not in df.columns
    assert tuple(df.columns) == EXPECTED_MADDEN_HEADER[1:]


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


def test_real_madden_assigns_per_season_ids():
    """End-to-end smoke: load real 2024 Madden, assign IDs, confirm count and range."""
    from nflpredictor.databuild.ids import assign_raw_madden_ids

    madden = load_raw_madden(RAW_DIR / "madden_2024.csv")
    with_ids = assign_raw_madden_ids(madden, 2024)
    assert with_ids.iloc[0]["madden_id"] == "2024-00001"
    assert with_ids.iloc[-1]["madden_id"] == "2024-02316"
    assert with_ids["madden_id"].is_unique
