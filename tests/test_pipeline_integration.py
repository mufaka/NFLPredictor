"""End-to-end integration test against the tiny fixture (DB-TEST-03)."""

from __future__ import annotations

import json
import pathlib

import pytest

from nflpredictor.databuild.pipeline import run_build


FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures"
RAW_TINY = FIXTURE_DIR / "raw_tiny"
EXPECTED = FIXTURE_DIR / "expected"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("integration")
    run_build(RAW_TINY, out)
    return out


@pytest.mark.parametrize(
    "filename",
    ["madden_2024.csv", "box_scores_2024.csv", "player_id_mapping.csv"],
)
def test_csv_outputs_match_expected(built, filename: str):
    actual = (built / filename).read_bytes()
    expected = (EXPECTED / filename).read_bytes()
    assert actual == expected, f"{filename} drifted from fixture-expected output"


def test_manifest_matches_expected_modulo_timestamp_and_git(built):
    actual = json.loads((built / "build_manifest.json").read_text(encoding="utf-8"))
    expected = json.loads((EXPECTED / "build_manifest.json").read_text(encoding="utf-8"))
    # Build timestamp and git commit will differ between runs; everything else
    # is deterministic from the raw inputs.
    actual.pop("build_timestamp_utc")
    expected.pop("build_timestamp_utc")
    actual.pop("git_commit", None)
    expected.pop("git_commit", None)
    assert actual == expected


def test_counts_exercise_every_tier(built):
    """Sanity-check: the fixture is constructed to hit each tier at least once."""
    manifest = json.loads((built / "build_manifest.json").read_text(encoding="utf-8"))
    counts = manifest["counts"]
    assert counts["tier1_matches"] >= 1, "fixture should exercise tier 1 (override)"
    assert counts["tier2_matches"] >= 1, "fixture should exercise tier 2 (team+name)"
    assert counts["tier3_matches"] >= 1, "fixture should exercise tier 3 (post-trade)"
    assert counts["tier4_matches"] >= 1, "fixture should exercise tier 4 (fuzzy)"
    assert counts["unmatched_appended_rows"] >= 1, "fixture should exercise unmatched path"
    assert counts["position_mismatches_logged"] >= 1, "fixture should exercise position mismatch"
