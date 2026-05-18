from __future__ import annotations

import datetime as _dt
import json
import pathlib

from nflpredictor.databuild.manifest import (
    build_manifest,
    compute_sha256,
    try_get_git_commit,
    utc_timestamp,
    write_manifest,
)


def test_compute_sha256_matches_known_value(tmp_path: pathlib.Path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"hello\n")
    # echo -n 'hello\n' | sha256sum -> 5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03
    assert compute_sha256(f) == (
        "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    )


def test_utc_timestamp_iso8601_z():
    ts = utc_timestamp(_dt.datetime(2026, 5, 17, 14, 32, 9, tzinfo=_dt.timezone.utc))
    assert ts == "2026-05-17T14:32:09Z"


def test_try_get_git_commit_outside_repo(tmp_path: pathlib.Path):
    """Outside a git repo, the helper returns None (DB-MAN-01: nullable)."""
    assert try_get_git_commit(tmp_path) is None


def test_build_manifest_shape(tmp_path: pathlib.Path):
    src = tmp_path / "src.csv"
    src.write_text("a\n1\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    out.write_text("b\n2\n", encoding="utf-8")
    manifest = build_manifest(
        raw_inputs={"raw/src.csv": src},
        outputs={"processed/out.csv": out},
        counts={
            "raw_madden_rows": 10,
            "unmatched_appended_rows": 2,
            "total_madden_rows_processed": 12,
            "box_score_games": 1,
            "total_starter_slots": 44,
            "tier1_matches": 0,
            "tier2_matches": 40,
            "tier3_matches": 2,
            "tier4_matches": 0,
            "unmatched_players_unique": 2,
            "position_mismatches_logged": 1,
        },
        repo_dir=tmp_path,
        now=_dt.datetime(2026, 5, 17, 0, 0, 0, tzinfo=_dt.timezone.utc),
    )
    assert manifest["build_timestamp_utc"] == "2026-05-17T00:00:00Z"
    assert manifest["normalization_version"] == "v1"
    assert set(manifest["source_sha256"].keys()) == {"raw/src.csv"}
    assert set(manifest["output_sha256"].keys()) == {"processed/out.csv"}
    assert manifest["git_commit"] is None  # tmp_path is not a git repo


def test_write_manifest_is_sorted_indented_with_trailing_newline(tmp_path: pathlib.Path):
    path = tmp_path / "m.json"
    write_manifest({"b": 2, "a": 1, "c": {"y": 2, "x": 1}}, path)
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    parsed = json.loads(raw)
    assert list(parsed.keys()) == ["a", "b", "c"]
    assert list(parsed["c"].keys()) == ["x", "y"]
    # Indent of 2 (DB-MAN-04).
    assert '"a": 1' in raw
    assert raw.splitlines()[1].startswith("  ")
