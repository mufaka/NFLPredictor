"""EV-TEST-06: full-build integration test against the synthetic fixture.

Runs ``run_evaluation_build`` against the checked-in Phase 5 fixture into a
temp dir and asserts byte-equality of the metric tables + manifest (modulo
``build_timestamp_utc`` and ``git_commit``) against the expected snapshot.
PNG existence is asserted but byte-identity is not (EV-NF-02).
"""

from __future__ import annotations

import json
import pathlib
import shutil

from nflpredictor.evaluate.outputs import (
    BREAKDOWNS_DIRNAME,
    EVALUATION_DIRNAME,
    EVALUATION_MANIFEST_BASENAME,
    METRICS_HEADLINE_BASENAME,
    PLOTS_DIRNAME,
)
from nflpredictor.evaluate.pipeline import (
    EVALUATION_CONFIG_BASENAME,
    run_evaluation_build,
)
from nflpredictor.evaluate.sources import (
    PHASE2_FEATURES_FLAT_BASENAME,
    PHASE2_FEATURES_POS_BASENAME,
    PHASE2_MANIFEST_BASENAME,
    PHASE2_VOCAB_BASENAME,
    PHASE3_MANIFEST_BASENAME,
    PHASE3_SPLITS_BASENAME,
    PHASE4_MANIFEST_BASENAME,
    PHASE4_PREDICTIONS_DIRNAME,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "evaluate"
EXPECTED = FIXTURE / "expected"


def _stage_processed(working: pathlib.Path) -> None:
    """Lay every Phase 2 / Phase 3 / Phase 4 fixture input into one processed dir."""
    working.mkdir(parents=True, exist_ok=True)
    for name in (
        PHASE2_FEATURES_FLAT_BASENAME,
        PHASE2_FEATURES_POS_BASENAME,
        PHASE2_VOCAB_BASENAME,
        PHASE2_MANIFEST_BASENAME,
    ):
        shutil.copy2(FIXTURE / "raw_phase2" / name, working / name)
    for name in (PHASE3_SPLITS_BASENAME, PHASE3_MANIFEST_BASENAME):
        shutil.copy2(FIXTURE / "raw_phase3" / name, working / name)
    shutil.copy2(
        FIXTURE / "raw_phase4" / PHASE4_MANIFEST_BASENAME,
        working / PHASE4_MANIFEST_BASENAME,
    )
    pred_dst = working / PHASE4_PREDICTIONS_DIRNAME
    pred_dst.mkdir()
    for p in sorted((FIXTURE / "raw_phase4" / PHASE4_PREDICTIONS_DIRNAME).glob("*.parquet")):
        shutil.copy2(p, pred_dst / p.name)


def _stage_raw(raw_dir: pathlib.Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        FIXTURE / "raw" / EVALUATION_CONFIG_BASENAME,
        raw_dir / EVALUATION_CONFIG_BASENAME,
    )


def _blank_volatile_fields(manifest: dict) -> dict:
    """Return a copy with build_timestamp_utc + git_commit blanked."""
    out = dict(manifest)
    out["build_timestamp_utc"] = "FIXTURE_TIMESTAMP"
    out["git_commit"] = "FIXTURE_GIT_COMMIT"
    return out


def test_integration_byte_identical_to_expected(tmp_path: pathlib.Path) -> None:
    processed = tmp_path / "processed"
    raw = tmp_path / "raw"
    _stage_processed(processed)
    _stage_raw(raw)

    run_evaluation_build(raw, processed, repo_dir=tmp_path)

    eval_dir = processed / EVALUATION_DIRNAME

    # Headline JSON byte-identical to expected.
    assert (eval_dir / METRICS_HEADLINE_BASENAME).read_bytes() == (
        EXPECTED / METRICS_HEADLINE_BASENAME
    ).read_bytes()

    # Every expected breakdown parquet byte-identical.
    expected_breakdowns = sorted((EXPECTED / BREAKDOWNS_DIRNAME).glob("*.parquet"))
    actual_breakdowns = sorted((eval_dir / BREAKDOWNS_DIRNAME).glob("*.parquet"))
    assert [p.name for p in actual_breakdowns] == [p.name for p in expected_breakdowns]
    for actual, expected in zip(actual_breakdowns, expected_breakdowns):
        assert actual.read_bytes() == expected.read_bytes(), f"divergent: {actual.name}"

    # Manifest content equal modulo timestamp + git_commit (EV-MAN-04, EV-NF-06).
    actual_manifest = json.loads((eval_dir / EVALUATION_MANIFEST_BASENAME).read_text())
    expected_manifest = json.loads((EXPECTED / EVALUATION_MANIFEST_BASENAME).read_text())
    assert _blank_volatile_fields(actual_manifest) == _blank_volatile_fields(expected_manifest)

    # PNG existence (byte-equality is NOT asserted here; that's the determinism
    # test's job and it lives within a single matplotlib wheel — EV-NF-02).
    plots = sorted((eval_dir / PLOTS_DIRNAME).glob("*.png"))
    assert len(plots) > 0
    for p in plots:
        # Every PNG is non-empty and starts with the magic byte sequence.
        data = p.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n"), f"not a PNG: {p.name}"
        assert len(data) > 0
