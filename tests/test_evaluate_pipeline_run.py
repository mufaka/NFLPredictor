"""EV-TEST-09: pinned-identity test against the real Phase 4 outputs.

Skips when ``Data/processed/predictions/`` is empty — the heavy real-data
run is expected post-Phase 4 on the CUDA training machine, not in CPU CI.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pytest

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
REAL_PROCESSED = REPO_ROOT / "Data" / "processed"
REAL_RAW = REPO_ROOT / "Data" / "raw"
REAL_PREDICTIONS_DIR = REAL_PROCESSED / PHASE4_PREDICTIONS_DIRNAME


def _skip_if_no_real_phase4() -> None:
    if not REAL_PREDICTIONS_DIR.exists() or not any(REAL_PREDICTIONS_DIR.glob("*.parquet")):
        pytest.skip(
            "Data/processed/predictions/ is empty; the real Phase 4 run is "
            "expected post-train on the CUDA machine."
        )


def _stage_real(working: pathlib.Path) -> None:
    """Copy Phase 2 + Phase 3 + Phase 4 real artifacts into ``working``."""
    working.mkdir(parents=True, exist_ok=True)
    for name in (
        PHASE2_FEATURES_FLAT_BASENAME,
        PHASE2_FEATURES_POS_BASENAME,
        PHASE2_VOCAB_BASENAME,
        PHASE2_MANIFEST_BASENAME,
        PHASE3_SPLITS_BASENAME,
        PHASE3_MANIFEST_BASENAME,
        PHASE4_MANIFEST_BASENAME,
    ):
        shutil.copy2(REAL_PROCESSED / name, working / name)
    pred_dst = working / PHASE4_PREDICTIONS_DIRNAME
    pred_dst.mkdir()
    for p in sorted(REAL_PREDICTIONS_DIR.glob("*.parquet")):
        shutil.copy2(p, pred_dst / p.name)


def test_real_phase4_build_emits_expected_surface(tmp_path: pathlib.Path) -> None:
    _skip_if_no_real_phase4()
    processed = tmp_path / "processed"
    raw = tmp_path / "raw"
    _stage_real(processed)
    raw.mkdir()
    shutil.copy2(
        REAL_RAW / EVALUATION_CONFIG_BASENAME, raw / EVALUATION_CONFIG_BASENAME
    )

    run_evaluation_build(raw, processed, repo_dir=REPO_ROOT)

    eval_dir = processed / EVALUATION_DIRNAME
    headline = json.loads((eval_dir / METRICS_HEADLINE_BASENAME).read_text())
    training_manifest = json.loads((processed / PHASE4_MANIFEST_BASENAME).read_text())
    expected_combos = sorted(
        k[len(f"{PHASE4_PREDICTIONS_DIRNAME}/") : -len(".parquet")]
        for k in training_manifest["output_sha256"].keys()
        if k.startswith(f"{PHASE4_PREDICTIONS_DIRNAME}/")
    )

    # (a) every combination in training_manifest.json appears in metrics_headline.json
    assert sorted(headline["combinations"].keys()) == expected_combos

    # (b) every S1 combo has both val and test entries.
    for combo_id, per_combo in headline["combinations"].items():
        if combo_id.endswith("__s1"):
            assert set(per_combo.keys()) == {"val", "test"}, combo_id

    # (c) every S3 combo has fold_0..fold_<n-1> and pooled.
    fold_count = None
    for combo_id, per_combo in headline["combinations"].items():
        if not combo_id.endswith("__s3"):
            continue
        keys = set(per_combo.keys())
        assert "pooled" in keys, combo_id
        fold_keys = {k for k in keys if k.startswith("fold_")}
        if fold_count is None:
            fold_count = len(fold_keys)
        else:
            assert len(fold_keys) == fold_count, combo_id
        # fold_<i> indices contiguous from 0.
        indices = sorted(int(k.split("_", 1)[1]) for k in fold_keys)
        assert indices == list(range(len(indices))), combo_id

    # (d) every enabled breakdown parquet exists and covers every combination.
    breakdowns = sorted((eval_dir / BREAKDOWNS_DIRNAME).glob("*.parquet"))
    assert {p.name for p in breakdowns} == {
        "by_team.parquet", "by_week.parquet", "by_home_away.parquet",
        "by_surface.parquet", "by_roof.parquet",
    }
    import pyarrow.parquet as pq
    for p in breakdowns:
        df = pq.read_table(p, columns=["combination_id"]).to_pandas()
        assert sorted(df["combination_id"].unique()) == expected_combos, p.name

    # (e) manifest's output_sha256 enumerates every on-disk output file.
    manifest = json.loads((eval_dir / EVALUATION_MANIFEST_BASENAME).read_text())
    on_disk = set()
    on_disk.add(METRICS_HEADLINE_BASENAME)
    for p in breakdowns:
        on_disk.add(f"{BREAKDOWNS_DIRNAME}/{p.name}")
    for p in sorted((eval_dir / PLOTS_DIRNAME).glob("*.png")):
        on_disk.add(f"{PLOTS_DIRNAME}/{p.name}")
    assert set(manifest["output_sha256"].keys()) == on_disk
