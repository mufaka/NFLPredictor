"""EV-TEST-07: two consecutive runs produce byte-identical outputs.

Asserts:
- ``metrics_headline.json`` byte-equal across runs.
- Every breakdown parquet byte-equal across runs.
- Manifest content byte-equal modulo ``build_timestamp_utc`` (EV-NF-06).
- Every PNG byte-equal within the pinned matplotlib wheel on the same machine
  (EV-NF-02). If raw byte-equality flakes, falls back to pixel-equality after
  a font-cache warm-up render — the relaxation documented in plan §4.4.
"""

from __future__ import annotations

import json
import pathlib

from nflpredictor.evaluate.outputs import (
    BREAKDOWNS_DIRNAME,
    EVALUATION_DIRNAME,
    EVALUATION_MANIFEST_BASENAME,
    METRICS_HEADLINE_BASENAME,
    PLOTS_DIRNAME,
)
from nflpredictor.evaluate.pipeline import run_evaluation_build
from nflpredictor.evaluate.plots import warmup_render

from .test_evaluate_integration import _stage_processed, _stage_raw


def _run_to(temp: pathlib.Path) -> pathlib.Path:
    processed = temp / "processed"
    raw = temp / "raw"
    _stage_processed(processed)
    _stage_raw(raw)
    run_evaluation_build(raw, processed, repo_dir=temp)
    return processed / EVALUATION_DIRNAME


def test_two_runs_byte_identical(tmp_path: pathlib.Path) -> None:
    warmup_render()  # prime matplotlib's font cache before either run.

    a_root = tmp_path / "run_a"
    b_root = tmp_path / "run_b"
    a_dir = _run_to(a_root)
    b_dir = _run_to(b_root)

    # Metric tables — byte identical, no caveats.
    assert (a_dir / METRICS_HEADLINE_BASENAME).read_bytes() == (
        b_dir / METRICS_HEADLINE_BASENAME
    ).read_bytes()
    a_breakdowns = sorted((a_dir / BREAKDOWNS_DIRNAME).glob("*.parquet"))
    b_breakdowns = sorted((b_dir / BREAKDOWNS_DIRNAME).glob("*.parquet"))
    assert [p.name for p in a_breakdowns] == [p.name for p in b_breakdowns]
    for a, b in zip(a_breakdowns, b_breakdowns):
        assert a.read_bytes() == b.read_bytes()

    # Manifest — content equal modulo build_timestamp_utc.
    a_mf = json.loads((a_dir / EVALUATION_MANIFEST_BASENAME).read_text())
    b_mf = json.loads((b_dir / EVALUATION_MANIFEST_BASENAME).read_text())
    a_mf.pop("build_timestamp_utc")
    b_mf.pop("build_timestamp_utc")
    assert a_mf == b_mf

    # PNG byte-identity within the pinned matplotlib wheel.
    a_plots = sorted((a_dir / PLOTS_DIRNAME).glob("*.png"))
    b_plots = sorted((b_dir / PLOTS_DIRNAME).glob("*.png"))
    assert [p.name for p in a_plots] == [p.name for p in b_plots]
    for a, b in zip(a_plots, b_plots):
        if a.read_bytes() == b.read_bytes():
            continue
        # Plan §4.4 fallback: pixel-equality after warm-up.
        from PIL import Image
        with Image.open(a) as ia, Image.open(b) as ib:
            assert ia.tobytes() == ib.tobytes(), (
                f"PNG {a.name} differs across runs in bytes and pixels — "
                "non-deterministic render"
            )
