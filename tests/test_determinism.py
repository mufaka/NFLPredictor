"""DB-NF-01: byte-identical reruns on identical inputs."""

from __future__ import annotations

import json
import pathlib

import pytest

from nflpredictor.databuild.pipeline import run_build


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "Data" / "raw"


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    a = tmp_path_factory.mktemp("a")
    b = tmp_path_factory.mktemp("b")
    run_build(RAW_DIR, a)
    run_build(RAW_DIR, b)
    return a, b


@pytest.mark.parametrize(
    "filename",
    ["madden_2024.csv", "box_scores_2024.csv", "player_id_mapping.csv"],
)
def test_csv_outputs_byte_identical(two_runs, filename: str):
    a, b = two_runs
    assert (a / filename).read_bytes() == (b / filename).read_bytes()


def test_manifest_identical_modulo_timestamp(two_runs):
    a, b = two_runs
    ma = json.loads((a / "build_manifest.json").read_text(encoding="utf-8"))
    mb = json.loads((b / "build_manifest.json").read_text(encoding="utf-8"))
    # DB-NF-04: only the build_timestamp_utc may differ between runs.
    ma.pop("build_timestamp_utc")
    mb.pop("build_timestamp_utc")
    assert ma == mb
    # Output hashes inside the manifests also match — confirms the file bytes
    # really are identical from the build's perspective as well.
    assert ma["output_sha256"] == mb["output_sha256"]
