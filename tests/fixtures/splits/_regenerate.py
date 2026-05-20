"""Regenerate the tiny Phase 3 integration-test fixture (multi-season).

Run from the repo root with the venv active:

    python -m tests.fixtures.splits._regenerate

The script slices the real Phase 2 ``features_flat_all.parquet`` to a
small subset — a few games per season, all six seasons — keeping only
the columns Phase 3 reads (``GameId``, ``season``). The slice is sampled
deterministically by lexicographic GameId order. A stub
``feature_manifest.json`` is generated whose ``output_sha256`` entry for
the slice equals its actual SHA so ``verify_phase2_outputs`` accepts the
fixture. A fixture ``splits_config.yaml`` enabling **both** strategies is
written so the snapshot covers season_holdout and loso_cv, and the split
build runs into ``expected/`` with the manifest timestamp blanked.

Outputs:
    tests/fixtures/splits/raw_phase2/
        features_flat_all.parquet
        feature_manifest.json
    tests/fixtures/splits/raw/
        splits_config.yaml
    tests/fixtures/splits/expected/
        splits_all.json
        splits_manifest.json (timestamp + git_commit blanked)
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from nflpredictor.splits.pipeline import (
    PHASE2_FEATURES_FLAT_BASENAME,
    PHASE2_MANIFEST_BASENAME,
    SPLITS_ARTIFACT_BASENAME,
    SPLITS_CONFIG_BASENAME,
    SPLITS_MANIFEST_BASENAME,
    run_split_build,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
REAL_PROCESSED = REPO_ROOT / "Data" / "processed"
FIXTURE_DIR = pathlib.Path(__file__).parent
RAW_PHASE2 = FIXTURE_DIR / "raw_phase2"
RAW_SPLITS = FIXTURE_DIR / "raw"
EXPECTED = FIXTURE_DIR / "expected"

GAMES_PER_SEASON = 4  # 4 × 6 seasons = 24 games — small enough to check in.

FIXTURE_CONFIG = """\
splits_version: "v2"

strategies: [season_holdout, loso_cv]

train_seasons: [2020, 2021, 2022, 2023]
val_season:  2024
test_season: 2025
"""


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture() -> None:
    src_path = REAL_PROCESSED / PHASE2_FEATURES_FLAT_BASENAME
    df = pq.read_table(src_path, columns=["GameId", "season"]).to_pandas()

    df_sorted = df.sort_values(["season", "GameId"], kind="stable")
    slice_df = (
        df_sorted.groupby("season", group_keys=False)
        .head(GAMES_PER_SEASON)
        .reset_index(drop=True)
    )

    expected_count = GAMES_PER_SEASON * 6
    if len(slice_df) != expected_count:
        raise SystemExit(
            f"expected {expected_count} games in slice; got {len(slice_df)}"
        )

    for d in (RAW_PHASE2, RAW_SPLITS, EXPECTED):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    slice_path = RAW_PHASE2 / PHASE2_FEATURES_FLAT_BASENAME
    pq.write_table(pa.Table.from_pandas(slice_df, preserve_index=False), slice_path)

    fake_phase2_manifest = {
        "build_timestamp_utc": "fixture",
        "git_commit": "fixture-phase2",
        "normalization_version": "v2",
        "output_sha256": {
            f"Data/processed/{PHASE2_FEATURES_FLAT_BASENAME}": _sha256(slice_path),
        },
    }
    manifest_path = RAW_PHASE2 / PHASE2_MANIFEST_BASENAME
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(fake_phase2_manifest, f, sort_keys=True, indent=2)
        f.write("\n")

    (RAW_SPLITS / SPLITS_CONFIG_BASENAME).write_text(FIXTURE_CONFIG, encoding="utf-8")

    working = FIXTURE_DIR / "_working"
    if working.exists():
        shutil.rmtree(working)
    working.mkdir()
    for src in (slice_path, manifest_path):
        shutil.copy2(src, working / src.name)

    run_split_build(RAW_SPLITS, working, repo_dir=REPO_ROOT)

    shutil.copy2(working / SPLITS_ARTIFACT_BASENAME, EXPECTED / SPLITS_ARTIFACT_BASENAME)

    splits_manifest = json.loads((working / SPLITS_MANIFEST_BASENAME).read_text())
    splits_manifest["build_timestamp_utc"] = "FIXTURE_TIMESTAMP"
    splits_manifest["git_commit"] = "FIXTURE_GIT_COMMIT"
    with (EXPECTED / SPLITS_MANIFEST_BASENAME).open("w", encoding="utf-8") as f:
        json.dump(splits_manifest, f, sort_keys=True, indent=2)
        f.write("\n")

    shutil.rmtree(working)

    print(
        "Regenerated Phase 3 fixture:",
        *(p.relative_to(REPO_ROOT) for p in sorted(FIXTURE_DIR.rglob("*")) if p.is_file()),
        sep="\n  ",
        file=sys.stderr,
    )


if __name__ == "__main__":
    build_fixture()
