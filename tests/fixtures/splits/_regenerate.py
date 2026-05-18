"""Regenerate the tiny Phase 3 integration-test fixture.

Run from the repo root with the venv active:

    python -m tests.fixtures.splits._regenerate

The script slices the real Phase 2 ``features_flat_2024.parquet`` to a
small subset — two games per NFL week (36 games total) — keeping only
the columns Phase 3 reads (``GameId``, ``week``). The slice is sampled
deterministically by lexicographic GameId order so the regeneration is
itself reproducible. A stub ``feature_manifest.json`` is generated
whose ``output_sha256`` entry for the slice equals its actual SHA so
``verify_phase2_outputs`` accepts the fixture. The shipped
``splits_config.yaml`` is mirrored verbatim, and the split build runs
into ``expected/`` with the manifest timestamp blanked.

The fixture is pinned to the Phase 2 parquet bytes — bump it whenever
Phase 2 outputs change.

Outputs:
    tests/fixtures/splits/raw_phase2/
        features_flat_2024.parquet
        feature_manifest.json
    tests/fixtures/splits/raw/
        splits_config.yaml
    tests/fixtures/splits/expected/
        splits_2024.json
        splits_manifest.json (timestamp + git_commit blanked)
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys

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
REAL_RAW = REPO_ROOT / "Data" / "raw"
FIXTURE_DIR = pathlib.Path(__file__).parent
RAW_PHASE2 = FIXTURE_DIR / "raw_phase2"
RAW_SPLITS = FIXTURE_DIR / "raw"
EXPECTED = FIXTURE_DIR / "expected"

GAMES_PER_WEEK = 2  # 2 × 18 weeks = 36 games — small enough to check in.


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture() -> None:
    # 1. Read the real Phase 2 parquet — only the columns Phase 3 reads.
    src_path = REAL_PROCESSED / PHASE2_FEATURES_FLAT_BASENAME
    df = pq.read_table(src_path, columns=["GameId", "week"]).to_pandas()

    # 2. Pick the first GAMES_PER_WEEK GameIds (lex order) per week.
    df_sorted = df.sort_values(["week", "GameId"], kind="stable")
    slice_df = (
        df_sorted.groupby("week", group_keys=False)
        .head(GAMES_PER_WEEK)
        .reset_index(drop=True)
    )

    expected_count = GAMES_PER_WEEK * 18
    if len(slice_df) != expected_count:
        raise SystemExit(
            f"expected {expected_count} games in slice; got {len(slice_df)} "
            f"(missing coverage for some weeks?)"
        )

    # 3. Wipe + repopulate fixture dirs.
    for d in (RAW_PHASE2, RAW_SPLITS, EXPECTED):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # 4. Write the sliced parquet.
    import pyarrow as pa

    slice_path = RAW_PHASE2 / PHASE2_FEATURES_FLAT_BASENAME
    pq.write_table(pa.Table.from_pandas(slice_df, preserve_index=False), slice_path)

    # 5. Stub feature_manifest.json with the slice's SHA so the hash gate accepts it.
    fake_phase2_manifest = {
        "build_timestamp_utc": "fixture",
        "git_commit": "fixture-phase2",
        "normalization_version": "v1",
        "output_sha256": {
            f"Data/processed/{PHASE2_FEATURES_FLAT_BASENAME}": _sha256(slice_path),
        },
    }
    manifest_path = RAW_PHASE2 / PHASE2_MANIFEST_BASENAME
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(fake_phase2_manifest, f, sort_keys=True, indent=2)
        f.write("\n")

    # 6. Mirror the shipped splits_config.yaml verbatim.
    shutil.copy2(
        REAL_RAW / SPLITS_CONFIG_BASENAME, RAW_SPLITS / SPLITS_CONFIG_BASENAME
    )

    # 7. Run the split build into a working dir; copy outputs to expected/.
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
