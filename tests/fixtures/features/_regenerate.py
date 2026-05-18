"""Regenerate the tiny Phase 2 integration-test fixture.

Run from the repo root with the venv active:

    python -m tests.fixtures.features._regenerate

The script slices Phase 1's real outputs to one game (the 2024-09-05
Chiefs/Ravens opener) and the Madden rows referenced by that game's
44 starter slots. It also generates a mini ``build_manifest.json``
whose ``output_sha256`` entries match the sliced CSVs so that
``verify_phase1_outputs`` accepts the fixture. Then it runs the
feature build against the slice and captures the resulting four
artifacts under ``expected/``.

Outputs:
    tests/fixtures/features/raw_phase1/
        box_scores_2024.csv
        madden_2024.csv
        build_manifest.json
    tests/fixtures/features/raw/
        feature_config.yaml
    tests/fixtures/features/expected/
        features_flat_2024.parquet
        features_pos_2024.parquet
        feature_vocab.json
        feature_manifest.json (timestamp blanked)

The script is run on demand, not from pytest; check in the artifacts.
The expected outputs are pyarrow-version-sensitive — bump
``normalization_version`` in ``feature_config.yaml`` and regenerate
when ``pyproject.toml``'s ``pyarrow`` pin changes.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys

import pandas as pd

from nflpredictor.features.pipeline import (
    PHASE1_BOX_SCORES_BASENAME,
    PHASE1_MADDEN_BASENAME,
    PHASE1_MANIFEST_BASENAME,
    FEATURE_MANIFEST_BASENAME,
    run_feature_build,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
REAL_PROCESSED = REPO_ROOT / "Data" / "processed"
REAL_RAW = REPO_ROOT / "Data" / "raw"
FIXTURE_DIR = pathlib.Path(__file__).parent
RAW_PHASE1 = FIXTURE_DIR / "raw_phase1"
RAW_FEATURES = FIXTURE_DIR / "raw"
EXPECTED = FIXTURE_DIR / "expected"

# The fixture slice — one game, all 44 starter slots filled.
FIXTURE_GAME_ID = "202409050kan"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def build_fixture() -> None:
    box = pd.read_csv(
        REAL_PROCESSED / PHASE1_BOX_SCORES_BASENAME,
        dtype=str,
        keep_default_na=False,
    )
    madden = pd.read_csv(
        REAL_PROCESSED / PHASE1_MADDEN_BASENAME,
        dtype=str,
        keep_default_na=False,
    )

    tiny_box = box[box["GameId"] == FIXTURE_GAME_ID].reset_index(drop=True)
    if len(tiny_box) != 1:
        raise SystemExit(f"expected exactly one row for game {FIXTURE_GAME_ID!r}")

    # Collect every madden_id referenced in the 44 starter slots.
    referenced_ids: set[str] = set()
    for unit in ("Off", "Def"):
        for i in range(1, 12):
            for side in ("Home", "Away"):
                slot = f"{side}{unit}{i:02d}"
                referenced_ids.add(str(tiny_box.iloc[0][f"{slot}_ID"]))
    referenced_ids.discard("")

    tiny_madden = madden[madden["madden_id"].isin(referenced_ids)].reset_index(drop=True)
    if len(tiny_madden) != len(referenced_ids):
        missing = referenced_ids - set(tiny_madden["madden_id"])
        raise SystemExit(f"madden lookup missing rows for: {sorted(missing)}")

    # Wipe + repopulate fixture dirs.
    for d in (RAW_PHASE1, RAW_FEATURES, EXPECTED):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    box_path = RAW_PHASE1 / PHASE1_BOX_SCORES_BASENAME
    madden_path = RAW_PHASE1 / PHASE1_MADDEN_BASENAME
    _write_csv(tiny_box, box_path)
    _write_csv(tiny_madden, madden_path)

    # Build a minimal Phase 1 manifest that verify_phase1_outputs accepts.
    fake_phase1_manifest = {
        "build_timestamp_utc": "fixture",
        "normalization_version": "v1",
        "git_commit": "fixture-phase1",
        "output_sha256": {
            f"Data/processed/{PHASE1_BOX_SCORES_BASENAME}": _sha256(box_path),
            f"Data/processed/{PHASE1_MADDEN_BASENAME}": _sha256(madden_path),
        },
    }
    manifest_path = RAW_PHASE1 / PHASE1_MANIFEST_BASENAME
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(fake_phase1_manifest, f, sort_keys=True, indent=2)
        f.write("\n")

    # Copy the feature_config.yaml as-is so the fixture tracks the shipped default.
    shutil.copy2(REAL_RAW / "feature_config.yaml", RAW_FEATURES / "feature_config.yaml")

    # Run the feature build into a working dir and capture the outputs.
    working = FIXTURE_DIR / "_working"
    if working.exists():
        shutil.rmtree(working)
    working.mkdir()
    for src in (box_path, madden_path, manifest_path):
        shutil.copy2(src, working / src.name)

    run_feature_build(RAW_FEATURES, working, repo_dir=REPO_ROOT)

    # Move the four artifacts into expected/. Blank the timestamp so the
    # checked-in copy stays diff-stable.
    for name in (
        "features_flat_2024.parquet",
        "features_pos_2024.parquet",
        "feature_vocab.json",
    ):
        shutil.copy2(working / name, EXPECTED / name)

    feature_manifest = json.loads((working / FEATURE_MANIFEST_BASENAME).read_text())
    feature_manifest["build_timestamp_utc"] = "FIXTURE_TIMESTAMP"
    with (EXPECTED / FEATURE_MANIFEST_BASENAME).open("w", encoding="utf-8") as f:
        json.dump(feature_manifest, f, sort_keys=True, indent=2)
        f.write("\n")

    shutil.rmtree(working)

    print(
        "Regenerated Phase 2 fixture:",
        *(p.relative_to(REPO_ROOT) for p in sorted(FIXTURE_DIR.rglob("*")) if p.is_file()),
        sep="\n  ",
        file=sys.stderr,
    )


if __name__ == "__main__":
    build_fixture()
