"""Regenerate the tiny Phase 2 integration-test fixture (multi-season).

Run from the repo root with the venv active:

    python -m tests.fixtures.features._regenerate

The script slices Phase 1's real combined outputs to two games — one
2024 game and one 2025 game — and the Madden rows referenced by those
games' starter slots. It generates a mini ``build_manifest.json`` whose
``output_sha256`` entries match the sliced CSVs so ``verify_phase1_outputs``
accepts the fixture, then runs the feature build against the slice and
captures the four artifacts under ``expected/``.

Outputs:
    tests/fixtures/features/raw_phase1/
        box_scores_all.csv
        madden_all.csv
        build_manifest.json
    tests/fixtures/features/raw/
        feature_config.yaml
    tests/fixtures/features/expected/
        features_flat_all.parquet
        features_pos_all.parquet
        feature_vocab.json
        feature_manifest.json (timestamp blanked)

The script is run on demand, not from pytest; check in the artifacts.
The expected outputs are pyarrow-version-sensitive — bump
``normalization_version`` in ``feature_config.yaml`` and regenerate when
``pyproject.toml``'s ``pyarrow`` pin changes.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys

import pandas as pd

from nflpredictor.features.pipeline import (
    FEATURE_MANIFEST_BASENAME,
    FLAT_PARQUET_BASENAME,
    PHASE1_BOX_SCORES_BASENAME,
    PHASE1_MADDEN_BASENAME,
    PHASE1_MANIFEST_BASENAME,
    POS_PARQUET_BASENAME,
    run_feature_build,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
REAL_PROCESSED = REPO_ROOT / "Data" / "processed"
REAL_RAW = REPO_ROOT / "Data" / "raw"
FIXTURE_DIR = pathlib.Path(__file__).parent
RAW_PHASE1 = FIXTURE_DIR / "raw_phase1"
RAW_FEATURES = FIXTURE_DIR / "raw"
EXPECTED = FIXTURE_DIR / "expected"

# One 2024 game; the 2025 game is the first 2025-season row of the file.
FIXTURE_GAME_2024 = "202409050kan"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def build_fixture() -> None:
    box = pd.read_csv(
        REAL_PROCESSED / PHASE1_BOX_SCORES_BASENAME, dtype=str, keep_default_na=False
    )
    madden = pd.read_csv(
        REAL_PROCESSED / PHASE1_MADDEN_BASENAME, dtype=str, keep_default_na=False
    )

    first_2025 = box[box["season"] == "2025"]["GameId"].iloc[0]
    game_ids = [FIXTURE_GAME_2024, first_2025]
    tiny_box = box[box["GameId"].isin(game_ids)].reset_index(drop=True)
    if len(tiny_box) != len(game_ids):
        raise SystemExit(f"expected one row each for {game_ids}")

    # Collect every madden_id referenced in the two games' starter slots.
    referenced_ids: set[str] = set()
    for _, row in tiny_box.iterrows():
        for unit in ("Off", "Def"):
            for i in range(1, 12):
                for side in ("Home", "Away"):
                    referenced_ids.add(str(row[f"{side}{unit}{i:02d}_ID"]))
    referenced_ids.discard("")

    tiny_madden = madden[madden["madden_id"].isin(referenced_ids)].reset_index(drop=True)
    if len(tiny_madden) != len(referenced_ids):
        missing = referenced_ids - set(tiny_madden["madden_id"])
        raise SystemExit(f"madden lookup missing rows for: {sorted(missing)}")

    for d in (RAW_PHASE1, RAW_FEATURES, EXPECTED):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    box_path = RAW_PHASE1 / PHASE1_BOX_SCORES_BASENAME
    madden_path = RAW_PHASE1 / PHASE1_MADDEN_BASENAME
    _write_csv(tiny_box, box_path)
    _write_csv(tiny_madden, madden_path)

    fake_phase1_manifest = {
        "build_timestamp_utc": "fixture",
        "normalization_version": "v2",
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

    shutil.copy2(REAL_RAW / "feature_config.yaml", RAW_FEATURES / "feature_config.yaml")

    working = FIXTURE_DIR / "_working"
    if working.exists():
        shutil.rmtree(working)
    working.mkdir()
    for src in (box_path, madden_path, manifest_path):
        shutil.copy2(src, working / src.name)

    run_feature_build(RAW_FEATURES, working, repo_dir=REPO_ROOT)

    for name in (FLAT_PARQUET_BASENAME, POS_PARQUET_BASENAME, "feature_vocab.json"):
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
