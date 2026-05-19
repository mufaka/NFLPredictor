"""Regenerate the tiny Phase 4 integration-test fixture.

Run from the repo root with the venv active:

    python -m tests.fixtures.train._regenerate

The script slices the real Phase 2 outputs to a small subset — two games per
NFL week (36 games total) — and rebuilds the chain:

    sliced features  →  Phase 3 run_split_build  →  splits_2024.json
                       Phase 4 run_training_build →  predictions/*.parquet
                                                    training_manifest.json

The slice is sampled deterministically by lexicographic GameId order so the
regeneration itself is reproducible. The full ``feature_vocab.json`` is
reused (sliced indices are a subset of the full integer range, all valid
under the full vocab). A trimmed ``training_config.yaml`` keeps the
integration test fast (``max_epochs=10``).

The fixture is pinned to (a) the Phase 2 parquet bytes and (b) the local
PyTorch wheel — bump it whenever Phase 2 outputs change or PyTorch is
upgraded, otherwise byte-equality tests will diverge.

Outputs:
    tests/fixtures/train/raw_phase2/
        features_flat_2024.parquet
        features_pos_2024.parquet
        feature_vocab.json
        feature_manifest.json
    tests/fixtures/train/raw_phase3/
        splits_2024.json
        splits_manifest.json
    tests/fixtures/train/raw/
        splits_config.yaml
        training_config.yaml
    tests/fixtures/train/expected/
        predictions/*.parquet × 12
        training_manifest.json (timestamp + git_commit blanked)
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from nflpredictor.splits.pipeline import (
    SPLITS_ARTIFACT_BASENAME,
    SPLITS_CONFIG_BASENAME,
    SPLITS_MANIFEST_BASENAME,
    run_split_build,
)
from nflpredictor.train.pipeline import (
    TRAINING_CONFIG_BASENAME,
    TRAINING_MANIFEST_BASENAME,
    run_training_build,
)
from nflpredictor.train.sources import (
    PHASE2_FEATURES_FLAT_BASENAME,
    PHASE2_FEATURES_POS_BASENAME,
    PHASE2_MANIFEST_BASENAME,
    PHASE2_VOCAB_BASENAME,
    PHASE3_SPLITS_BASENAME,
)
from nflpredictor.train.outputs import (
    PREDICTIONS_DIRNAME,
    TRAINING_LOSS_CURVES_BASENAME,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
REAL_PROCESSED = REPO_ROOT / "Data" / "processed"
REAL_RAW = REPO_ROOT / "Data" / "raw"

FIXTURE_DIR = pathlib.Path(__file__).parent
RAW_PHASE2 = FIXTURE_DIR / "raw_phase2"
RAW_PHASE3 = FIXTURE_DIR / "raw_phase3"
RAW_CONFIG = FIXTURE_DIR / "raw"
EXPECTED = FIXTURE_DIR / "expected"

GAMES_PER_WEEK = 2  # 2 × 18 weeks = 36 games — small enough to check in.


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pick_game_ids() -> list[str]:
    src = REAL_PROCESSED / PHASE2_FEATURES_FLAT_BASENAME
    df = pq.read_table(src, columns=["GameId", "week"]).to_pandas()
    chosen = (
        df.sort_values(["week", "GameId"], kind="mergesort")
        .groupby("week", group_keys=False)
        .head(GAMES_PER_WEEK)
        .sort_values("GameId", kind="mergesort")
    )
    return chosen["GameId"].astype(str).tolist()


def _slice_parquet(src: pathlib.Path, dst: pathlib.Path, gids: set[str]) -> None:
    df = pq.read_table(src).to_pandas()
    sub = df[df["GameId"].isin(gids)].sort_values("GameId", kind="mergesort").reset_index(drop=True)
    pq.write_table(
        pa.Table.from_pandas(sub, preserve_index=False),
        dst,
        compression="snappy",
        row_group_size=1024,
    )


def _trimmed_training_config() -> dict:
    cfg = yaml.safe_load((REAL_RAW / TRAINING_CONFIG_BASENAME).read_text())
    # Force CPU so the fixture is portable across dev machines.
    cfg["device"] = "cpu"
    cfg["linear"]["max_epochs"] = 10
    cfg["linear"]["early_stop_patience"] = 100
    cfg["mlp"]["max_epochs"] = 10
    cfg["mlp"]["early_stop_patience"] = 100
    return cfg


def build_fixture() -> None:
    # 0. Wipe + repopulate every fixture dir.
    for d in (RAW_PHASE2, RAW_PHASE3, RAW_CONFIG, EXPECTED):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # 1. Pick GameIds (36 total) and slice both feature parquets.
    gids = _pick_game_ids()
    gid_set = set(gids)
    flat_path = RAW_PHASE2 / PHASE2_FEATURES_FLAT_BASENAME
    pos_path = RAW_PHASE2 / PHASE2_FEATURES_POS_BASENAME
    _slice_parquet(REAL_PROCESSED / PHASE2_FEATURES_FLAT_BASENAME, flat_path, gid_set)
    _slice_parquet(REAL_PROCESSED / PHASE2_FEATURES_POS_BASENAME, pos_path, gid_set)

    # 2. Reuse full vocab (sliced cat values are a subset; all valid).
    vocab_path = RAW_PHASE2 / PHASE2_VOCAB_BASENAME
    shutil.copy2(REAL_PROCESSED / PHASE2_VOCAB_BASENAME, vocab_path)

    # 3. Stub feature_manifest.json with the sliced SHAs.
    feature_manifest = {
        "build_timestamp_utc": "fixture",
        "git_commit": "fixture-phase2",
        "normalization_version": "v1",
        "output_sha256": {
            f"Data/processed/{PHASE2_FEATURES_FLAT_BASENAME}": _sha256(flat_path),
            f"Data/processed/{PHASE2_FEATURES_POS_BASENAME}": _sha256(pos_path),
            f"Data/processed/{PHASE2_VOCAB_BASENAME}": _sha256(vocab_path),
        },
    }
    fmanifest_path = RAW_PHASE2 / PHASE2_MANIFEST_BASENAME
    with fmanifest_path.open("w", encoding="utf-8") as f:
        json.dump(feature_manifest, f, sort_keys=True, indent=2)
        f.write("\n")

    # 4. Mirror the shipped splits_config.yaml and build the fixture's Phase 3 outputs.
    shutil.copy2(REAL_RAW / SPLITS_CONFIG_BASENAME, RAW_CONFIG / SPLITS_CONFIG_BASENAME)
    # Run Phase 3 against the sliced features. It writes splits artifacts next to them.
    run_split_build(RAW_CONFIG, RAW_PHASE2, repo_dir=REPO_ROOT)
    # Move the Phase 3 outputs to raw_phase3/ so the train fixture has clean roles.
    for name in (SPLITS_ARTIFACT_BASENAME, SPLITS_MANIFEST_BASENAME):
        (RAW_PHASE2 / name).rename(RAW_PHASE3 / name)

    # 5. Write the trimmed training_config.yaml.
    tcfg = _trimmed_training_config()
    tcfg_path = RAW_CONFIG / TRAINING_CONFIG_BASENAME
    with tcfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(tcfg, f, sort_keys=False)

    # 6. Run Phase 4 against the staged inputs. Train needs everything in one processed dir.
    working = FIXTURE_DIR / "_working"
    if working.exists():
        shutil.rmtree(working)
    working.mkdir()
    for src in (flat_path, pos_path, vocab_path, fmanifest_path):
        shutil.copy2(src, working / src.name)
    for name in (SPLITS_ARTIFACT_BASENAME, SPLITS_MANIFEST_BASENAME):
        shutil.copy2(RAW_PHASE3 / name, working / name)
    run_training_build(RAW_CONFIG, working, repo_dir=REPO_ROOT)

    # 7. Move expected outputs (12 prediction parquets + loss-curve sidecar + manifest) into expected/.
    (EXPECTED / PREDICTIONS_DIRNAME).mkdir(parents=True)
    for p in sorted((working / PREDICTIONS_DIRNAME).glob("*.parquet")):
        shutil.copy2(p, EXPECTED / PREDICTIONS_DIRNAME / p.name)
    # Phase 6 sidecar (DD-LC-02).
    shutil.copy2(
        working / TRAINING_LOSS_CURVES_BASENAME,
        EXPECTED / TRAINING_LOSS_CURVES_BASENAME,
    )
    # Blank the build-run-specific fields in the manifest so byte-equality tests pin to the rest.
    manifest = json.loads((working / TRAINING_MANIFEST_BASENAME).read_text())
    manifest["build_timestamp_utc"] = "FIXTURE_TIMESTAMP"
    manifest["git_commit"] = "FIXTURE_GIT_COMMIT"
    with (EXPECTED / TRAINING_MANIFEST_BASENAME).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, sort_keys=True, indent=2)
        f.write("\n")

    shutil.rmtree(working)

    print(
        "Regenerated Phase 4 fixture:",
        *(p.relative_to(REPO_ROOT) for p in sorted(FIXTURE_DIR.rglob("*")) if p.is_file()),
        sep="\n  ",
        file=sys.stderr,
    )


if __name__ == "__main__":
    build_fixture()
