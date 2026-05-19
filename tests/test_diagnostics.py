"""DD-INT-04 / DD-TEST-09: unit tests for nflpredictor.diagnostics helpers.

Test strategy: the train fixture under ``tests/fixtures/train/`` already
contains the Phase 2 / Phase 3 / Phase 4 artifacts the helpers need
(features parquets, vocab, splits, predictions, training manifest, loss
curves). The Phase 1 outputs (``box_scores_2024.csv``, ``player_id_mapping.csv``)
are not in that fixture, so:
  * raw box scores come straight from ``Data/raw/box_scores_2024.csv`` — a
    checked-in repo file the train fixture's synthetic ``GameId`` codes
    were intentionally drawn from (e.g. ``202411280dal``).
  * a minimal ``player_id_mapping.csv`` is hand-written into the staged
    processed directory covering a handful of the test game's starter
    ``_ID`` codes; the helper tolerates unmapped rows gracefully.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nflpredictor.diagnostics import encoding, loss_curves, trace
from nflpredictor.diagnostics.loss_curves import LossCurvesIntegrityError


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "train"
REAL_RAW = REPO_ROOT / "Data" / "raw"

# Picked because it appears in the train fixture in:
#   * S1.val (one prediction per S1 combination)
#   * S3 fold k=12 / fold_index=6 (one prediction per S3 combination)
# So a single GameId exercises every per-combination output path.
TEST_GAME_ID = "202411280dal"


# A small, fully-deterministic mapping covering one starter from each side/unit
# of the test game. The helper tolerates unmapped rows (madden_id / note return
# None), so a partial mapping is sufficient to exercise the join.
PLAYER_MAPPING_FIXTURE = pd.DataFrame([
    {"box_score_id": "RushCo00", "madden_id": "2024-FIX01", "note": "fixture"},
    {"box_score_id": "DowdRi01", "madden_id": "2024-FIX02", "note": "fixture"},
    {"box_score_id": "GolsCh00", "madden_id": "2024-FIX03", "note": "fixture"},
    {"box_score_id": "LockDr00", "madden_id": "2024-FIX04", "note": "fixture"},
    {"box_score_id": "LawrDe03", "madden_id": "2024-FIX05", "note": "fixture"},
])


@pytest.fixture(scope="session")
def staged_processed_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Build a composite processed dir that unifies the train fixture's split layout."""
    out = tmp_path_factory.mktemp("diagnostics_processed")

    # Phase 2 outputs.
    for name in ("features_flat_2024.parquet", "features_pos_2024.parquet",
                 "feature_vocab.json", "feature_manifest.json"):
        shutil.copy2(TRAIN_FIXTURE / "raw_phase2" / name, out / name)

    # Phase 3 outputs.
    for name in ("splits_2024.json", "splits_manifest.json"):
        shutil.copy2(TRAIN_FIXTURE / "raw_phase3" / name, out / name)

    # Phase 4 outputs.
    shutil.copy2(TRAIN_FIXTURE / "expected" / "training_manifest.json",
                 out / "training_manifest.json")
    shutil.copy2(TRAIN_FIXTURE / "expected" / "training_loss_curves.parquet",
                 out / "training_loss_curves.parquet")
    pred_dst = out / "predictions"
    pred_dst.mkdir()
    for p in sorted((TRAIN_FIXTURE / "expected" / "predictions").glob("*.parquet")):
        shutil.copy2(p, pred_dst / p.name)

    # Minimal player_id_mapping.csv covering the test game's offensive
    # starters from both teams (see PLAYER_MAPPING_FIXTURE).
    PLAYER_MAPPING_FIXTURE.to_csv(out / "player_id_mapping.csv", index=False)

    return out


# ---------------------------------------------------------------------------
# trace.py
# ---------------------------------------------------------------------------


def test_load_raw_game_returns_single_row() -> None:
    row = trace.load_raw_game(TEST_GAME_ID, raw_dir=REAL_RAW)
    assert row["GameId"] == TEST_GAME_ID
    assert row["HomeTeam"] == "Dallas Cowboys"
    assert row["AwayTeam"] == "New York Giants"


def test_load_raw_game_unknown_raises() -> None:
    with pytest.raises(KeyError, match="GameId 'not-a-real-id'"):
        trace.load_raw_game("not-a-real-id", raw_dir=REAL_RAW)


def test_resolve_starters_returns_44_slots(staged_processed_dir: pathlib.Path) -> None:
    df = trace.resolve_starters(
        TEST_GAME_ID, raw_dir=REAL_RAW, processed_dir=staged_processed_dir
    )
    assert len(df) == 44
    assert list(df.columns) == [
        "slot", "side", "unit", "position", "name",
        "box_score_id", "madden_id", "note",
    ]
    # Side / unit decomposition.
    assert set(df["side"].unique()) == {"Home", "Away"}
    assert set(df["unit"].unique()) == {"Off", "Def"}
    assert (df["side"] == "Home").sum() == 22
    assert (df["unit"] == "Off").sum() == 22


def test_resolve_starters_joins_mapping(staged_processed_dir: pathlib.Path) -> None:
    df = trace.resolve_starters(
        TEST_GAME_ID, raw_dir=REAL_RAW, processed_dir=staged_processed_dir
    )
    home_qb = df[df["slot"] == "HomeOff01"].iloc[0]
    assert home_qb["position"] == "QB"
    assert home_qb["name"] == "Cooper Rush"
    assert home_qb["box_score_id"] == "RushCo00"
    assert home_qb["madden_id"] == "2024-FIX01"
    assert home_qb["note"] == "fixture"

    # A slot whose _ID is not in the fixture mapping must report None for
    # both madden_id and note (graceful missing-mapping handling).
    home_off02 = df[df["slot"] == "HomeOff02"].iloc[0]
    assert home_off02["name"] == "Rico Dowdle"
    # Rico Dowdle (DowdRi01) IS in the fixture mapping.
    assert home_off02["madden_id"] == "2024-FIX02"
    # Pick a slot definitely outside the fixture mapping.
    unmapped = df[
        ~df["box_score_id"].isin(PLAYER_MAPPING_FIXTURE["box_score_id"])
    ].iloc[0]
    assert pd.isna(unmapped["madden_id"])
    assert pd.isna(unmapped["note"])


def test_lookup_split_membership_test_game(staged_processed_dir: pathlib.Path) -> None:
    out = trace.lookup_split_membership(TEST_GAME_ID, processed_dir=staged_processed_dir)
    assert out == {"S1": "val", "S3_folds": [12], "S3_test": False}


def test_lookup_split_membership_unknown_returns_blank(
    staged_processed_dir: pathlib.Path,
) -> None:
    out = trace.lookup_split_membership(
        "999999999xxx", processed_dir=staged_processed_dir
    )
    assert out == {"S1": None, "S3_folds": [], "S3_test": False}


def test_lookup_predictions_returns_row_per_combination(
    staged_processed_dir: pathlib.Path,
) -> None:
    df = trace.lookup_predictions(TEST_GAME_ID, processed_dir=staged_processed_dir)
    # The test game appears once in every prediction parquet (12 combinations).
    assert len(df) == 12
    expected_columns = [
        "combination_id", "slice",
        "pred_home", "pred_away", "true_home", "true_away",
        "residual_home", "residual_away",
    ]
    assert list(df.columns) == expected_columns
    # All 12 combinations represented exactly once.
    assert df["combination_id"].is_unique
    # S1 slices land as "val" / "test"; S3 slices land as "fold_<k>".
    s1_rows = df[df["combination_id"].str.endswith("__s1")]
    s3_rows = df[df["combination_id"].str.endswith("__s3")]
    assert (s1_rows["slice"] == "val").all()
    assert (s3_rows["slice"] == "fold_12").all()
    # Residuals = pred − true (consistency check on a couple of rows).
    sample = df.iloc[0]
    assert sample["residual_home"] == pytest.approx(
        sample["pred_home"] - sample["true_home"], rel=1e-12
    )
    assert sample["residual_away"] == pytest.approx(
        sample["pred_away"] - sample["true_away"], rel=1e-12
    )


def test_lookup_predictions_empty_when_no_predictions_dir(
    tmp_path: pathlib.Path, staged_processed_dir: pathlib.Path,
) -> None:
    """Dev machine without a Phase 4 run sees an empty frame, not an exception."""
    # Stage a processed dir with features + splits but no predictions/.
    bare = tmp_path / "bare_processed"
    bare.mkdir()
    for name in ("features_flat_2024.parquet", "splits_2024.json"):
        shutil.copy2(staged_processed_dir / name, bare / name)
    out = trace.lookup_predictions(TEST_GAME_ID, processed_dir=bare)
    assert out.empty
    assert list(out.columns) == [
        "combination_id", "slice",
        "pred_home", "pred_away", "true_home", "true_away",
        "residual_home", "residual_away",
    ]


# ---------------------------------------------------------------------------
# encoding.py
# ---------------------------------------------------------------------------


def test_encode_one_game_flat_returns_row(staged_processed_dir: pathlib.Path) -> None:
    row = encoding.encode_one_game_flat(TEST_GAME_ID, processed_dir=staged_processed_dir)
    assert row["GameId"] == TEST_GAME_ID
    assert "home_score" in row.index
    assert "away_score" in row.index
    assert "week" in row.index


def test_encode_one_game_pos_returns_row(staged_processed_dir: pathlib.Path) -> None:
    row = encoding.encode_one_game_pos(TEST_GAME_ID, processed_dir=staged_processed_dir)
    assert row["GameId"] == TEST_GAME_ID


def test_encode_one_game_unknown_raises(staged_processed_dir: pathlib.Path) -> None:
    with pytest.raises(KeyError, match="GameId 'nope'"):
        encoding.encode_one_game_flat("nope", processed_dir=staged_processed_dir)


def test_explain_categorical_high_card_routes_to_embedding(
    staged_processed_dir: pathlib.Path,
) -> None:
    out = encoding.explain_categorical(
        "home_team_code", "dal", processed_dir=staged_processed_dir
    )
    assert out["vocab_key"] == "team_codes"
    assert out["routing"] == "embedding"
    assert out["embedding_table"] == "team_codes"
    assert out["vocab_size"] == 32
    # NULL_BUMP = 1; "dal" sits at vocab index 8 → integer_code 9.
    assert out["integer_code"] == 9


def test_explain_categorical_low_card_routes_to_one_hot(
    staged_processed_dir: pathlib.Path,
) -> None:
    out = encoding.explain_categorical(
        "roof", "outdoors", processed_dir=staged_processed_dir
    )
    assert out["vocab_key"] == "roof"
    assert out["routing"] == "one_hot"
    assert out["integer_code"] == 2  # vocab index 1 + NULL_BUMP


def test_explain_categorical_numeric_column_raises(
    staged_processed_dir: pathlib.Path,
) -> None:
    with pytest.raises(ValueError, match="not categorical"):
        encoding.explain_categorical("week", "1", processed_dir=staged_processed_dir)


def test_explain_categorical_unknown_value_raises(
    staged_processed_dir: pathlib.Path,
) -> None:
    with pytest.raises(ValueError, match="not present in vocab"):
        encoding.explain_categorical(
            "roof", "no-such-roof", processed_dir=staged_processed_dir
        )


# ---------------------------------------------------------------------------
# loss_curves.py
# ---------------------------------------------------------------------------


def test_load_loss_curves_returns_expected_schema(
    staged_processed_dir: pathlib.Path,
) -> None:
    df = loss_curves.load_loss_curves(staged_processed_dir)
    assert list(df.columns) == [
        "combination_id", "fold", "epoch", "train_loss", "val_loss", "val_mae",
    ]
    assert len(df) > 0
    # Every row's combination_id appears in the manifest's training_summaries.
    manifest = json.loads(
        (staged_processed_dir / "training_manifest.json").read_text()
    )
    assert set(df["combination_id"]).issubset(set(manifest["training_summaries"]))


def test_load_loss_curves_detects_sha_mismatch(
    tmp_path: pathlib.Path, staged_processed_dir: pathlib.Path,
) -> None:
    """Tampering with the parquet must surface as a clear error, not silently mislead."""
    bad = tmp_path / "bad_processed"
    bad.mkdir()
    for name in ("training_manifest.json", "training_loss_curves.parquet"):
        shutil.copy2(staged_processed_dir / name, bad / name)
    # Re-write the parquet with a single dummy row so its SHA diverges.
    table = pa.Table.from_pydict({
        "combination_id": ["fake__none__s1"],
        "fold": [0],
        "epoch": [1],
        "train_loss": [0.0],
        "val_loss": [0.0],
        "val_mae": [0.0],
    })
    pq.write_table(table, bad / "training_loss_curves.parquet")

    with pytest.raises(LossCurvesIntegrityError, match="SHA-256 mismatch"):
        loss_curves.load_loss_curves(bad)


def test_load_loss_curves_verify_sha_false_skips_check(
    tmp_path: pathlib.Path, staged_processed_dir: pathlib.Path,
) -> None:
    bad = tmp_path / "bad_processed"
    bad.mkdir()
    shutil.copy2(
        staged_processed_dir / "training_manifest.json",
        bad / "training_manifest.json",
    )
    # Write a 1-row parquet whose SHA does not match the manifest.
    table = pa.Table.from_pydict({
        "combination_id": ["fake__none__s1"],
        "fold": [0],
        "epoch": [1],
        "train_loss": [0.0],
        "val_loss": [0.0],
        "val_mae": [0.0],
    })
    pq.write_table(table, bad / "training_loss_curves.parquet")

    df = loss_curves.load_loss_curves(bad, verify_sha=False)
    assert len(df) == 1
    assert df["combination_id"].iloc[0] == "fake__none__s1"


def test_load_loss_curves_missing_file_raises(tmp_path: pathlib.Path) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(FileNotFoundError):
        loss_curves.load_loss_curves(bare)


def test_filter_curves_by_combination_id(
    staged_processed_dir: pathlib.Path,
) -> None:
    df = loss_curves.load_loss_curves(staged_processed_dir)
    target = df["combination_id"].iloc[0]
    out = loss_curves.filter_curves(df, combination_id=target)
    assert (out["combination_id"] == target).all()
    assert list(out.columns) == list(df.columns)


def test_filter_curves_by_fold(staged_processed_dir: pathlib.Path) -> None:
    df = loss_curves.load_loss_curves(staged_processed_dir)
    out = loss_curves.filter_curves(df, fold=0)
    assert (out["fold"] == 0).all()


def test_filter_curves_both(staged_processed_dir: pathlib.Path) -> None:
    df = loss_curves.load_loss_curves(staged_processed_dir)
    target = df["combination_id"].iloc[0]
    out = loss_curves.filter_curves(df, combination_id=target, fold=0)
    assert (out["combination_id"] == target).all()
    assert (out["fold"] == 0).all()
