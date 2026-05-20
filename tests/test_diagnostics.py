"""DD-INT-04 / DD-TEST-09: unit tests for nflpredictor.diagnostics helpers.

Test strategy: the train fixture under ``tests/fixtures/train/`` contains the
Phase 2 / Phase 3 / Phase 4 artifacts the helpers need (features parquets,
vocab, splits, predictions, training manifest, loss curves). The Phase 1
outputs (raw box scores, player_id_mapping) are not in that fixture, so raw
box scores come straight from ``Data/raw/box_scores_<YYYY>.csv`` and a minimal
``player_id_mapping.csv`` is hand-written into the staged processed directory.
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

# The 2024 Chiefs/Ravens opener — present in the train fixture's
# season_holdout.val slice, so it exercises every per-combination output path.
TEST_GAME_ID = "202409050kan"


# A small, deterministic mapping covering a few of the test game's starters.
# resolve_starters tolerates unmapped rows (madden_id / note return None).
PLAYER_MAPPING_FIXTURE = pd.DataFrame([
    {"season": "2024", "box_score_id": "MahoPa00", "madden_id": "2024-FIX01", "note": "fixture"},
    {"season": "2024", "box_score_id": "PachIs00", "madden_id": "2024-FIX02", "note": "fixture"},
    {"season": "2024", "box_score_id": "JackLa00", "madden_id": "2024-FIX03", "note": "fixture"},
    {"season": "2024", "box_score_id": "DannMi00", "madden_id": "2024-FIX04", "note": "fixture"},
])


@pytest.fixture(scope="session")
def staged_processed_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Build a composite processed dir that unifies the train fixture's layout."""
    out = tmp_path_factory.mktemp("diagnostics_processed")

    for name in ("features_flat_all.parquet", "features_pos_all.parquet",
                 "feature_vocab.json", "feature_manifest.json"):
        shutil.copy2(TRAIN_FIXTURE / "raw_phase2" / name, out / name)

    for name in ("splits_all.json", "splits_manifest.json"):
        shutil.copy2(TRAIN_FIXTURE / "raw_phase3" / name, out / name)

    shutil.copy2(TRAIN_FIXTURE / "expected" / "training_manifest.json",
                 out / "training_manifest.json")
    shutil.copy2(TRAIN_FIXTURE / "expected" / "training_loss_curves.parquet",
                 out / "training_loss_curves.parquet")
    pred_dst = out / "predictions"
    pred_dst.mkdir()
    for p in sorted((TRAIN_FIXTURE / "expected" / "predictions").glob("*.parquet")):
        shutil.copy2(p, pred_dst / p.name)

    PLAYER_MAPPING_FIXTURE.to_csv(out / "player_id_mapping.csv", index=False)

    return out


# ---------------------------------------------------------------------------
# trace.py
# ---------------------------------------------------------------------------


def test_load_raw_game_returns_single_row() -> None:
    row = trace.load_raw_game(TEST_GAME_ID, raw_dir=REAL_RAW)
    assert row["GameId"] == TEST_GAME_ID
    assert row["HomeTeam"] == "Kansas City Chiefs"
    assert row["AwayTeam"] == "Baltimore Ravens"


def test_load_raw_game_unknown_raises() -> None:
    with pytest.raises(KeyError, match="GameId '202409050xxx'"):
        trace.load_raw_game("202409050xxx", raw_dir=REAL_RAW)


def test_resolve_starters_returns_44_slots(staged_processed_dir: pathlib.Path) -> None:
    df = trace.resolve_starters(
        TEST_GAME_ID, raw_dir=REAL_RAW, processed_dir=staged_processed_dir
    )
    assert len(df) == 44
    assert list(df.columns) == [
        "slot", "side", "unit", "position", "name",
        "box_score_id", "madden_id", "note",
    ]
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
    assert home_qb["name"] == "Patrick Mahomes"
    assert home_qb["box_score_id"] == "MahoPa00"
    assert home_qb["madden_id"] == "2024-FIX01"
    assert home_qb["note"] == "fixture"

    # A slot whose _ID is not in the fixture mapping reports None.
    unmapped = df[
        ~df["box_score_id"].isin(PLAYER_MAPPING_FIXTURE["box_score_id"])
    ].iloc[0]
    assert pd.isna(unmapped["madden_id"])
    assert pd.isna(unmapped["note"])


def test_lookup_split_membership_val_game(staged_processed_dir: pathlib.Path) -> None:
    out = trace.lookup_split_membership(TEST_GAME_ID, processed_dir=staged_processed_dir)
    # The 2024 opener is in season_holdout.val; the fixture has no loso_cv.
    assert out == {
        "season_holdout": "val",
        "loso_cv_val_seasons": [],
        "loso_cv_in_test": False,
    }


def test_lookup_split_membership_unknown_returns_blank(
    staged_processed_dir: pathlib.Path,
) -> None:
    out = trace.lookup_split_membership(
        "999999999xxx", processed_dir=staged_processed_dir
    )
    assert out == {
        "season_holdout": None,
        "loso_cv_val_seasons": [],
        "loso_cv_in_test": False,
    }


def test_lookup_predictions_returns_row_per_combination(
    staged_processed_dir: pathlib.Path,
) -> None:
    df = trace.lookup_predictions(TEST_GAME_ID, processed_dir=staged_processed_dir)
    # The test game appears once in every season_holdout prediction parquet
    # (6 combinations in the default config).
    assert len(df) == 6
    expected_columns = [
        "combination_id", "slice",
        "pred_home", "pred_away", "true_home", "true_away",
        "residual_home", "residual_away",
    ]
    assert list(df.columns) == expected_columns
    assert df["combination_id"].is_unique
    # season_holdout slices land as "val" for a val-slice game.
    assert (df["slice"] == "val").all()
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
    bare = tmp_path / "bare_processed"
    bare.mkdir()
    for name in ("features_flat_all.parquet", "splits_all.json"):
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
    assert "season" in row.index


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


def test_explain_categorical_low_card_routes_to_one_hot(
    staged_processed_dir: pathlib.Path,
) -> None:
    out = encoding.explain_categorical(
        "roof", "outdoors", processed_dir=staged_processed_dir
    )
    assert out["vocab_key"] == "roof"
    assert out["routing"] == "one_hot"


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
    manifest = json.loads(
        (staged_processed_dir / "training_manifest.json").read_text()
    )
    assert set(df["combination_id"]).issubset(set(manifest["training_summaries"]))


def test_load_loss_curves_detects_sha_mismatch(
    tmp_path: pathlib.Path, staged_processed_dir: pathlib.Path,
) -> None:
    bad = tmp_path / "bad_processed"
    bad.mkdir()
    for name in ("training_manifest.json", "training_loss_curves.parquet"):
        shutil.copy2(staged_processed_dir / name, bad / name)
    table = pa.Table.from_pydict({
        "combination_id": ["fake__none__season_holdout"],
        "fold": [0], "epoch": [1],
        "train_loss": [0.0], "val_loss": [0.0], "val_mae": [0.0],
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
    table = pa.Table.from_pydict({
        "combination_id": ["fake__none__season_holdout"],
        "fold": [0], "epoch": [1],
        "train_loss": [0.0], "val_loss": [0.0], "val_mae": [0.0],
    })
    pq.write_table(table, bad / "training_loss_curves.parquet")
    df = loss_curves.load_loss_curves(bad, verify_sha=False)
    assert len(df) == 1
    assert df["combination_id"].iloc[0] == "fake__none__season_holdout"


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
