"""Per-game trace helpers: raw box-score row → starter resolution → split bucket → predictions.

Powers the Phase 6 pipeline-walkthrough notebook's Section 2/4/5 cells. Heavy
joining and lookup logic lives here (DD-INT-01) so the notebook stays small
and the helpers stay testable (DD-INT-04).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pandas as pd


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPO_ROOT / "Data" / "raw"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "Data" / "processed"

PLAYER_ID_MAPPING_BASENAME = "player_id_mapping.csv"
SPLITS_BASENAME = "splits_all.json"
FEATURES_FLAT_BASENAME = "features_flat_all.parquet"
PREDICTIONS_DIRNAME = "predictions"


def _season_of(game_id: str) -> int:
    """NFL season of a GameId — month >= 8 maps to the calendar year, else year-1."""
    year, month = int(game_id[:4]), int(game_id[4:6])
    return year if month >= 8 else year - 1


def _starter_slot_names() -> list[str]:
    """The 44 canonical starter slot prefixes (HomeOff01..HomeDef11, AwayOff01..AwayDef11)."""
    out: list[str] = []
    for side in ("Home", "Away"):
        for unit in ("Off", "Def"):
            for i in range(1, 12):
                out.append(f"{side}{unit}{i:02d}")
    return out


def load_raw_game(
    game_id: str,
    raw_dir: pathlib.Path = DEFAULT_RAW_DIR,
) -> pd.Series:
    """Return the raw box-score row for ``game_id``.

    The season is derived from the ``GameId`` date prefix and used to pick
    the right ``box_scores_<YYYY>.csv`` file. Raises ``KeyError`` if no row
    matches.
    """
    path = raw_dir / f"box_scores_{_season_of(game_id)}.csv"
    df = pd.read_csv(path)
    matched = df[df["GameId"] == game_id]
    if matched.empty:
        raise KeyError(f"GameId {game_id!r} not found in {path}")
    if len(matched) > 1:
        raise RuntimeError(
            f"GameId {game_id!r} matches {len(matched)} rows in {path}; expected one"
        )
    return matched.iloc[0]


def resolve_starters(
    game_id: str,
    raw_dir: pathlib.Path = DEFAULT_RAW_DIR,
    processed_dir: pathlib.Path = DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    """One row per starter slot with Madden-ID provenance.

    Columns: ``slot``, ``side``, ``unit``, ``position``, ``name``,
    ``box_score_id``, ``madden_id``, ``note``. ``madden_id`` and ``note`` are
    ``None`` when no mapping row exists for that ``box_score_id``.
    """
    row = load_raw_game(game_id, raw_dir)
    mapping = pd.read_csv(processed_dir / PLAYER_ID_MAPPING_BASENAME)
    by_box_id = mapping.drop_duplicates("box_score_id").set_index("box_score_id")

    records: list[dict[str, Any]] = []
    for slot in _starter_slot_names():
        box_id = row.get(f"{slot}_ID")
        box_id_str = box_id if isinstance(box_id, str) and box_id else None
        madden_id: Any = None
        note: Any = None
        if box_id_str is not None and box_id_str in by_box_id.index:
            m = by_box_id.loc[box_id_str]
            madden_id = m["madden_id"]
            note = m["note"]
        records.append({
            "slot": slot,
            "side": slot[:4],
            "unit": slot[4:7],
            "position": row.get(f"{slot}_Position"),
            "name": row.get(f"{slot}_Name"),
            "box_score_id": box_id_str,
            "madden_id": madden_id,
            "note": note,
        })
    return pd.DataFrame.from_records(records)


def lookup_split_membership(
    game_id: str,
    processed_dir: pathlib.Path = DEFAULT_PROCESSED_DIR,
) -> dict[str, Any]:
    """Return the split membership of ``game_id``.

    Shape: ``{"season_holdout": "train"|"val"|"test"|None,
    "loso_cv_val_seasons": [s, ...], "loso_cv_in_test": bool}``. The
    ``loso_cv_*`` fields are empty/False when the splits artifact does not
    carry the (opt-in) ``loso_cv`` strategy.
    """
    splits = json.loads((processed_dir / SPLITS_BASENAME).read_text())

    sh_bucket: str | None = None
    season_holdout = splits.get("season_holdout")
    if season_holdout:
        for name in ("train", "val", "test"):
            if game_id in season_holdout[name]:
                sh_bucket = name
                break

    loso_val_seasons: list[int] = []
    loso_in_test = False
    loso_cv = splits.get("loso_cv")
    if loso_cv:
        for fold in loso_cv["folds"]:
            if game_id in fold["val"]:
                loso_val_seasons.append(int(fold["val_season"]))
        loso_in_test = game_id in loso_cv["test"]

    return {
        "season_holdout": sh_bucket,
        "loso_cv_val_seasons": loso_val_seasons,
        "loso_cv_in_test": loso_in_test,
    }


def lookup_predictions(
    game_id: str,
    processed_dir: pathlib.Path = DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    """One row per ``(combination_id, slice)`` that emitted a prediction for ``game_id``.

    Columns: ``combination_id``, ``slice`` (season_holdout: ``"val"``/``"test"``;
    loso_cv: ``"fold_<val_season>"``), ``pred_home``, ``pred_away``,
    ``true_home``, ``true_away``, ``residual_home``, ``residual_away``.

    Returns an empty frame (with the correct columns) when no predictions
    directory is present.
    """
    features = pd.read_parquet(
        processed_dir / FEATURES_FLAT_BASENAME,
        columns=["GameId", "home_score", "away_score"],
    )
    label_row = features[features["GameId"] == game_id]
    if label_row.empty:
        raise KeyError(
            f"GameId {game_id!r} not found in {processed_dir / FEATURES_FLAT_BASENAME}"
        )
    true_home = float(label_row["home_score"].iloc[0])
    true_away = float(label_row["away_score"].iloc[0])

    splits = json.loads((processed_dir / SPLITS_BASENAME).read_text())
    fold_season_by_index: dict[int, int] = {}
    loso_cv = splits.get("loso_cv")
    if loso_cv:
        fold_season_by_index = {
            int(f["fold_index"]): int(f["val_season"]) for f in loso_cv["folds"]
        }

    pred_dir = processed_dir / PREDICTIONS_DIRNAME
    records: list[dict[str, Any]] = []
    if pred_dir.exists():
        for path in sorted(pred_dir.glob("*.parquet")):
            combination_id = path.stem
            is_holdout = combination_id.endswith("__season_holdout")
            df = pd.read_parquet(path)
            hits = df[df["GameId"] == game_id]
            if hits.empty:
                continue
            for _, hit in hits.iterrows():
                if is_holdout:
                    slice_label = str(hit["slice"])
                else:
                    fold_idx = int(hit["fold_index"])
                    slice_label = f"fold_{fold_season_by_index.get(fold_idx, fold_idx)}"
                ph = float(hit["pred_home"])
                pa = float(hit["pred_away"])
                records.append({
                    "combination_id": combination_id,
                    "slice": slice_label,
                    "pred_home": ph,
                    "pred_away": pa,
                    "true_home": true_home,
                    "true_away": true_away,
                    "residual_home": ph - true_home,
                    "residual_away": pa - true_away,
                })

    if not records:
        return pd.DataFrame({
            "combination_id": pd.Series(dtype="object"),
            "slice": pd.Series(dtype="object"),
            "pred_home": pd.Series(dtype="float64"),
            "pred_away": pd.Series(dtype="float64"),
            "true_home": pd.Series(dtype="float64"),
            "true_away": pd.Series(dtype="float64"),
            "residual_home": pd.Series(dtype="float64"),
            "residual_away": pd.Series(dtype="float64"),
        })
    return pd.DataFrame.from_records(records)
