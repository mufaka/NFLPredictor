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

BOX_SCORES_BASENAME = "box_scores_2024.csv"
PLAYER_ID_MAPPING_BASENAME = "player_id_mapping.csv"
SPLITS_BASENAME = "splits_2024.json"
FEATURES_FLAT_BASENAME = "features_flat_2024.parquet"
PREDICTIONS_DIRNAME = "predictions"


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
    """Return the ``box_scores_2024.csv`` row for ``game_id``.

    Raises ``KeyError`` if no row matches.
    """
    path = raw_dir / BOX_SCORES_BASENAME
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

    Columns: ``slot`` (e.g. ``"HomeOff01"``), ``side`` (``"Home"``/``"Away"``),
    ``unit`` (``"Off"``/``"Def"``), ``position``, ``name``, ``box_score_id``,
    ``madden_id``, ``note``. ``box_score_id`` is ``None`` when the raw cell
    was blank; ``madden_id`` and ``note`` are ``None`` when no mapping row
    exists for that ``box_score_id`` (rare, but possible for unmatched
    starters appended during Phase 1).
    """
    row = load_raw_game(game_id, raw_dir)
    mapping = pd.read_csv(processed_dir / PLAYER_ID_MAPPING_BASENAME)
    by_box_id = mapping.set_index("box_score_id")

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
    """Return the S1 bucket and S3 fold(s) that contain ``game_id``.

    Shape: ``{"S1": "train"|"val"|"test"|None, "S3_folds": [k, ...], "S3_test": bool}``.
    ``S3_folds`` lists the ``k`` values (6–14) of every S3 fold whose ``val``
    slice contains the game; an S3 train-only game returns an empty list.
    """
    splits = json.loads((processed_dir / SPLITS_BASENAME).read_text())

    s1_bucket: str | None = None
    for name in ("train", "val", "test"):
        if game_id in splits["S1"][name]:
            s1_bucket = name
            break

    s3_folds_with_game: list[int] = []
    for fold in splits["S3"]["folds"]:
        if game_id in fold["val"]:
            s3_folds_with_game.append(int(fold["k"]))
    s3_in_test = game_id in splits["S3"]["test"]

    return {
        "S1": s1_bucket,
        "S3_folds": s3_folds_with_game,
        "S3_test": s3_in_test,
    }


def lookup_predictions(
    game_id: str,
    processed_dir: pathlib.Path = DEFAULT_PROCESSED_DIR,
) -> pd.DataFrame:
    """One row per ``(combination_id, slice)`` that emitted a prediction for ``game_id``.

    Columns: ``combination_id``, ``slice`` (S1: ``"val"``/``"test"``;
    S3: ``"fold_<k>"`` using each fold's ``k`` value from ``splits_2024.json``),
    ``pred_home``, ``pred_away`` (at face value from the prediction parquet),
    ``true_home``, ``true_away`` (from the Phase 2 features row),
    ``residual_home`` = ``pred_home − true_home``, ``residual_away`` analogous.

    Returns an empty frame (with the correct columns) if no predictions
    directory is present — the dev machine without a Phase 4 run is the
    canonical empty case.
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
    fold_k_by_index = {int(f["fold_index"]): int(f["k"]) for f in splits["S3"]["folds"]}

    pred_dir = processed_dir / PREDICTIONS_DIRNAME
    records: list[dict[str, Any]] = []
    if pred_dir.exists():
        for path in sorted(pred_dir.glob("*.parquet")):
            combination_id = path.stem
            is_s1 = combination_id.endswith("__s1")
            df = pd.read_parquet(path)
            hits = df[df["GameId"] == game_id]
            if hits.empty:
                continue
            for _, hit in hits.iterrows():
                if is_s1:
                    slice_label = str(hit["slice"])
                else:
                    fold_idx = int(hit["fold_index"])
                    slice_label = f"fold_{fold_k_by_index[fold_idx]}"
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
