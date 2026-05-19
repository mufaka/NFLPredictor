"""Feature build pipeline (§5)."""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Optional

import pandas as pd

from nflpredictor.databuild.manifest import compute_sha256

from .config import (
    FeatureConfig,
    load_feature_config,
    validate_madden_columns_exist,
)
from .flat import assemble_flat
from .game_level import assemble_game_level
from .manifest import build_feature_manifest, write_feature_manifest
from .officials import OFFICIAL_COLUMN_NAMES, assemble_officials
from .outputs import write_parquet, write_vocab
from .pos import CANONICAL_BPOS_SLOTS, assemble_pos
from .slots import CANONICAL_SLOTS, build_madden_lookup
from .vocab import Vocabulary, build_vocabulary, encode_column


PHASE1_MADDEN_BASENAME = "madden_2024.csv"
PHASE1_BOX_SCORES_BASENAME = "box_scores_2024.csv"
PHASE1_MANIFEST_BASENAME = "build_manifest.json"

FEATURE_CONFIG_BASENAME = "feature_config.yaml"

FLAT_PARQUET_BASENAME = "features_flat_2024.parquet"
POS_PARQUET_BASENAME = "features_pos_2024.parquet"
FEATURE_VOCAB_BASENAME = "feature_vocab.json"
FEATURE_MANIFEST_BASENAME = "feature_manifest.json"


class Phase1OutputMismatchError(ValueError):
    """Raised when an on-disk Phase 1 output diverges from the manifest hash (FE-IN-04)."""


def verify_phase1_outputs(processed_dir: pathlib.Path) -> dict:
    """Verify Phase 1 outputs against ``build_manifest.json`` (FE-IN-04).

    Returns the parsed manifest dict so the caller can propagate provenance into
    Phase 2's own manifest.
    """
    manifest_path = processed_dir / PHASE1_MANIFEST_BASENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Phase 1 manifest not found at {manifest_path}; "
            "run `python -m nflpredictor.databuild` first"
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected = manifest.get("output_sha256")
    if not isinstance(expected, dict):
        raise Phase1OutputMismatchError(
            "build_manifest.json is missing the 'output_sha256' map"
        )

    for basename in (PHASE1_MADDEN_BASENAME, PHASE1_BOX_SCORES_BASENAME):
        on_disk = processed_dir / basename
        if not on_disk.exists():
            raise FileNotFoundError(
                f"Phase 1 output {on_disk} not found; "
                "run `python -m nflpredictor.databuild` first"
            )
        manifest_key = f"Data/processed/{basename}"
        if manifest_key not in expected:
            raise Phase1OutputMismatchError(
                f"build_manifest.json output_sha256 does not record {manifest_key}"
            )
        actual = compute_sha256(on_disk)
        if actual != expected[manifest_key]:
            raise Phase1OutputMismatchError(
                f"Phase 1 output {manifest_key} hash mismatch: "
                f"manifest={expected[manifest_key]!r}, disk={actual!r}. "
                "Re-run `python -m nflpredictor.databuild` to regenerate."
            )
    return manifest


# ---------------------------------------------------------------------------
# Vocabulary collection
# ---------------------------------------------------------------------------

# game-level field → vocab key (categorical fields only).
_GAME_LEVEL_VOCAB_KEY: dict[str, str] = {
    "day_of_week": "day_of_week",
    "stadium": "stadium",
    "roof": "roof",
    "surface": "surface",
    "home_team_code": "team_codes",
    "away_team_code": "team_codes",
    "home_coach": "coaches",
    "away_coach": "coaches",
}


# feature_vocab.json schema tag — bumped when the JSON layout changes.
VOCAB_VERSION = "v2"


def build_column_vocab_keys(
    config: FeatureConfig,
    *,
    has_officials: bool,
) -> dict[str, str]:
    """Compute the ``column_name → vocab_key`` map for both shapes (FE-VOC-07).

    Phase 4's encoder reads this map directly, replacing the old pattern-match
    on column-name suffixes. The map covers every categorical column across
    both shapes:

    - Game-level categoricals — only those toggled on in ``config.game_features.include``.
    - Officials — ``official_*`` columns when ``has_officials`` is True.
    - Per-slot Madden categoricals — for each Madden column in
      ``madden_categorical_columns``, one entry per slot in both
      ``CANONICAL_SLOTS`` (flat) and ``CANONICAL_BPOS_SLOTS`` (pos).
    - Per-slot canonical positions — ``{slot}_position`` columns (flat only).

    Numeric columns (Madden ratings, ``week``, ``days_rest_*``, ``_matched``,
    ``_present``, weather floats, scores) deliberately do not appear here.
    """
    from .flat import snake_case
    from .officials import OFFICIAL_COLUMN_NAMES

    out: dict[str, str] = {}

    for field in config.game_features.include:
        key = _GAME_LEVEL_VOCAB_KEY.get(field)
        if key is not None:
            out[field] = key

    if has_officials:
        for col in OFFICIAL_COLUMN_NAMES:
            out[col] = "officials"

    for slot in CANONICAL_SLOTS:
        out[f"{slot}_position"] = "positions"

    for col in config.madden_categorical_columns:
        snake = snake_case(col)
        for slot in CANONICAL_SLOTS:
            out[f"{slot}_madden_{snake}"] = col
        for slot in CANONICAL_BPOS_SLOTS:
            out[f"{slot}_madden_{snake}"] = col

    return out


def collect_observations(
    box_scores_df: pd.DataFrame,
    madden_df: pd.DataFrame,
    game_level_df: pd.DataFrame,
    officials_df: Optional[pd.DataFrame],
    config: FeatureConfig,
) -> dict[str, set[str]]:
    """Gather every categorical value the vocabulary sidecar needs (FE-VOC-03)."""
    obs: dict[str, set[str]] = {}
    include = config.game_features.include

    for field in include:
        key = _GAME_LEVEL_VOCAB_KEY.get(field)
        if key is None:
            continue
        existing = obs.setdefault(key, set())
        existing.update(str(v) for v in game_level_df[field])

    if officials_df is not None:
        officials_obs: set[str] = set()
        for col in OFFICIAL_COLUMN_NAMES:
            officials_obs.update(str(v) if v is not None else "" for v in officials_df[col])
        obs["officials"] = officials_obs

    positions: set[str] = set()
    for slot in CANONICAL_SLOTS:
        positions.update(str(v) for v in box_scores_df[f"{slot}_Position"])
    obs["positions"] = positions

    for col in config.madden_categorical_columns:
        obs[col] = {str(v) for v in madden_df[col]}

    return obs


def encode_game_level(
    game_level_df: pd.DataFrame, config: FeatureConfig, vocab: Vocabulary
) -> pd.DataFrame:
    """Replace categorical raw-string columns with their integer codes."""
    out = game_level_df.copy()
    for field in config.game_features.include:
        key = _GAME_LEVEL_VOCAB_KEY.get(field)
        if key is None:
            continue
        out[field] = encode_column(out[field], key, vocab)
    return out


def encode_officials(officials_df: pd.DataFrame, vocab: Vocabulary) -> pd.DataFrame:
    """Replace each ``official_*`` column with integer codes (FE-OFF-02)."""
    out = officials_df.copy()
    for col in OFFICIAL_COLUMN_NAMES:
        out[col] = encode_column(out[col], "officials", vocab)
    return out


# ---------------------------------------------------------------------------
# Column-count breakdown for the manifest
# ---------------------------------------------------------------------------


def _flat_column_counts(config: FeatureConfig) -> dict[str, int]:
    n_madden_cols = len(config.madden_columns)
    return {
        "game_id": 1,
        "game_level": len(config.game_features.include),
        "weather": 4 if config.game_features.weather == "parsed" else 0,
        "officials": 7 if config.game_features.officials == "included" else 0,
        "slot_madden": 44 * n_madden_cols,
        "slot_position": 44,
        "slot_matched": 44,
        "labels": 2,
    }


def _pos_column_counts(config: FeatureConfig) -> dict[str, int]:
    n_madden_cols = len(config.madden_columns)
    return {
        "game_id": 1,
        "game_level": len(config.game_features.include),
        "weather": 4 if config.game_features.weather == "parsed" else 0,
        "officials": 7 if config.game_features.officials == "included" else 0,
        "slot_madden": 58 * n_madden_cols,
        "slot_present": 58,
        "slot_matched": 58,
        "labels": 2,
    }


def _with_total(breakdown: dict[str, int]) -> dict[str, int]:
    return {"total": sum(breakdown.values()), **breakdown}


# ---------------------------------------------------------------------------
# Section combination
# ---------------------------------------------------------------------------


def _combine_sections(
    box_scores_df: pd.DataFrame,
    game_level_encoded: pd.DataFrame,
    weather_df: Optional[pd.DataFrame],
    officials_encoded: Optional[pd.DataFrame],
    shape_df: pd.DataFrame,
) -> pd.DataFrame:
    """Stitch the per-section frames in FE-OUT-06 order and append labels."""
    sections: list[pd.DataFrame] = [game_level_encoded.reset_index(drop=True)]
    if weather_df is not None:
        sections.append(weather_df.drop(columns=["GameId"]).reset_index(drop=True))
    if officials_encoded is not None:
        sections.append(officials_encoded.drop(columns=["GameId"]).reset_index(drop=True))
    sections.append(shape_df.drop(columns=["GameId"]).reset_index(drop=True))
    combined = pd.concat(sections, axis=1)
    combined["home_score"] = box_scores_df["HomeScore"].astype("float64").values
    combined["away_score"] = box_scores_df["AwayScore"].astype("float64").values
    return combined


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_feature_build(
    raw_dir: pathlib.Path,
    processed_dir: pathlib.Path,
    *,
    repo_dir: Optional[pathlib.Path] = None,
) -> None:
    """Run the Phase 2 feature build end-to-end (§5)."""
    if repo_dir is None:
        repo_dir = processed_dir.parent.parent  # Data/processed/.. -> repo root

    config_path = raw_dir / FEATURE_CONFIG_BASENAME
    config = load_feature_config(config_path)

    phase1_manifest = verify_phase1_outputs(processed_dir)

    box_scores_df = pd.read_csv(
        processed_dir / PHASE1_BOX_SCORES_BASENAME,
        dtype=str,
        keep_default_na=False,
    )
    madden_df = pd.read_csv(
        processed_dir / PHASE1_MADDEN_BASENAME,
        dtype=str,
        keep_default_na=False,
    )

    validate_madden_columns_exist(config, list(madden_df.columns))

    # Sort by GameId so every emitted artifact shares a stable row order.
    box_scores_df = box_scores_df.sort_values(
        "GameId", kind="stable", ignore_index=True
    )

    game_level_df = assemble_game_level(box_scores_df, config.game_features.include)

    weather_df = None
    if config.game_features.weather == "parsed":
        from .weather import parse_weather_column

        weather_df = parse_weather_column(box_scores_df)

    officials_df = None
    if config.game_features.officials == "included":
        officials_df = assemble_officials(box_scores_df)

    observations = collect_observations(
        box_scores_df, madden_df, game_level_df, officials_df, config
    )
    vocab = build_vocabulary(observations)

    game_level_encoded = encode_game_level(game_level_df, config, vocab)
    officials_encoded = (
        encode_officials(officials_df, vocab) if officials_df is not None else None
    )

    madden_lookup = build_madden_lookup(madden_df)

    flat_path = processed_dir / FLAT_PARQUET_BASENAME
    pos_path = processed_dir / POS_PARQUET_BASENAME

    feature_outputs: dict[str, pathlib.Path] = {}
    column_counts: dict[str, dict[str, int]] = {}

    if "flat" in config.slot_shapes:
        flat_df = assemble_flat(box_scores_df, madden_lookup, config, vocab)
        combined_flat = _combine_sections(
            box_scores_df, game_level_encoded, weather_df, officials_encoded, flat_df
        )
        # Pin GameId as first column for FE-OUT-03.
        combined_flat = combined_flat.assign(
            GameId=box_scores_df["GameId"].astype(str).values
        )
        combined_flat = combined_flat[
            ["GameId", *[c for c in combined_flat.columns if c != "GameId"]]
        ]
        write_parquet(combined_flat, flat_path)
        feature_outputs[f"Data/processed/{FLAT_PARQUET_BASENAME}"] = flat_path
        column_counts[FLAT_PARQUET_BASENAME] = _with_total(_flat_column_counts(config))
        print(
            f"features_flat_2024.parquet: {len(combined_flat)} rows × "
            f"{len(combined_flat.columns)} columns",
            file=sys.stderr,
        )
    else:
        flat_path.unlink(missing_ok=True)

    if "pos" in config.slot_shapes:
        pos_df = assemble_pos(box_scores_df, madden_lookup, config, vocab)
        combined_pos = _combine_sections(
            box_scores_df, game_level_encoded, weather_df, officials_encoded, pos_df
        )
        combined_pos = combined_pos.assign(
            GameId=box_scores_df["GameId"].astype(str).values
        )
        combined_pos = combined_pos[
            ["GameId", *[c for c in combined_pos.columns if c != "GameId"]]
        ]
        write_parquet(combined_pos, pos_path)
        feature_outputs[f"Data/processed/{POS_PARQUET_BASENAME}"] = pos_path
        column_counts[POS_PARQUET_BASENAME] = _with_total(_pos_column_counts(config))
        print(
            f"features_pos_2024.parquet: {len(combined_pos)} rows × "
            f"{len(combined_pos.columns)} columns",
            file=sys.stderr,
        )
    else:
        pos_path.unlink(missing_ok=True)

    column_vocab_keys = build_column_vocab_keys(
        config, has_officials=officials_df is not None
    )
    vocab_path = processed_dir / FEATURE_VOCAB_BASENAME
    write_vocab(vocab, column_vocab_keys, VOCAB_VERSION, vocab_path)
    feature_outputs[f"Data/processed/{FEATURE_VOCAB_BASENAME}"] = vocab_path

    print(
        "vocab sizes: " + ", ".join(f"{k}={v}" for k, v in sorted(vocab.sizes().items())),
        file=sys.stderr,
    )

    # Manifest is written last so output SHAs hash the on-disk bytes (FE-MAN-04).
    manifest = build_feature_manifest(
        config_path=config_path,
        normalization_version=config.normalization_version,
        phase1_outputs={
            f"Data/processed/{PHASE1_BOX_SCORES_BASENAME}": processed_dir / PHASE1_BOX_SCORES_BASENAME,
            f"Data/processed/{PHASE1_MADDEN_BASENAME}": processed_dir / PHASE1_MADDEN_BASENAME,
        },
        phase1_manifest=phase1_manifest,
        feature_outputs=feature_outputs,
        column_counts=column_counts,
        vocab=vocab,
        repo_dir=repo_dir,
    )
    write_feature_manifest(manifest, processed_dir / FEATURE_MANIFEST_BASENAME)
