"""End-to-end orchestration (§5.1, EV-MAN-01..07, EV-NF-04..08)."""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Optional

import matplotlib
import numpy
import pyarrow

from .breakdowns import ALL_SPECS, build_breakdown_table
from .config import EvaluationConfig, load_evaluation_config
from .manifest import (
    build_evaluation_manifest,
    build_evaluation_summary,
    compute_sha256,
    write_evaluation_manifest,
)
from .metrics import build_headline_metrics, join_predictions_with_labels
from .outputs import (
    BREAKDOWNS_DIRNAME,
    EVALUATION_MANIFEST_BASENAME,
    METRICS_HEADLINE_BASENAME,
    PLOTS_DIRNAME,
    cleanup_disabled_outputs,
    ensure_evaluation_dirs,
    write_breakdown_parquet,
    write_metrics_headline,
)
from .plots import render_all_plots
from .sources import (
    PHASE2_FEATURES_FLAT_BASENAME,
    PHASE2_FEATURES_POS_BASENAME,
    PHASE2_TRACKED_OUTPUTS,
    PHASE2_VOCAB_BASENAME,
    PHASE3_SPLITS_BASENAME,
    PHASE4_MANIFEST_BASENAME,
    PHASE4_PREDICTIONS_DIRNAME,
    assert_label_parity,
    enumerate_combinations,
    load_features_flat,
    load_features_pos,
    load_prediction_parquet,
    load_splits,
    load_vocab,
    validate_prediction_coverage,
    verify_phase2_outputs,
    verify_phase3_outputs,
    verify_phase4_outputs,
)


EVALUATION_CONFIG_BASENAME: str = "evaluation_config.yaml"


def run_evaluation_build(
    raw_dir: pathlib.Path,
    processed_dir: pathlib.Path,
    *,
    repo_dir: Optional[pathlib.Path] = None,
) -> None:
    """Execute the Phase 5 build stages 1–12 from the plan."""
    if repo_dir is None:
        repo_dir = processed_dir.parent.parent

    # 1. Verify upstream outputs (EV-IN-06, EV-IN-07, EV-IN-08).
    phase2_manifest = verify_phase2_outputs(processed_dir)
    phase3_manifest = verify_phase3_outputs(processed_dir)  # noqa: F841 — kept for symmetry
    phase4_manifest = verify_phase4_outputs(processed_dir)

    # 2. Load + validate evaluation config (EV-IN-04, EV-IN-09).
    config_path = raw_dir / EVALUATION_CONFIG_BASENAME
    config = load_evaluation_config(config_path)

    # 3. Load Phase 2 features (flat + pos) + vocab + Phase 3 splits;
    #    verify label parity (EV-NF-03).
    features_flat = load_features_flat(processed_dir)
    features_pos = load_features_pos(processed_dir)
    assert_label_parity(features_flat, features_pos)
    vocab = load_vocab(processed_dir)
    splits = load_splits(processed_dir)

    # 4. Enumerate combinations; load + validate each prediction parquet.
    combinations = enumerate_combinations(phase4_manifest, processed_dir)
    if not combinations:
        raise RuntimeError(
            f"training_manifest.json at {processed_dir / PHASE4_MANIFEST_BASENAME} "
            "lists no prediction parquets; nothing to evaluate"
        )

    joined_by_combo: dict[str, Any] = {}
    strategy_by_combo: dict[str, str] = {}
    for key in combinations:
        preds = load_prediction_parquet(key)
        validate_prediction_coverage(preds, key, splits)
        joined = join_predictions_with_labels(preds, _features_with_breakdown_cols(features_flat))
        joined_by_combo[key.combination_id] = joined
        strategy_by_combo[key.combination_id] = key.strategy

    ordered_pairs = [(k.combination_id, k.strategy) for k in combinations]

    # 5. Headline metrics structure.
    headline = build_headline_metrics(ordered_pairs, joined_by_combo)

    # 6. Per-breakdown tables (only for enabled dims).
    breakdown_tables_by_dim: dict[str, Any] = {}
    for spec in ALL_SPECS:
        if not config.breakdowns.is_enabled(spec.name):
            continue
        breakdown_tables_by_dim[spec.name] = build_breakdown_table(
            spec, ordered_pairs, joined_by_combo, vocab
        )

    # 7. Ensure output dirs exist; clean up stale disabled outputs (EV-OUT-05, EV-OUT-06).
    evaluation_dir, breakdowns_dir, plots_dir = ensure_evaluation_dirs(processed_dir)
    cleanup_disabled_outputs(evaluation_dir, config)

    # 8. Write headline JSON.
    headline_path = evaluation_dir / METRICS_HEADLINE_BASENAME
    write_metrics_headline(headline, headline_path)

    # 9. Write enabled breakdown parquets.
    for spec in ALL_SPECS:
        if not config.breakdowns.is_enabled(spec.name):
            continue
        df = breakdown_tables_by_dim[spec.name]
        write_breakdown_parquet(df, spec, breakdowns_dir / spec.parquet_basename)

    # 10. Render enabled plots.
    plot_paths = render_all_plots(
        headline=headline,
        breakdown_tables_by_dim=breakdown_tables_by_dim,
        joined_by_combo=joined_by_combo,
        strategy_by_combo=strategy_by_combo,
        plot_cfg=config.plots,
        headline_metric=config.headline_metric,
        plots_dir=plots_dir,
    )

    # 11. Compute output SHAs against the on-disk artifacts.
    output_sha256: dict[str, str] = {}
    output_sha256[METRICS_HEADLINE_BASENAME] = compute_sha256(headline_path)
    for spec in ALL_SPECS:
        if not config.breakdowns.is_enabled(spec.name):
            continue
        bp = breakdowns_dir / spec.parquet_basename
        output_sha256[f"{BREAKDOWNS_DIRNAME}/{spec.parquet_basename}"] = compute_sha256(bp)
    for png_path in sorted(plot_paths):
        output_sha256[f"{PLOTS_DIRNAME}/{png_path.name}"] = compute_sha256(png_path)

    # 12. Build + write the manifest LAST (EV-MAN-04).
    evaluation_config_sha256 = compute_sha256(config_path)
    phase2_source_sha256 = {
        f"Data/processed/{basename}": compute_sha256(processed_dir / basename)
        for basename in PHASE2_TRACKED_OUTPUTS
    }
    phase3_source_sha256 = {
        f"Data/processed/{PHASE3_SPLITS_BASENAME}": compute_sha256(
            processed_dir / PHASE3_SPLITS_BASENAME
        ),
    }
    phase4_source_sha256: dict[str, str] = {
        PHASE4_MANIFEST_BASENAME: compute_sha256(processed_dir / PHASE4_MANIFEST_BASENAME),
    }
    for key in combinations:
        rel = f"{PHASE4_PREDICTIONS_DIRNAME}/{key.parquet_path.name}"
        phase4_source_sha256[rel] = compute_sha256(key.parquet_path)

    evaluation_summary = build_evaluation_summary(headline, config.headline_metric)
    manifest = build_evaluation_manifest(
        evaluation_version=config.evaluation_version,
        headline_metric=config.headline_metric,
        evaluation_config_sha256=evaluation_config_sha256,
        phase2_source_sha256=phase2_source_sha256,
        phase3_source_sha256=phase3_source_sha256,
        phase4_source_sha256=phase4_source_sha256,
        output_sha256=output_sha256,
        phase4_manifest_git_commit=phase4_manifest.get("git_commit"),
        combination_ids=[k.combination_id for k in combinations],
        evaluation_summary=evaluation_summary,
        matplotlib_version=matplotlib.__version__,
        numpy_version=numpy.__version__,
        pyarrow_version=pyarrow.__version__,
        repo_dir=repo_dir,
    )
    write_evaluation_manifest(manifest, evaluation_dir / EVALUATION_MANIFEST_BASENAME)

    # 13. Per-combination headline-metric table to stdout (EV-NF-08).
    _log_headline_table(headline, config.headline_metric)
    _log(f"wrote manifest: {evaluation_dir / EVALUATION_MANIFEST_BASENAME}")


def _features_with_breakdown_cols(features_flat: "pd.DataFrame"):  # type: ignore[name-defined]
    """Return features_flat narrowed to the columns Phase 5 needs.

    Labels + breakdown keys: ``GameId, home_score, away_score, week, surface,
    roof, home_team_code, away_team_code``. Keeping the joined frame narrow
    avoids any accidental dependency on the full ~200-column matrix.
    """
    needed = [
        "GameId", "home_score", "away_score",
        "week", "surface", "roof",
        "home_team_code", "away_team_code",
    ]
    return features_flat[needed]


# ---------------------------------------------------------------------------
# Logging helpers (EV-NF-08)
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(msg, file=sys.stdout, flush=True)


def _log_headline_table(headline: dict[str, Any], headline_metric: str) -> None:
    """Print a per-(combination, slice) headline-metric matrix to stdout."""
    combo_block = headline.get("combinations", {})
    _log(f"per-combination headline metric ({headline_metric}):")
    for combo_id in sorted(combo_block.keys()):
        per_combo = combo_block[combo_id]
        for slice_name in sorted(per_combo.keys()):
            cell = per_combo[slice_name]
            value = cell.get(headline_metric)
            if value is None:
                value_str = "  n/a"
            else:
                value_str = f"{float(value):.4f}"
            n_games = cell.get("n_games", 0)
            _log(f"  {combo_id:<40s} {slice_name:<10s} {headline_metric}={value_str}  n={n_games}")
