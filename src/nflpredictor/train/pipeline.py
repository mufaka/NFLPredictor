"""Combination enumeration + top-level orchestration (§5, TR-RUNG-07, TR-MAN-*)."""

from __future__ import annotations

import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

# Setting CUBLAS_WORKSPACE_CONFIG before any torch import that allocates CUDA
# tensors is part of the per-device determinism contract (TR-TRAIN-02). We set
# it via the module-level sentinel so even importers that bypass run_training_build
# inherit the right value.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402  — must happen AFTER the env-var setdefault above.

from .config import TrainingConfig, load_training_config
from .encoders import (
    ColumnClassification,
    FeatureEncoder,
    classify_columns,
)
from .manifest import (
    build_training_manifest,
    build_training_summary_s1,
    build_training_summary_s3,
    write_training_manifest,
)
from .outputs import (
    LOSS_CURVES_COLUMNS,
    PREDICTIONS_DIRNAME,
    TRAINING_LOSS_CURVES_BASENAME,
    combination_filename,
    ensure_predictions_dir,
    write_loss_curves,
    write_s1_predictions,
    write_s3_predictions,
)
from .predict import (
    LearnedComboResult,
    run_learned_combo_s1,
    run_learned_combo_s3,
    run_trivial_combo_s1,
    run_trivial_combo_s3,
)
from .sources import (
    PHASE2_FEATURES_FLAT_BASENAME,
    PHASE2_FEATURES_POS_BASENAME,
    PHASE2_TRACKED_OUTPUTS,
    PHASE2_VOCAB_BASENAME,
    PHASE3_SPLITS_BASENAME,
    assert_label_parity,
    compute_sha256,
    high_card_vocab_keys,
    load_features,
    load_splits_artifact,
    load_vocab,
    verify_phase2_outputs,
    verify_phase3_outputs,
    verify_strategy_availability,
)
from .train_loop import ResolvedDevice, resolve_device


TRIVIAL_RUNGS: frozenset[str] = frozenset({"mean", "team_mean"})
LEARNED_RUNGS: frozenset[str] = frozenset({"linear", "mlp"})


TRAINING_CONFIG_BASENAME = "training_config.yaml"
TRAINING_MANIFEST_BASENAME = "training_manifest.json"


# --------------------------------------------------------------------------
# Combination enumeration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Combination:
    """One (rung, shape, strategy) the training build will train and predict for."""

    rung: str
    shape: str       # "none" for trivial rungs; "flat" | "pos" for learned rungs
    strategy: str    # "S1" or "S3"

    @property
    def is_trivial(self) -> bool:
        return self.rung in TRIVIAL_RUNGS

    @property
    def is_learned(self) -> bool:
        return self.rung in LEARNED_RUNGS

    @property
    def manifest_key(self) -> str:
        """Combination id used in manifest keys and parquet filenames (TR-OUT-01)."""
        from .outputs import RUNG_FILE_PREFIX
        return f"{RUNG_FILE_PREFIX[self.rung]}__{self.shape}__{self.strategy.lower()}"


def enumerate_combinations(config: TrainingConfig) -> list[Combination]:
    """Cartesian product of ``rungs × shapes × strategies``, filtered per TR-RUNG-07.

    - Trivial rungs (``mean``, ``team_mean``) pair only with ``shape = "none"``
      (config.shapes is irrelevant for them).
    - Learned rungs (``linear``, ``mlp``) pair with each entry in
      ``config.shapes``.

    Order is stable: outer iteration over config.rungs, then config.shapes
    (learned rungs only), then config.strategies. This keeps manifest /
    log layout predictable across runs.
    """
    combos: list[Combination] = []
    for rung in config.rungs:
        if rung in TRIVIAL_RUNGS:
            for strategy in config.strategies:
                combos.append(Combination(rung=rung, shape="none", strategy=strategy))
        elif rung in LEARNED_RUNGS:
            for shape in config.shapes:
                for strategy in config.strategies:
                    combos.append(
                        Combination(rung=rung, shape=shape, strategy=strategy)
                    )
        else:
            # config.py rejects unknown rungs; defense-in-depth.
            raise ValueError(f"unknown rung in config: {rung!r}")
    return combos


# --------------------------------------------------------------------------
# Pipeline orchestration (§5.2 of the plan)
# --------------------------------------------------------------------------


def _val_mae_from_predictions(
    predictions: pd.DataFrame,
    labels_lookup: dict[str, tuple[float, float]],
    *,
    slice_filter: Optional[str] = None,
) -> float:
    """Compute MAE per TR-MAN-04: mean over the (N, 2) absolute-error matrix.

    ``labels_lookup`` maps GameId → (home_score, away_score). When
    ``slice_filter`` is set, only rows whose ``slice`` column matches are
    considered (used for S1 val MAE).
    """
    df = predictions
    if slice_filter is not None:
        df = df[df["slice"] == slice_filter]
    if df.empty:
        return float("nan")
    abs_errs: list[float] = []
    for _, row in df.iterrows():
        true_h, true_a = labels_lookup[str(row["GameId"])]
        abs_errs.append(abs(float(row["pred_home"]) - true_h))
        abs_errs.append(abs(float(row["pred_away"]) - true_a))
    return float(sum(abs_errs) / len(abs_errs))


def _build_labels_lookup(feature_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        str(gid): (float(home), float(away))
        for gid, home, away in zip(
            feature_df["GameId"],
            feature_df["home_score"],
            feature_df["away_score"],
        )
    }


def _log_summary(line: str) -> None:
    print(line, file=sys.stdout, flush=True)


def _collect_loss_curve_rows(
    combo: Combination,
    result: LearnedComboResult,
    splits: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flatten a learned combination's loss-curve records into parquet rows (DD-LC-01).

    For S1 the single fold uses ``fold = 0``. For S3 each fold is tagged with
    its expanding-window ``k`` value (6..14) drawn from the splits artifact,
    matching Spec-Phase6 §4.1.
    """
    rows: list[dict[str, Any]] = []
    if combo.strategy == "S1":
        train_result = result.train_results[0]
        for record in train_result.loss_curve:
            rows.append({
                "combination_id": combo.manifest_key,
                "fold": 0,
                "epoch": record.epoch,
                "train_loss": record.train_loss,
                "val_loss": record.val_loss,
                "val_mae": record.val_mae,
            })
        return rows

    # S3 — train_results[i] aligns positionally with splits["S3"]["folds"][i].
    folds = splits["S3"]["folds"]
    if len(result.train_results) != len(folds):
        raise RuntimeError(
            f"S3 train_results count ({len(result.train_results)}) does not "
            f"match S3 fold count ({len(folds)}) for combo {combo.manifest_key!r}"
        )
    for fold_entry, train_result in zip(folds, result.train_results):
        k_value = int(fold_entry["k"])
        for record in train_result.loss_curve:
            rows.append({
                "combination_id": combo.manifest_key,
                "fold": k_value,
                "epoch": record.epoch,
                "train_loss": record.train_loss,
                "val_loss": record.val_loss,
                "val_mae": record.val_mae,
            })
    return rows


def _run_one_combo(
    combo: Combination,
    *,
    features_by_shape: dict[str, pd.DataFrame],
    classifications_by_shape: dict[str, ColumnClassification],
    encoders_factory_by_shape: dict[str, "object"],  # callable () → FeatureEncoder
    splits: dict[str, Any],
    config: TrainingConfig,
    device: ResolvedDevice,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Train + predict one combination; return (predictions_df, manifest_summary, loss_curve_rows).

    ``loss_curve_rows`` is empty for trivial rungs (DD-LC-01) and lists one
    dict per (fold, epoch) for learned rungs.
    """
    shape_for_features = "flat" if combo.is_trivial else combo.shape
    feature_df = features_by_shape[shape_for_features]
    labels_lookup = _build_labels_lookup(feature_df)

    if combo.is_trivial:
        if combo.strategy == "S1":
            preds = run_trivial_combo_s1(combo.rung, feature_df, splits)
            val_mae = _val_mae_from_predictions(preds, labels_lookup, slice_filter="val")
            summary = build_training_summary_s1(None, val_mae)
            return preds, summary, []
        else:
            preds = run_trivial_combo_s3(combo.rung, feature_df, splits)
            # Per-fold val MAE in the fold's order.
            per_fold_mae: list[float] = []
            for fold in splits["S3"]["folds"]:
                fold_idx = int(fold["fold_index"])
                sub = preds[preds["fold_index"] == fold_idx]
                per_fold_mae.append(_val_mae_from_predictions(sub, labels_lookup))
            per_fold_results: list[Any] = [None] * len(per_fold_mae)
            summary = build_training_summary_s3(per_fold_results, per_fold_mae)
            return preds, summary, []

    # Learned rung
    classification = classifications_by_shape[combo.shape]
    encoder_factory = encoders_factory_by_shape[combo.shape]

    if combo.strategy == "S1":
        result: LearnedComboResult = run_learned_combo_s1(
            combo.rung,
            feature_df,
            splits,
            classification=classification,
            encoder_factory=encoder_factory,
            linear_hp=config.linear,
            mlp_hp=config.mlp,
            seed=config.seed,
            device=device,
            labels_lookup=labels_lookup,
        )
        val_mae = _val_mae_from_predictions(result.predictions, labels_lookup, slice_filter="val")
        summary = build_training_summary_s1(result.train_results[0], val_mae)
        loss_rows = _collect_loss_curve_rows(combo, result, splits)
        return result.predictions, summary, loss_rows

    # S3
    result_s3: LearnedComboResult = run_learned_combo_s3(
        combo.rung,
        feature_df,
        splits,
        classification=classification,
        encoder_factory=encoder_factory,
        linear_hp=config.linear,
        mlp_hp=config.mlp,
        seed=config.seed,
        device=device,
        labels_lookup=labels_lookup,
    )
    per_fold_mae = []
    for fold in splits["S3"]["folds"]:
        fold_idx = int(fold["fold_index"])
        sub = result_s3.predictions[result_s3.predictions["fold_index"] == fold_idx]
        per_fold_mae.append(_val_mae_from_predictions(sub, labels_lookup))
    summary = build_training_summary_s3(list(result_s3.train_results), per_fold_mae)
    loss_rows = _collect_loss_curve_rows(combo, result_s3, splits)
    return result_s3.predictions, summary, loss_rows


def run_training_build(
    raw_dir: pathlib.Path,
    processed_dir: pathlib.Path,
    *,
    repo_dir: Optional[pathlib.Path] = None,
) -> None:
    """Run the Phase 4 training build end-to-end (§5.1)."""
    if repo_dir is None:
        repo_dir = processed_dir.parent.parent

    # 1. (Already done at module import: os.environ.setdefault for CUBLAS.)

    # 2. Verify upstream artifacts (TR-IN-05, TR-IN-06).
    phase2_manifest = verify_phase2_outputs(processed_dir)
    phase3_manifest = verify_phase3_outputs(processed_dir)

    # 3. Load vocab; derive high-card keys; load training config (TR-IN-03, TR-CFG-01..10).
    vocab = load_vocab(processed_dir)
    high_card_keys = high_card_vocab_keys(vocab)
    config_path = raw_dir / TRAINING_CONFIG_BASENAME
    config = load_training_config(config_path, high_card_keys)

    # 4. Resolve device (TR-CFG-10).
    device = resolve_device(config.device)
    _log_summary(
        f"device: requested={config.device} resolved={device.name}"
        + (f" cuda={device.cuda_device_name} cuda_version={device.cuda_version}"
           if device.name == "cuda" else "")
    )

    # 5. Verify strategy availability (TR-IN-08).
    splits = load_splits_artifact(processed_dir)
    verify_strategy_availability(splits, config.strategies)

    # 6. Load features for both shapes; verify label parity (TR-NF-02).
    features_by_shape = {
        shape: load_features(processed_dir, shape) for shape in ("flat", "pos")
    }
    assert_label_parity(features_by_shape["flat"], features_by_shape["pos"])

    classifications_by_shape: dict[str, ColumnClassification] = {
        shape: classify_columns(list(features_by_shape[shape].columns), vocab)
        for shape in ("flat", "pos")
    }

    # Encoder factories: each call returns a fresh encoder (so per-combination
    # seeding produces matching init weights).
    def _make_factory(shape: str):
        cls = classifications_by_shape[shape]
        def factory() -> FeatureEncoder:
            return FeatureEncoder(cls, vocab, config.embedding_dims)
        return factory
    encoders_factory_by_shape = {
        "flat": _make_factory("flat"),
        "pos": _make_factory("pos"),
    }

    # 7. Enumerate combinations.
    combos = enumerate_combinations(config)
    _log_summary(f"enumerated {len(combos)} combinations:")

    # 8. Dispatch per combination; collect predictions + summaries + loss-curve rows.
    out_dir = ensure_predictions_dir(processed_dir)
    training_summaries: dict[str, Any] = {}
    loss_curve_rows: list[dict[str, Any]] = []
    for combo in combos:
        predictions, summary, combo_loss_rows = _run_one_combo(
            combo,
            features_by_shape=features_by_shape,
            classifications_by_shape=classifications_by_shape,
            encoders_factory_by_shape=encoders_factory_by_shape,
            splits=splits,
            config=config,
            device=device,
        )

        # 9. Write the parquet for this combination.
        filename = combination_filename(combo.rung, combo.shape, combo.strategy)
        out_path = out_dir / filename
        if combo.strategy == "S1":
            write_s1_predictions(predictions, out_path)
        else:
            write_s3_predictions(predictions, out_path)

        training_summaries[combo.manifest_key] = summary
        loss_curve_rows.extend(combo_loss_rows)
        _log_summary(_format_combo_log_line(combo, summary))

    # 9b. Write the Phase 6 loss-curve sidecar parquet (DD-LC-02).
    loss_curves_path = processed_dir / TRAINING_LOSS_CURVES_BASENAME
    loss_curves_df = pd.DataFrame(loss_curve_rows, columns=list(LOSS_CURVES_COLUMNS))
    write_loss_curves(loss_curves_df, loss_curves_path)
    _log_summary(
        f"wrote loss curves: {loss_curves_path} ({len(loss_curve_rows)} rows)"
    )

    # 10. Compute output SHAs against the on-disk parquets.
    output_sha256: dict[str, str] = {}
    for combo in combos:
        key = f"{PREDICTIONS_DIRNAME}/{combination_filename(combo.rung, combo.shape, combo.strategy)}"
        output_sha256[key] = compute_sha256(
            out_dir / combination_filename(combo.rung, combo.shape, combo.strategy)
        )
    output_sha256[TRAINING_LOSS_CURVES_BASENAME] = compute_sha256(loss_curves_path)

    # 11. Build + write manifest LAST (TR-MAN-06).
    training_config_sha256 = compute_sha256(config_path)
    phase2_source_sha256 = {
        f"Data/processed/{basename}": compute_sha256(processed_dir / basename)
        for basename in PHASE2_TRACKED_OUTPUTS
    }
    phase3_source_sha256 = {
        f"Data/processed/{PHASE3_SPLITS_BASENAME}": compute_sha256(
            processed_dir / PHASE3_SPLITS_BASENAME
        ),
    }
    manifest = build_training_manifest(
        config=config,
        resolved_device=device,
        torch_version=torch.__version__,
        phase2_source_sha256=phase2_source_sha256,
        phase3_source_sha256=phase3_source_sha256,
        output_sha256=output_sha256,
        training_config_sha256=training_config_sha256,
        phase2_manifest_git_commit=phase2_manifest.get("git_commit"),
        phase3_manifest_git_commit=phase3_manifest.get("git_commit"),
        training_summaries=training_summaries,
        repo_dir=repo_dir,
    )
    write_training_manifest(manifest, processed_dir / TRAINING_MANIFEST_BASENAME)
    _log_summary(f"wrote manifest: {processed_dir / TRAINING_MANIFEST_BASENAME}")


def _format_combo_log_line(combo: Combination, summary: dict[str, Any]) -> str:
    """One-line per-combination summary for stdout (TR-NF-07)."""
    base = f"  {combo.manifest_key:<40s}"
    if "fold_count" in summary:  # S3
        return (
            f"{base}  S3 folds={summary['fold_count']}  "
            f"mean_val_mae={summary['mean_val_mae']:.4f}"
        )
    # S1
    epochs = summary.get("epochs_trained")
    if epochs is None:
        return f"{base}  S1 val_mae={summary['val_mae']:.4f}  (trivial)"
    return (
        f"{base}  S1 val_mae={summary['val_mae']:.4f}  "
        f"epochs={epochs} best={summary['best_epoch']} "
        f"early={summary['stopped_early']}"
    )
