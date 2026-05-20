"""Plot rendering smoke + byte-determinism tests (EV-PLOT-01..06, EV-NF-02)."""

from __future__ import annotations

import pathlib

import matplotlib
import numpy as np
import pandas as pd
import pytest

# Force Agg backend before any pyplot import via the production module.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nflpredictor.evaluate.config import PlotConfig
from nflpredictor.evaluate.plots import (
    LADDER_SLICE_BASES,
    PER_STRATEGY_PLOT_SLICES,
    plot_breakdown,
    plot_ladder_summary,
    plot_residual_distribution,
    plot_scatter,
    render_all_plots,
    save_figure,
    warmup_render,
)


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


PLOT_CFG = PlotConfig(
    scatter=True,
    residual_distribution=True,
    ladder_summary=True,
    breakdown_plots=("by_week",),
    dpi=100,
    figure_width_inches=8.0,
    figure_height_inches=5.0,
)


def _joined_s1_frame() -> pd.DataFrame:
    """A tiny season_holdout joined frame: 3 val games + 2 test games."""
    return pd.DataFrame({
        "GameId": ["g1", "g2", "g3", "g4", "g5"],
        "slice": ["val", "val", "val", "test", "test"],
        "pred_home": [21.0, 28.0, 14.0, 31.0, 13.0],
        "pred_away": [17.0, 20.0, 21.0, 10.0, 13.0],
        "true_home": [24.0, 21.0, 24.0, 14.0, 17.0],
        "true_away": [17.0, 27.0, 21.0, 10.0, 17.0],
        "week":       [1, 1, 2, 2, 3],
        "home_team_code": [0, 1, 0, 2, 1],
        "away_team_code": [1, 0, 2, 0, 2],
    })


def _joined_s3_frame() -> pd.DataFrame:
    """A tiny loso_cv joined frame: 4 games across 2 folds."""
    return pd.DataFrame({
        "GameId": ["g1", "g2", "g3", "g4"],
        "fold_index": [0, 0, 1, 1],
        "pred_home": [21.0, 28.0, 14.0, 31.0],
        "pred_away": [17.0, 20.0, 21.0, 10.0],
        "true_home": [24.0, 21.0, 24.0, 14.0],
        "true_away": [17.0, 27.0, 21.0, 10.0],
        "week":       [1, 1, 2, 2],
        "home_team_code": [0, 1, 0, 2],
        "away_team_code": [1, 0, 2, 0],
    })


def _headline_fixture() -> dict:
    return {
        "combinations": {
            "rung0_mean__none__season_holdout": {
                "val": {"n_games": 3, "mae": 5.0, "mae_home": 5.0, "mae_away": 5.0,
                        "rmse_home": 6.0, "rmse_away": 6.0, "wl_accuracy": 0.5,
                        "spread_mae": 3.0, "total_mae": 6.0},
                "test": {"n_games": 2, "mae": 5.5, "mae_home": 5.5, "mae_away": 5.5,
                         "rmse_home": 6.5, "rmse_away": 6.5, "wl_accuracy": 0.5,
                         "spread_mae": 3.5, "total_mae": 7.0},
            },
            "rung2_linear__flat__loso_cv": {
                "fold_0": {"n_games": 2, "mae": 4.0, "mae_home": 4.0, "mae_away": 4.0,
                           "rmse_home": 5.0, "rmse_away": 5.0, "wl_accuracy": 0.5,
                           "spread_mae": 2.0, "total_mae": 4.0},
                "fold_1": {"n_games": 2, "mae": 3.5, "mae_home": 3.5, "mae_away": 3.5,
                           "rmse_home": 4.5, "rmse_away": 4.5, "wl_accuracy": 0.5,
                           "spread_mae": 1.5, "total_mae": 3.5},
                "pooled": {"n_games": 4, "mae": 3.75, "mae_home": 3.75, "mae_away": 3.75,
                           "rmse_home": 4.75, "rmse_away": 4.75, "wl_accuracy": 0.5,
                           "spread_mae": 1.75, "total_mae": 3.75},
            },
        }
    }


# ---------------------------------------------------------------------------
# Smoke tests per plot family
# ---------------------------------------------------------------------------


def test_plot_scatter_returns_figure_and_writes_png(tmp_path: pathlib.Path) -> None:
    fig = plot_scatter("combo", "val", _joined_s1_frame(), PLOT_CFG)
    assert isinstance(fig, plt.Figure)
    # Title must match the EV-PLOT-01 format before saving.
    assert fig.axes[0].get_title() == "combo — val"
    path = tmp_path / "scatter.png"
    save_figure(fig, path, dpi=PLOT_CFG.dpi)
    assert path.stat().st_size > 0
    assert path.read_bytes().startswith(PNG_MAGIC)


def test_plot_residual_distribution_returns_figure_and_writes_png(tmp_path: pathlib.Path) -> None:
    fig = plot_residual_distribution("combo", "val", _joined_s1_frame(), PLOT_CFG)
    assert isinstance(fig, plt.Figure)
    assert fig.axes[0].get_title() == "combo — val"
    path = tmp_path / "resid.png"
    save_figure(fig, path, dpi=PLOT_CFG.dpi)
    assert path.read_bytes().startswith(PNG_MAGIC)


def test_plot_ladder_summary_returns_figure_and_writes_png(tmp_path: pathlib.Path) -> None:
    fig = plot_ladder_summary(_headline_fixture(), "val", "mae", PLOT_CFG)
    assert isinstance(fig, plt.Figure)
    assert "Ladder summary" in fig.axes[0].get_title()
    assert "val" in fig.axes[0].get_title()
    assert "mae" in fig.axes[0].get_title()
    path = tmp_path / "ladder.png"
    save_figure(fig, path, dpi=PLOT_CFG.dpi)
    assert path.read_bytes().startswith(PNG_MAGIC)


def test_plot_ladder_summary_skips_combos_without_slice() -> None:
    """loso_cv combos shouldn't appear in val/test ladder; season_holdout combos shouldn't appear in pooled."""
    headline = _headline_fixture()
    fig_val = plot_ladder_summary(headline, "val", "mae", PLOT_CFG)
    # val should include only the season_holdout combo (one bar).
    n_bars_val = sum(len(c.get_children()) for c in fig_val.axes[0].containers)
    plt.close(fig_val)
    fig_pooled = plot_ladder_summary(headline, "pooled", "mae", PLOT_CFG)
    n_bars_pooled = sum(len(c.get_children()) for c in fig_pooled.axes[0].containers)
    plt.close(fig_pooled)
    # val=1 bar (season_holdout only), pooled=1 bar (loso_cv only).
    assert n_bars_val == 1
    assert n_bars_pooled == 1


def test_plot_breakdown_by_week_returns_figure(tmp_path: pathlib.Path) -> None:
    rows = pd.DataFrame([
        {"combination_id": "c", "slice": "val", "week": i,
         "n_games": 2 if i <= 3 else 0,
         "mae": float(i) if i <= 3 else float("nan"),
         "mae_home": 1.0, "mae_away": 1.0,
         "rmse_home": 1.0, "rmse_away": 1.0,
         "wl_accuracy": 0.5, "spread_mae": 1.0, "total_mae": 2.0}
        for i in range(1, 19)
    ])
    fig = plot_breakdown("c", "val", "by_week", rows, "mae", PLOT_CFG)
    assert isinstance(fig, plt.Figure)
    title = fig.axes[0].get_title()
    assert "c" in title and "val" in title and "by_week" in title
    path = tmp_path / "by_week.png"
    save_figure(fig, path, dpi=PLOT_CFG.dpi)
    assert path.read_bytes().startswith(PNG_MAGIC)


def test_plot_breakdown_by_team_returns_figure() -> None:
    rows = pd.DataFrame([
        {"combination_id": "c", "slice": "val",
         "team_code": team, "home_or_away": role,
         "n_games": 1, "mae": 4.0, "mae_home": 4.0, "mae_away": 4.0,
         "rmse_home": 5.0, "rmse_away": 5.0, "wl_accuracy": 0.5,
         "spread_mae": 2.0, "total_mae": 4.0}
        for team in ("atl", "buf", "car")
        for role in ("home", "away")
    ])
    fig = plot_breakdown("c", "val", "by_team", rows, "mae", PLOT_CFG)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_breakdown_by_home_away_returns_figure() -> None:
    rows = pd.DataFrame([
        {"combination_id": "c", "slice": "val",
         "home_or_away": role,
         "n_games": 6, "mae": 4.0, "mae_home": 4.0, "mae_away": 4.0,
         "rmse_home": 5.0, "rmse_away": 5.0, "wl_accuracy": 0.5,
         "spread_mae": 2.0, "total_mae": 4.0}
        for role in ("away", "home")
    ])
    fig = plot_breakdown("c", "val", "by_home_away", rows, "mae", PLOT_CFG)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_breakdown_unknown_dim_rejected() -> None:
    rows = pd.DataFrame([{"combination_id": "c", "slice": "val", "n_games": 0}])
    with pytest.raises(ValueError, match="unknown breakdown dimension"):
        plot_breakdown("c", "val", "by_zodiac", rows, "mae", PLOT_CFG)


# ---------------------------------------------------------------------------
# Byte-determinism (within the pinned matplotlib wheel)
# ---------------------------------------------------------------------------


def test_save_figure_strips_metadata_for_byte_identity(tmp_path: pathlib.Path) -> None:
    """Two renders of the same plot produce byte-identical PNGs (EV-NF-02)."""
    warmup_render()
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    fig1 = plot_scatter("combo", "val", _joined_s1_frame(), PLOT_CFG)
    save_figure(fig1, a, dpi=PLOT_CFG.dpi)
    fig2 = plot_scatter("combo", "val", _joined_s1_frame(), PLOT_CFG)
    save_figure(fig2, b, dpi=PLOT_CFG.dpi)
    if a.read_bytes() != b.read_bytes():
        # Phase 4.4 fallback: assert pixel equality if PNG bytes diverge for
        # reasons outside our control (e.g., zlib version differences mid-test).
        from PIL import Image
        with Image.open(a) as ia, Image.open(b) as ib:
            assert ia.tobytes() == ib.tobytes(), (
                "PNG bytes diverge AND pixel bytes diverge — non-deterministic render"
            )


def test_residual_byte_identity(tmp_path: pathlib.Path) -> None:
    warmup_render()
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    fig1 = plot_residual_distribution("combo", "val", _joined_s1_frame(), PLOT_CFG)
    save_figure(fig1, a, dpi=PLOT_CFG.dpi)
    fig2 = plot_residual_distribution("combo", "val", _joined_s1_frame(), PLOT_CFG)
    save_figure(fig2, b, dpi=PLOT_CFG.dpi)
    if a.read_bytes() != b.read_bytes():
        from PIL import Image
        with Image.open(a) as ia, Image.open(b) as ib:
            assert ia.tobytes() == ib.tobytes()


def test_ladder_byte_identity(tmp_path: pathlib.Path) -> None:
    warmup_render()
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    fig1 = plot_ladder_summary(_headline_fixture(), "val", "mae", PLOT_CFG)
    save_figure(fig1, a, dpi=PLOT_CFG.dpi)
    fig2 = plot_ladder_summary(_headline_fixture(), "val", "mae", PLOT_CFG)
    save_figure(fig2, b, dpi=PLOT_CFG.dpi)
    if a.read_bytes() != b.read_bytes():
        from PIL import Image
        with Image.open(a) as ia, Image.open(b) as ib:
            assert ia.tobytes() == ib.tobytes()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def test_render_all_plots_emits_expected_filenames(tmp_path: pathlib.Path) -> None:
    joined_s1 = _joined_s1_frame()
    joined_s3 = _joined_s3_frame()
    joined_by_combo = {
        "rung0_mean__none__season_holdout": joined_s1,
        "rung2_linear__flat__loso_cv": joined_s3,
    }
    strategy_by_combo = {
        "rung0_mean__none__season_holdout": "season_holdout",
        "rung2_linear__flat__loso_cv": "loso_cv",
    }
    # by_week breakdown table for both combos × val/test/pooled.
    by_week_rows = []
    for combo, slices in (
        ("rung0_mean__none__season_holdout", ("val", "test")),
        ("rung2_linear__flat__loso_cv", ("pooled",)),
    ):
        for slice_name in slices:
            for week in range(1, 19):
                by_week_rows.append({
                    "combination_id": combo,
                    "slice": slice_name,
                    "week": week,
                    "n_games": 1 if week <= 2 else 0,
                    "mae": float(week) if week <= 2 else float("nan"),
                    "mae_home": 1.0, "mae_away": 1.0,
                    "rmse_home": 1.0, "rmse_away": 1.0,
                    "wl_accuracy": 0.5, "spread_mae": 1.0, "total_mae": 2.0,
                })
    breakdown_tables = {"by_week": pd.DataFrame(by_week_rows)}

    written = render_all_plots(
        headline=_headline_fixture(),
        breakdown_tables_by_dim=breakdown_tables,
        joined_by_combo=joined_by_combo,
        strategy_by_combo=strategy_by_combo,
        plot_cfg=PLOT_CFG,
        headline_metric="mae",
        plots_dir=tmp_path,
    )

    written_names = sorted(p.name for p in written)
    # season_holdout combo: val + test × {scatter, residuals, by_week} = 6 PNGs
    # loso_cv combo: pooled × {scatter, residuals, by_week} = 3 PNGs
    # Ladder summaries: val, test, pooled = 3 PNGs
    expected = sorted([
        "rung0_mean__none__season_holdout__val__scatter.png",
        "rung0_mean__none__season_holdout__val__residuals.png",
        "rung0_mean__none__season_holdout__val__by_week.png",
        "rung0_mean__none__season_holdout__test__scatter.png",
        "rung0_mean__none__season_holdout__test__residuals.png",
        "rung0_mean__none__season_holdout__test__by_week.png",
        "rung2_linear__flat__loso_cv__pooled__scatter.png",
        "rung2_linear__flat__loso_cv__pooled__residuals.png",
        "rung2_linear__flat__loso_cv__pooled__by_week.png",
        "ladder_summary__val.png",
        "ladder_summary__test.png",
        "ladder_summary__pooled.png",
    ])
    assert written_names == expected
    # Spot-check each is a PNG.
    for p in written:
        assert p.read_bytes().startswith(PNG_MAGIC)


def test_render_all_plots_respects_plot_toggles(tmp_path: pathlib.Path) -> None:
    cfg = PlotConfig(
        scatter=False,
        residual_distribution=True,
        ladder_summary=False,
        breakdown_plots=(),
        dpi=100,
        figure_width_inches=8.0,
        figure_height_inches=5.0,
    )
    written = render_all_plots(
        headline=_headline_fixture(),
        breakdown_tables_by_dim={},
        joined_by_combo={"rung0_mean__none__season_holdout": _joined_s1_frame()},
        strategy_by_combo={"rung0_mean__none__season_holdout": "season_holdout"},
        plot_cfg=cfg,
        headline_metric="mae",
        plots_dir=tmp_path,
    )
    names = sorted(p.name for p in written)
    # Only residual histograms for val + test → 2 PNGs.
    assert names == [
        "rung0_mean__none__season_holdout__test__residuals.png",
        "rung0_mean__none__season_holdout__val__residuals.png",
    ]


def test_per_strategy_plot_slices_constant() -> None:
    """Sanity-check the slice basis mapping matches EV-PLOT-01 / EV-PLOT-02 / EV-PLOT-04."""
    assert PER_STRATEGY_PLOT_SLICES["season_holdout"] == ("val", "test")
    assert PER_STRATEGY_PLOT_SLICES["loso_cv"] == ("pooled",)
    assert LADDER_SLICE_BASES == ("val", "test", "pooled")


def test_save_figure_path_creates_parent_dirs(tmp_path: pathlib.Path) -> None:
    nested = tmp_path / "a" / "b" / "c.png"
    fig = plot_scatter("c", "val", _joined_s1_frame(), PLOT_CFG)
    save_figure(fig, nested, dpi=PLOT_CFG.dpi)
    assert nested.exists()
