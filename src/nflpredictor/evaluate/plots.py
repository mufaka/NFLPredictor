"""Calibration plot rendering (§3.7, EV-PLOT-01..07).

Determinism contract (EV-NF-02): metric content is byte-deterministic
unconditionally; PNG byte identity holds only within a pinned matplotlib
wheel on the same platform. ``save_figure`` strips the two PNG tEXt chunks
that matplotlib otherwise writes (``Software`` + ``Creation Time``) so the
output bytes don't bake in the matplotlib version string or wall-clock.

The Agg backend is selected at import time (``EV-PLOT-05``) before any
``matplotlib.pyplot`` import so headless rendering is locked in even if a
caller has previously selected an interactive backend.
"""

from __future__ import annotations

import pathlib
from typing import Any, Optional

import matplotlib  # noqa: E402

# Required for headless determinism (EV-PLOT-05). Must run before any pyplot
# import — including the one immediately below.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from .config import PlotConfig


# Bin specification for residual histograms — fixed so binning is deterministic
# across runs (EV-PLOT-06). NFL score residuals routinely live in ±30 points.
_RESIDUAL_BIN_EDGES: np.ndarray = np.arange(-30.0, 32.0, 2.0)

# Slice bases that the per-combination plot families render for, per strategy
# (EV-PLOT-01 / EV-PLOT-02 / EV-PLOT-04).
PER_STRATEGY_PLOT_SLICES: dict[str, tuple[str, ...]] = {
    "season_holdout": ("val", "test"),
    "loso_cv": ("pooled",),
}

# Slice bases for the ladder summary (EV-PLOT-03).
LADDER_SLICE_BASES: tuple[str, ...] = ("val", "test", "pooled")


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def save_figure(fig: plt.Figure, path: pathlib.Path, *, dpi: int) -> None:
    """Save ``fig`` to ``path`` with PNG metadata stripped, then close it.

    Per EV-PLOT-05: ``metadata={"Software": None, "Creation Time": None}``
    removes the two tEXt chunks matplotlib writes by default, which otherwise
    bake in the matplotlib version string and the wall-clock time of the run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi,
        metadata={"Software": None, "Creation Time": None},
    )
    plt.close(fig)


def _figsize(plot_cfg: PlotConfig) -> tuple[float, float]:
    return (plot_cfg.figure_width_inches, plot_cfg.figure_height_inches)


# ---------------------------------------------------------------------------
# Per-combination scatter (EV-PLOT-01)
# ---------------------------------------------------------------------------


def _slice_games(joined: pd.DataFrame, slice_name: str) -> pd.DataFrame:
    """Filter the joined predictions frame to one slice basis.

    ``"pooled"`` returns the entire frame (loso_cv pooled = concat of all folds).
    Otherwise the frame is filtered by the ``slice`` column (season_holdout val / test).
    """
    if slice_name == "pooled":
        return joined
    if "slice" not in joined.columns:
        raise ValueError(
            f"cannot slice frame by {slice_name!r}: 'slice' column not present "
            "(is this an loso_cv frame? loso_cv only renders the 'pooled' basis)"
        )
    return joined.loc[joined["slice"] == slice_name]


def plot_scatter(
    combination_id: str,
    slice_name: str,
    joined: pd.DataFrame,
    plot_cfg: PlotConfig,
) -> plt.Figure:
    """Predicted-vs-actual scatter with home + away overlaid (EV-PLOT-01)."""
    games = _slice_games(joined, slice_name)
    fig, ax = plt.subplots(figsize=_figsize(plot_cfg))
    if len(games) == 0:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        ax.scatter(games["true_home"], games["pred_home"], label="Home", alpha=0.7)
        ax.scatter(games["true_away"], games["pred_away"], label="Away", alpha=0.7)
        all_scores = np.concatenate([
            games["true_home"].to_numpy(),
            games["pred_home"].to_numpy(),
            games["true_away"].to_numpy(),
            games["pred_away"].to_numpy(),
        ])
        lo = float(np.min(all_scores))
        hi = float(np.max(all_scores))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", label="y = x")
    ax.set_xlabel("Actual score")
    ax.set_ylabel("Predicted score")
    ax.set_title(f"{combination_id} — {slice_name}")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-combination residual histogram (EV-PLOT-02)
# ---------------------------------------------------------------------------


def plot_residual_distribution(
    combination_id: str,
    slice_name: str,
    joined: pd.DataFrame,
    plot_cfg: PlotConfig,
) -> plt.Figure:
    """Histogram of ``pred − true`` residuals, home + away overlaid (EV-PLOT-02)."""
    games = _slice_games(joined, slice_name)
    fig, ax = plt.subplots(figsize=_figsize(plot_cfg))
    if len(games) > 0:
        home_resid = (games["pred_home"] - games["true_home"]).to_numpy()
        away_resid = (games["pred_away"] - games["true_away"]).to_numpy()
        ax.hist(
            home_resid,
            bins=_RESIDUAL_BIN_EDGES,
            alpha=0.5,
            label="Home",
        )
        ax.hist(
            away_resid,
            bins=_RESIDUAL_BIN_EDGES,
            alpha=0.5,
            label="Away",
        )
    ax.set_xlabel("Residual (pred − true)")
    ax.set_ylabel("Count")
    ax.set_title(f"{combination_id} — {slice_name}")
    ax.axvline(0.0, linestyle="--", color="gray")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Ladder summary (EV-PLOT-03)
# ---------------------------------------------------------------------------


def _combinations_with_slice(
    headline: dict[str, Any], slice_basis: str
) -> list[tuple[str, float]]:
    """Return ``(combination_id, headline_value)`` pairs for combos that have ``slice_basis``.

    Sorted lexicographically on combination id (EV-PLOT-03).
    """
    out: list[tuple[str, float]] = []
    combo_block = headline.get("combinations", {})
    for combo_id in sorted(combo_block.keys()):
        per_combo = combo_block[combo_id]
        if slice_basis in per_combo:
            cell = per_combo[slice_basis]
            value = cell.get("mae")  # caller passes the right metric key separately
            out.append((combo_id, value))
    return out


def plot_ladder_summary(
    headline: dict[str, Any],
    slice_basis: str,
    headline_metric: str,
    plot_cfg: PlotConfig,
) -> plt.Figure:
    """Per-combination bar chart of the configured headline metric (EV-PLOT-03).

    Combinations whose per-combo block does not contain ``slice_basis`` are
    omitted (e.g., loso_cv combos when ``slice_basis == "test"``). Bars are sorted
    lexicographically.
    """
    combo_block = headline.get("combinations", {})
    pairs: list[tuple[str, Optional[float]]] = []
    for combo_id in sorted(combo_block.keys()):
        per_combo = combo_block[combo_id]
        if slice_basis in per_combo:
            cell = per_combo[slice_basis]
            pairs.append((combo_id, cell.get(headline_metric)))

    fig, ax = plt.subplots(figsize=_figsize(plot_cfg))
    if pairs:
        ids = [p[0] for p in pairs]
        # None → NaN so matplotlib renders an empty slot.
        values = [float("nan") if p[1] is None else float(p[1]) for p in pairs]
        x_positions = np.arange(len(ids))
        ax.bar(x_positions, values)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(ids, rotation=45, ha="right")
    ax.set_ylabel(headline_metric)
    ax.set_title(f"Ladder summary — {slice_basis} {headline_metric}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-breakdown plots (EV-PLOT-04)
# ---------------------------------------------------------------------------


def _values_for(rows: pd.DataFrame, metric: str) -> np.ndarray:
    """Pull a metric column out of a breakdown row frame as float64, NaN-tolerant."""
    return rows[metric].astype("float64").to_numpy()


def plot_breakdown(
    combination_id: str,
    slice_name: str,
    dim_name: str,
    breakdown_rows: pd.DataFrame,
    headline_metric: str,
    plot_cfg: PlotConfig,
) -> plt.Figure:
    """Per-breakdown plot for one (combination, slice, dim) (EV-PLOT-04).

    Renders different chart types per dimension:
    - ``by_week``: line chart, x = week, y = headline_metric.
    - ``by_team``: horizontal bar chart, y = team_code (sorted lex), x = headline_metric.
                   Both home and away rows shown (color-distinguished).
    - ``by_home_away``: two-bar chart, x = role, y = headline_metric.
    """
    fig, ax = plt.subplots(figsize=_figsize(plot_cfg))
    title = f"{combination_id} — {slice_name} — {dim_name} ({headline_metric})"

    if dim_name == "by_week":
        weeks = breakdown_rows["week"].astype("int64").to_numpy()
        order = np.argsort(weeks)
        ax.plot(weeks[order], _values_for(breakdown_rows.iloc[order], headline_metric), marker="o")
        ax.set_xlabel("Week")
        ax.set_ylabel(headline_metric)
    elif dim_name == "by_team":
        rows = breakdown_rows.sort_values(["team_code", "home_or_away"], kind="mergesort")
        teams = list(rows["team_code"].astype(str))
        roles = list(rows["home_or_away"].astype(str))
        labels = [f"{t} ({r})" for t, r in zip(teams, roles)]
        values = _values_for(rows, headline_metric)
        y_positions = np.arange(len(labels))
        ax.barh(y_positions, values)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # alphabetical top-down
        ax.set_xlabel(headline_metric)
    elif dim_name == "by_home_away":
        rows = breakdown_rows.sort_values("home_or_away", kind="mergesort")
        roles = list(rows["home_or_away"].astype(str))
        values = _values_for(rows, headline_metric)
        x_positions = np.arange(len(roles))
        ax.bar(x_positions, values)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(roles)
        ax.set_ylabel(headline_metric)
    else:
        raise ValueError(
            f"unknown breakdown dimension for plotting: {dim_name!r}; "
            f"expected one of {{by_week, by_team, by_home_away}}"
        )

    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def render_all_plots(
    *,
    headline: dict[str, Any],
    breakdown_tables_by_dim: dict[str, pd.DataFrame],
    joined_by_combo: dict[str, pd.DataFrame],
    strategy_by_combo: dict[str, str],
    plot_cfg: PlotConfig,
    headline_metric: str,
    plots_dir: pathlib.Path,
) -> list[pathlib.Path]:
    """Render every enabled plot and return the list of written PNG paths.

    Combinations are iterated in lexicographic order. Per-combination plot
    slice bases come from ``PER_STRATEGY_PLOT_SLICES`` for each combo's
    strategy. The ladder-summary slice bases come from ``LADDER_SLICE_BASES``;
    summaries are rendered only when at least one combination contributes a
    row for that basis.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []

    # Per-combination plots
    for combination_id in sorted(joined_by_combo.keys()):
        strategy = strategy_by_combo[combination_id]
        joined = joined_by_combo[combination_id]
        for slice_name in PER_STRATEGY_PLOT_SLICES[strategy]:
            if plot_cfg.scatter:
                path = plots_dir / f"{combination_id}__{slice_name}__scatter.png"
                fig = plot_scatter(combination_id, slice_name, joined, plot_cfg)
                save_figure(fig, path, dpi=plot_cfg.dpi)
                written.append(path)
            if plot_cfg.residual_distribution:
                path = plots_dir / f"{combination_id}__{slice_name}__residuals.png"
                fig = plot_residual_distribution(combination_id, slice_name, joined, plot_cfg)
                save_figure(fig, path, dpi=plot_cfg.dpi)
                written.append(path)
            for dim_name in plot_cfg.breakdown_plots:
                table = breakdown_tables_by_dim.get(dim_name)
                if table is None:
                    continue
                rows = table[
                    (table["combination_id"] == combination_id)
                    & (table["slice"] == slice_name)
                ]
                if len(rows) == 0:
                    continue
                path = plots_dir / f"{combination_id}__{slice_name}__{dim_name}.png"
                fig = plot_breakdown(
                    combination_id, slice_name, dim_name, rows, headline_metric, plot_cfg
                )
                save_figure(fig, path, dpi=plot_cfg.dpi)
                written.append(path)

    # Ladder summary plots
    if plot_cfg.ladder_summary:
        for slice_basis in LADDER_SLICE_BASES:
            # Only render if at least one combination contributes.
            contributing = [
                combo_id
                for combo_id, per_combo in headline.get("combinations", {}).items()
                if slice_basis in per_combo
            ]
            if not contributing:
                continue
            path = plots_dir / f"ladder_summary__{slice_basis}.png"
            fig = plot_ladder_summary(headline, slice_basis, headline_metric, plot_cfg)
            save_figure(fig, path, dpi=plot_cfg.dpi)
            written.append(path)

    return written


def warmup_render() -> None:
    """Render a throwaway figure to prime font caches (used by tests + first run).

    Matplotlib lazily builds its font cache on first use; for byte-identity
    tests across two successive renders this warm-up guarantees the cache
    state is identical between the two calls.
    """
    fig, ax = plt.subplots(figsize=(1.0, 1.0))
    ax.plot([0, 1], [0, 1])
    ax.set_title("warmup")
    plt.close(fig)
