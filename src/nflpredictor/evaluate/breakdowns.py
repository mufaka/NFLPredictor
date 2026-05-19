"""Breakdown aggregation (§3.6, EV-BRK-01..07, EV-OUT-02).

Each breakdown dimension partitions the games in a cell by a categorical
column and recomputes the metric stack per partition. The output is a long
DataFrame: one row per ``(combination_id, slice, *breakdown_key_columns)``
triple, with the value universe (e.g., all 32 teams, weeks 1–18, every
vocab entry for surface/roof) emitted unconditionally so the file is
"complete by construction" per EV-BRK-06. Empty cells get ``n_games == 0``
and null metric values per EV-MET-09.

Row order across the stacked frame is lexicographic on
``(combination_id, slice, *breakdown_key_columns)`` per EV-OUT-02.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .metrics import METRIC_KEYS, compute_cell_metrics, iter_cells


# Hard-coded universes for dimensions not driven by the Phase 2 vocab.
WEEK_UNIVERSE: tuple[int, ...] = tuple(range(1, 19))   # 2024 regular season: weeks 1–18.
HOME_AWAY_UNIVERSE: tuple[str, ...] = ("away", "home")  # lexicographic for EV-OUT-02.

# NULL sentinel encoded by Phase 2 (FE-NULL-01) — rows with this code for a
# categorical column did not have a value at feature-build time. Excluded
# from every breakdown so we don't emit a "null surface" bucket.
NULL_SENTINEL: int = -1


@dataclass(frozen=True)
class BreakdownSpec:
    """One breakdown dimension's identity + parquet contract."""

    name: str                                # "by_team", "by_week", ...
    breakdown_key_columns: tuple[str, ...]   # extra columns beyond combination_id/slice
    parquet_basename: str                    # "by_team.parquet" etc.


BY_TEAM = BreakdownSpec(
    name="by_team",
    breakdown_key_columns=("team_code", "home_or_away"),
    parquet_basename="by_team.parquet",
)
BY_WEEK = BreakdownSpec(
    name="by_week",
    breakdown_key_columns=("week",),
    parquet_basename="by_week.parquet",
)
BY_HOME_AWAY = BreakdownSpec(
    name="by_home_away",
    breakdown_key_columns=("home_or_away",),
    parquet_basename="by_home_away.parquet",
)
BY_SURFACE = BreakdownSpec(
    name="by_surface",
    breakdown_key_columns=("surface_code", "surface_label"),
    parquet_basename="by_surface.parquet",
)
BY_ROOF = BreakdownSpec(
    name="by_roof",
    breakdown_key_columns=("roof_code", "roof_label"),
    parquet_basename="by_roof.parquet",
)

ALL_SPECS: tuple[BreakdownSpec, ...] = (
    BY_TEAM,
    BY_WEEK,
    BY_HOME_AWAY,
    BY_SURFACE,
    BY_ROOF,
)

SPEC_BY_NAME: dict[str, BreakdownSpec] = {s.name: s for s in ALL_SPECS}


# ---------------------------------------------------------------------------
# Per-spec aggregation
# ---------------------------------------------------------------------------


def _metric_row(
    games: pd.DataFrame,
    base: dict[str, Any],
) -> dict[str, Any]:
    """Compute the canonical 8-metric stack over ``games`` and merge into ``base``."""
    if len(games) == 0:
        cell = compute_cell_metrics(np.empty(0), np.empty(0), np.empty(0), np.empty(0))
    else:
        cell = compute_cell_metrics(
            games["pred_home"].to_numpy(dtype=np.float64, copy=False),
            games["pred_away"].to_numpy(dtype=np.float64, copy=False),
            games["true_home"].to_numpy(dtype=np.float64, copy=False),
            games["true_away"].to_numpy(dtype=np.float64, copy=False),
        )
    out = dict(base)
    out.update(cell)
    return out


def _aggregate_by_team(
    combination_id: str,
    joined: pd.DataFrame,
    strategy: str,
    vocab: dict[str, list[str]],
) -> pd.DataFrame:
    """Each game contributes (home_team, 'home') + (away_team, 'away') (EV-BRK-01)."""
    team_codes_vocab = vocab.get("team_codes", [])
    rows: list[dict[str, Any]] = []
    for cell in iter_cells(joined, strategy):
        # Build {(team_code_int, "home"|"away") -> sub-frame} via masks.
        for role, code_column in (("home", "home_team_code"), ("away", "away_team_code")):
            for code_int, team_label in enumerate(team_codes_vocab):
                mask = cell.games[code_column].to_numpy(copy=False) == code_int
                sub = cell.games.loc[mask]
                base = {
                    "combination_id": combination_id,
                    "slice": cell.slice_name,
                    "team_code": team_label,
                    "home_or_away": role,
                }
                rows.append(_metric_row(sub, base))
    return pd.DataFrame(rows)


def _aggregate_by_week(
    combination_id: str,
    joined: pd.DataFrame,
    strategy: str,
    vocab: dict[str, list[str]],
) -> pd.DataFrame:
    """Each game contributes one row keyed by week (EV-BRK-02)."""
    rows: list[dict[str, Any]] = []
    for cell in iter_cells(joined, strategy):
        weeks_series = cell.games["week"].to_numpy(copy=False)
        for week in WEEK_UNIVERSE:
            mask = weeks_series == week
            sub = cell.games.loc[mask]
            base = {
                "combination_id": combination_id,
                "slice": cell.slice_name,
                "week": int(week),
            }
            rows.append(_metric_row(sub, base))
    return pd.DataFrame(rows)


def _aggregate_by_home_away(
    combination_id: str,
    joined: pd.DataFrame,
    strategy: str,
    vocab: dict[str, list[str]],
) -> pd.DataFrame:
    """Each game contributes two rows: home + away (EV-BRK-03).

    Both rows compute the standard 8-metric stack over the same cell games;
    the breakdown_key column ``home_or_away`` distinguishes them. mae_home /
    rmse_home naturally surface the home-side view; mae_away / rmse_away the
    away-side view; wl_accuracy / spread_mae / total_mae are whole-game per
    EV-MET-04..06 and consequently identical across the two rows.
    """
    rows: list[dict[str, Any]] = []
    for cell in iter_cells(joined, strategy):
        for role in HOME_AWAY_UNIVERSE:
            base = {
                "combination_id": combination_id,
                "slice": cell.slice_name,
                "home_or_away": role,
            }
            rows.append(_metric_row(cell.games, base))
    return pd.DataFrame(rows)


def _aggregate_by_categorical(
    combination_id: str,
    joined: pd.DataFrame,
    strategy: str,
    vocab: dict[str, list[str]],
    *,
    feature_column: str,
    vocab_key: str,
    code_column_name: str,
    label_column_name: str,
) -> pd.DataFrame:
    """Generic surface/roof aggregator (EV-BRK-04, EV-BRK-05).

    Each game contributes one row keyed by its categorical code; the universe
    is ``vocab[vocab_key]`` so absent values still get a null-metric row.
    NULL-coded games (sentinel ``-1`` from Phase 2) are excluded.
    """
    values = vocab.get(vocab_key, [])
    rows: list[dict[str, Any]] = []
    for cell in iter_cells(joined, strategy):
        codes_series = cell.games[feature_column].to_numpy(copy=False)
        for code_int, label in enumerate(values):
            mask = codes_series == code_int
            sub = cell.games.loc[mask]
            base = {
                "combination_id": combination_id,
                "slice": cell.slice_name,
                code_column_name: int(code_int),
                label_column_name: label,
            }
            rows.append(_metric_row(sub, base))
    return pd.DataFrame(rows)


def _aggregate_by_surface(
    combination_id: str,
    joined: pd.DataFrame,
    strategy: str,
    vocab: dict[str, list[str]],
) -> pd.DataFrame:
    return _aggregate_by_categorical(
        combination_id,
        joined,
        strategy,
        vocab,
        feature_column="surface",
        vocab_key="surface",
        code_column_name="surface_code",
        label_column_name="surface_label",
    )


def _aggregate_by_roof(
    combination_id: str,
    joined: pd.DataFrame,
    strategy: str,
    vocab: dict[str, list[str]],
) -> pd.DataFrame:
    return _aggregate_by_categorical(
        combination_id,
        joined,
        strategy,
        vocab,
        feature_column="roof",
        vocab_key="roof",
        code_column_name="roof_code",
        label_column_name="roof_label",
    )


_AGGREGATORS: dict[str, Callable[..., pd.DataFrame]] = {
    "by_team": _aggregate_by_team,
    "by_week": _aggregate_by_week,
    "by_home_away": _aggregate_by_home_away,
    "by_surface": _aggregate_by_surface,
    "by_roof": _aggregate_by_roof,
}


def aggregate_breakdown(
    combination_id: str,
    joined: pd.DataFrame,
    strategy: str,
    spec: BreakdownSpec,
    vocab: dict[str, list[str]],
) -> pd.DataFrame:
    """Apply ``spec``'s aggregation to a single combination's joined predictions."""
    aggregator = _AGGREGATORS[spec.name]
    return aggregator(combination_id, joined, strategy, vocab)


def build_breakdown_table(
    spec: BreakdownSpec,
    ordered_combinations: Iterable[tuple[str, str]],
    joined_by_combo: dict[str, pd.DataFrame],
    vocab: dict[str, list[str]],
) -> pd.DataFrame:
    """Stack one breakdown table across every combination.

    ``ordered_combinations`` is an iterable of ``(combination_id, strategy)``
    pairs in the lexicographic order they should appear in the output. The
    returned frame's row order is sorted lexicographically on
    ``(combination_id, slice, *spec.breakdown_key_columns)`` per EV-OUT-02.
    """
    frames: list[pd.DataFrame] = []
    for combination_id, strategy in ordered_combinations:
        joined = joined_by_combo[combination_id]
        frames.append(aggregate_breakdown(combination_id, joined, strategy, spec, vocab))
    if not frames:
        # Empty schema with the right columns so callers can write a 0-row parquet.
        columns = (
            ["combination_id", "slice", *spec.breakdown_key_columns, "n_games", *METRIC_KEYS]
        )
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    stacked = pd.concat(frames, ignore_index=True)
    sort_columns = ["combination_id", "slice", *spec.breakdown_key_columns]
    return stacked.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
