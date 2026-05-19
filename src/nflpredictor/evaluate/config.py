"""evaluation_config.yaml loader + validator (§3.2, §4.1)."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

import yaml


ALLOWED_HEADLINE_METRICS: frozenset[str] = frozenset({
    "mae", "rmse", "wl_accuracy", "spread_mae", "total_mae",
})

ALLOWED_BREAKDOWN_KEYS: frozenset[str] = frozenset({
    "by_team", "by_week", "by_home_away", "by_surface", "by_roof",
})

ALLOWED_BREAKDOWN_PLOT_KEYS: frozenset[str] = frozenset({
    "by_week", "by_team", "by_home_away",
})

_TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    "evaluation_version",
    "headline_metric",
    "breakdowns",
    "plots",
})

_PLOT_KEYS: frozenset[str] = frozenset({
    "scatter",
    "residual_distribution",
    "ladder_summary",
    "breakdown_plots",
    "dpi",
    "figure_width_inches",
    "figure_height_inches",
})


@dataclass(frozen=True)
class BreakdownToggles:
    by_team: bool
    by_week: bool
    by_home_away: bool
    by_surface: bool
    by_roof: bool

    def is_enabled(self, name: str) -> bool:
        return bool(getattr(self, name))


@dataclass(frozen=True)
class PlotConfig:
    scatter: bool
    residual_distribution: bool
    ladder_summary: bool
    breakdown_plots: tuple[str, ...]
    dpi: int
    figure_width_inches: float
    figure_height_inches: float


@dataclass(frozen=True)
class EvaluationConfig:
    evaluation_version: str
    headline_metric: str
    breakdowns: BreakdownToggles
    plots: PlotConfig


class EvaluationConfigError(ValueError):
    """Raised for malformed evaluation_config.yaml content (§3.2)."""


def load_evaluation_config(path: pathlib.Path) -> EvaluationConfig:
    """Read, parse, and validate evaluation_config.yaml (EV-IN-04, EV-IN-09, EV-CFG-01..08)."""
    if not path.exists():
        raise FileNotFoundError(f"evaluation_config.yaml not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)  # EV-SEC-04
    if not isinstance(raw, dict):
        raise EvaluationConfigError(
            f"evaluation_config.yaml must be a mapping at the top level; "
            f"got {type(raw).__name__}"
        )
    return _parse(raw)


def _parse(raw: dict[str, Any]) -> EvaluationConfig:
    unknown = set(raw.keys()) - _TOP_LEVEL_KEYS
    if unknown:
        raise EvaluationConfigError(
            f"unknown top-level keys in evaluation_config.yaml: {sorted(unknown)}"
        )
    missing = _TOP_LEVEL_KEYS - set(raw.keys())
    if missing:
        raise EvaluationConfigError(
            f"missing required top-level keys: {sorted(missing)}"
        )

    breakdowns = _parse_breakdowns(raw["breakdowns"])
    plots = _parse_plots(raw["plots"], breakdowns)

    return EvaluationConfig(
        evaluation_version=_parse_evaluation_version(raw["evaluation_version"]),
        headline_metric=_parse_headline_metric(raw["headline_metric"]),
        breakdowns=breakdowns,
        plots=plots,
    )


def _parse_evaluation_version(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationConfigError(
            f"evaluation_version must be a non-empty string; got {value!r}"
        )
    return value


def _parse_headline_metric(value: Any) -> str:
    if not isinstance(value, str) or value not in ALLOWED_HEADLINE_METRICS:
        raise EvaluationConfigError(
            f"headline_metric must be one of {sorted(ALLOWED_HEADLINE_METRICS)}; "
            f"got {value!r}"
        )
    return value


def _parse_breakdowns(block: Any) -> BreakdownToggles:
    if not isinstance(block, dict):
        raise EvaluationConfigError(
            f"breakdowns must be a mapping; got {type(block).__name__}"
        )
    unknown = set(block.keys()) - ALLOWED_BREAKDOWN_KEYS
    if unknown:
        raise EvaluationConfigError(
            f"unknown keys in breakdowns: {sorted(unknown)}"
        )
    out: dict[str, bool] = {}
    for key in ALLOWED_BREAKDOWN_KEYS:
        value = block.get(key, False)
        if not isinstance(value, bool):
            raise EvaluationConfigError(
                f"breakdowns.{key} must be a boolean; got {value!r}"
            )
        out[key] = value
    return BreakdownToggles(**out)


def _parse_plots(block: Any, breakdowns: BreakdownToggles) -> PlotConfig:
    if not isinstance(block, dict):
        raise EvaluationConfigError(
            f"plots must be a mapping; got {type(block).__name__}"
        )
    unknown = set(block.keys()) - _PLOT_KEYS
    if unknown:
        raise EvaluationConfigError(
            f"unknown keys in plots: {sorted(unknown)}"
        )
    missing = _PLOT_KEYS - set(block.keys())
    if missing:
        raise EvaluationConfigError(
            f"missing required keys in plots: {sorted(missing)}"
        )

    scatter = _parse_bool("plots.scatter", block["scatter"])
    residual_distribution = _parse_bool(
        "plots.residual_distribution", block["residual_distribution"]
    )
    ladder_summary = _parse_bool("plots.ladder_summary", block["ladder_summary"])
    breakdown_plots = _parse_breakdown_plots(block["breakdown_plots"], breakdowns)
    dpi = _positive_int("plots.dpi", block["dpi"])
    figure_width_inches = _positive_float(
        "plots.figure_width_inches", block["figure_width_inches"]
    )
    figure_height_inches = _positive_float(
        "plots.figure_height_inches", block["figure_height_inches"]
    )

    return PlotConfig(
        scatter=scatter,
        residual_distribution=residual_distribution,
        ladder_summary=ladder_summary,
        breakdown_plots=breakdown_plots,
        dpi=dpi,
        figure_width_inches=figure_width_inches,
        figure_height_inches=figure_height_inches,
    )


def _parse_breakdown_plots(
    value: Any, breakdowns: BreakdownToggles
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise EvaluationConfigError(
            f"plots.breakdown_plots must be a list; got {type(value).__name__}"
        )
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, str):
            raise EvaluationConfigError(
                f"plots.breakdown_plots entries must be strings; got {entry!r}"
            )
        if entry not in ALLOWED_BREAKDOWN_PLOT_KEYS:
            raise EvaluationConfigError(
                f"plots.breakdown_plots entry {entry!r} must be one of "
                f"{sorted(ALLOWED_BREAKDOWN_PLOT_KEYS)}"
            )
        if entry in seen:
            raise EvaluationConfigError(
                f"duplicate entry in plots.breakdown_plots: {entry!r}"
            )
        seen.add(entry)
        if not breakdowns.is_enabled(entry):
            raise EvaluationConfigError(
                f"plots.breakdown_plots entry {entry!r} requires breakdowns.{entry} "
                f"to be true; got false"
            )
    return tuple(value)


def _parse_bool(field: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise EvaluationConfigError(f"{field} must be a boolean; got {value!r}")
    return value


def _positive_int(field: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationConfigError(f"{field} must be an integer; got {value!r}")
    if value <= 0:
        raise EvaluationConfigError(f"{field} must be > 0; got {value}")
    return value


def _positive_float(field: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationConfigError(f"{field} must be a number; got {value!r}")
    fvalue = float(value)
    if not (fvalue > 0):
        raise EvaluationConfigError(f"{field} must be > 0; got {value}")
    return fvalue
