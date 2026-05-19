"""EV-TEST-01: validate evaluation_config.yaml schema enforcement."""

from __future__ import annotations

import pathlib

import pytest
import yaml

from nflpredictor.evaluate.config import (
    EvaluationConfigError,
    load_evaluation_config,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "Data" / "raw" / "evaluation_config.yaml"


def _baseline() -> dict:
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write(tmp_path: pathlib.Path, data: dict) -> pathlib.Path:
    p = tmp_path / "evaluation_config.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return p


# (a) default v1 config — must accept
def test_default_v1_config_accepts() -> None:
    cfg = load_evaluation_config(DEFAULT_CONFIG_PATH)
    assert cfg.evaluation_version == "v1"
    assert cfg.headline_metric == "mae"
    assert cfg.breakdowns.by_team is True
    assert cfg.breakdowns.by_week is True
    assert cfg.breakdowns.by_home_away is True
    assert cfg.breakdowns.by_surface is True
    assert cfg.breakdowns.by_roof is True
    assert cfg.plots.scatter is True
    assert cfg.plots.residual_distribution is True
    assert cfg.plots.ladder_summary is True
    assert cfg.plots.breakdown_plots == ("by_week",)
    assert cfg.plots.dpi == 100
    assert cfg.plots.figure_width_inches == 8.0
    assert cfg.plots.figure_height_inches == 5.0


# (b) unknown top-level key
def test_unknown_top_level_key_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["surprise_key"] = 42
    with pytest.raises(EvaluationConfigError, match="unknown top-level keys"):
        load_evaluation_config(_write(tmp_path, data))


# (c) headline_metric outside the allowed set
def test_bad_headline_metric_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["headline_metric"] = "rocket"
    with pytest.raises(EvaluationConfigError, match="headline_metric"):
        load_evaluation_config(_write(tmp_path, data))


# (d) unknown breakdown key
def test_unknown_breakdown_key_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["breakdowns"]["by_zodiac"] = True
    with pytest.raises(EvaluationConfigError, match="unknown keys in breakdowns"):
        load_evaluation_config(_write(tmp_path, data))


# (e) plots.breakdown_plots entry whose corresponding breakdowns.<dim> is false
def test_breakdown_plot_without_breakdown_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["breakdowns"]["by_week"] = False
    # breakdown_plots still contains "by_week" → must fail
    with pytest.raises(EvaluationConfigError, match="requires breakdowns.by_week"):
        load_evaluation_config(_write(tmp_path, data))


# (f) non-positive plots.dpi
def test_nonpositive_dpi_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["plots"]["dpi"] = 0
    with pytest.raises(EvaluationConfigError, match=r"plots.dpi"):
        load_evaluation_config(_write(tmp_path, data))


def test_negative_dpi_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["plots"]["dpi"] = -1
    with pytest.raises(EvaluationConfigError, match=r"plots.dpi"):
        load_evaluation_config(_write(tmp_path, data))


# Valid trim: breakdowns.by_team=false with breakdown_plots=[by_week] still ok.
def test_trim_combination_accepts(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["breakdowns"]["by_team"] = False
    cfg = load_evaluation_config(_write(tmp_path, data))
    assert cfg.breakdowns.by_team is False
    assert cfg.breakdowns.by_week is True
    assert cfg.plots.breakdown_plots == ("by_week",)


# Extra coverage — cheap and helpful.

def test_missing_top_level_key_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data.pop("plots")
    with pytest.raises(EvaluationConfigError, match="missing required top-level keys"):
        load_evaluation_config(_write(tmp_path, data))


def test_unknown_breakdown_plot_entry_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["plots"]["breakdown_plots"] = ["by_surface"]
    with pytest.raises(EvaluationConfigError, match="plots.breakdown_plots entry 'by_surface'"):
        load_evaluation_config(_write(tmp_path, data))


def test_duplicate_breakdown_plot_entry_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["plots"]["breakdown_plots"] = ["by_week", "by_week"]
    with pytest.raises(EvaluationConfigError, match="duplicate"):
        load_evaluation_config(_write(tmp_path, data))


def test_nonbool_breakdown_value_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["breakdowns"]["by_team"] = "yes"
    with pytest.raises(EvaluationConfigError, match="breakdowns.by_team"):
        load_evaluation_config(_write(tmp_path, data))


def test_missing_evaluation_version_rejected(tmp_path: pathlib.Path) -> None:
    data = _baseline()
    data["evaluation_version"] = ""
    with pytest.raises(EvaluationConfigError, match="evaluation_version"):
        load_evaluation_config(_write(tmp_path, data))


def test_missing_breakdowns_dim_defaults_false(tmp_path: pathlib.Path) -> None:
    """An omitted breakdown key defaults to false (EV-CFG-04)."""
    data = _baseline()
    data["breakdowns"] = {"by_week": True}
    # remove by_week from breakdown_plots so the cross-toggle check passes
    data["plots"]["breakdown_plots"] = ["by_week"]
    cfg = load_evaluation_config(_write(tmp_path, data))
    assert cfg.breakdowns.by_week is True
    assert cfg.breakdowns.by_team is False
    assert cfg.breakdowns.by_surface is False


def test_missing_evaluation_config_file_rejected(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_evaluation_config(tmp_path / "missing.yaml")
