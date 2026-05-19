# Phase 5: Evaluation Specification

## 1. Introduction

### 1.1 Purpose

This specification defines the **Evaluation** phase of the NFL Predictor project. Phase 5 consumes Phase 4's prediction artifacts plus Phase 2's labels and breakdown columns and emits a fixed set of metric artifacts: a headline-metrics JSON, per-breakdown parquet tables, calibration plot PNGs, and an evaluation manifest.

Phases 1, 2, 3, and 4 are implemented and frozen; this spec assumes their outputs as given. Phase 6 (Error Analysis & Iteration) and beyond remain exploratory in `Docs/Idea.md` and are intentionally not specified yet — Phase 5 closes the contract one layer up the stack by producing the metric surface that an error-analysis phase can bind to.

### 1.2 Scope

In scope:

- A fifth deterministic build step (separate from Phases 1, 2, 3, and 4) that reads Phase 2's feature matrices, Phase 3's split artifact, Phase 4's prediction parquets, and a hand-edited `Data/raw/evaluation_config.yaml`, and writes a headline-metrics JSON, per-dimension breakdown parquets, calibration plot PNGs, and an evaluation manifest into `Data/processed/evaluation/`.
- A fixed v1 metric set: **MAE**, **RMSE**, **W/L accuracy**, **spread MAE**, **total MAE** — each computed per `(combination, slice)`, with home/away decomposition where meaningful.
- A fixed v1 breakdown-dimension set: **by team**, **by week**, **by home/away**, **by surface**, **by roof** — each toggleable via the YAML config.
- A fixed v1 plot set: **predicted-vs-actual scatter**, **residual distribution**, and **ladder summary** — plus optional per-breakdown plots (error by week, error by team) toggleable via the YAML config.
- Evaluation of **every prediction parquet emitted by Phase 4 on every run**, with no special test-slice gating: S1.val + S1.test + every S3 fold val are computed unconditionally.
- An evaluation manifest recording source SHAs (Phase 2, Phase 3, Phase 4, the config), output SHAs, headline-metric values per combination, matplotlib version, and git commit.
- Testing requirements that protect determinism (for metric outputs), source-hash pinning, and combination-coverage contracts.

Out of scope:

- **Rung selection.** Phase 5 reports per-combination metrics on every emitted slice; the human decides which `(rung, shape)` is the "chosen" model. No "best model" field is emitted.
- **External benchmarks (Vegas closing lines, prior-season models, third-party prediction services).** v1 evaluates only the rungs Phase 4 emits.
- **Error attribution / saliency / SHAP / Integrated Gradients.** Feature attribution is a Phase 6 concern.
- **Worst-N narrative analysis.** Phase 5 emits raw per-game residuals as a byproduct of the breakdown tables; tagging the worst games with likely causes (injury, weather extreme, blowout) is a Phase 6 act.
- **Re-training, model amendments, or any modification of Phase 4 outputs.** Phase 5 is read-only with respect to `Data/processed/predictions/`.
- **Cross-PNG-renderer byte determinism.** Metric JSONs and breakdown parquets are byte-deterministic given pinned inputs; PNG byte-identity holds only for the same pinned matplotlib wheel on the same platform.
- **Probabilistic / interval predictions.** Phase 4 emits point predictions only; Phase 5 evaluates point predictions only.
- **Classification-style metrics beyond W/L** (e.g., precision/recall, ATS hit rate, ROI). The v1 derived metric is W/L accuracy only.

### 1.3 Definitions and Acronyms

| Term | Definition |
|------|------------|
| Combination | A `(rung, feature_shape, strategy)` triple — identical to the Phase 4 term. Phase 5 keys all outputs by the same combination ids (the Phase 4 prediction filenames without extension). |
| Slice | One of `val`, `test`, or `fold_<i>` (the latter for S3 only). Slice is the scoring granularity inside one combination. |
| Strategy | One of `S1` or `S3`, named identically to Phase 3 and Phase 4. |
| Headline metric | **Per-side MAE averaged**: `mean(|pred_home − true_home| + |pred_away − true_away|) / 2`. Same definition as Phase 4's TR-MAN-04 so Phase 5's val MAE numerically matches Phase 4's. |
| Breakdown | A partition of an evaluated slice's games by one categorical dimension (team, week, home/away, surface, roof). Metrics are recomputed per partition cell. |
| Pooled (S3) | Metrics computed by concatenating all S3 fold val predictions into a single pool, then evaluating once. Distinct from "mean of per-fold metrics." Both shapes are emitted. |
| Calibration plot | A deterministic PNG rendered by matplotlib that visualizes prediction quality (scatter, residual distribution, ladder summary). |
| Evaluation config | The hand-edited `Data/raw/evaluation_config.yaml` declaring breakdown toggles, plot toggles, headline metric, and plot rendering parameters. |
| Headline metrics file | `Data/processed/evaluation/metrics_headline.json` — one entry per `(combination, slice)`. |
| Breakdown parquet | One parquet per breakdown dimension under `Data/processed/evaluation/breakdowns/`, stacking all combinations and slices for that dimension. |
| Evaluation manifest | `Data/processed/evaluation/evaluation_manifest.json` — provenance for the evaluation run. |

### 1.4 Design Principles

- **Contract first.** The headline-metrics JSON schema, the breakdown parquet schemas, and the evaluation manifest are stable, documented contracts. Phase 6 (and any external consumer) binds to them, not to the build internals.
- **Determinism (metric content).** Identical inputs (Phase 2 outputs + Phase 3 outputs + Phase 4 outputs + `evaluation_config.yaml`) shall produce byte-identical metric JSONs and byte-identical breakdown parquets across runs. The only permitted source of run-to-run drift in these artifacts is the manifest's `build_timestamp_utc`. PNG byte-identity is a softer guarantee — see TR-NF-02 (Phase 5 NF block).
- **Source-hash pinning.** The build refuses to run if any upstream artifact's on-disk SHA-256 does not match the SHA recorded in the corresponding upstream manifest. Drift fails fast; it never silently propagates into metrics.
- **One responsibility per phase.** Phase 5 computes metrics and renders plots. It does not select the winning rung, does not retrain anything, does not attribute errors to features, and does not propose Phase 6 hypotheses. Those are Phase 6 concerns.
- **No metric leakage across phases.** Phase 4 already computes val MAE for its manifest (TR-MAN-04). Phase 5 recomputes val MAE from scratch using the formula in §3.4 and shall produce numerically identical values; the cross-check is enforced in tests (EV-TEST-08).
- **Configuration over code edits.** Toggling a breakdown dimension, swapping the headline metric, or turning off PNG rendering is a YAML edit. Adding a new metric or a new breakdown dimension is a code change accompanied by a config-schema change and an `evaluation_version` bump.
- **Test slice is touched, but not interpreted.** Phase 5 computes test MAE because Phase 4 has already emitted test predictions; computing it is just arithmetic over a parquet that already exists on disk. No special "test mode" or opt-in flag is required. Test metrics appear in the headline file with `slice = "test"`; choosing to *read* them and freeze the final model is a human act.
- **Plot content is data, not narrative.** PNGs surface what the tables already say. Anything that requires prose or judgment is a Phase 6 deliverable, not a Phase 5 plot.

### 1.5 Relationship to Phases 1–4

```
Data/raw/{box_scores, madden, ..., feature_config}
    │  python -m nflpredictor.databuild
    ▼
Data/processed/{madden_2024.csv, box_scores_2024.csv,
                player_id_mapping.csv, build_manifest.json}
    │  python -m nflpredictor.features
    ▼
Data/processed/{features_flat_2024.parquet, features_pos_2024.parquet,
                feature_vocab.json, feature_manifest.json}
    │  python -m nflpredictor.splits
    ▼
Data/processed/{splits_2024.json, splits_manifest.json}
    │  python -m nflpredictor.train
    ▼
Data/processed/predictions/{<rung>__<shape>__<strategy>.parquet × 12,
                            training_manifest.json}
    │  python -m nflpredictor.evaluate
    ▼
Data/processed/evaluation/{metrics_headline.json,
                           breakdowns/<dim>.parquet × N_dims,
                           plots/<combo>__<plot>.png × M_plots,
                           evaluation_manifest.json}
```

The Phase 5 build refuses to run if any of Phase 2's tracked outputs (`features_flat_2024.parquet`, `features_pos_2024.parquet`, `feature_vocab.json`), Phase 3's output (`splits_2024.json`), or any Phase 4 prediction parquet listed in `training_manifest.json → output_sha256` is missing or has an on-disk SHA-256 that disagrees with its upstream manifest. This guarantees that evaluation outputs are always pinned to a specific Phase 4 run on top of specific Phase 2 and Phase 3 builds.

---

## 2. Technology Additions

| Layer | Technology | Purpose |
|-------|------------|---------|
| Metric arithmetic | `numpy` | All metric computations (MAE, RMSE, W/L, spread, total). |
| Data wrangling | `pandas` (or `polars`, implementation choice) | Joins predictions against features for breakdown dimensions; constructs per-cell aggregates. |
| Parquet I/O | `pyarrow` | Reads Phase 2 features and Phase 4 predictions; writes per-breakdown parquets. |
| JSON I/O | stdlib `json` | Reads Phase 3 splits and the upstream manifests; writes the headline metrics file and the evaluation manifest. |
| YAML I/O | `pyyaml` (`safe_load`) | Reads `Data/raw/evaluation_config.yaml`. |
| Plotting | `matplotlib` (Agg backend, pinned in venv) | Renders the calibration plot PNGs. |

The build is executable from the repository root via `python -m nflpredictor.evaluate` and requires no interactive input. `matplotlib` is a new dependency relative to Phases 1–4; the venv pin is recorded in the project's lockfile and the matplotlib version is written to `evaluation_manifest.json` so cross-version drift in PNG bytes is auditable.

---

## 3. Functional Requirements

### 3.1 Input Contract

| ID | Requirement |
|----|-------------|
| EV-IN-01 | The build shall read `Data/processed/features_flat_2024.parquet`, `Data/processed/features_pos_2024.parquet`, `Data/processed/feature_vocab.json`, and `Data/processed/feature_manifest.json` as Phase 2 inputs. It uses `features_flat_2024.parquet` as the canonical source for labels (`home_score`, `away_score`) and breakdown columns; `features_pos_2024.parquet` is consulted only to verify label parity with `features_flat_2024.parquet` (EV-NF-02). |
| EV-IN-02 | The build shall read `Data/processed/splits_2024.json` and `Data/processed/splits_manifest.json` as Phase 3 inputs. |
| EV-IN-03 | The build shall read every Phase 4 prediction parquet enumerated in `training_manifest.json → output_sha256` (excluding the manifest itself) plus `Data/processed/training_manifest.json` as Phase 4 inputs. |
| EV-IN-04 | The build shall read `Data/raw/evaluation_config.yaml` as a secondary input. The file is required; the build shall fail fast if it is missing. |
| EV-IN-05 | The build shall not modify any file under `Data/raw/`, any Phase 1, Phase 2, Phase 3, or Phase 4 output under `Data/processed/`, or any Phase 4 prediction parquet under `Data/processed/predictions/`. |
| EV-IN-06 | The build shall recompute the SHA-256 of each Phase 2 tracked output (`features_flat_2024.parquet`, `features_pos_2024.parquet`, `feature_vocab.json`) and compare each to the corresponding `output_sha256` entry in `feature_manifest.json`. On any mismatch the build shall fail fast with a clear error naming the divergent file. |
| EV-IN-07 | The build shall recompute the SHA-256 of `splits_2024.json` and compare it to the corresponding `output_sha256` entry in `splits_manifest.json`. On mismatch the build shall fail fast. |
| EV-IN-08 | The build shall recompute the SHA-256 of each Phase 4 prediction parquet and compare it to the corresponding `output_sha256` entry in `training_manifest.json`. On mismatch the build shall fail fast with a clear error naming the divergent file. |
| EV-IN-09 | The build shall reject (fail-fast) any `evaluation_config.yaml` whose schema does not satisfy §3.2. |
| EV-IN-10 | The build shall verify that every prediction parquet's `GameId` set is a subset of `splits_2024.json → S1.all_games`, and that every S1 parquet's `slice = "val"` rows exactly match `S1.val`, every `slice = "test"` rows exactly match `S1.test`, and every S3 parquet's `fold_index = i` rows exactly match `S3.folds[i].val`. On mismatch the build shall fail fast. (This is a defensive cross-check; Phase 4 already enforces these contracts on emission, but Phase 5 re-validates because the parquets may have been touched in transit.) |

### 3.2 Evaluation Config Schema

| ID | Requirement |
|----|-------------|
| EV-CFG-01 | The `evaluation_config.yaml` schema shall include the top-level keys defined in §4.1. Unknown top-level keys shall cause the build to fail fast. |
| EV-CFG-02 | `evaluation_version` (string) is recorded in the manifest. Any change to metric definitions, breakdown layout, plot inventory, or output structure requires bumping this version. |
| EV-CFG-03 | `headline_metric` (string) shall be one of `"mae"`, `"rmse"`, `"wl_accuracy"`, `"spread_mae"`, `"total_mae"`. It controls only which metric is highlighted in stdout summaries and on the ladder-summary plot; **all five metrics are always computed and emitted to the metrics file regardless of this setting**. |
| EV-CFG-04 | `breakdowns` shall be an object whose keys are a subset of `{by_team, by_week, by_home_away, by_surface, by_roof}` and whose values are booleans. Any key set to `true` triggers emission of the corresponding breakdown parquet; `false` (or absent) suppresses it. Unknown keys shall cause the build to fail fast. |
| EV-CFG-05 | `plots.scatter` (bool), `plots.residual_distribution` (bool), `plots.ladder_summary` (bool) shall control whether each plot family is rendered. |
| EV-CFG-06 | `plots.breakdown_plots` shall be a list whose entries are a subset of `{by_week, by_team, by_home_away}`. Each listed entry triggers emission of the corresponding per-combination breakdown plot. Unknown entries shall cause the build to fail fast. A breakdown plot may only appear in `plots.breakdown_plots` if the corresponding `breakdowns.<dim>` is `true`; otherwise the build shall fail fast. |
| EV-CFG-07 | `plots.dpi` shall be a positive integer (default `100`). `plots.figure_width_inches` and `plots.figure_height_inches` shall be positive floats (defaults `8.0` and `5.0`). These are recorded in the manifest. |
| EV-CFG-08 | A reference default config shall ship in the repository (`Data/raw/evaluation_config.yaml`) and reproduce the v1 defaults declared in §3.3. |

### 3.3 v1 Default Evaluation Config

The default `evaluation_config.yaml` shipped in the repo declares the following. Editing this file (and only this file) is how toggles and rendering parameters change.

| Field | v1 Default |
|-------|-----------|
| `evaluation_version` | `"v1"` |
| `headline_metric` | `"mae"` |
| `breakdowns.by_team` | `true` |
| `breakdowns.by_week` | `true` |
| `breakdowns.by_home_away` | `true` |
| `breakdowns.by_surface` | `true` |
| `breakdowns.by_roof` | `true` |
| `plots.scatter` | `true` |
| `plots.residual_distribution` | `true` |
| `plots.ladder_summary` | `true` |
| `plots.breakdown_plots` | `["by_week"]` |
| `plots.dpi` | `100` |
| `plots.figure_width_inches` | `8.0` |
| `plots.figure_height_inches` | `5.0` |

### 3.4 Metric Definitions

All metrics in this section are computed over the games in a single `(combination, slice)` cell. `pred_home`, `pred_away` come from the relevant Phase 4 prediction parquet; `true_home`, `true_away` come from `features_flat_2024.parquet`'s label columns joined on `GameId`.

| ID | Requirement |
|----|-------------|
| EV-MET-01 | **MAE (headline metric, per-side averaged).** Defined as `mean(|pred_home − true_home| + |pred_away − true_away|) / 2` taken over the games in the cell. Numerically identical to Phase 4's TR-MAN-04 for the val slice; the cross-check is enforced by EV-TEST-08. |
| EV-MET-02 | **MAE (per-side).** `mae_home = mean(|pred_home − true_home|)`, `mae_away = mean(|pred_away − true_away|)`. Both are emitted alongside the headline MAE. |
| EV-MET-03 | **RMSE (per-side).** `rmse_home = sqrt(mean((pred_home − true_home)²))`, `rmse_away = sqrt(mean((pred_away − true_away)²))`. Both are emitted. No averaged-RMSE field is emitted (averaging RMSEs is ambiguous; both sides are exposed instead). |
| EV-MET-04 | **W/L accuracy.** Predicted winner = `sign(pred_home − pred_away)`; actual winner = `sign(true_home − true_away)`. `wl_accuracy = mean(predicted_winner == actual_winner)`. Ties on the prediction side (`pred_home == pred_away`) shall count as incorrect unless the actual result is also a tie; ties on the actual side (`true_home == true_away`) shall count as correct only if the prediction is also a tie. Tied games are rare but the rule is explicit so v1 is reproducible. |
| EV-MET-05 | **Spread MAE.** `spread_mae = mean(|(pred_home − pred_away) − (true_home − true_away)|)`. |
| EV-MET-06 | **Total MAE.** `total_mae = mean(|(pred_home + pred_away) − (true_home + true_away)|)`. |
| EV-MET-07 | **n_games** shall be emitted alongside every metric cell: the count of games contributing to that cell. For S1 cells `n_games` equals the cell's slice size; for S3 per-fold cells it equals that fold's val size; for S3 pooled cells it equals the sum across folds (with repeats if a GameId appears in multiple folds — the S3 expanding-window protocol allows this). |
| EV-MET-08 | All metric floats shall be serialized to JSON with 6-digit precision (Python's default `json.dump` writes IEEE 754 round-trip; the build shall not truncate further). Parquet float columns retain `float64` precision. |
| EV-MET-09 | When `n_games == 0` for a cell (a possible edge case for empty breakdown buckets — e.g., a team with no games in a given val slice), the metric values shall be JSON `null` (parquet `null`) and `n_games` shall be `0`. The build shall not raise on empty cells. |

### 3.5 Combination Enumeration and Slice Handling

| ID | Requirement |
|----|-------------|
| EV-COMB-01 | The build shall enumerate combinations by reading the prediction-parquet filenames listed in `training_manifest.json → output_sha256` (i.e., the keys under `predictions/`). The combination id is the filename stem (e.g., `rung2_linear__flat__s1`). The build does not re-derive combinations from `training_config.yaml`. |
| EV-COMB-02 | For an **S1** combination (filename ends with `__s1`), the build shall compute metrics independently for `slice = "val"` and `slice = "test"`, populating both rows in the headline file and contributing both slices to every relevant breakdown. |
| EV-COMB-03 | For an **S3** combination (filename ends with `__s3`), the build shall compute metrics independently for each `fold_index ∈ {0..fold_count-1}` and additionally compute one **pooled** metric set (EV-MET-07 defines `n_games`). Both per-fold and pooled rows shall appear in the headline file and contribute to every relevant breakdown. |
| EV-COMB-04 | The headline file and every breakdown parquet shall carry every combination present in `training_manifest.json → output_sha256`. The build shall fail fast if any combination's parquet is missing from disk despite being listed in the manifest. |
| EV-COMB-05 | Combination ordering in all outputs is lexicographic on the combination id (the parquet filename stem). Slice ordering within a combination is: S1 → `["val", "test"]`; S3 → `["fold_0", "fold_1", …, "fold_<n-1>", "pooled"]`. Stable ordering is required by EV-NF-01 (byte determinism). |

### 3.6 Breakdown Dimensions

Each enabled breakdown dimension produces one parquet under `Data/processed/evaluation/breakdowns/`. The parquet stacks every `(combination, slice, breakdown_value)` triple for which `n_games > 0`, plus rows where `n_games == 0` if the breakdown value appeared in the full season but is absent from the slice (so the parquet is complete by-construction across the dataset's value universe).

| ID | Requirement |
|----|-------------|
| EV-BRK-01 | **by_team.** Each game contributes two rows: one keyed by `home_team_code`, one by `away_team_code`. `home_or_away` column distinguishes them. Metrics within a team cell are computed over the contributions where that team appears in the indicated role; per-side metrics (`mae_home`, `mae_away`) are reported relative to the home/away role of *the game*, not the role of the queried team. (Rationale: a team's offensive performance contributes to `mae_home` when it plays at home and `mae_away` when away; the breakdown surfaces both views.) |
| EV-BRK-02 | **by_week.** Each game contributes one row keyed by `week` (integer 1–18 in the 2024 universe). Week is read from `features_flat_2024.parquet`'s `week` column. |
| EV-BRK-03 | **by_home_away.** Each game contributes two rows: one with `home_or_away = "home"` reporting `mae_home`, `rmse_home`, etc.; one with `home_or_away = "away"` reporting `mae_away`, `rmse_away`, etc. The remaining columns (`wl_accuracy`, `spread_mae`, `total_mae`) are computed on the whole game and are duplicated across the two rows. |
| EV-BRK-04 | **by_surface.** Each game contributes one row keyed by `surface` (the categorical code from `features_flat_2024.parquet`'s `surface` column; the int → label lookup uses `feature_vocab.json → surface`). The parquet carries both the integer code and the human-readable label so the file is self-describing. |
| EV-BRK-05 | **by_roof.** Same shape as `by_surface` but keyed by `roof`. |
| EV-BRK-06 | Every breakdown parquet shall include the columns enumerated in §4.3 — at minimum: `combination_id`, `slice`, `breakdown_value` (typed per dimension), `breakdown_label` (string, the human-readable form where applicable), `n_games`, plus the five metric families. |
| EV-BRK-07 | Per-team breakdown values shall use the 3-letter PFR team codes (the same `team_codes` vocab Phase 2 uses for `HomeTeamCode` / `AwayTeamCode`). |

### 3.7 Plot Outputs

All plots are PNGs rendered with the matplotlib Agg backend, written to `Data/processed/evaluation/plots/`. The build shall create the directory if it does not exist and shall overwrite existing PNGs matching the §3.10 filename pattern.

| ID | Requirement |
|----|-------------|
| EV-PLOT-01 | **Predicted-vs-actual scatter.** One PNG per `(combination, slice)` cell where `slice ∈ {"val", "test"}` for S1 combinations and `slice = "pooled"` for S3 combinations. The plot overlays two scatter series (home and away) on a single axis: x = actual score, y = predicted score; identity line `y = x` is drawn. Filename: `<combination_id>__<slice>__scatter.png`. |
| EV-PLOT-02 | **Residual distribution.** One PNG per `(combination, slice)` cell on the same slice basis as EV-PLOT-01. The plot is a histogram of `pred − true` residuals with two series (home, away) overlaid. Filename: `<combination_id>__<slice>__residuals.png`. |
| EV-PLOT-03 | **Ladder summary.** A single PNG `ladder_summary__<slice>.png` is rendered per slice basis `slice ∈ {"val", "test", "pooled"}` — three files total. Each is a grouped bar chart with one bar per combination (lexicographic order), height = `headline_metric` value for that `(combination, slice)`; combinations whose parquet does not contribute to the given slice are omitted. The headline metric shown is the value of `config.headline_metric`. |
| EV-PLOT-04 | **Per-breakdown plots (configurable).** For each entry in `plots.breakdown_plots`, one PNG per `(combination, slice)` cell on the same slice basis as EV-PLOT-01 is rendered, showing the chosen metric (`headline_metric`) across the breakdown values as a line chart (for `by_week`) or bar chart (for `by_team`, `by_home_away`). Filename: `<combination_id>__<slice>__<dim>.png`. |
| EV-PLOT-05 | Plot rendering shall use `matplotlib.use("Agg")` and `plt.savefig(..., dpi=config.plots.dpi, metadata={"Software": None, "Creation Time": None})` (or the equivalent metadata-stripping idiom for the matplotlib version in use). All figures shall be closed via `plt.close(fig)` after saving. |
| EV-PLOT-06 | Plot text content (titles, axis labels, legend entries) shall be deterministic given pinned inputs. Random colors, jitter, or any non-deterministic styling shall not be used. Default matplotlib colors and styles are acceptable. |
| EV-PLOT-07 | The matplotlib version shall be captured in the evaluation manifest (EV-MAN-01) so cross-version PNG byte drift is auditable. PNG byte-identity is not asserted across matplotlib versions or platforms (see EV-NF-02). |

### 3.8 Output Encoding

| ID | Requirement |
|----|-------------|
| EV-OUT-01 | The build shall write `Data/processed/evaluation/metrics_headline.json` with the schema defined in §4.2. JSON output shall use sorted keys, two-space indentation, UTF-8, and Unix line endings (`\n`). |
| EV-OUT-02 | The build shall write one parquet per enabled breakdown dimension into `Data/processed/evaluation/breakdowns/`. Filenames: `by_team.parquet`, `by_week.parquet`, `by_home_away.parquet`, `by_surface.parquet`, `by_roof.parquet`. Each file's row order shall be lexicographic on `(combination_id, slice, breakdown_value)`. |
| EV-OUT-03 | Parquet output shall use compression `snappy`, row-group size `1024`, and the `pyarrow` writer (the same writer profile as Phase 2 features and Phase 4 predictions). |
| EV-OUT-04 | The build shall write PNGs into `Data/processed/evaluation/plots/` per the filenames enumerated in §3.7. |
| EV-OUT-05 | The build shall create `Data/processed/evaluation/`, `Data/processed/evaluation/breakdowns/`, and `Data/processed/evaluation/plots/` if they do not exist. Existing files in those directories whose names match the §3.10 patterns shall be overwritten; other files shall not be touched. |
| EV-OUT-06 | If a breakdown is toggled off in the config, the build shall delete any stale parquet under `Data/processed/evaluation/breakdowns/` whose name corresponds to that disabled breakdown, so the directory contents always reflect the current config. The same applies to plot families toggled off. |

### 3.9 Evaluation Manifest

| ID | Requirement |
|----|-------------|
| EV-MAN-01 | The build shall emit `Data/processed/evaluation/evaluation_manifest.json` containing at minimum the keys: `build_timestamp_utc`, `evaluation_version`, `headline_metric`, `evaluation_config_sha256`, `phase2_source_sha256` (object: input filename → SHA), `phase3_source_sha256` (object: input filename → SHA), `phase4_source_sha256` (object: input filename relative to `Data/processed/` → SHA, including `training_manifest.json` and every prediction parquet), `output_sha256` (object: output filename relative to `Data/processed/evaluation/` → SHA, including `metrics_headline.json`, every breakdown parquet, and every PNG), `git_commit` (or `null`), `phase4_manifest_git_commit`, `matplotlib_version`, `numpy_version`, `pyarrow_version`, `combination_ids` (sorted list of evaluated combinations), `evaluation_summary`. |
| EV-MAN-02 | `evaluation_summary` shall be an object keyed by combination id, with each value containing the headline-metric value per evaluated slice: for S1 keys, `{"val": <float or null>, "test": <float or null>}`; for S3 keys, `{"pooled": <float or null>, "mean_per_fold": <float or null>, "per_fold": [<float>, …]}`. This is a navigation aid; the full per-metric matrix lives in `metrics_headline.json`. |
| EV-MAN-03 | The manifest's `build_timestamp_utc` shall be in ISO 8601 UTC format. |
| EV-MAN-04 | The manifest shall be written **last** — after every metric JSON, breakdown parquet, and PNG — so each output SHA can be computed against the on-disk file. |
| EV-MAN-05 | The manifest shall be written with sorted keys and stable two-space indentation. Byte-identical inputs and a pinned matplotlib wheel shall produce a byte-identical manifest modulo `build_timestamp_utc`. |
| EV-MAN-06 | `evaluation_config_sha256` shall be the SHA-256 of the raw `Data/raw/evaluation_config.yaml` file bytes (not the parsed/normalized form). This mirrors Phases 1–4's source-hash discipline. |
| EV-MAN-07 | No test-slice metric is gated, redacted, or otherwise distinguished in the manifest. S1's test slice appears in `evaluation_summary` and in `metrics_headline.json` unconditionally; the human chooses when to read it. |

---

## 4. Data Model

### 4.1 `Data/raw/evaluation_config.yaml`

```yaml
evaluation_version: "v1"

headline_metric: "mae"   # one of "mae" | "rmse" | "wl_accuracy" | "spread_mae" | "total_mae"

breakdowns:
  by_team:      true
  by_week:      true
  by_home_away: true
  by_surface:   true
  by_roof:      true

plots:
  scatter:              true
  residual_distribution: true
  ladder_summary:       true
  breakdown_plots:      [by_week]
  dpi:                  100
  figure_width_inches:  8.0
  figure_height_inches: 5.0
```

### 4.2 `Data/processed/evaluation/metrics_headline.json` Structure

```json
{
  "combinations": {
    "rung0_mean__none__s1": {
      "val": {
        "n_games": 45,
        "mae": 9.87,
        "mae_home": 9.21,
        "mae_away": 10.53,
        "rmse_home": 11.84,
        "rmse_away": 13.05,
        "wl_accuracy": 0.53,
        "spread_mae": 13.21,
        "total_mae": 14.62
      },
      "test": { "n_games": 48, "mae": 10.04, "…": "…" }
    },
    "rung2_linear__flat__s3": {
      "fold_0": { "n_games": 5, "mae": 8.93, "…": "…" },
      "fold_1": { "n_games": 5, "mae": 9.12, "…": "…" },
      "…": "…",
      "pooled": { "n_games": 45, "mae": 8.71, "…": "…" }
    }
  }
}
```

Key ordering inside each cell is lexicographic on the metric name (per EV-OUT-01's sorted-keys rule); the order shown above is illustrative.

### 4.3 Breakdown Parquet Schemas

**`by_team.parquet`**

| Column | Type | Description |
|--------|------|-------------|
| `combination_id` | string | Phase 4 combination id (filename stem). |
| `slice` | string | One of `val`, `test`, `fold_<i>`, `pooled`. |
| `team_code` | string | 3-letter PFR team code. |
| `home_or_away` | string | `"home"` or `"away"` — the role the team played in the contributing games. |
| `n_games` | int32 | Game count for this cell. |
| `mae` | float64 | Headline-formula MAE over the cell (EV-MET-01). |
| `mae_home` | float64 | Per-side home MAE over the cell. |
| `mae_away` | float64 | Per-side away MAE over the cell. |
| `rmse_home` | float64 | |
| `rmse_away` | float64 | |
| `wl_accuracy` | float64 | |
| `spread_mae` | float64 | |
| `total_mae` | float64 | |

**`by_week.parquet`**

Same shape but with `week` (int8, 1–18) replacing `team_code` + `home_or_away`.

**`by_home_away.parquet`**

Same shape but with `home_or_away` (string, `"home"` or `"away"`) as the sole breakdown key.

**`by_surface.parquet`** and **`by_roof.parquet`**

Same shape but with two breakdown columns: `<dim>_code` (int8, integer code from Phase 2) and `<dim>_label` (string, the human-readable label from `feature_vocab.json`).

Row order in every breakdown parquet: lexicographic on `(combination_id, slice, <primary breakdown key>)`.

### 4.4 `Data/processed/evaluation/evaluation_manifest.json` Structure

```json
{
  "build_timestamp_utc": "2026-05-22T17:08:00Z",
  "evaluation_version": "v1",
  "headline_metric": "mae",
  "evaluation_config_sha256": "…",
  "phase2_source_sha256": {
    "features_flat_2024.parquet": "…",
    "features_pos_2024.parquet": "…",
    "feature_vocab.json": "…"
  },
  "phase3_source_sha256": {
    "splits_2024.json": "…"
  },
  "phase4_source_sha256": {
    "training_manifest.json": "…",
    "predictions/rung0_mean__none__s1.parquet": "…",
    "predictions/rung0_mean__none__s3.parquet": "…",
    "…": "…"
  },
  "output_sha256": {
    "metrics_headline.json": "…",
    "breakdowns/by_team.parquet": "…",
    "breakdowns/by_week.parquet": "…",
    "plots/rung2_linear__flat__s1__val__scatter.png": "…",
    "…": "…"
  },
  "git_commit": "…",
  "phase4_manifest_git_commit": "…",
  "matplotlib_version": "3.9.2",
  "numpy_version": "2.1.3",
  "pyarrow_version": "17.0.0",
  "combination_ids": [
    "rung0_mean__none__s1",
    "rung0_mean__none__s3",
    "…"
  ],
  "evaluation_summary": {
    "rung0_mean__none__s1": { "val": 9.87, "test": 10.04 },
    "rung2_linear__flat__s3": {
      "pooled": 8.71,
      "mean_per_fold": 8.69,
      "per_fold": [8.93, 9.12, "…"]
    }
  }
}
```

### 4.5 Output Files

| File | Type | Description |
|------|------|-------------|
| `Data/processed/evaluation/metrics_headline.json` | JSON | Full metric matrix per `(combination, slice)`. |
| `Data/processed/evaluation/breakdowns/by_team.parquet` | Parquet | (Conditional on `breakdowns.by_team`.) |
| `Data/processed/evaluation/breakdowns/by_week.parquet` | Parquet | (Conditional on `breakdowns.by_week`.) |
| `Data/processed/evaluation/breakdowns/by_home_away.parquet` | Parquet | (Conditional on `breakdowns.by_home_away`.) |
| `Data/processed/evaluation/breakdowns/by_surface.parquet` | Parquet | (Conditional on `breakdowns.by_surface`.) |
| `Data/processed/evaluation/breakdowns/by_roof.parquet` | Parquet | (Conditional on `breakdowns.by_roof`.) |
| `Data/processed/evaluation/plots/<combination_id>__<slice>__scatter.png` | PNG | One per `(combination, slice)` per EV-PLOT-01. |
| `Data/processed/evaluation/plots/<combination_id>__<slice>__residuals.png` | PNG | One per `(combination, slice)` per EV-PLOT-02. |
| `Data/processed/evaluation/plots/ladder_summary__<slice>.png` | PNG | One per slice basis per EV-PLOT-03. |
| `Data/processed/evaluation/plots/<combination_id>__<slice>__<dim>.png` | PNG | One per `(combination, slice, dim)` per EV-PLOT-04 (conditional on `plots.breakdown_plots`). |
| `Data/processed/evaluation/evaluation_manifest.json` | JSON | Build provenance + per-combination headline summary. |

For the v1 default config against Phase 4's v1 default output (12 prediction parquets — 6 S1 + 6 S3), the output set is: 1 headline JSON + 5 breakdown parquets + (12 × ~3 plots per combination + 3 ladder summary plots) ≈ 40 PNGs + 1 manifest. The exact PNG count is a function of which plot families are enabled.

---

## 5. Build Pipeline Design

The pipeline is described conceptually; implementation may organize it differently as long as §3's contract is honored. Recommended module layout, mirroring Phase 1's `databuild/`, Phase 2's `features/`, Phase 3's `splits/`, and Phase 4's `train/`:

```
src/nflpredictor/evaluate/
    __init__.py
    __main__.py        # entry point: python -m nflpredictor.evaluate
    pipeline.py        # top-level orchestration
    config.py          # evaluation_config.yaml load + validation
    sources.py         # Phase 2 + Phase 3 + Phase 4 input load + SHA verification
    metrics.py         # MAE/RMSE/W-L/spread/total computations (§3.4)
    breakdowns.py      # per-dimension aggregation (§3.6)
    plots.py           # matplotlib renderers (§3.7)
    outputs.py         # JSON + parquet + PNG writers
    manifest.py        # evaluation_manifest.json construction
```

### 5.1 Conceptual Stages

1. **Load and validate inputs.** Read `evaluation_config.yaml`; validate per §3.2 and §4.1. Read `feature_manifest.json`, `splits_manifest.json`, and `training_manifest.json`; verify Phase 2, Phase 3, and Phase 4 output SHAs against on-disk files (EV-IN-06, EV-IN-07, EV-IN-08).
2. **Load features, splits, and predictions into memory.** Read `features_flat_2024.parquet` (labels + breakdown columns), `splits_2024.json`, and every prediction parquet enumerated in `training_manifest.json`. Verify label parity across `flat` and `pos` Phase 2 parquets (EV-NF-02). Cross-validate prediction GameId coverage per EV-IN-10.
3. **Enumerate `(combination, slice)` cells.** Per EV-COMB-01..05.
4. **Compute metrics per cell.** Apply §3.4 to every cell. Build the headline-metrics structure.
5. **Compute breakdowns.** For each enabled dimension, partition the games in every cell by the breakdown value and recompute the metric stack per partition. Stack into the per-dimension parquet shape (§4.3).
6. **Render plots.** For each enabled plot family, render the PNG set per §3.7.
7. **Emit outputs.** Write `metrics_headline.json`, every enabled breakdown parquet, every enabled PNG. Per EV-OUT-06, delete any stale outputs for disabled breakdowns / plot families.
8. **Emit manifest.** Compute output SHAs and per-combination headline summaries; write `evaluation_manifest.json` last.

### 5.2 Determinism Boundaries

- The only non-deterministic input is the wall-clock timestamp written to `evaluation_manifest.json`.
- Metric content (the headline JSON, the breakdown parquets, and every numeric field in the manifest) is deterministic given fixed inputs — the computations are pure numpy/pandas arithmetic. Byte-identity of these artifacts is required (EV-NF-01).
- PNG byte-identity is a softer guarantee. Within the same pinned matplotlib wheel on the same platform, repeated runs produce byte-identical PNGs (EV-PLOT-05 strips PNG metadata for this purpose). Across matplotlib versions or platforms (Linux vs macOS rendering, font availability differences), PNG bytes may differ. The manifest records `matplotlib_version` so a downstream consumer can tell whether two PNG sets are comparable byte-for-byte.
- Any drift in Phase 2, Phase 3, or Phase 4 produces an EV-IN-06 / EV-IN-07 / EV-IN-08 hash mismatch and a hard failure rather than silent metric drift.

---

## 6. Integration / Endpoint / Tooling Design

Not applicable. The evaluation build is a single-shot offline script with no network surface, no API, and no UI.

---

## 7. Changes to Existing Requirements

`Docs/Idea.md` §"Phase 5: Evaluation" enumerated open questions. The decisions encoded in this spec are:

- **Vegas closing-line comparison**: **declined for v1** and removed from the open-questions list. The project's stated goal is to predict scores from compositional / physical inputs and beat trivial baselines; adding an external strong-baseline comparison is a different question and not in scope. If a future iteration wants this, it becomes a Phase 5 amendment with a `evaluation_version` bump.
- **Metric set in v1**: MAE (headline, per-side averaged), per-side MAE/RMSE, W/L accuracy, spread MAE, total MAE. All five are always computed; `config.headline_metric` only controls which one is highlighted in stdout summaries and on the ladder-summary plot.
- **Breakdown dimensions**: by team, by week, by home/away, by surface, by roof — all on by default, individually toggleable.
- **Plot inventory**: predicted-vs-actual scatter, residual distribution, ladder summary (all on by default); per-week error plot on by default; per-team and per-home/away error plots available but off by default.
- **Slice scope**: every prediction parquet on every run. Test-slice metrics are computed unconditionally; there is no `--include-test` flag. Phase 4's choice to defer test MAE to Phase 5 (TR-MAN-03) is fully discharged here: `metrics_headline.json` carries `slice = "test"` rows alongside `slice = "val"` rows for every S1 combination.
- **Headline-MAE formula equals Phase 4's val-MAE formula**: EV-MET-01 reuses TR-MAN-04 verbatim, and EV-TEST-08 enforces numerical agreement.

The Phase 1, 2, 3, and 4 specs are unchanged. `Docs/Idea.md` §"Phase 5" and §"Remaining Open Questions → Phase 5" require a follow-on edit to mark the open question as resolved; this spec is the source of truth in the interim.

---

## 8. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| EV-NF-01 | The build shall be deterministic for metric content: identical inputs (Phase 2 outputs, Phase 3 outputs, Phase 4 outputs, `evaluation_config.yaml`) shall produce byte-identical `metrics_headline.json`, byte-identical breakdown parquets, and a byte-identical manifest (excluding `build_timestamp_utc`) across runs. |
| EV-NF-02 | The build shall be deterministic for PNG content within a pinned matplotlib wheel on the same platform: rerunning with the same matplotlib version produces byte-identical PNGs. Cross-version or cross-platform PNG byte-identity is not asserted; `matplotlib_version` in the manifest is the auditability lever. |
| EV-NF-03 | At startup, the build shall verify that `features_flat_2024.parquet` and `features_pos_2024.parquet` carry identical `(GameId, home_score, away_score)` triples (same set, same label values). On mismatch the build shall fail fast. This duplicates Phase 4's TR-NF-02 but is repeated here because Phase 5 binds to `flat` for labels and an undetected divergence would silently corrupt metrics. |
| EV-NF-04 | The build shall be re-runnable: it shall produce correct output regardless of whether the headline JSON, breakdown parquets, PNGs, or manifest exist, are stale, or are absent. Stale outputs matching the §3.10 patterns shall be overwritten; stale outputs for now-disabled breakdowns or plots shall be deleted per EV-OUT-06. |
| EV-NF-05 | The build shall complete in under 2 minutes for the v1 default config against the v1 Phase 4 output (12 combinations, ~40 PNGs) on a modern 8-core laptop. PNG rendering dominates wall clock; metric computation is sub-second. |
| EV-NF-06 | The `build_timestamp_utc` field in `evaluation_manifest.json` is the only permitted source of run-to-run output drift in metric artifacts under fixed inputs and matplotlib wheel. |
| EV-NF-07 | The build shall emit a non-zero exit code on any fatal error. |
| EV-NF-08 | The build shall log per-combination headline-metric values to stdout in a stable, scannable format so the metric matrix is visible without opening `metrics_headline.json`. |
| EV-NF-09 | All JSON outputs shall use UTF-8 encoding with Unix line endings (`\n`). |
| EV-NF-10 | All Parquet outputs shall use the `pyarrow` writer with `snappy` compression and the row-group profile declared in EV-OUT-03. |
| EV-NF-11 | The build shall not download any model weights, dataset, font, or other resource at runtime. All inputs are local files. matplotlib font caching is permitted (it is a one-time cache populated from installed system fonts). |

---

## 9. UI Requirements

Not applicable. The plot PNGs are read by humans but are static artifacts written to disk; there is no interactive UI surface.

---

## 10. Testing Requirements

| ID | Requirement |
|----|-------------|
| EV-TEST-01 | A unit test shall verify `evaluation_config.yaml` validation against: (a) the default v1 config (must accept); (b) an unknown top-level key (must reject); (c) `headline_metric` outside the allowed set (must reject); (d) an unknown breakdown key (must reject); (e) a `plots.breakdown_plots` entry whose corresponding `breakdowns.<dim>` is `false` (must reject); (f) a non-positive `plots.dpi` (must reject). |
| EV-TEST-02 | A unit test shall verify each metric definition (§3.4) against a small synthetic prediction/label fixture with hand-computed expected values. MAE, per-side MAE, per-side RMSE, W/L accuracy (including the tie-handling rules in EV-MET-04), spread MAE, and total MAE shall each match the hand-computed expectation to float64 round-trip precision. |
| EV-TEST-03 | A unit test shall verify the breakdown aggregation: a synthetic 6-game cell partitioned by team yields the expected per-team metric rows, including `home_or_away` doubling per EV-BRK-01, and empty cells yield `n_games == 0` with null metrics per EV-MET-09. |
| EV-TEST-04 | A unit test shall verify the headline JSON schema and key ordering (§4.2): combinations sorted lexicographically, slices ordered per EV-COMB-05, metric keys sorted alphabetically inside each cell. |
| EV-TEST-05 | A unit test shall verify the breakdown parquet schemas and row ordering (§4.3): every required column present at the expected type; row order matches EV-OUT-02. |
| EV-TEST-06 | An integration test shall run the full Phase 5 build against a small synthetic Phase 2 + Phase 3 + Phase 4 stand-in (~30 games, 2 combinations) and assert that the headline JSON, every breakdown parquet, and the manifest match a checked-in expected snapshot (excluding `build_timestamp_utc`). PNG existence is asserted; PNG byte-equality is not (per EV-NF-02). |
| EV-TEST-07 | A determinism test shall run the evaluation build twice in succession against identical Phase 2, Phase 3, Phase 4, and config inputs and assert byte-equality of `metrics_headline.json`, every breakdown parquet, and the manifest (excluding `build_timestamp_utc`) in the same pinned matplotlib wheel. PNG byte-equality is asserted within the same pinned matplotlib wheel on the same platform per EV-NF-02. |
| EV-TEST-08 | A cross-phase agreement test shall, against a synthetic Phase 4 stand-in, compute Phase 5's headline MAE for every `S1.val` cell and assert it equals the value Phase 4 recorded in its `training_summaries.<combo>.val_mae` field to float64 precision. The same equality shall hold per-fold for S3 combinations against `training_summaries.<combo>.per_fold[i].val_mae`. (This test prevents Phase 5's formula from silently drifting from Phase 4's.) |
| EV-TEST-09 | A pinned-identity test shall run the real Phase 5 build against the actual Phase 2, Phase 3, and Phase 4 outputs and assert: (a) every combination in `training_manifest.json` appears in `metrics_headline.json`; (b) every S1 combination has both `val` and `test` entries; (c) every S3 combination has `fold_0` through `fold_<n-1>` and `pooled` entries; (d) every enabled breakdown parquet exists and covers every combination; (e) the manifest's `output_sha256` exactly enumerates the files on disk. This test skips when `Data/processed/predictions/` is empty (the heavy real-data run is expected post-Phase 4 on the CUDA machine, not in CPU CI). |
| EV-TEST-10 | A source-pinning test shall verify that an EV-IN-06 hash mismatch (e.g., a hand-edited `features_flat_2024.parquet`), an EV-IN-07 hash mismatch (e.g., a hand-edited `splits_2024.json`), and an EV-IN-08 hash mismatch (e.g., a hand-edited prediction parquet) each cause the build to fail fast with a clear error naming the divergent file. |
| EV-TEST-11 | A test shall verify the test-slice handling: starting from a fresh `Data/processed/evaluation/` directory, a single Phase 5 run produces `slice = "test"` metric entries for every S1 combination without any flag, argument, or config change. |

---

## 11. Security Considerations

| ID | Consideration |
|----|---------------|
| EV-SEC-01 | The build operates exclusively on local files under the repository root. It shall not make any network requests, including matplotlib font downloads or telemetry. |
| EV-SEC-02 | The build shall not write to any path outside `Data/processed/evaluation/`. |
| EV-SEC-03 | No input or output of this build contains credentials, PII beyond publicly available game schedules and player names, or other sensitive data. |
| EV-SEC-04 | `evaluation_config.yaml` is user-editable; the build validates its schema (§3.2) but is not required to defend against adversarial input — the only consumer is the build script in the same repository. YAML loading shall use `yaml.safe_load`. |
| EV-SEC-05 | The build shall not deserialize any pickle, joblib, or `torch.load` artifact at runtime. v1 reads only parquet, JSON, and YAML. |

---

## 12. Future Considerations

The following are explicitly out of scope for Phase 5 v1 and recorded so they are not lost:

- **External baselines.** Comparing against Vegas closing lines, prior-season "predict last year's score" baselines, or third-party prediction services is deferred. Adding any would be a Phase 5 amendment with a new input-contract block and a `evaluation_version` bump.
- **Probabilistic / interval metrics.** Phase 4 emits point predictions only. If a future ladder rung produces score distributions (e.g., quantile regression, a Gaussian head), Phase 5 will need to add proper scoring rules (CRPS, log-likelihood) and prediction-interval coverage metrics.
- **Calibration plots beyond v1.** Reliability diagrams, Q-Q plots of residuals, predicted-vs-actual KDE overlays, and per-team residual fan charts are all reasonable additions; v1 intentionally keeps the plot inventory short.
- **Cross-combination diff views.** A side-by-side "rung k vs rung k+1 residual diff" plot or table would directly support the Phase 6 question "did the upgrade earn its keep." Deferred to Phase 6.
- **Aggregation across multiple training runs.** v1 evaluates a single Phase 4 output set. A future amendment could compare two runs (different seeds, different configs) by reading two `training_manifest.json` snapshots. Out of scope until Phase 6's iteration loop demands it.
- **HTML / dashboard renderings.** PNGs are the v1 surface. A Jupyter notebook or static HTML dashboard that mounts these artifacts is left to ad-hoc exploration; no programmatic dashboard generator is in scope.
- **Cross-matplotlib-version PNG byte determinism.** Not promised. The manifest's `matplotlib_version` records the wheel used; changing it is a deliberate environment update.
- **Multi-season evaluation.** When Phase 2/3/4 outputs cover more than 2024, the evaluation build will need a `--season` flag or per-season config wiring. v1 stays 2024-only.

---

## 13. References

- [Idea.md](./Idea.md) — Source idea document; §"Phase 5: Evaluation" enumerates the open questions this spec resolves.
- [Spec-Phase1-DataBuild.md](./Spec-Phase1-DataBuild.md) — Phase 1 specification.
- [Spec-Phase2-FeatureEngineering.md](./Spec-Phase2-FeatureEngineering.md) — Phase 2 specification. Phase 5 reads its outputs for labels and breakdown columns and binds to its `feature_manifest.json`.
- [Spec-Phase3-Splits.md](./Spec-Phase3-Splits.md) — Phase 3 specification. Phase 5 reads its splits artifact for slice composition and binds to its `splits_manifest.json`.
- [Spec-Phase4-BaselineLadder.md](./Spec-Phase4-BaselineLadder.md) — Phase 4 specification. Phase 5 reads its prediction parquets and binds to its `training_manifest.json`; the headline-MAE formula (EV-MET-01) reuses Phase 4's TR-MAN-04 verbatim.
- [Plan-Phase1-DataBuild.md](./Plan-Phase1-DataBuild.md), [Plan-Phase2-FeatureEngineering.md](./Plan-Phase2-FeatureEngineering.md), [Plan-Phase3-Splits.md](./Plan-Phase3-Splits.md), [Plan-Phase4-BaselineLadder.md](./Plan-Phase4-BaselineLadder.md) — Stylistic precedent for the Phase 5 implementation plan to follow.
- `Data/processed/predictions/*.parquet`, `Data/processed/training_manifest.json` — Phase 4 outputs; primary inputs to Phase 5.
- `Data/processed/features_flat_2024.parquet`, `Data/processed/feature_vocab.json` — Phase 2 outputs; secondary inputs to Phase 5 (labels + breakdown dimension lookup).
- `Data/processed/splits_2024.json` — Phase 3 output; secondary input to Phase 5 (slice membership cross-validation).
- `Data/processed/feature_manifest.json`, `Data/processed/splits_manifest.json`, `Data/processed/training_manifest.json` — Upstream manifests; consulted for source-hash pinning per EV-IN-06, EV-IN-07, EV-IN-08.
- [CLAUDE.md](../CLAUDE.md) — Repository-level notes on venv, dataset shape, and the Phase 1–4 build conventions Phase 5 mirrors.
