# Phase 6 — Reading the Phase 5 Outputs

> **Revision note (multi-year).** Updated for the 2020–2025 migration: the split strategies are `season_holdout` (val/test slices) and `loso_cv` (per-fold slices), replacing the week-based `S1`/`S3`. Some week-anchored examples are kept as illustrations of *what to watch for* and may not map literally to the season-holdout layout.

A field guide to every artifact `python -m nflpredictor.evaluate` emits, written so a first-time reader can open `Data/processed/evaluation/` and form a calibrated opinion of what they're looking at.

---

## How to use this document

Phase 5 writes one JSON, five breakdown parquets, and a folder of PNGs into `Data/processed/evaluation/`. This guide gives **one structured entry per artifact** — 10 entries in total. Each entry follows the same four sub-section template:

* **What it shows** — the mechanical description: what's in the file, what the columns or axes mean, and how the values were computed.
* **What good looks like** — calibration intuition: what a healthy artifact would look like for this project's regression problem.
* **Red flags** — concrete anti-patterns to watch for. Not "the model is bad" — specific, recognizable signatures.
* **Action to consider** — when a red flag fires, what to investigate in **Phase 7** (Error Analysis & Iteration). Phase 6 is about *reading* the outputs; **Phase 7 is where actions get taken**.

The guide is **interpretation-focused, not improvement-focused**. Picking a knob to turn — adjusting `hidden_dim`, swapping the optimizer, changing the loss — is a Phase 7 conversation. The goal here is to learn to look at the artifacts and know what they're telling you before you reach for any knob.

> **Note on embedded PNGs.** The plot screenshots referenced below live under `Data/processed/evaluation/plots/` and are produced by the real Phase 5 run on the CUDA machine. On the CPU-only dev box the links will appear broken until the run has been executed; the file paths themselves are stable. Re-rendering plots in this guide is intentionally out of scope (DD-RG-04 — the doc references the canonical artifacts, it does not copy them).

After the 10 artifact entries, two short cross-cutting sections:

* **How to read across combinations** — why delta comparisons (rung 2 flat vs. rung 2 pos, rung 2 vs. rung 3) are the primary lens, and why absolute metric values are secondary.
* **Vocabulary** — a one-page glossary mapping Phase 5 terms (`combination_id`, `slice`, `fold`, `pooled`, `headline metric`) back to their Spec-Phase5 definitions so this doc reads stand-alone.

---

## The 10 artifacts

### 1. metrics_headline.json

#### What it shows

The full per-`(combination, slice)` metric matrix. One JSON object with a top-level `"combinations"` key; each combination id maps to a slice → metric-cell dict. For an **season_holdout** combination the slices are `"val"` and `"test"`; for an **loso_cv** combination they are `"fold_0".."fold_<n-1>"` plus `"pooled"`. Each cell carries `n_games` plus the five metric families:

* `mae` — the **headline metric**, per-side averaged: `mean(|pred_home − true_home| + |pred_away − true_away|) / 2`. Numerically identical to Phase 4's `training_summaries.<combo>.val_mae` for season_holdout val cells (the cross-check is enforced by `tests/test_evaluate_cross_phase.py`).
* `mae_home`, `mae_away` — the same MAE split per side.
* `rmse_home`, `rmse_away` — per-side RMSE. No averaged-RMSE field exists; averaging RMSEs is ambiguous, so both sides are reported instead.
* `wl_accuracy` — predicted-winner accuracy. `sign(pred_home − pred_away)` vs. `sign(true_home − true_away)`.
* `spread_mae` — MAE of the predicted spread vs. the actual spread.
* `total_mae` — MAE of the predicted total points vs. the actual total.

Cells with `n_games == 0` carry JSON `null` for every metric.

#### What good looks like

For NFL scores, **per-side MAE in the high single digits to low double digits** is the realistic ballpark for a season-level model. A team's score is roughly N(23, σ≈10) before any conditioning, so a model that just predicts the league mean already achieves an MAE around 8–9; anything substantially below that is the model earning its keep. **W/L accuracy** in the 55–65 % band is a typical zone for a learned model that has signal but isn't claiming clairvoyance. **Spread MAE** should track per-side MAE × √2 if the home/away errors are uncorrelated; lower than that means the model captures something about who-beats-whom even when its absolute score predictions are off; higher than that means the model is fighting itself.

#### Red flags

* `wl_accuracy == 0.5` exactly, on a slice with `n_games > 10`. The model is no better than a coin flip — likely degenerate (always predicting the same outcome) or the slice is too small for the metric to mean anything.
* Per-side `mae_home` and `mae_away` differing by more than ~1.5 points. Asymmetric error across home/away is a calibration problem (the model has learned the home-field bias one-sidedly).
* `total_mae > 2 × mae`. Means home and away residuals are *correlated* — when the model overshoots one team, it tends to overshoot the other. Likely a global score-level miscalibration (e.g., the model expects every game to be high-scoring).
* `n_games` cell values that don't match the expected slice size from `splits_all.json`. Indicates a join bug between predictions and labels.

#### Action to consider

* Pull the offending combination's prediction parquet directly and compare its rows against the expected slice GameIds; if a join mismatch shows up here, the bug is in Phase 4's emission, not Phase 5's read.
* For per-side asymmetry, inspect `breakdowns/by_home_away.parquet` for the same combination and confirm the asymmetry is consistent across slices.
* For `wl_accuracy ≈ 0.5`, plot the predicted-spread distribution; if it collapses near zero the model has learned to hedge and Phase 7's first lever is the loss function (MAE penalizes confidence less than a calibrated-classification head would).

---

### 2. breakdowns/by_team.parquet

#### What it shows

One row per `(combination_id, slice, team_code, home_or_away)` cell, where `home_or_away` records the role the team played in the contributing games. Each team contributes two rows per combination/slice pair (one home, one away). The metric columns (`mae`, `mae_home`, `mae_away`, `rmse_home`, `rmse_away`, `wl_accuracy`, `spread_mae`, `total_mae`) report values **only over games where that team appears in that role**; per-side metrics (`mae_home`, `mae_away`) are reported relative to the home/away role of *the game*, not the queried team.

The parquet is universe-complete: every team appears in both roles for every `(combination, slice)`, with `n_games == 0` and `null` metrics where there are no contributing games.

#### What good looks like

Per-team MAE should cluster around the league-wide value reported in `metrics_headline.json` for the same combination/slice. A spread of ±2–4 points across teams within a slice is normal small-sample noise (per-team cells are small samples; the spread is high because `n_games` is tiny per cell). On `pooled` slices for loso_cv combinations, the per-team `n_games` grows and the spread tightens.

#### Red flags

* A single team consistently 5+ points above the league MAE across multiple slices and multiple combinations — that team is systematically harder for the model and the bias is real, not noise.
* Wide divergence between a team's home rows and away rows on the same combination — the model has learned something about the team's home-field behavior that doesn't generalize away (or vice versa).
* `n_games == 0` for a team in a role you'd expect to be populated. Means the team had no games in that slice — usually fine for season_holdout val (small slice) but suspicious for `pooled`.

#### Action to consider

* For chronic per-team bias, pull the contributing games for that team via `nflpredictor.diagnostics.trace.lookup_predictions` and look for a shared feature (a specific QB, a roof type, a divisional matchup); chase that thread in Phase 7's error analysis.
* For home/away divergence within a team, cross-reference against `by_home_away.parquet` to see whether the asymmetry is team-specific or league-wide.

---

### 3. breakdowns/by_week.parquet

#### What it shows

One row per `(combination_id, slice, week)` cell, where `week` is the integer 1–18 read from `features_flat_all.parquet`'s `week` column. Each game contributes one row to its week's cell. The parquet is universe-complete across all 18 weeks even when a slice contains only a subset.

#### What good looks like

Per-week MAE should be roughly stationary across the season. Some week-to-week noise is expected (each by_week cell is a small sample), but a clear *trend* — e.g., MAE rising linearly week-over-week — is not. On `pooled` slices the noise damps and any real seasonal effect becomes more visible.

#### Red flags

* MAE monotonically increasing across the val slice's weeks (13 → 14 → 15) on the **season_holdout** combinations. Possibly the model is overfit to the early-season training data and degrades as the season's underlying distribution shifts (injuries, eliminated teams resting starters).
* A single week with MAE 5+ points above its neighbors. That week may have had a high-variance event (a blowout, an unusual weather game) that the model couldn't represent.
* Test-slice weeks (16–18) systematically worse than val-slice weeks (13–15) on the same combination. Indicates the model is *brittle to recency*; investigate whether the underlying training cut-off should be moved later (Phase 7 splits-config conversation).

#### Action to consider

* For monotonic per-week degradation, the **loso_cv** strategy is the proximate test: if the per-fold MAE varies across held-out seasons (more training data, more recent val slice), the model is generalizing in time; if not, the model is memorizing recent weeks.
* For single-week outliers, use `lookup_predictions(game_id=…)` for each game that week and check the residual distribution per game — a single huge residual will dominate the week's mean MAE.

---

### 4. breakdowns/by_home_away.parquet

#### What it shows

Two rows per `(combination_id, slice)` cell. Row 1: `home_or_away = "home"`, reporting `mae_home`, `rmse_home`, etc. Row 2: `home_or_away = "away"`, reporting the away counterparts. The remaining columns (`wl_accuracy`, `spread_mae`, `total_mae`) are game-level and are duplicated across the two rows.

#### What good looks like

Per-side MAE should differ by less than ~1.5 points between home and away on the same combination/slice. The mean home score in the NFL is about 3 points above the mean away score (the home-field advantage); a model that hasn't learned this will show `mae_home > mae_away` for a degenerate mean-only predictor and roughly equal MAEs for a calibrated learned model. The trivial rung 0 baseline pins one ceiling for what "unlearned home/away" looks like; the rung 1 team-mean baseline pins another (it learns per-team but not home/away).

#### Red flags

* Persistent `mae_home > mae_away + 2` across all combinations and slices. The encoder isn't using the home/away signal at all and the model is treating every game symmetrically.
* The trivial-rung row matches the learned-rung row to ~0.1 point. Means the learned rung is *not* improving on the unconditional baseline — for `by_home_away` specifically that's a strong signal of underfitting.

#### Action to consider

* If learned-rung MAE matches trivial-rung MAE within noise on every breakdown, the model isn't learning. Phase 7's first lever is to look at the loss curves (`Docs/Phase6-TrainingDynamics.md`) and check whether the model is converging at all.
* For persistent home/away asymmetry on learned rungs, check `feature_vocab.json` — is `team_codes` shared correctly between home and away? A vocab-routing bug here would show up exactly this way.

---

### 5. breakdowns/by_surface.parquet

#### What it shows

One row per `(combination_id, slice, surface_code, surface_label)` cell, where `surface_code` is the integer code from Phase 2's `surface` column and `surface_label` is the human-readable form from `feature_vocab.json → surface` (e.g., `"grass"`, `"fieldturf"`, `"matrixturf"`). The parquet is universe-complete across every surface that appeared in the season.

#### What good looks like

Per-surface MAE should not vary systematically across surfaces. The headline result is a uniform color across the vocabulary; surface is a low-cardinality categorical (≤ 8 entries, so the encoder routes it through one-hot) and its predictive value is real but small (kicker accuracy, fumble rates differ slightly between turf and grass). For most slices the per-surface `n_games` is 1–3, so a 5-point spread on a single cell is almost certainly noise.

#### Red flags

* A specific surface consistently 3+ points worse than the league mean across multiple slices and multiple combinations. The surface vocabulary maps to a behavior the model isn't capturing.
* A surface with `n_games > 0` on the train slice but no rows in val/test — indicates the val/test universe excluded that surface, which means the model never had a chance to be evaluated against it. Investigate the splits config.

#### Action to consider

* For systematic surface-correlated error, pull the actual `surface_label` values via `feature_vocab.json` and check whether two label variants (e.g., `"fieldturf"` and `"field turf"`) are colliding into one vocab entry — a Phase 1 normalization bug masking as a Phase 4 modeling issue.

---

### 6. breakdowns/by_roof.parquet

#### What it shows

Same shape as `by_surface.parquet` but keyed by `roof_code` + `roof_label` (e.g., `"dome"`, `"outdoors"`, `"retractable roof (closed)"`, `"retractable roof (open)"`). One row per cell; universe-complete across every roof type that appeared in the season.

#### What good looks like

Indoor games (dome / retractable-roof-closed) should have lower per-side MAE than outdoor games for a well-calibrated model — indoor games carry less weather noise and are more predictable. The expected gap is small (~0.5–1.0 points). Outdoor games with adverse weather will widen the residual distribution but the mean MAE shouldn't move dramatically because the weather features (`weather_temp_f`, `weather_humidity_pct`, `weather_wind_mph`) capture some of that variance.

#### Red flags

* Indoor MAE > outdoor MAE on any combination. The model is using the weather features in the *wrong direction* — possibly the encoder is feeding weather signals into indoor games where they're set to a sentinel value, and the model has learned to over-react to the sentinel.
* Identical MAE across every roof type to ~0.1 point. The model is ignoring the roof signal entirely. Combined with similar flatness on `by_surface`, indicates the low-card categoricals aren't pulling weight.

#### Action to consider

* For misordered indoor/outdoor MAE, inspect the weather columns directly in `features_flat_all.parquet` for indoor games — are they `null`-filled, sentinel-filled, or carrying real (incorrect) values? The fix is in Phase 2, not Phase 4.
* For flat per-roof MAE, this is part of the broader "low-card categoricals aren't pulling weight" pattern; look at the residual-distribution plots to see whether the model is making one global prediction shape regardless of the inputs.

---

### 7. plots/&lt;combination_id&gt;__&lt;slice&gt;__scatter.png

#### What it shows

A predicted-vs-actual scatter plot for one `(combination, slice)` cell. X-axis: actual score. Y-axis: predicted score. Two series overlaid: home (one color) and away (another). The identity line `y = x` is drawn so perfectly-calibrated predictions land on the line; over-predictions sit above it, under-predictions sit below.

One file per season_holdout val/test slice (`__val__scatter.png`, `__test__scatter.png`) and one per loso_cv pooled slice (`__pooled__scatter.png`).

**Representative example:**

![scatter — rung2_linear flat season_holdout val](../Data/processed/evaluation/plots/rung2_linear__flat__season_holdout__val__scatter.png)

#### What good looks like

Points cluster around the identity line with roughly symmetric scatter above and below. The cloud should be wider in the high-score tails (40+) than near the mean (~23) because absolute residuals are larger for high-variance games. The home and away series should overlap; if they form two separate clouds offset along the y-axis, the model has a systematic home/away bias.

#### Red flags

* The cloud is a horizontal band — predictions cluster around one y value regardless of x. The model is making roughly the same prediction for every game (the league-mean baseline shape).
* The cloud is wider above the identity line than below (or vice versa). The model has a directional bias — over-predicts or under-predicts on average.
* Outliers far from the line at extreme actual values (true score > 40 or < 10). The model has no representation for blowouts or shutouts and falls back to its prior. This is *expected* for the trivial rungs but should improve on rung 2 / rung 3.

#### Action to consider

* For a horizontal-band shape, this is the canonical "model isn't learning" signature; cross-reference against `Docs/Phase6-TrainingDynamics.md`'s loss-curve patterns to see whether training even converged.
* For directional bias, the simplest diagnostic is the residual-distribution plot (next entry) — confirm the bias there before reaching for fixes.

---

### 8. plots/&lt;combination_id&gt;__&lt;slice&gt;__residuals.png

#### What it shows

A histogram of `pred − true` residuals over the same `(combination, slice)` cell as the scatter plot, with two series overlaid: home and away. The histogram bins are auto-chosen by matplotlib; titles and axis labels are deterministic per EV-PLOT-06.

One file per season_holdout val/test slice and one per loso_cv pooled slice.

**Representative example:**

![residuals — rung2_linear flat season_holdout val](../Data/processed/evaluation/plots/rung2_linear__flat__season_holdout__val__residuals.png)

#### What good looks like

A roughly bell-shaped distribution centered on zero. The standard deviation should sit somewhere in the 9–12 point range for this problem. Home and away histograms should overlap closely; their means should both be near zero.

#### Red flags

* Histogram center shifted off zero (e.g., centered around +3). The model has a global directional bias — it's over-predicting (or under-predicting) on average.
* Strongly asymmetric distribution — a long tail on one side and a short tail on the other. The L1 (MAE) loss isn't penalizing the tail, so the model has learned to be miscalibrated in one direction to win on the short side.
* Two visible peaks (bimodal). The model has effectively learned two different prediction modes and is switching between them; could indicate a categorical feature collapsing two true clusters into one embedding.

#### Action to consider

* For a shifted center, the fix is calibration — could be addressed in post-processing (subtract the mean bias) or by re-balancing the training set if early-season vs. late-season scoring rates differ from the train distribution.
* For asymmetric or bimodal shapes, the issue is more fundamental than calibration; Phase 7's tools are loss-function changes (Huber, MSE) or richer feature encodings.

---

### 9. plots/&lt;combination_id&gt;__&lt;slice&gt;__by_week.png

#### What it shows

A line chart of the `headline_metric` (default: `mae`) across `week` for the named `(combination, slice)` cell. One line per metric series; the x-axis is `week`, the y-axis is the metric value. Configurable via `evaluation_config.yaml`'s `plots.breakdown_plots` list — `by_week` is enabled in the v1 default.

This plot complements the `by_week.parquet` breakdown: the parquet has the numbers, the plot has the visual trend.

**Representative example:**

![by_week — rung2_linear flat season_holdout val](../Data/processed/evaluation/plots/rung2_linear__flat__season_holdout__val__by_week.png)

#### What good looks like

A roughly flat line with week-over-week noise (each a single by_week cell is a small sample, so single-week noise is large). On `pooled` slices for loso_cv combinations the noise damps; a flat line on `pooled` is the gold standard. No monotonic trend.

#### Red flags

* Strong upward trend across the val slice (weeks 13 → 14 → 15) — model is degrading week-over-week, suggesting train/val distribution shift.
* A single week's MAE 3+ points above its neighbors. Look at that week's games specifically.
* The line for the `__test__` slice is uniformly higher than the `__val__` slice line for the same combination. Suggests the model overfit to the val slice (e.g., implicit early-stopping leakage through hyperparameter selection).

#### Action to consider

* For monotonic per-week degradation, the immediate diagnostic is to plot the same line for an **loso_cv** combination — if loso_cv's per-fold MAE is stable across held-out seasons, the model is generalizing; if loso_cv trends too, the issue is in the data.
* For a single-week spike, pull that week's games via `nflpredictor.diagnostics.trace.lookup_predictions` and look at the residual distribution for those games specifically.

---

### 10. plots/ladder_summary__{val,test,pooled}.png

#### What it shows

A grouped bar chart with one bar per **combination** in lexicographic order; bar height is the `headline_metric` value for that combination at the named slice (`val`, `test`, or `pooled`). Three files total: `ladder_summary__val.png`, `ladder_summary__test.png`, `ladder_summary__pooled.png`. Combinations whose parquet does not contribute to the given slice are omitted (e.g., loso_cv combinations don't appear in `__val.png` because their natural slice is `pooled`).

The chart is the **single most important snapshot** of the model ladder — it's where you eyeball "did this rung earn its keep over the previous rung."

**Representative example:**

![ladder summary — val](../Data/processed/evaluation/plots/ladder_summary__val.png)

#### What good looks like

A monotonic descent (lower is better for MAE) as you read left-to-right across the rungs: rung 0 (mean baseline) → rung 1 (team-mean baseline) → rung 2 (linear) → rung 3 (MLP). Rung 1 should beat rung 0 by a meaningful margin (team-mean adds real information). Rung 2 should beat rung 1 by a smaller but real margin (the encoder is adding something). Rung 3 should beat rung 2 by a smaller margin still (the MLP captures non-linearities the linear model can't). The flat-vs-pos shape comparison should show one consistently lower than the other for the same rung; that delta is the "is the position-grouped representation worth it" answer.

#### Red flags

* Rung 1 (team-mean) ≈ rung 0 (overall mean). Either the team feature isn't connecting through to the prediction, or the season is unusually flat (every team scoring near the league mean).
* Rung 2 ≥ rung 1. The encoder is adding noise rather than signal — likely an embedding-dimension or weight-init issue.
* Rung 3 ≥ rung 2 by more than ~0.5 point. The MLP is overfitting on a dataset this small (272 games); the linear rung is winning the bias-variance trade.
* The flat and pos bars for the same learned rung differ by more than 2 points. One representation is dominating; investigate why (`features_pos_all.parquet`'s per-position aggregation may be losing information that `features_flat_all.parquet` retains, or vice versa).
* The `__test__.png` shape doesn't match the `__val__.png` shape (e.g., rung 3 wins on val but loses on test). Strong indicator of val-slice overfit via hyperparameter selection.

#### Action to consider

* For ladder reversals (a higher rung losing to a lower rung), the diagnostic is the training-dynamics doc and notebook (`Docs/Phase6-TrainingDynamics.md`) — look at the loss curves for the regressing rung first.
* For val/test shape divergence, the lever is in Phase 3 (splits config) or Phase 4 (training config — particularly `max_epochs` and `lr`); Phase 7 is the conversation.

---

## How to read across combinations

The single most useful question you can ask of these artifacts is not *"what's the MAE for rung 3 MLP pos on val?"* — it's *"how does that number compare to the rung below it, and to the same rung on the other shape?"* The absolute MAE value is a function of the underlying problem's difficulty (NFL scores have ~σ=10 of irreducible noise per side); the value alone doesn't tell you whether the model is good. **Delta comparisons** do.

The three delta lenses worth carrying around:

1. **Rung-to-rung delta on the same shape.** E.g., `rung3_mlp__flat__season_holdout.val.mae` vs. `rung2_linear__flat__season_holdout.val.mae`. Answers: *did the extra capacity earn its keep?* A delta of < 0.2 points on this dataset (272 games) is essentially noise; you need ≥ 0.5 to call it a real improvement and ≥ 1.0 to call it large.

2. **Shape delta on the same rung.** E.g., `rung2_linear__pos__season_holdout.val.mae` vs. `rung2_linear__flat__season_holdout.val.mae`. Answers: *is the position-grouped feature shape more informative than the slot-indexed shape?* This is the cleanest A/B in the entire ladder because the only thing changing is the feature layout. A consistent shape preference across rungs and slices is real signal.

3. **Strategy delta on the same combination.** E.g., `rung2_linear__flat__season_holdout.val.mae` vs. `rung2_linear__flat__loso_cv.pooled.mae`. Answers: *does the leave-one-season-out protocol agree with the fixed-week-boundary protocol about which combinations win?* Disagreement is a flag that the season_holdout val slice is small enough to be lucky in one direction.

For each lens, the `ladder_summary__*.png` plots are the headline; the per-`(combination, slice)` rows in `metrics_headline.json` are the receipts.

**A word on absolute values.** A model with `mae = 8.5` on val is doing *something* on this dataset — but "something" relative to what? The rung 0 baseline (always predict the league mean) gives a useful upper bound: typically MAE ≈ 8.5–9.0 on this problem. A learned rung that doesn't beat the baseline by at least 1 full point is not earning its keep. A learned rung that beats it by 3+ is doing real work. The middle ground (0–1 point improvement) is the most common case and the most interesting to interrogate.

---

## Vocabulary

A one-page glossary for terms used throughout this guide. Every term links back to its canonical definition in `Spec-Phase5-Evaluation.md` or `Spec-Phase4-BaselineLadder.md`.

* **combination_id** — the filename stem of a Phase 4 prediction parquet, e.g., `rung2_linear__flat__season_holdout`. Encodes rung (`rung0_mean`, `rung1_team_mean`, `rung2_linear`, `rung3_mlp`), feature shape (`flat`, `pos`, or `none` for the rungs that don't take features), and splits strategy (`s1` or `s3`). Spec source: Spec-Phase4 §3.2 (TR-OUT-01) and Spec-Phase5 §3.5 (EV-COMB-01).

* **slice** — the named subset of games over which a metric is computed. For season_holdout combinations: `"val"` or `"test"`. For loso_cv combinations: `"fold_0".."fold_<n-1>"` (per-fold val metrics) or `"pooled"` (concatenated-fold metrics). Spec source: Spec-Phase5 §3.5 (EV-COMB-02 / EV-COMB-03).

* **fold** — one of the leave-one-season-out `(train, val)` partitions loso_cv emits. Each fold is identified by its `val_season` (the held-out season) and its zero-based `fold_index`. Spec source: Spec-Phase3 §3.4 and Spec-Phase4 §3.5.

* **pooled** — the metric set computed by concatenating every fold's val predictions into one universe, then running the metric formulas once. Different from the *mean* of per-fold metrics (the difference is usually small but real). Spec source: Spec-Phase5 §3.5 (EV-COMB-03) and §3.4 (EV-MET-07).

* **headline metric** — the single metric chosen by `evaluation_config.yaml`'s `headline_metric` field as the headline summary across artifacts (default: `mae`). Affects the `ladder_summary__*.png` bar heights and the `by_week` line-plot y-axis. Spec source: Spec-Phase5 §3.4 (EV-MET-01).

* **rung** — one position in Phase 4's model ladder. Rung 0 = league mean. Rung 1 = per-team mean. Rung 2 = `nn.Linear` over the encoded feature vector. Rung 3 = small MLP over the same encoded vector. Spec source: Spec-Phase4 §3.4.

* **feature shape** — the layout of the encoded feature vector. `flat` is slot-indexed (one column per `(HomeOff01..HomeDef11)` slot × Madden attribute). `pos` is position-indexed (one column per canonical position × Madden attribute). `none` is for the trivial rungs that don't read features. Spec source: Spec-Phase2 §3.5 and Spec-Phase4 §3.3.

* **strategy** — season_holdout (fixed week boundaries) or loso_cv (leave-one-season-out folds). The third axis of the combination matrix, alongside rung and feature shape. Spec source: Spec-Phase3 §3.4.

* **delta** — a difference between two metric cells used as a comparison lens. The three lenses worth carrying around are documented above in "How to read across combinations."

* **universe-complete** (breakdown parquets) — the property that every value in the breakdown's key universe appears as a row for every `(combination, slice)`, even when `n_games == 0`. Makes the parquet self-describing and downstream joins safer. Spec source: Spec-Phase5 §3.4 (EV-MET-09) and §3.6 (EV-BRK-01..06).
