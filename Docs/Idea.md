# NFL Predictor — Project Idea

## Overview

This is the umbrella idea document for the NFL Predictor project — a learning exercise in training a real-world model end-to-end with deliberate best practices for disparate data. It supersedes `Docs/Overview.md`.

The project's arc is split into seven phases. Decisions made early in the document constrain later phases.

1. **Data Build** — produce processed training data from two disparate raw sources.
2. **Feature Engineering & Model Inputs** — decide which columns the model actually sees, and how categorical signals are encoded.
3. **Splits** — define a time-aware partition into train / validation / test.
4. **Baseline & Model Ladder** — PyTorch-native progression from trivial baselines up to candidate models.
5. **Evaluation** — score predictions and derive secondary metrics (W/L, spread, total).
6. **Error Analysis & Iteration** — diagnose where the model misses and decide whether to add Madden columns or change shape.
7. **2025 Test (deferred)** — future aspiration if 2025 data becomes available; out of scope for this iteration.

The Data Build phase (Phase 1) is the most worked-out section; later phases are exploratory and intentionally lighter on commitments. Because this is a learning project, the document also surfaces *why* each option exists, so the choice can favor teaching value over pure expedience.

## Human-Provided Direction

Explicit decisions and constraints already provided. These are treated as fixed for this iteration of the idea.

- **Framework**: PyTorch is the modeling framework. This supersedes the linear/RF/GBT ladder currently listed in `Overview.md` → "Candidate Models". The ladder is still useful as a *baseline reference*, but the production model will be a PyTorch model. (Reconciling this with `Overview.md` is an open question — see below.)
- **Order of operations**: The training data is finalized *before* any model code is written. No "design the model first, then shape the data to it."
- **Training data shape**: Two processed files form the training data — a processed Madden file and a processed box-scores file — joined by `madden_id`. A third file, a mapping/audit artifact, is produced alongside them for troubleshooting only; consumers do not need it to train. The earlier instinct to materialize a single wide training table is superseded — that read of "all rankings alongside the box scores" turned out to be a held assumption, not a requirement. The relational pair gives the same fidelity with less duplication.
- **Madden as player registry**: Madden is the canonical source of player identity. The build step adds a `madden_id` column to the Madden file. Names and team-name mapping live there, and only there.
- **ID format**: `2024-00001`-style strings (year prefix + zero-padded sequence). The year prefix encodes Madden vintage; sequence is assigned at build time and stable across re-runs of the same source.
- **Box-score IDs are rewritten in the processed copy**: The build emits a processed box-scores file that is structurally identical to the raw file, except the per-slot `_ID` columns now hold `madden_id` values. The raw box-scores file in `Data/raw/` is preserved untouched (it keeps the PFR-style `_ID` values as the historical source of truth). The processed file has no blank `_ID` cells — every starter resolves to a `madden_id` because unmatched players get a row in the Madden file too.
- **Mapping file is for troubleshooting, not for the join**: A `player_id_mapping.csv` artifact records `(box_score_id, madden_id, note)` for every starter the build encountered. The `note` column captures resolution provenance — e.g., `"raw _ID blank; resolved by name 'T.J. Watt' on team pit"` or `"deterministic name match"` or `"manual override"`. This file exists for humans to audit and debug; the model code joins `processed/box_scores_2024.csv` and `processed/madden_2024.csv` directly on `madden_id` and never reads the mapping file.
- **Build resolves blanks via name lookup**: When a raw box-score `_ID` is blank (e.g., `T.J. Watt`), the build uses name + team to find the right Madden row and writes that `madden_id` into the processed box-scores file. The blank does not propagate forward — but it does get recorded in the mapping file's `note` column.
- **Unmatched players become null Madden rows with a flag**: Players appearing in box scores but missing from raw Madden get added to the processed Madden file with null rating values and `matched = 0`. The `matched` flag enables filtering and lets us A/B different matching strategies later. Existing Madden rows carry `matched = 1`.
- **Null-fill happens at the end of the Madden build**: After unmatched rows are added, a final build step fills null rating values with column averages (computed over `matched = 1` rows only, so unmatched rows don't pollute their own fill values).
- **Madden file keeps all original columns**: The processed Madden file carries the full 69 source columns plus `madden_id` and `matched`. The earlier "v1 slice / reserve pool" framing is moot — the training step picks whichever columns it wants from the full set. The initial training run will start with `Overall Rating` + `Archetype`; expanding is a one-line change in the training code.
- **No precomputed team aggregates**: Bench-depth / roster-mean signals are deliberately not included. They would require an averaging choice (uniform vs. weighted) that introduces uncontrolled parameters; if needed, they can be derived later from the Madden file at training time.
- **Manual overrides via versioned file**: A `Data/player_overrides.csv` (or equivalent) is read by the build step and edited by hand for ambiguous cases. It is itself a versioned, audit-friendly artifact.
- **File layout**: Source files move to `Data/raw/`; build outputs land in `Data/processed/`.
- **PyTorch-native baseline ladder**: The Overview's linear/RF/GBT ladder is replaced with a PyTorch-native progression (mean predictor → `nn.Linear` → small MLP → larger / attention-over-slots). This is closure on the C-option choice.
- **Scope of this round**: 2024 data only. The build's contract is intentionally general so future seasons could be ingested without redesign, but 2025 is not part of training, evaluation, or methodology for this iteration.
- **Madden source contract**: The current Madden CSV's *provider* may change later, but its *schema* (the 69-column shape) is treated as the canonical input contract for roster-level ratings. Alternative providers will be mapped into this shape rather than the other way around.
- **Targets**: `home_score` and `away_score` (regression). Win/loss, spread, and total are derived from predicted scores rather than modeled directly.
- **Splits**: Time-aware. Train, val, and test all carve out of 2024 — see Phase 3.
- **Scope**: Single-developer learning project. Reproducibility and clear documentation matter; production infrastructure (registry, serving, monitoring) does not.
- **Canonical example intent**: Decisions should favor approaches that teach broadly applicable best practices for disparate-data ML, even when a shortcut would work for this specific dataset.

## Problem Statement

The end-to-end problem: produce a PyTorch model that predicts `home_score` and `away_score` for an NFL game given only information knowable before kickoff. The model is trained, validated, and tested against time-disjoint slices of the 2024 season (272 games total — see Phase 3 for the partition). The build contract is general enough that future seasons could be ingested without redesign, but that portability is a design discipline rather than a methodology requirement.

Two raw files describe the same world from different angles and do not share a primary key:

- `Data/box_scores_2024.csv` — 272 games, 164 columns. Identifies players by Pro-Football-Reference-style IDs (e.g. `MahoPa00`) and team codes (`kan`, `rav`). Carries the labels we want to predict.
- `Data/maddennfl24fullplayerratings.csv` — 2,368 players, 69 columns. Identifies players by `Full Name` and `Team` (nicknames like `49ers`, `Ravens`). Carries the rating signal we want as features.

The training-data work — Phase 1 — is therefore a **build step** that produces two processed files: a Madden file augmented with generated `madden_id` values, and a box-scores file whose lineup `_ID` columns reference those same IDs (no blanks). After the build, joining is a trivial ID lookup at model-fit time. Real-world frictions the build step has to absorb:

- Team representation differs (3-letter PFR code vs. franchise nickname). Resolved by a hand-maintained team-name lookup used during ID assignment.
- Player names will not match exactly (Jr./Sr. suffixes, hyphens, periods in initials, accented characters, "III"/"II", informal vs. legal names). Resolved by name normalization plus a manual-overrides file for the residual cases.
- At least one box-score row in the sample has a blank `_ID` (`T.J. Watt`), so the source side is itself imperfect. The build needs a documented rule for how blank source IDs get resolved (by name lookup) or recorded as legitimately missing.
- Madden is a roster snapshot, not a per-week record, so it cannot represent in-season trades, injuries, or rookies signed after the snapshot. Players who appear in box scores but not in Madden need an explicit policy (no rating row vs. a placeholder row with null ratings).
- `Archetype` is categorical with many levels. Its encoding for the model is deferred to model-fit time — the build step just emits raw strings.

The idea is to settle the build-step's identity policy, file shapes, and the documentation/contract around them before writing any model.

## Goals

End-to-end:

- Predict `home_score` and `away_score` for the held-out test slice of 2024 (Weeks 16–18) using a PyTorch model trained on Weeks 1–12 and tuned against Weeks 13–15, with derived metrics (W/L, spread, total) computed from the score predictions.
- The model must beat genuinely-dumb baselines (predict the league mean; predict each team's season mean) by a non-trivial margin on the validation slice, or the project re-examines its features rather than its architecture.
- Establish a clear, reproducible pipeline: raw data → build step → processed data → splits → training run → frozen model → test-slice evaluation. Each handoff is a documented contract.
- The project reads as a teaching example. Decisions and their reasoning are recorded inline or in `Docs/`.

Phase 1 (Data Build):

- Produce the finalized training artifacts under `Data/processed/`:
  - `madden_2024.csv` — all 69 original Madden columns + `madden_id` (e.g. `2024-00001`) + `matched` flag, with null ratings filled with column averages.
  - `box_scores_2024.csv` — structurally identical to the raw box-scores file, except every `_ID` column now holds a `madden_id`. No blanks.
  - `player_id_mapping.csv` — audit/troubleshooting artifact: `(box_score_id, madden_id, note)` for every starter the build encountered. Not required by model code.
- The build step that emits these is deterministic, re-runnable, and reads from `Data/raw/` (which includes the raw sources plus a hand-edited `player_overrides.csv` for ambiguous matches).
- Define a stable input *contract* (column shape, types, allowed values) that future seasons and alternate ratings providers can be coerced into without redesigning the schema.

## Non-Goals

- Building a serving or inference pipeline beyond what's needed to score the held-out test slice of 2024.
- Scraping or refreshing the box-score or Madden sources. Both files are treated as given inputs for this round.
- Predicting player-level outcomes (yards, touchdowns, interceptions, etc.). The labels are team scores only.
- Replacing the Madden source. If the provider changes later, that's a separate effort that maps the new source onto the existing 69-column contract.
- Modeling betting markets directly (Vegas spreads/totals as inputs). They could be added later as a feature group but are not in scope.
- In-season retraining or online learning. The model is trained once on the train slice of 2024 and frozen before evaluation against the val or test slices.

## Current System Context

- `Data/box_scores_2024.csv` — 272 rows × 164 cols. One row per game. 11 offense + 11 defense slots per side per game (44 starters total). Each slot has `_Position`, `_Name`, `_ID`. Includes officials, weather, coach names, stadium, surface, roof.
- `Data/maddennfl24fullplayerratings.csv` — 2,368 rows × 69 cols. One row per Madden roster player. Includes general ratings (Speed, Awareness…), position-specific ratings (Throw Accuracy Short/Mid/Deep, Man/Zone Coverage…), `Jersey Number`, `Height`, `Weight`, `Age`, `Birthdate`, `Years Pro`, `College`, ` Total Salary `, ` Signing Bonus ` (note the surrounding whitespace in those two headers).
- `Docs/Overview.md` — the project's parent document. Establishes targets, methodology, evaluation plan, baselines, and a reproducibility checklist. References a tree/linear model ladder that the PyTorch direction now supersedes.
- No source code, no chosen language toolchain in-repo yet. `CLAUDE.md` notes the project is greenfield and flags that the stack hasn't been committed to outside of "PyTorch will be used."

## Phase 1: Data Build

> **Status**: Implementation complete (2026-05-17). The build is implemented at `src/nflpredictor/databuild/` and produces the four artifacts in `Data/processed/` via `python -m nflpredictor.databuild`. Real-data run summary: 7,863 tier-2, 2,405 tier-3, 90 tier-4 matches; 271 unique unmatched players appended; 534 position mismatches logged across 11,968 starter slots (272 games × 44). The sections below are retained for reference so the reasoning behind the chosen direction stays visible. **Chosen**: A3-with-A2-bootstrap, B3, C2.

Two design dimensions dominate this stage: **(A) how player rows get matched** and **(B) what the unified table looks like physically**. They are largely independent; any A can pair with any B. A third dimension — **(C) how PyTorch reconciles with the Overview's model ladder** — is a Phase 4 decision recorded here for historical reasons.

### A. Player-mapping strategy

**Chosen: A3 bootstrapped by A2.** The build step runs the A2 tiered-with-fuzzy match logic to populate the crosswalk file (which then plays the role A3 describes). The crosswalk becomes the durable artifact; A2 is just how it gets populated initially. Manual overrides override the automated match output.

#### Approach A1: Deterministic-only join on (normalized team, normalized name)

Normalize team representations through a hand-maintained lookup (`kan` ↔ `Chiefs`, `rav` ↔ `Ravens`, `sfo` ↔ `49ers`, …), normalize names (strip suffixes, fold accents, collapse punctuation), and join on the resulting tuple.

Potential advantages:
- Simplest to implement and audit.
- Easy to spot unmatched rows: anything that falls through is a name we have to look at by hand.
- Fully reproducible — same input always yields the same join.

Potential concerns:
- Will leave some legitimate players unmatched (different legal names, mid-season call-ups, players Madden lists under a different team than they started for in a given game).
- Doesn't gracefully handle the trade/practice-squad cases Madden cannot represent at all.

#### Approach A2: Tiered match with fuzzy fallback

Layered matching: (1) exact normalized-name match within team; (2) exact normalized-name match league-wide (handles in-season trades where Madden's team is stale); (3) fuzzy match (Levenshtein / token-set ratio) with a confidence threshold; (4) anything below threshold escalates to a manual-overrides file.

Potential advantages:
- Catches more players automatically.
- Manual-overrides file becomes the single source of truth for ambiguous cases and is itself a versionable artifact.
- The tier that matched is auditable per-player, which is good teaching material.

Potential concerns:
- Fuzzy matching introduces a quality knob (threshold) that has to be defended.
- More moving parts; more places for the join to silently regress.

#### Approach A3: External ID crosswalk

Build or borrow a third table that maps PFR player IDs to a stable external identifier (e.g. nflverse, Sleeper, ESPN ID), then map Madden rows to the same identifier. Join through the crosswalk.

Potential advantages:
- The most "real-world correct" pattern — identifier-based joins are robust to name and team drift.
- Generalizes well if a second ratings provider is ever added.

Potential concerns:
- Requires an additional dataset that isn't in-repo today and may need network fetching.
- Higher initial effort; possibly disproportionate for 272 games and 2,368 players.

### B. Physical shape of the training data

**Chosen: B3 — relational.** The build emits a processed Madden file (`madden_id` + `matched` + all 69 columns; nulls filled) and a processed box-scores file (structurally identical to raw, but with `_ID` columns rewritten to `madden_id` values; no blanks). The two are joined directly on `madden_id` at training time. A separate mapping/audit file is also emitted for troubleshooting but is not part of the join. The raw files in `Data/raw/` are preserved untouched. B1 and B2 are retained below as considered alternatives.

#### Approach B3 (leading): Two relational files joined by `player_id`

Persist two processed artifacts:

- A Madden file with an added `player_id` column (e.g., `P00001`-style explicit strings) plus the v1 slice of `Overall Rating` + `Archetype`. One row per Madden player. The full source Madden file remains on disk for additive-column expansion.
- A box-scores file structurally identical to the source, except every starter slot's `_ID` column carries the new `player_id` instead of (or alongside — see open question) the PFR-style ID.

A model-fit-time step joins lineup IDs against the Madden table to produce whatever shape the model needs (wide, long, embedding-lookup, etc.) without changing what's persisted on disk.

Potential advantages:
- No data duplication — each Madden row appears once, not 44 times across games.
- Adding or removing a Madden column is a one-file change with no downstream re-shape work.
- The hard work of name → ID mapping happens once, in the build step. After that, joins are trivial ID lookups.
- Provider swap (Madden → some other ratings source) is a single-file replacement that preserves the IDs and the box-score side untouched.
- Mirrors how real ML data pipelines tend to be organized (entity tables + event tables), which fits the project's "canonical example" goal.

Potential concerns:
- Two artifacts must be versioned together — the build step has to emit them as a matched pair.
- The model code has to do a join step at fit time. Cheap with pandas/polars but it does push more responsibility onto the model-fit boundary.
- Players who appear in box scores but not in Madden need an explicit policy (placeholder row, sentinel ID, or null on the box-score side).

#### Approach B1 (alternative): Wide table, slot-indexed columns

For each of the 44 starter slots, prefix the chosen Madden columns: `HomeOff01_Madden_OverallRating`, `HomeOff01_Madden_Archetype`, …, `AwayDef11_Madden_Archetype` — ~88 player columns total with the v1 slice; scales linearly as more Madden columns are added.

Potential advantages:
- Direct correspondence to box-score slot positions — easy to inspect per-row.
- No model-fit-time join needed.

Potential concerns:
- Duplication: each player's ratings repeat across every game they start.
- Slot index (`Off01` vs `Off02`) is positional, not semantically meaningful.
- Adding/removing a Madden column triggers a 44× column rename.

#### Approach B2 (alternative): Wide table, position-indexed columns

Group starters by position (QB, RB1, RB2, WR1..WR4, …) rather than by slot index. Same shape as B1 but with semantically named columns.

Potential advantages:
- Columns carry meaning (`Home_QB_Madden_OverallRating` is interpretable across games).
- Reduces the slot-permutation noise of B1.

Potential concerns:
- Requires a stable position taxonomy and a rule for unusual formations (3 TE sets, no fullback).
- Same duplication and rename-on-expansion costs as B1.

### C. Reconciling PyTorch with the Overview's model ladder

**Chosen: C2 — PyTorch-native ladder.** Mean predictor → linear `nn.Linear` → small MLP → larger MLP or attention over starter slots. The original tree/linear ladder in `Overview.md` is superseded; `Overview.md` should be updated to match (or noted as legacy).

C1 (keep tree/linear models as cross-paradigm baselines) was considered but declined — single-developer learning project, stays in one framework.

## Phase 2: Feature Engineering & Model Inputs

> **Status**: Implementation complete (2026-05-18). The feature build is implemented at `src/nflpredictor/features/` and produces four artifacts in `Data/processed/` via `python -m nflpredictor.features` (gated on a Phase 1 source-hash check). Real-data run summary: 272 games × 202 columns for B-flat, 272 × 258 for B-pos; 9 vocab keys (Archetype=46, coaches=35, day_of_week=6, officials=121, positions=23, roof=4, stadium=34, surface=6, team_codes=32); ~18 B-pos OL/LB bucket-overflow warnings on real lineups; wall-clock ~1.2s. The sections below are retained for reference so the reasoning behind the chosen direction stays visible. **Choices**: 2A deferred-to-config, 2B both B-flat and B-pos emitted, 2C parsed weather + included officials, 2D integer codes + vocab sidecar.

At training time, the joined view of a single game contains:

- **Game-level box-score columns** (week, day-of-week, start time, stadium, attendance, duration, roof, surface, weather string, coach names, officials, home/away team codes, scores).
- **44 starter rows** (22 home + 22 away, each with position + the resolved `madden_id` and a join into the Madden file's 69 columns).

The model can't consume all of this raw. Three decisions shape what does get fed in.

### 2A. Which Madden columns to include

**Initial choice**: `Overall Rating` (numeric, 0–99) and `Archetype` (categorical). The remaining 67 columns are available in the processed Madden file and switched on by changing the training code's column-selection list — no rebuild needed.

Potential additions in iteration (named here so they're not forgotten): `Speed`, `Awareness`, `Throw Accuracy Short/Mid/Deep` (for QBs), `Man Coverage`/`Zone Coverage` (for DBs). Each addition is a feature-engineering experiment with a recorded before/after metric.

### 2B. How starter slots are presented to the model

Options:

- **B-flat (slot-indexed flat vector)** — concatenate all 44 starters' Madden columns into one long vector per game. Position-dependent; the model has to learn that `Off01` is the QB.
- **B-pos (position-indexed)** — group by position (`HomeQB_*`, `HomeRB1_*`, …). More semantic; needs a stable position taxonomy and handling for unusual formations.
- **B-set (permutation-invariant set)** — feed 22 home + 22 away starter vectors as a set, with a permutation-invariant aggregator (mean pool, attention). Most flexible; the strongest fit for the eventual "attention over slots" model.

These are model-shape decisions, not file-shape decisions — switching between them is a training-code change against the same processed files.

### 2C. Game-level features

Decide per column whether it's an input feature, a label, or excluded:

- **Labels**: `HomeScore`, `AwayScore`.
- **Excluded** (post-game / leakage): `Duration`, `Attendance` (somewhat — fills as the game progresses).
- **Pre-kickoff features**: week, day-of-week, start time, stadium, roof, surface, weather string (needs parsing — temperature, wind, humidity), home/away team codes, coach names, days of rest per team (derived from prior game date).
- **Officials**: known at kickoff. Whether they meaningfully affect score totals is an empirical question; could start excluded and add back as a feature-engineering experiment.

**Deliberately excluded by premise** — in-season form features (team W/L record, rolling point differentials, recent-game momentum). The project's premise is that *compositional and physical* state — team roster (Madden), venue, weather, officials, home/away — can produce a reasonable prediction on its own. W/L record is unmistakably correlated with outcomes, but it bundles strength-of-schedule with player-sentiment and momentum effects ("winning is contagious") that we don't want a v1 roster-strength model to absorb implicitly. Subjective and momentum-style factors are left to a follow-on model that can target them explicitly and be ablated against this one.

Open question: weather is free-text (`"67 degrees, relative humidity 53%, wind 8 mph"`). Decide whether to parse into structured columns (temp, humidity, wind), one-hot a few categories, or skip for v1.

### 2D. Categorical encoding

`Archetype`, team codes, coach names, position codes, stadium, surface, roof are categorical. Options applied at training time (the processed file stores raw strings):

- **One-hot** — simple, sparse, no parameters to learn. Reasonable for low-cardinality fields (roof, surface).
- **Learned embeddings** (`nn.Embedding`) — better for high-cardinality fields with semantic structure (Archetype, team codes). Requires deciding embedding dimensions.
- **Target encoding** — encode by mean of label conditional on category. Powerful but leakage-prone; only safe if computed from training data only and applied to validation/test as a frozen lookup.

## Phase 3: Splits

> **Status**: Implementation complete (2026-05-18). The split build is implemented at `src/nflpredictor/splits/` and produces two artifacts in `Data/processed/` via `python -m nflpredictor.splits` (gated on a Phase 2 source-hash check). Real-data run summary: S1 train=179 / val=45 / test=48 (272 total); S3 fold_count=9 with `k ∈ {6..14}` and S3.test ≡ S1.test; wall-clock ~0.4s. The sections below are retained for reference so the reasoning behind the chosen direction stays visible.

### Why time-aware

The evaluation contract is: train on the early part of 2024, validate on a middle slice, score the final test slice. All three live within the 2024 season — 2025 is not in scope for this iteration. Shuffled k-fold would answer a different question (within-season interpolation) and would overstate test performance by removing the temporal gap between training and held-out data rather than measuring across it. Validating temporally is the only honest way to know whether the model has learned the signal or memorized the games.

Four reasons time-aware is the right framing for *this* dataset, not just the generic "no leakage" answer:

1. **Games within a season aren't i.i.d.** Week 12 outcomes are conditioned on Weeks 1–11 (injury attrition, coordinator adjustments, midseason trades, coaching turnover). Shuffling lets the model see downstream state.
2. **Madden ratings are a frozen preseason snapshot.** Their predictive power decays as rosters diverge from it. Shuffling makes the decay symmetric across train and held-out slices, so the model never has to model it. Time splits put validation and test in the staler-Madden regime — exactly where roster robustness matters most.
3. **End-of-season is a structural regime change.** Weeks 16–18 over-index on rested starters, eliminated-team experimentation, worse weather, degraded fields. Shuffling smears that regime into training as if it were typical.
4. **272 games is small enough that shuffling hides per-team leakage.** Each team appears ~17 times; shuffled folds let the model learn each team's identity from the same distribution it's then asked to predict against. Time splits force train, val, and test to draw from different points in each team's season trajectory.

### Acknowledged bias

Time-aware splits are not free. Naming the biases so error analysis stays honest:

- **Validation and test are regimes, not samples.** Weeks 13–18 over-index on rested starters, bad weather, and northern outdoor stadiums. Tuning to validation MAE risks fitting late-season quirks; the test metric we ultimately quote is itself measured against a non-random slice of the season.
- **Training is also biased.** Early-season is where Madden is freshest, injuries fewest, weather mildest. The model learns from a "clean" slice and is asked to generalize to a harder one; training metrics will systematically beat validation, and that gap is partly distributional, not just generalization error.
- **The late-season regime appears zero times in training.** Whatever distinguishes Weeks 13–18 — weather-suppressed totals, blowout dynamics, tank jobs — has no training examples; the model can only extrapolate into it.
- **Small-data carve-out.** ~33% of games (val + test) are off-limits to training. With 272 games total, training has ~180 — a real cost, but the cost of inflated test metrics from a shuffled split would be larger.
- **Hyperparameter selection is noisy.** ~45 val games is small; close metric deltas between candidate models may be late-season noise. Expanding-window CV (S3) is the lever for this when it becomes the binding constraint.

### Why we stick with it anyway

1. **The evaluation we care about IS temporal generalization.** The question we're answering is "can the model predict games it didn't see, given a chronological gap from the games it did see." Shuffled validation removes the gap rather than measuring across it — that's a different question entirely.
2. **The biases are diagnosable, not hidden.** Slice analysis (Phase 6) separates "model is wrong" from "model is wrong in the late-season regime specifically." Reading *deltas* across the baseline ladder (Phase 4) is more robust to validation-slice quirks than any single absolute metric.

### The chosen partition

- **Train**: Weeks 1–12 (~180 games, ~66%).
- **Val**: Weeks 13–15 (~45 games, ~17%).
- **Test**: Weeks 16–18 (~44 games, ~17%).

Both strategies ship in v1:

- **3-S1 (single-fold)** — One fixed partition with the boundaries above.
- **3-S3 (expanding-window CV)** — Test slice is fixed (Weeks 16–18, identical to S1). Within Weeks 1–15, expanding-window folds rotate: train Weeks 1..k, val Week k+1, for k ∈ {6, …, 14}. Produces 9 folds for stable comparison of close models without disturbing the test slice.

See `Docs/Spec-Phase3-Splits.md` for the deterministic build that emits `Data/processed/splits_2024.json` from this configuration.

## Phase 4: Baseline & Model Ladder

> **Status**: Implementation complete (2026-05-18). The training build is implemented at `src/nflpredictor/train/` and produces 12 prediction parquets (one per `(rung, feature_shape, strategy)` combination) plus `training_manifest.json` in `Data/processed/` via `python -m nflpredictor.train` (gated on a Phase 2 + Phase 3 source-hash check). Fixture-data run summary (36 games, max_epochs=10): rung 0 mean S1.val_mae≈7.09; rung 1 team_mean S1.val_mae≈6.32; rung 2 linear S1.val_mae≈8.20 (flat) / 6.25 (pos); rung 3 MLP S1.val_mae≈7.37 (flat) / 7.68 (pos). The full real-data run is expected from the CUDA training machine; the dev CPU resolves and runs but takes ~30–60 minutes. **Choices**: rungs 0–3 (rung 4 attention deferred), both `flat` and `pos` per learned rung, single multi-output regressor with MAE loss, both S1 (headline) + S3 (tiebreaker), one-hot for low-card categoricals + learned `nn.Embedding` for high-card with the `vocab_size + 1` NULL slot at index 0, predictions-only outputs (no checkpoints in v1).

Each rung must be evaluated against the same splits and metrics. A rung is only worth keeping if it beats the one below it by a meaningful margin on validation.

- **Rung 0 — Mean predictor.** Predict `(mean home score in training, mean away score in training)` for every game. The "you can't be worse than this and still be a model" floor. ~12 points MAE per side, roughly.
- **Rung 1 — Team-mean predictor.** Predict each team's training-set mean score, contextualized by home/away. Still no Madden signal at all; isolates "knowing which team is playing" as a feature.
- **Rung 2 — Linear (`nn.Linear`).** Flat feature vector in, 2 outputs. The smallest possible PyTorch model. This is where Madden ratings first influence predictions.
- **Rung 3 — Small MLP.** 1–2 hidden layers (128–256 units), ReLU/GELU, dropout. The first place nonlinearity earns its keep.
- **Rung 4 — Attention over starter slots.** Treat 22 home + 22 away starters as two sets, apply a small transformer/attention encoder, aggregate per side, predict scores. The "interesting" model; only justified if rung 3's residuals suggest it.

The ladder is intentionally bounded — rung 4 is the upper end of complexity worth considering for 272 training games.

Open questions:

- Single multi-output regressor (one model, 2 outputs) vs. two independent regressors (one for home, one for away)? Single is more common and shares representation; independent is easier to debug.
- Loss: MSE (penalizes large misses, standard), MAE (more interpretable, more robust to outliers), or Huber (compromise)? Worth deliberate choice rather than defaulting.
- Optimizer / schedule: Adam with a small constant LR is a fine default for this scale; nothing fancy needed unless rung 3 stalls.

## Phase 5: Evaluation

> **Status**: Direction inherited from Overview; refinements are exploratory.

Primary regression metrics (computed on home and away separately, then averaged):

- **MAE** — most interpretable ("we miss by N points on average"). The headline metric.
- **RMSE** — penalizes large misses. Use to spot games where the model is confidently wrong.

Derived from score predictions:

- **W/L accuracy** — sign of `home_score − away_score`. The most intuitive sanity check.
- **Spread MAE** — `(home_score − away_score)` predicted vs. actual.
- **Total MAE** — `(home_score + away_score)` predicted vs. actual. Useful because a model can be good at one and bad at the other.

Calibration plots:

- Predicted score distribution vs. actual score distribution.
- Error by week, by team, by home/away, by surface/roof.

Open question: do we measure against Vegas closing lines (when available) as an "external benchmark"? Closing lines are a strong, hard-to-beat baseline; including them is honest but discouraging early on.

## Phase 6: Error Analysis & Iteration

> **Status**: Exploratory.

The error-analysis loop is what makes Phase 2's "add more Madden columns" decisions tractable. Two complementary lenses:

- **Worst-N analysis** — pull the 20 highest-error games. Tag each with a likely cause (injury, weather extreme, blowout dynamics, late-season tank job, key starter we couldn't match). Recurring tags become candidate feature additions or matching-quality investigations.
- **Slice analysis** — error by team, by week, by surface, by `matched=0` density. If error correlates with unmatched-player density, the null-fill choice is biting us and should be revisited.

Feature attribution (since we're PyTorch, not tree-based):

- Gradient-based attribution (saliency, Integrated Gradients via `captum`) is the natural PyTorch fit.
- Ablation studies (zero out a feature group, measure metric delta) are heavier but more interpretable for a learning project.

The iteration loop:

1. Train current model on current features → record metrics.
2. Worst-N + slice analysis on validation predictions.
3. Hypothesis: "this feature group might help" → add it, retrain, compare.
4. If no improvement after N tries, stop adding features and call the model frozen.

Open question: how many iteration cycles before declaring the model frozen for the test-slice evaluation? Setting a soft cap (e.g., "5 feature-engineering cycles") prevents endless tinkering.

## Phase 7: 2025 Test (deferred)

> **Status**: Out of scope for the current iteration. The evaluation universe is 2024 only — see Phase 3. A future iteration may apply the frozen Phase 4 model to 2025 data if and when that dataset becomes available; nothing below is committed to.

Sketch of what a future 2025 evaluation could look like, retained for traceability:

- Receive a raw 2025 box-scores file in the same schema as 2024.
- Run the build with 2025 raw inputs; emit a 2025 processed Madden file, `Data/processed/box_scores_2025.csv`, and a 2025 mapping file.
- Apply the frozen model. Compare predicted vs. actual scores game-by-game.

Future open questions (deferred):

- **Madden vintage**: keep Madden 24 (frozen-input simplicity, stale ratings) or ingest Madden 25 and rebuild player IDs (more realistic, but requires the ID-assignment step to be stable across rebuilds)?
- **Drift handling**: a 2025 rookie or trade not in Madden 24 lands in the `matched=0` bucket and sees league-average ratings. Worth measuring frequency and error correlation.
- **Reporting**: format and home for a "2025 results" writeup.

## Relevant Considerations

### Build Artifacts and Layout

Concretely, the build produces this layout:

```
Data/
  raw/
    box_scores_2024.csv               # untouched source (PFR-style _IDs preserved)
    maddennfl24fullplayerratings.csv  # untouched source
    player_overrides.csv              # hand-edited; input to the build for ambiguous matches
  processed/
    madden_2024.csv                   # full Madden columns + madden_id + matched; nulls filled with column averages
    box_scores_2024.csv               # raw shape; _ID columns now hold madden_id values; no blanks
    player_id_mapping.csv             # audit only: (box_score_id, madden_id, note) — not used by training
    build_manifest.json               # source SHAs, build timestamp, name-normalization version (regression-defense)
```

The build step is a single deterministic script that reads `raw/` and emits `processed/`. The training code reads only `processed/madden_2024.csv` and `processed/box_scores_2024.csv` and joins them on `madden_id`. The mapping file is for human troubleshooting only. The `build_manifest.json` exists so the model-fit step can verify it's joining against a known build.

### Data and State Considerations

- **Header hygiene**: Madden's ` Total Salary ` and ` Signing Bonus ` headers carry surrounding whitespace. Strip on read; document the strip.
- **Type coercion**: `Birthdate` appears to be a serialized number (e.g. `33857`) — likely Excel epoch days. Decide a target type before downstream code treats it as ordinal.
- **Currency columns**: Salary fields are strings with commas and possibly currency markers. Decide whether to parse to numeric or drop.
- **Missingness map**: Build it once on the joined table; commit it as a `Docs/DataDictionary.md` companion artifact alongside the training file.
- **Madden snapshot date**: Madden 24 ratings are a single point-in-time. They can't capture mid-season injuries, trades, or rookie progression. Decide whether the training table notes "Madden snapshot vintage" as a column so a future Madden 25 join doesn't silently break temporal alignment.
- **Bench depth**: Box scores only enumerate starters. Madden lists rosters. Decide whether the training table needs *any* bench-strength signal (e.g. team-mean Madden rating across the full roster) on top of the per-starter columns.
- **Future-data portability**: 2025 evaluation is not in scope, but the build contract should remain general enough that future seasons could be ingested without redesigning the schema. The contract — column names, types, allowed values — is a deliverable in itself.

### Operational Considerations

- **File format**: At ~3,000 columns × 272 rows, Parquet is much friendlier than CSV (typed, compressed, faster to load with `pandas` or `polars`, lossless for floats). CSV remains an easy human-readable export.
- **Reproducibility**: The Overview already calls for fixed seeds and versioned splits. Versioning the *join recipe* (mapping tables + manual overrides) belongs in the same envelope.
- **Manual overrides**: A human-edited `Data/player_overrides.csv` (or equivalent) is itself a versioned artifact and an audit log of every judgment call.

### Learning-Value Considerations

Because this is a canonical-example project, the two-file relational direction (B3) carries a few teaching benefits worth naming explicitly:

- **Separation of storage from model input.** Real ML pipelines almost always look like "entities + events" rather than "one giant denormalized table." Doing it that way here builds the right mental model.
- **The mapping problem is solved exactly once.** Every name-normalization decision, every team-code translation, every override lives in the build step. The model code never re-litigates them.
- **Provider swap becomes a real exercise.** When (or if) the Madden source changes, the test is whether a new file can drop into the same shape and IDs — which is exactly the contract this project says it wants to teach.
- **The build step itself is a documentable artifact.** A small, deterministic script that turns two raw files into two processed files is one of the most generally useful patterns in applied ML, and it'll read cleanly in this repo.

## Tradeoffs or Risks

- **Two artifacts to keep in sync**: With B3 leading, the Madden file and box-scores file must be versioned together. A model-fit-time join that silently uses a stale Madden file is a real failure mode. Defense: the build step emits a `build_manifest.json` or similar that records the source SHAs, build timestamp, and chosen slice; the model load step verifies it.
- **Build-step regressions**: The build now carries all the name/team-mapping logic. A subtle change to normalization (different Unicode policy, different suffix handling) can silently relabel players. A regression test fixture — a small set of "this name should map to this ID" assertions — keeps that under control.
- **Madden as a frozen snapshot**: Treating ratings as static for a full season is a known approximation. Worth being explicit about it now rather than discovering it during error analysis.
- **Name-match silent failures**: A join that returns "no match" is easy to handle; a join that returns the *wrong* player (two players sharing a name) is the dangerous failure mode. The manual-overrides file is the main defense; tier-of-match logging is the diagnostic.
- **PyTorch-native baselines** (C2): the project loses a cross-paradigm sanity check (linear regression / GBT). The PyTorch ladder must include genuinely-dumb baselines (mean predictor, single `nn.Linear`) and hold itself to beating them, or the value of "always run a simple baseline" is lost. Worth being deliberate about that in the spec.
- **Average-fill on unmatched players**: filling nulls with column averages over matched rows is a reasonable default, but it does mean the model will see unmatched players as "league-average everywhere." If many unmatched players are deep-bench / practice-squad replacements, average-fill *overstates* their ratings. The `matched` flag is the lever that lets us see the bias and try alternatives (e.g., position-mean fill, or fill with a "below-average" sentinel).
- **Scope creep into modeling**: It will be tempting to start designing the PyTorch model while the build is being written. The stated direction is to finalize data first; the document should keep reasserting that boundary.

## Resolved Decisions (Phase 1)

All twelve Phase-1 open questions have been answered. Recorded here for traceability; the consequences are folded into **Human-Provided Direction** and **Goals** above.

1. **Matching strategy** — A3 (external crosswalk) bootstrapped by A2 (tiered with fuzzy fallback).
2. **PFR ID retention** — Raw box-scores file untouched (keeps PFR IDs). Processed box-scores file has `_ID` columns rewritten to `madden_id` values, with no blanks. A `player_id_mapping.csv` audit file records the `(box_score_id, madden_id, note)` trail for troubleshooting but is not consumed by training code.
3. **Unmatched players** — Added to processed Madden with null ratings + `matched = 0`; nulls filled with column averages computed over matched rows.
4. **Source `_ID` resolution** — Name lookup is the resolution mechanism throughout the build.
5. **ID format** — `2024-00001`-style strings (see #9).
6. **Overview ladder** — Replace with PyTorch-native progression (C2).
7. **Reserve-column workflow** — Moot. Processed Madden carries all original columns; training step selects which to use.
8. **Bench-depth aggregates** — Declined. No precomputed roster averages in the build; would introduce uncontrolled parameters.
9. **ID/vintage encoding** — Year-prefixed (`2024-`) IDs encode the Madden vintage in the ID itself; no separate vintage column needed.
10. **Manual-overrides** — Versioned `player_overrides.csv` (in `Data/raw/`); hand-edited; read by the build.
11. **File layout** — `Data/raw/` for sources, `Data/processed/` for build outputs.
12. **2024-only first pass** — Confirmed. The user has a separate application that may produce 2025 data later; that's a future iteration, out of scope here.

## Remaining Open Questions

Phase 1 is concrete enough to move to a specification. Later phases have intentional looseness — they're exploratory and depend on what Phase 1's data ends up looking like in practice.

**Phase 1 (small, non-blocking)**:

- **Mapping-file scope** — should `player_id_mapping.csv` include one row per (game, slot) starter occurrence, or be deduplicated to one row per unique `(box_score_id, madden_id)` pair?
- **Position consistency check** — should the build verify that a starter's box-score `Position` and Madden `Position` agree, and emit a warning (or `note` entry) when they don't?

**Phase 2 (Feature Engineering)**:

- Slot presentation: B-flat, B-pos, or B-set? This couples tightly to Phase 4 — B-set only makes sense if rung 4 (attention) is on the table.
- Weather string: parse to structured (temp/wind/humidity), bucket into categories, or skip for v1?
- Categorical encoding strategy per field (one-hot vs. embedding vs. target encoding). Reasonable to defer until rung 2 actually runs.
- Officials: include as features, or excluded?

**Phase 3 (Splits)**: Resolved by `Docs/Spec-Phase3-Splits.md` — both S1 and S3 ship in v1; the test slice (Weeks 16–18) is the held-out partition and no separate final-final fallback is reserved.

**Phase 4 (Model Ladder)**: Resolved by `Docs/Spec-Phase4-BaselineLadder.md` — single multi-output regressor with `nn.L1Loss` (MAE); rungs 0–3 (mean → team_mean → `nn.Linear` → small MLP) ship in v1 with rung 4 (attention) deferred; both `flat` and `pos` shapes trained per learned rung; both `S1` and `S3` strategies consumed; one-hot for low-card categoricals + learned `nn.Embedding` for high-card with a NULL slot at index 0; predictions-only outputs.

**Phase 5 (Evaluation)**:

- Include Vegas closing-line comparison as an external benchmark?

**Phase 6 (Error Analysis)**:

- Soft cap on iteration cycles before freezing the model?

**Phase 7 (2025 Test, deferred)**:

These items become live only if a future iteration brings 2025 data into scope. Recorded for traceability:

- Stay on Madden 24 ratings, or ingest Madden 25 when available?
- Format and home for a "2025 results" writeup.

## References or Related Artifacts

- `Docs/Overview.md` — parent project document; defines the full methodology, target metrics, and the existing (non-PyTorch) model ladder this idea must reconcile with.
- `Data/box_scores_2024.csv` — labels source; lineup truth for the join.
- `Data/maddennfl24fullplayerratings.csv` — feature source; canonical 69-column shape.
- `CLAUDE.md` — repository-level notes on dataset shape and join gotchas (team-code vs. nickname, name normalization, header whitespace).
