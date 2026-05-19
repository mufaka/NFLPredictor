# Config Reference

This is the single source of truth for the four YAML files under `Data/raw/` that
drive the pipeline. Each file is parsed and validated by a dedicated loader; an
unknown key, missing key, wrong type, or out-of-range value is a hard error.

| File | Loader | Used by phase |
| --- | --- | --- |
| `feature_config.yaml` | `src/nflpredictor/features/config.py` | Phase 2 — feature engineering |
| `splits_config.yaml` | `src/nflpredictor/splits/config.py` | Phase 3 — train/val/test splits |
| `training_config.yaml` | `src/nflpredictor/train/config.py` | Phase 4 — baseline & model ladder |
| `evaluation_config.yaml` | `src/nflpredictor/evaluate/config.py` | Phase 5 — evaluation |

Every loader rejects unknown top-level keys and unknown keys inside any nested
mapping; mistyped names won't silently no-op.

## After editing any config

Each phase pins SHA-256 hashes of every upstream output in its manifest and
refuses to run when those hashes don't match what's on disk. So a feature-config
edit invalidates Phase 2's outputs, which invalidates Phase 3, Phase 4, and
Phase 5. The fix is to re-run every downstream phase in order:

```bash
source .venv/bin/activate
python -m nflpredictor.features
python -m nflpredictor.splits
python -m nflpredictor.train      # CUDA box for the real run
python -m nflpredictor.evaluate
```

The `*_version` string at the top of each YAML is a manual bump knob — change it
when the output layout (not the values) changes. The loaders don't interpret it;
it's surfaced verbatim in the corresponding manifest.

---

## `feature_config.yaml`

Top-level keys (all required):

| Key | Type | Notes |
| --- | --- | --- |
| `normalization_version` | non-empty string | Tag stamped into `feature_manifest.json`. |
| `madden_columns` | non-empty list of strings | Columns from `Data/processed/madden_2024.csv` to expand per-slot. |
| `madden_categorical_columns` | list of strings (may be empty) | Must be a subset of `madden_columns`. Any column whose Madden values are non-numeric **must** appear here, or the Phase 2 build raises `ValueError` from `slots.py`. |
| `game_features` | mapping | See below. |
| `slot_shapes` | non-empty list | Subset of `[flat, pos]`. |

### `madden_columns` — enumerable list

The validator accepts any column header from `Data/processed/madden_2024.csv`
(which is `maddennfl24fullplayerratings.csv` after Phase 1's whitespace strip on
` Total Salary ` and ` Signing Bonus `). The full set:

```
Team, Position, Full Name, Overall Rating, Jersey Number, Speed, Acceleration,
Strength, Agility, Awareness, Catching, Carrying, Throw Power, Kick Power,
Kick Accuracy, Run Block, Pass Block, Tackle, Break Tackle, Jumping,
Kick Return, Injury, Stamina, Toughness, Trucking, Change Of Direction,
Ball Carrier Vision, Stiff Arm, Spin Move, Juke Move, Impact Blocking,
Run Block Power, Run Block Finesse, Pass Block Power, Pass Block Finesse,
Lead Block, Break Sack, Throw Under Pressure, Power Moves, Finesse Moves,
Block Shedding, Pursuit, Play Recognition, Man Coverage, Zone Coverage,
Spectacular Catch, Catch In Traffic, Short Route Running, Medium Route Running,
Deep Route Running, Hit Power, Press, Release, Throw Accuracy Short,
Throw Accuracy Mid, Throw Accuracy Deep, Play Action, Throw On The Run,
Height, Weight, Age, Birthdate, Years Pro, Running Style, Archetype, College,
Total Salary, Signing Bonus, Player Handness
```

Non-numeric (must also appear in `madden_categorical_columns` if used): `Team`,
`Position`, `Full Name`, `Running Style`, `Archetype`, `College`,
`Player Handness`. `Birthdate` is parsed as a date string and is also
non-numeric.

Each entry in `madden_columns` becomes 22 columns in the `flat` shape
(`HomeOff01_<col>` … `AwayDef11_<col>`) and 29 home + 29 away columns in the
`pos` shape (one per canonical position slot).

### `game_features` block

| Key | Type | Allowed values |
| --- | --- | --- |
| `weather` | string | `parsed`, `skip` |
| `officials` | string | `included`, `skip` |
| `include` | list of strings | Subset of the closed set below; order is preserved. |

`game_features.include` accepts only these 12 identifiers (from
`KNOWN_GAME_LEVEL_FIELDS` in `features/config.py`):

```
week, day_of_week, start_hour, stadium, roof, surface,
home_team_code, away_team_code, home_coach, away_coach,
days_rest_home, days_rest_away
```

### Worked example: adding `Speed`, `Awareness`, and `Position`

```yaml
madden_columns:
  - "Overall Rating"
  - "Archetype"
  - "Speed"          # numeric — no entry in madden_categorical_columns
  - "Awareness"      # numeric
  - "Position"       # non-numeric — also goes below

madden_categorical_columns:
  - "Archetype"
  - "Position"
```

After saving, re-run `python -m nflpredictor.features` and the three
downstream phases. `Position` becomes a new vocab key in `feature_vocab.json`
(distinct from the existing `positions` key, which is the box-score canonical
taxonomy used by the `pos` shape). If its cardinality exceeds
`one_hot_threshold`, add either `Position: <dim>` or rely on `_default` in
`embedding_dims` — see the [`embedding_dims`](#embedding_dims-block) section.

---

## `splits_config.yaml`

Top-level keys (all required except `s3`, which becomes required when `S3` is in
`strategies`):

| Key | Type | Notes |
| --- | --- | --- |
| `splits_version` | non-empty string | Tag stamped into `splits_manifest.json`. |
| `strategies` | non-empty list | Subset of `[S1, S3]`; duplicates rejected. |
| `train_weeks` | `[start, end]` ints | `1 ≤ start ≤ end ≤ 18`. |
| `val_weeks` | `[start, end]` ints | Must satisfy `val_weeks[0] == train_weeks[1] + 1`. |
| `test_weeks` | `[start, end]` ints | Must satisfy `test_weeks[0] == val_weeks[1] + 1`. |
| `s3.k_start` | int | Required iff `S3 ∈ strategies`. `1 ≤ k_start < val_weeks[1]`. |

The three week ranges must be strictly increasing and **contiguous**: the
validator rejects any gap or overlap. `s3.k_start` is the number of training
weeks at the smallest expanding-window fold; the number of folds is
`val_weeks[1] - k_start` for the v1 layout (k_start=6 → folds for k ∈ {6..14}).

If `s3` is present but `S3` is not in `strategies`, the loader raises — the two
must be kept in lockstep.

---

## `training_config.yaml`

Top-level keys:

| Key | Required | Type | Notes |
| --- | --- | --- | --- |
| `training_version` | yes | non-empty string | Tag stamped into `training_manifest.json`. |
| `seed` | yes | non-negative int | Drives PyTorch / numpy / Python RNGs. |
| `device` | yes | string | `auto`, `cpu`, or `cuda`. `auto` → CUDA if available, else CPU. |
| `one_hot_threshold` | no (default 8) | positive int | Cardinality boundary for categorical encoding: vocab size ≤ this → one-hot; > this → learned embedding. |
| `rungs` | yes | non-empty list | Subset of `[mean, team_mean, linear, mlp]`; duplicates rejected. |
| `shapes` | yes | non-empty list | Subset of `[flat, pos]`. |
| `strategies` | yes | non-empty list | Subset of `[S1, S3]`. |
| `linear` | yes | mapping | See below. |
| `mlp` | yes | mapping | See below. |
| `embedding_dims` | yes | mapping | See below. |

### `linear` block (all keys required)

| Key | Type | Constraint |
| --- | --- | --- |
| `lr` | float | `> 0` |
| `batch_size` | int | `> 0` |
| `max_epochs` | int | `> 0` |
| `early_stop_patience` | int | `> 0` |

### `mlp` block (all keys required)

| Key | Type | Constraint |
| --- | --- | --- |
| `lr` | float | `> 0` |
| `batch_size` | int | `> 0` |
| `max_epochs` | int | `> 0` |
| `early_stop_patience` | int | `> 0` |
| `hidden_dim` | int | `> 0` |
| `activation` | string | `gelu` or `relu` |
| `dropout` | float | `0.0 ≤ dropout < 1.0` |

### `embedding_dims` block

Mapping of `vocab_key → embedding_dim` (positive int) plus an optional
`_default` fallback. The loader requires an entry for **every
high-cardinality vocab key** — either an explicit `<key>: <dim>` entry or a
`_default: <dim>` that covers any unspecified key. "High-cardinality" means
`vocab.size > one_hot_threshold` (default 8); the threshold is configurable via
the top-level `one_hot_threshold` knob. Vocab sizes are pulled live from
`feature_vocab.json` at train time, so the required key set moves as the
feature config changes.

For the shipped feature config (Archetype + game-level defaults), the
high-cardinality vocab keys are:

| Vocab key | Why it's high-card |
| --- | --- |
| `Archetype` | Madden archetype labels (Pro QB, Field General, …). |
| `team_codes` | 32 NFL team codes; backs `home_team_code` and `away_team_code`. |
| `coaches` | Head coaches across the 2024 season. |
| `officials` | Officiating crew members. |
| `positions` | Canonical position taxonomy used by the `pos` shape. |
| `stadium` | NFL stadiums (~32). |

Low-card keys (one-hot encoded, no embedding row needed under the default
threshold): `roof`, `surface`, `day_of_week`. Any new Madden categorical
column added via `feature_config.yaml` automatically becomes a new vocab key
(named after the Madden column, e.g. `Position`, `College`). If its vocab
grows past `one_hot_threshold`, the loader requires an embedding dim — pick
between adding an explicit entry or relying on `_default`.

Example with the fallback:

```yaml
embedding_dims:
  _default:    8     # covers any new high-card vocab key automatically
  Archetype:   16    # explicit overrides for keys that warrant more capacity
  team_codes:   8
```

Without `_default`, the missing-key error names the offending vocab key:

```
TrainingConfigError: embedding_dims is missing required high-cardinality
vocab keys: ['Position']; either add explicit entries or add a '_default'
fallback.
```

**Why per vocab key, not per column?** A single `Archetype: 16` entry backs
all 44 per-slot `_madden_archetype` columns — they share one `nn.Embedding`.
The "Field General" archetype gets the same vector whether the column is
`HomeOff01_madden_archetype` or `AwayDef07_madden_archetype`. Per-column
embeddings would multiply parameter count without adding signal.

**How does the encoder know which column maps to which vocab key?** Phase 2
emits the `column_vocab_keys` map in `feature_vocab.json` (schema
`vocab_version: "v2"`). Phase 4's encoder reads it directly — there are no
hard-coded suffix rules on column names. Any new categorical column added via
`feature_config.yaml` is routed automatically.

---

## `evaluation_config.yaml`

Top-level keys (all required):

| Key | Type | Notes |
| --- | --- | --- |
| `evaluation_version` | non-empty string | Tag stamped into `evaluation_manifest.json`. |
| `headline_metric` | string | `mae`, `rmse`, `wl_accuracy`, `spread_mae`, or `total_mae`. All five metrics are always computed; this just picks which one the headline JSON foregrounds. |
| `breakdowns` | mapping | See below. |
| `plots` | mapping | See below. |

### `breakdowns` block

All five keys are optional (omitted → `false`). No unknown keys allowed.

| Key | Type | Notes |
| --- | --- | --- |
| `by_team` | bool | One row per team × combination × slice. |
| `by_week` | bool | One row per week × combination × slice. |
| `by_home_away` | bool | One row per side × combination × slice. |
| `by_surface` | bool | One row per surface × combination × slice. |
| `by_roof` | bool | One row per roof × combination × slice. |

### `plots` block (all keys required)

| Key | Type | Constraint |
| --- | --- | --- |
| `scatter` | bool | Per-combination scatter plots. |
| `residual_distribution` | bool | Per-combination residual histograms. |
| `ladder_summary` | bool | One PNG per slice (val/test/pooled). |
| `breakdown_plots` | list of strings | Subset of `[by_week, by_team, by_home_away]`; each entry **must** have the matching `breakdowns.<key>` set to `true`. Duplicates rejected. |
| `dpi` | int | `> 0` |
| `figure_width_inches` | float | `> 0` |
| `figure_height_inches` | float | `> 0` |

Note: `by_surface` and `by_roof` can be toggled in `breakdowns` but currently
have no per-breakdown plot variant.
