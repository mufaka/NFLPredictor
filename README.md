# NFLPredictor

NFL game outcome predictor — a learning project that builds a PyTorch model end-to-end on 2024 NFL data and applies it to 2025 once available. Phase 1 produces a normalized dataset joining 2024 box-score lineups to Madden NFL 24 player ratings; Phase 2 turns that into deterministic feature matrices ready for modeling; Phase 3 partitions the 2024 game universe into the train / val / test slices every downstream modeling phase binds to; Phase 4 trains the v1 PyTorch baseline ladder (mean → team-mean → `nn.Linear` → small MLP) against those splits and emits per-combination prediction parquets.

## Setup (first checkout)

Requires Python 3.11+.

```bash
git clone <repo-url>
cd NFLPredictor

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"
```

## Running the builds

All three builds are deterministic — identical inputs produce byte-identical outputs across reruns.

### Phase 1 — data build

```bash
source .venv/bin/activate
python -m nflpredictor.databuild
```

Reads `Data/raw/{box_scores_2024.csv, maddennfl24fullplayerratings.csv, player_overrides.csv}` and emits four artifacts into `Data/processed/`:

- `madden_2024.csv` — Madden roster with assigned `madden_id` and a `matched` flag (`0` for box-score players Madden didn't have; their numeric columns are null-filled with column means).
- `box_scores_2024.csv` — same shape as raw but every per-slot `_ID` now carries a `madden_id`.
- `player_id_mapping.csv` — audit trail: `(box_score_id, madden_id, note)` per resolved starter.
- `build_manifest.json` — SHA-256 hashes, normalization version, git commit, per-tier match counts.

### Phase 2 — feature engineering

```bash
source .venv/bin/activate
python -m nflpredictor.features
```

Refuses to run unless the Phase 1 outputs on disk match the SHAs recorded in `build_manifest.json`. Reads Phase 1's processed outputs plus `Data/raw/feature_config.yaml` and emits four artifacts into `Data/processed/`:

- `features_flat_2024.parquet` — slot-indexed feature matrix (272 rows × 202 columns for the v1 default config).
- `features_pos_2024.parquet` — position-indexed feature matrix (272 rows × 258 columns) grouped by canonical position taxonomy.
- `feature_vocab.json` — integer-code domain for every categorical column.
- `feature_manifest.json` — provenance: input/output SHAs, normalization version, per-shape column counts, vocab sizes.

`Data/raw/feature_config.yaml` is the knob for expanding the feature inventory — adding Madden columns or toggling weather/officials is a YAML edit, not a code change.

### Phase 3 — splits

```bash
source .venv/bin/activate
python -m nflpredictor.splits
```

Refuses to run unless `features_flat_2024.parquet` on disk matches the SHA recorded in `feature_manifest.json`. Reads that parquet's `(GameId, week)` columns plus `Data/raw/splits_config.yaml` and emits two artifacts into `Data/processed/`:

- `splits_2024.json` — split-assignment artifact. For the v1 boundaries (train Weeks 1–12 / val 13–15 / test 16–18), S1 partitions the 272 games into train=179 / val=45 / test=48. S3 emits 9 expanding-window `(train, val)` folds (`k ∈ {6..14}`) plus a `test` slice identical to S1's.
- `splits_manifest.json` — provenance: SHA-256 of the source parquet, the splits config, and the artifact; `splits_version`; per-strategy summaries; git commits.

`Data/raw/splits_config.yaml` is the knob for the split contract — moving week boundaries or toggling between S1/S3 is a YAML edit; a new strategy or change to artifact layout requires a `splits_version` bump.

### Phase 4 — baseline & model ladder

```bash
source .venv/bin/activate
python -m nflpredictor.train
```

Refuses to run unless every Phase 2 tracked output and `splits_2024.json` on disk match the SHAs recorded in their upstream manifests. Reads those outputs plus `Data/raw/training_config.yaml` and emits **12 prediction parquets + a training manifest** into `Data/processed/`:

- `predictions/<rung_id>__<shape>__<strategy>.parquet` × 12 — long-format predictions per `(rung, feature_shape, strategy)` combination. The v1 ladder is rungs 0–3 (mean → team_mean → `nn.Linear` → small MLP); each learned rung trains once per feature shape (`flat`, `pos`); each combination is run for both `S1` (single fold, val + test predictions) and `S3` (9 expanding-window folds, per-fold val predictions).
- `training_manifest.json` — provenance: SHA-256 of every Phase 2 / Phase 3 input, the training config, and each output parquet; per-combination val MAE summary; resolved `device`, `torch_version`, and (when CUDA) `cuda_device_name` + `cuda_version`. The manifest carries val MAE only — test MAE is never computed by Phase 4.

The training build is **device-aware, single-device**: `device: "auto"` (the v1 default) resolves to CUDA when available, else CPU. Determinism is per-device — byte-identical re-runs are guaranteed within the same resolved device + pinned PyTorch wheel; CPU↔CUDA byte equality is not asserted. The encoder bumps every categorical index by +1 so Phase 2's `NULL_SENTINEL = -1` lands in a reserved null slot at index 0; low-cardinality categoricals (≤ 8 entries) go through one-hot, high-cardinality through shared `nn.Embedding` lookups.

`Data/raw/training_config.yaml` is the knob — adjusting rungs, shapes, strategies, hyperparameters, embedding dims, or the device is a YAML edit; a new rung or output layout change requires a `training_version` bump.

## Tests

```bash
source .venv/bin/activate
pytest -q
```

## Layout

- `src/nflpredictor/databuild/` — Phase 1 build pipeline
- `src/nflpredictor/features/` — Phase 2 feature engineering pipeline
- `src/nflpredictor/splits/` — Phase 3 split-assignment pipeline
- `src/nflpredictor/train/` — Phase 4 baseline & model ladder
- `Data/raw/` — checked-in source CSVs and config YAML (one per phase)
- `Data/processed/` — build outputs (regenerated, not tracked)
- `Docs/` — project overview, phase plans, specs (Phases 1–4 implementation-complete; Phases 5–7 still exploratory in `Idea.md`)
