# NFLPredictor

NFL game outcome predictor — a learning project that builds a PyTorch model end-to-end on 2024 NFL data and applies it to 2025 once available. Phase 1 produces a normalized dataset joining 2024 box-score lineups to Madden NFL 24 player ratings; Phase 2 turns that into deterministic feature matrices ready for modeling; Phase 3 partitions the 2024 game universe into the train / val / test slices every downstream modeling phase binds to.

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

## Tests

```bash
source .venv/bin/activate
pytest -q
```

## Layout

- `src/nflpredictor/databuild/` — Phase 1 build pipeline
- `src/nflpredictor/features/` — Phase 2 feature engineering pipeline
- `src/nflpredictor/splits/` — Phase 3 split-assignment pipeline
- `Data/raw/` — checked-in source CSVs, `feature_config.yaml`, and `splits_config.yaml`
- `Data/processed/` — build outputs (regenerated, not tracked)
- `Docs/` — project overview, phase plans, specs (Phases 1–3 implementation-complete; Phases 4–7 still exploratory in `Idea.md`)
