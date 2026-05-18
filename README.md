# NFLPredictor

NFL game outcome predictor — a learning project that builds a PyTorch model end-to-end on 2024 NFL data and applies it to 2025 once available. Phase 1 produces a normalized dataset joining 2024 box-score lineups to Madden NFL 24 player ratings; Phase 2 turns that into deterministic feature matrices ready for modeling.

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

Both builds are deterministic — identical inputs produce byte-identical outputs across reruns.

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

## Tests

```bash
source .venv/bin/activate
pytest -q
```

## Layout

- `src/nflpredictor/databuild/` — Phase 1 build pipeline
- `src/nflpredictor/features/` — Phase 2 feature engineering pipeline
- `Data/raw/` — checked-in source CSVs and `feature_config.yaml`
- `Data/processed/` — build outputs (regenerated, not tracked)
- `Docs/` — project overview, phase plans, specs (Phases 1 + 2 implementation-complete; Phases 3–7 still exploratory in `Idea.md`)
