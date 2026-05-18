# NFLPredictor

NFL game outcome predictor. Phase 1 builds a normalized dataset
that joins 2024 box-score lineups to Madden NFL 24 player ratings.

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

## Running the data build

```bash
source .venv/bin/activate
python -m nflpredictor.databuild
```

Outputs land in `Data/processed/` (gitignored — reproducible from raw inputs).

## Tests

```bash
source .venv/bin/activate
pytest -q
```

## Layout

- `src/nflpredictor/` — package source
- `Data/raw/` — checked-in source CSVs
- `Data/processed/` — build outputs (regenerated, not tracked)
- `Docs/` — project overview, phase plans, specs
