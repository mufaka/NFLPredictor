# Phase 3: Splits Specification

> **Revision note (multi-year).** This specification was originally written to split a single season (2024) by week boundary, with an `S1` single-fold strategy and an `S3` expanding-window CV strategy. It has been **redesigned** for the six-season (2020–2025) dataset, per `Docs/Plan-MultiYear-Migration.md`. The week-based strategies are dropped and replaced by **season-holdout** splitting. Because the strategies are replaced wholesale, the `SP-S1-*` and `SP-S3-*` requirement IDs are retired; their replacements are `SP-SH-*` and `SP-LOSO-*`. `splits_version` is bumped to `v2`. The `SP-IN-*`, `SP-CFG-*`, `SP-OUT-*`, `SP-MAN-*`, `SP-NF-*`, and `SP-TEST-*` ID families are preserved.

## 1. Introduction

### 1.1 Purpose

This specification defines the **Splits** phase of the NFL Predictor project. Phase 3 partitions the six-season game universe into deterministic train, validation, and test slices and emits a small split-assignment artifact that every downstream phase (baselines, model ladder, evaluation, diagnostics) binds to.

Phases 1 and 2 emit six seasons of joined and feature-engineered data; this spec assumes their outputs as given.

### 1.2 Why Season-Holdout

Phase 3 splits by **whole season**, not by week. The reasoning is recorded here because it drove the redesign:

1. **No temporal structure to defend.** A Madden rating is a static preseason snapshot — a Week 1 game and a Week 18 game of the same season use the identical roster ratings. A game's week therefore carries no information the model can leak across, so the original week-boundary partition and expanding-window CV solved a problem this feature design does not have.
2. **Roster-memorization leakage.** A pooled or random split places the *same season's games* in both train and test. Because each team plays ~17 games per season, the model would see most of a team's outcomes paired with that team's exact roster ratings, and could memorize "this roster → this many points" instead of learning the general ratings→points relationship.
3. **Deployment realism.** The project's purpose is predicting an unseen future season from a fresh Madden snapshot. That is, by definition, a season-holdout evaluation. A test metric computed on a pooled split is optimistic relative to how the model is actually used.

Season-awareness is confined to the split *boundaries*. It is not a model feature; a feature-matrix row remains "this box score had these players."

### 1.3 Scope

In scope:

- A third deterministic build step (separate from Phases 1 and 2) that reads Phase 2's feature matrix and a hand-edited `Data/raw/splits_config.yaml` and writes a split-assignment artifact + manifest into `Data/processed/`.
- Two split strategies: **`season_holdout`** (a single fixed train/val/test partition by season; the default) and **`loso_cv`** (leave-one-season-out cross-validation; optional).
- A configurable assignment of seasons to roles; the v2 default is train 2020–2023 / val 2024 / test 2025.
- A splits manifest recording source SHAs, config SHA, output SHAs, and per-strategy game counts.
- Testing requirements that protect determinism and split disjointness.

Out of scope:

- Any model code, training loop, or metric computation. (Phase 4 and beyond.)
- Week-based, sliding-window, or any within-season split strategy. These were intentionally removed (see §1.2).
- Random or pooled train/val splits. The default holds out a whole season for validation; a pooled train/val split is noted as a future option (§12) but is not specified here.
- Re-running Phase 1 or Phase 2. The split build assumes Phase 2's outputs are present and trusts their `feature_manifest.json`.

### 1.4 Definitions and Acronyms

| Term | Definition |
|------|------------|
| Game universe | The distinct `GameId` values present in `features_flat_all.parquet` (≈1,626 games across six seasons). Phase 3 partitions this universe; it does not invent or filter games. |
| Season | An NFL season year, one of `2020`–`2025`, read from the `season` column of the Phase 2 feature matrix. |
| Split role | One of `train`, `val`, `test`. Under `season_holdout`, every game has exactly one role. |
| **`season_holdout`** | Single-fold strategy. One fixed partition: a set of train seasons, one val season, one test season. The default strategy. |
| **`loso_cv`** | Leave-one-season-out cross-validation. The test season is fixed; each fold holds out one of the remaining seasons as val and trains on the rest. |
| Fold | A single `(train, val)` pair within `loso_cv`. The number of folds equals the number of non-test seasons. |
| Splits config | The hand-edited `Data/raw/splits_config.yaml` declaring the season-to-role assignment and which strategies to emit. |
| Splits artifact | `Data/processed/splits_all.json` — the file Phase 4 reads to learn which games belong to which slice. |
| Splits manifest | `Data/processed/splits_manifest.json` — provenance for a split build. |

### 1.5 Design Principles

- **Contract first.** The splits artifact is a stable, documented contract. Phase 4 binds to that contract, not to the build's internals.
- **Determinism above all.** Identical inputs (Phase 2 outputs + `splits_config.yaml`) must produce byte-identical `splits_all.json` and manifest across runs. The only permitted source of drift is the manifest's build timestamp.
- **Auditable partition.** Every `GameId` in the universe appears in exactly one role under `season_holdout`. No game is silently dropped; the manifest's per-strategy summary states the counts.
- **The test season is never pooled.** Whatever season is assigned to `test` is excluded from training under every strategy, including each `loso_cv` fold. This is the load-bearing invariant of the redesign.
- **Loss-less preservation of upstream layers.** Phase 1 and Phase 2 outputs are read-only.
- **Configuration over code edits.** Re-assigning seasons to roles or toggling a strategy is a YAML edit. Adding a *new strategy* is a code change accompanied by a config-schema change and a `splits_version` bump.

### 1.6 Relationship to Phases 1 and 2

```
Data/raw/{box_scores_<YYYY>, madden_<YYYY>, ..., feature_config}
    │  python -m nflpredictor.databuild
    ▼
Data/processed/{madden_all.csv, box_scores_all.csv,
                player_id_mapping.csv, build_manifest.json}
    │  python -m nflpredictor.features
    ▼
Data/processed/{features_flat_all.parquet, features_pos_all.parquet,
                feature_vocab.json, feature_manifest.json}
    │  python -m nflpredictor.splits
    ▼
Data/processed/{splits_all.json, splits_manifest.json}
```

The Phase 3 build refuses to run if Phase 2's output is missing or if its on-disk SHA-256 hash disagrees with `feature_manifest.json`'s recorded output hash. This guarantees that a split file is always pinned to a specific Phase 2 build.

---

## 2. Technology Additions

| Layer | Technology | Purpose |
|-------|------------|---------|
| Build step | Python (same venv as Phases 1 and 2) | Reads Phase 2 outputs + `splits_config.yaml`; emits JSON. |
| Parquet read | `pyarrow` | Reads `features_flat_all.parquet` to obtain GameIds and seasons. |
| YAML I/O | `pyyaml` (`safe_load`) | Reads `Data/raw/splits_config.yaml`. |
| Storage format | JSON (two-space indent, trailing newline) | Both artifact and manifest. |

The build is executable from the repository root via `python -m nflpredictor.splits` and requires no interactive input.

---

## 3. Functional Requirements

### 3.1 Input Contract

| ID | Requirement |
|----|-------------|
| SP-IN-01 | The build shall read `Data/processed/features_flat_all.parquet` and `Data/processed/feature_manifest.json` as primary inputs. |
| SP-IN-02 | The build shall read `Data/raw/splits_config.yaml` as a secondary input. The file is required; the build shall fail fast if it is missing. |
| SP-IN-03 | The build shall not modify any file under `Data/raw/` or any Phase 1 or Phase 2 output under `Data/processed/`. |
| SP-IN-04 | The build shall recompute the SHA-256 of `features_flat_all.parquet` and compare it to the corresponding `output_sha256` entry in `feature_manifest.json`. On mismatch the build shall fail fast with a clear error naming the divergent file. |
| SP-IN-05 | The feature matrix shall contain a `season` column with integer values in `[2020, 2025]`. The build shall fail fast if the column is absent or contains values outside that range. |
| SP-IN-06 | The build shall reject (fail-fast) any `splits_config.yaml` whose schema does not satisfy §3.2. |

### 3.2 Splits Config Schema

| ID | Requirement |
|----|-------------|
| SP-CFG-01 | The `splits_config.yaml` schema shall include the top-level keys defined in §4.1. Unknown top-level keys shall cause the build to fail fast. |
| SP-CFG-02 | `strategies` is a non-empty list. Each entry shall be one of `season_holdout` or `loso_cv`. Order is preserved in the artifact. Duplicate entries shall cause the build to fail fast. |
| SP-CFG-03 | `train_seasons` is a non-empty list of distinct integer seasons. `val_season` and `test_season` are each a single integer season. All seasons shall lie in `[2020, 2025]`. |
| SP-CFG-04 | The three role assignments shall be mutually disjoint: `val_season ∉ train_seasons`, `test_season ∉ train_seasons`, and `val_season ≠ test_season`. Their union (`train_seasons ∪ {val_season} ∪ {test_season}`) shall equal exactly the set of seasons present in `features_flat_all.parquet`. No season present in the data may be left unassigned, and no assigned season may be absent from the data. |
| SP-CFG-05 | When `loso_cv` is listed in `strategies`, the rotation pool `train_seasons ∪ {val_season}` shall contain at least two seasons (so every fold has at least one train season and one val season). |
| SP-CFG-06 | `splits_version` (string) is recorded in the manifest. Any change to role-assignment semantics, fold construction, or artifact layout requires bumping this version. |
| SP-CFG-07 | A reference default config shall ship in the repository (`Data/raw/splits_config.yaml`) and reproduce the v2 assignment declared in §3.3. |

### 3.3 v2 Default Splits

The default `splits_config.yaml` shipped in the repo declares the following. Editing this file (and only this file) is how seasons move between roles or strategies toggle.

| Field | v2 Default |
|-------|-----------|
| `strategies` | `["season_holdout"]` |
| `train_seasons` | `[2020, 2021, 2022, 2023]` |
| `val_season` | `2024` |
| `test_season` | `2025` |
| `splits_version` | `"v2"` |

The default assigns the most recent season (`2025`) to `test`, which is the honest proxy for the deployment task — predicting a brand-new season. Re-assigning roles is a config edit, but assigning the latest available season to `test` is the recommended posture. `loso_cv` is available but off by default; enabling it adds the leave-one-season-out folds at ≈6× the training cost.

### 3.4 `season_holdout` Strategy

| ID | Requirement |
|----|-------------|
| SP-SH-01 | When `season_holdout` is in `strategies`, the build shall assign every `GameId` in the universe to exactly one of `train`, `val`, or `test` based on the game's `season` value: `train` if `season ∈ train_seasons`, `val` if `season == val_season`, `test` if `season == test_season`. |
| SP-SH-02 | The three `season_holdout` role lists shall partition the game universe: every `GameId` appears in exactly one list, and the union of the three lists equals the universe. |
| SP-SH-03 | Within each role list, `GameId` values shall be sorted lexicographically ascending. |

### 3.5 `loso_cv` Strategy

| ID | Requirement |
|----|-------------|
| SP-LOSO-01 | When `loso_cv` is in `strategies`, the build shall emit a single `test` list identical to `season_holdout`'s `test` list (the games of `test_season`). The test slice does not rotate across folds. |
| SP-LOSO-02 | The build shall emit one fold per season in the rotation pool `train_seasons ∪ {val_season}`. For the fold holding out season `s`: `val` is the set of games with `season == s`; `train` is the set of games whose season is in the pool but not `s`. The test season is never part of any fold's train or val. |
| SP-LOSO-03 | Folds shall be indexed from `0` in ascending `val_season` order (the fold holding out the earliest season is `fold_index 0`). |
| SP-LOSO-04 | Within each fold, `GameId` values in `train` and `val` shall each be sorted lexicographically ascending. |
| SP-LOSO-05 | For every fold, `train ∩ val = ∅` and `(train ∪ val) ∩ test = ∅`. |
| SP-LOSO-06 | The union of `(train ∪ val)` across all folds shall equal the set of all games whose season is in the rotation pool (equivalently, every game not in `test`). |

### 3.6 Output Encoding

| ID | Requirement |
|----|-------------|
| SP-OUT-01 | The build shall emit `Data/processed/splits_all.json` with the structure declared in §4.2. |
| SP-OUT-02 | The artifact shall be written with `json.dump(..., indent=2)` and a trailing newline. Top-level and per-strategy key order is fixed by §4.2; the build constructs the structure deterministically so identical inputs produce byte-identical output without relying on `sort_keys`. |
| SP-OUT-03 | The set of `GameId` values across `season_holdout`'s three role lists shall equal the set of `GameId` values in `features_flat_all.parquet`. |
| SP-OUT-04 | When `loso_cv` is emitted, the union of `(train ∪ val)` across all folds plus the `test` list shall equal the full game universe. |

### 3.7 Splits Manifest

| ID | Requirement |
|----|-------------|
| SP-MAN-01 | The build shall emit `Data/processed/splits_manifest.json` containing at minimum the keys: `build_timestamp_utc`, `splits_version`, `splits_config_sha256`, `phase2_source_sha256` (object: input filename → SHA), `output_sha256` (object: output filename → SHA), `git_commit` (or `null`), `phase2_manifest_git_commit` (the `git_commit` from `feature_manifest.json`, or `null`), `season_assignment` (object: `{"train_seasons": [...], "val_season": N, "test_season": N}`), `strategy_summaries`. |
| SP-MAN-02 | `strategy_summaries` is an object keyed by strategy name. For `season_holdout`, the value is `{"train_n": N, "val_n": N, "test_n": N}`. For `loso_cv`, the value is `{"test_n": N, "fold_count": N, "folds": [{"fold_index": i, "val_season": s, "train_n": N, "val_n": N}, …]}`. |
| SP-MAN-03 | The manifest's `build_timestamp_utc` shall be in ISO 8601 UTC format. |
| SP-MAN-04 | The manifest shall be written with sorted keys and stable two-space indentation; byte-identical inputs shall produce byte-identical manifests modulo the timestamp. |
| SP-MAN-05 | The manifest shall be written **last** — after the splits artifact — so the artifact's SHA can be computed against the on-disk file. |
| SP-MAN-06 | `splits_config_sha256` shall be the SHA-256 of the raw `Data/raw/splits_config.yaml` file bytes (not the parsed/normalized form). |

---

## 4. Data Model

### 4.1 `Data/raw/splits_config.yaml`

```yaml
splits_version: "v2"

strategies: [season_holdout]   # optionally also loso_cv

train_seasons: [2020, 2021, 2022, 2023]
val_season:  2024
test_season: 2025
```

### 4.2 `Data/processed/splits_all.json` Structure

```json
{
  "splits_version": "v2",
  "season_holdout": {
    "train": ["...", "..."],
    "val":   ["...", "..."],
    "test":  ["...", "..."]
  },
  "loso_cv": {
    "test": ["...", "..."],
    "folds": [
      {"fold_index": 0, "val_season": 2020, "train": ["..."], "val": ["..."]},
      {"fold_index": 1, "val_season": 2021, "train": ["..."], "val": ["..."]}
    ]
  }
}
```

- Top-level key order: `splits_version`, then each strategy in the order declared in `strategies`.
- Within `season_holdout`: roles ordered `train`, `val`, `test`.
- Within `loso_cv`: `test` first, then `folds` (an ordered list).
- Per-fold key order: `fold_index`, `val_season`, `train`, `val`.
- Every `GameId` list is lexicographically sorted ascending.
- A strategy not listed in `strategies` is absent from the artifact entirely.

### 4.3 Output Files

| File | Type | Description |
|------|------|-------------|
| `Data/processed/splits_all.json` | JSON | The split assignment artifact. |
| `Data/processed/splits_manifest.json` | JSON | Build provenance. |

---

## 5. Build Pipeline Design

The pipeline is described conceptually; implementation may organize it differently as long as §3's contract is honored. Recommended module layout:

```
src/nflpredictor/splits/
    __init__.py
    __main__.py          # entry point: python -m nflpredictor.splits
    pipeline.py          # top-level orchestration
    config.py            # splits_config.yaml load + validation
    season_holdout.py    # season_holdout partition logic
    loso_cv.py           # leave-one-season-out folds
    outputs.py           # JSON writers
    manifest.py          # splits_manifest.json construction
```

### 5.1 Conceptual Stages

1. **Load and validate inputs.** Read `splits_config.yaml`; validate per §3.2 and §4.1. Read `features_flat_all.parquet` for `GameId` + `season`; verify the parquet's SHA against `feature_manifest.json` (SP-IN-04).
2. **Build the `(GameId → season)` map.** This is the universe Phase 3 partitions. Confirm the seasons present match the config's assignment (SP-CFG-04).
3. **Compute the `season_holdout` partition.** Bucket each game by the role of its season.
4. **Compute `loso_cv` folds.** Reuse the `test` list; construct each fold's `(train, val)` from the `(GameId → season)` map.
5. **Sort.** Sort every `GameId` list lexicographically ascending.
6. **Emit artifact.** Write `splits_all.json` with the §4.2 structure.
7. **Emit manifest.** Compute output SHAs and per-strategy summaries; write `splits_manifest.json` last.

### 5.2 Determinism Boundaries

The only non-deterministic input is the wall-clock timestamp written to `splits_manifest.json`. The splits artifact is fully deterministic given the inputs. Phase 3's determinism story chains from Phase 2's: any drift in Phase 2 produces an SP-IN-04 hash mismatch and a hard failure rather than silent split drift.

---

## 6. Integration / Endpoint / Tooling Design

Not applicable. The split build is a single-shot offline script with no network surface, no API, and no UI.

---

## 7. Changes to Existing Requirements

This revision redesigns Phase 3 for the six-season dataset. The material contract changes are:

- The input is the combined `features_flat_all.parquet`; Phase 3 reads its `season` column, no longer `week`.
- The week-based `S1` and `S3` strategies are removed (see §1.2). They are replaced by `season_holdout` (default) and `loso_cv` (optional).
- The config declares a season-to-role assignment (`train_seasons` / `val_season` / `test_season`) instead of week ranges.
- The artifact is `splits_all.json`; `splits_version` is bumped to `v2`.

Phase 4 onward must be revised to read `splits_all.json` and the renamed strategy keys. The Phase 1 and Phase 2 specs were revised in the same migration.

---

## 8. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| SP-NF-01 | The build shall be deterministic: identical Phase 2 outputs and `splits_config.yaml` shall produce byte-identical artifact and manifest (excluding timestamp) across runs. |
| SP-NF-02 | The build shall be re-runnable: it shall produce correct output regardless of whether the two split artifacts exist, are stale, or are absent. Stale outputs shall be overwritten. |
| SP-NF-03 | The build shall complete in under 10 seconds on a modern laptop for the current data scale (≈1,626 games × 6 seasons). |
| SP-NF-04 | The `build_timestamp_utc` field in `splits_manifest.json` is the only permitted source of run-to-run output drift. |
| SP-NF-05 | The build shall emit a non-zero exit code on any fatal error. |
| SP-NF-06 | The build shall log per-strategy counts (`season_holdout`: train / val / test sizes; `loso_cv`: fold count + each fold's val season and train/val sizes) to stdout so the partition is visible without opening the manifest. |
| SP-NF-07 | All JSON outputs shall use UTF-8 encoding with Unix line endings (`\n`). |

---

## 9. UI Requirements

Not applicable.

---

## 10. Testing Requirements

| ID | Requirement |
|----|-------------|
| SP-TEST-01 | A unit test shall verify `splits_config.yaml` validation against (a) the default v2 config (must accept); (b) a config where `val_season` is also in `train_seasons` (must reject); (c) a config where `val_season == test_season` (must reject); (d) a config whose season assignment does not cover all seasons present in the data (must reject); (e) a config with an unknown top-level key (must reject); (f) `strategies = []` (must reject); (g) duplicate entries in `strategies` (must reject); (h) `loso_cv` enabled with a single-season rotation pool (must reject). |
| SP-TEST-02 | A unit test shall verify the `season_holdout` partition against a synthetic `(GameId, season)` fixture: every game appears in exactly one role; the three role lists are disjoint and union to the full universe; role assignments match the season-to-role map. |
| SP-TEST-03 | A unit test shall verify `loso_cv` fold construction against the same fixture: the `loso_cv` test list equals `season_holdout`'s test list element-for-element; one fold exists per rotation-pool season in ascending `val_season` order; each fold's `val` is exactly one season's games; no fold's train or val intersects `test`; each list is sorted. |
| SP-TEST-04 | An integration test shall run the full Phase 3 build against a small synthetic multi-season Phase 2 stand-in and assert that the splits artifact and manifest match a checked-in expected snapshot. |
| SP-TEST-05 | A determinism test shall run `run_split_build` twice in succession against identical Phase 2 outputs and identical `splits_config.yaml` and assert byte-equality of both the artifact and the manifest (excluding `build_timestamp_utc`). |
| SP-TEST-06 | A pinned-identity test shall run the real Phase 3 build against the actual Phase 2 outputs and assert: (a) `season_holdout`'s three lists union to the full game universe with no duplicates; (b) the train, val, test counts match the expected counts for the v2 assignment on real data; (c) no `GameId` of `test_season` appears in any `train` or `val` list under either strategy; (d) when `loso_cv` is enabled, `fold_count` equals the number of rotation-pool seasons. |
| SP-TEST-07 | A test shall verify that an SP-IN-04 hash mismatch (e.g., a hand-edited `features_flat_all.parquet`) causes the build to fail fast with a clear error. |
| SP-TEST-08 | A test shall verify that every `GameId` list in the artifact is sorted lexicographically ascending. |

---

## 11. Security Considerations

| ID | Consideration |
|----|---------------|
| SP-SEC-01 | The build operates exclusively on local files under the repository root. It shall not make any network requests. |
| SP-SEC-02 | The build shall not write to any path outside `Data/processed/`. |
| SP-SEC-03 | No input or output of this build contains credentials, PII beyond publicly available game schedules, or other sensitive data. |
| SP-SEC-04 | `splits_config.yaml` is user-editable; the build validates its schema (§3.2) but is not required to defend against adversarial input. YAML loading shall use `yaml.safe_load`. |

---

## 12. Future Considerations

The following are explicitly out of scope for Phase 3 v2 and recorded so they are not lost:

- **Pooled train/val split.** The val season is used only for early stopping and model selection, so pooling the non-test seasons and drawing a random train/val split is a defensible alternative to whole-season validation. It is not specified here; adding it would be a new strategy with a `splits_version` bump.
- **Rotating the test season.** `loso_cv` rotates only the val season. A stricter regime would also rotate which season is held out as test, producing a leave-one-season-out estimate of test performance. This multiplies training cost further and is deferred.
- **Newer seasons.** When a 2026+ season is added upstream, it becomes the natural `test_season` and the prior test season folds into the rotation pool — a pure config edit, no code change.
- **Stratified checks.** A future amendment could assert that each team appears a minimum number of times in each val season; on full-season holdout this is satisfied trivially, so no check ships in v2.

---

## 13. References

- [Plan-MultiYear-Migration.md](./Plan-MultiYear-Migration.md) — Cross-phase roadmap for the 2020–2025 migration; records the season-holdout decision.
- [Spec-Phase1-DataBuild.md](./Spec-Phase1-DataBuild.md) — Phase 1 specification.
- [Spec-Phase2-FeatureEngineering.md](./Spec-Phase2-FeatureEngineering.md) — Phase 2 specification. Phase 3 reads its `features_flat_all.parquet` and binds to its `feature_manifest.json`.
- `Data/processed/features_flat_all.parquet` — Phase 2 output; primary input to Phase 3.
- `Data/processed/feature_manifest.json` — Phase 2 manifest; consulted for source-hash pinning per SP-IN-04.
- [CLAUDE.md](../CLAUDE.md) — Repository-level notes on venv, dataset shape, and join gotchas.
