# Phase 4: Baseline & Model Ladder Implementation Plan

This document defines the phased implementation plan for the Baseline & Model Ladder phase of the NFL Predictor project, based on [Spec-Phase4-BaselineLadder.md](./Spec-Phase4-BaselineLadder.md). Each phase builds on the previous one and contains checkbox-tracked work items. Requirement IDs (`TR-*`) reference the corresponding entries in the specification.

This is a single-developer learning project. Phases are sized for one person to complete in sittings of an hour or two, with tests landing in the same phase as the code they cover. Phase 4 is larger than Phase 3 because the surface area is wider — PyTorch model code, two rung families (trivial vs learned), two feature shapes, two split strategies, an encoder layer, a training loop with early stopping, and twelve prediction parquets — so the plan is correspondingly longer.

---

## Progress Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Scaffolding, Config, and Upstream Hash Gates | Not Started |
| 2 | Feature/Split Loading and Categorical Encoder | Not Started |
| 3 | Trivial Rungs and Prediction Parquet Writers | Not Started |
| 4 | Learned Rungs and Training Loop | Not Started |
| 5 | Pipeline Orchestration, Manifest, and Real-Data Smoke Run | Not Started |
| 6 | Determinism Hardening, Integration Tests, and Documentation | Not Started |

---

## Current State

- Phases 1, 2, and 3 are complete; `Data/processed/{madden_2024.csv, box_scores_2024.csv, player_id_mapping.csv, build_manifest.json, features_flat_2024.parquet, features_pos_2024.parquet, feature_vocab.json, feature_manifest.json, splits_2024.json, splits_manifest.json}` are produced by their respective entry points.
- The Phase 4 specification is final at `Docs/Spec-Phase4-BaselineLadder.md` (all Phase 4 open questions in `Docs/Idea.md` resolved in-spec).
- No `src/nflpredictor/train/` module exists yet.
- No `Data/raw/training_config.yaml` exists yet; it must be created as part of Phase 1 of this plan.
- One new runtime dependency: `torch` (CPU wheel on the dev machine; a CUDA wheel may replace it on a training machine). `pyyaml`, `pyarrow`, `pandas`, and `numpy` are already declared from earlier phases.
- The build is **device-aware, single-device**: the v1 config defaults to `device: "auto"`, which resolves to CUDA when available and CPU otherwise. The dev machine has no GPU; actual training will happen on a separate CUDA machine. Determinism is per-device — byte equality is guaranteed within the same resolved device, not across devices.

---

## Guiding Principles

1. **Work inside the virtual environment.** All `pip`, `python`, and `pytest` commands run with the `.venv/` venv activated.
2. **Tests accompany every phase.** No phase is complete until the tests it introduces pass.
3. **Vertical slices where practical.** Each phase delivers something runnable or testable on its own.
4. **One responsibility per module.** Match the proposed `src/nflpredictor/train/` layout from §5 of the spec; do not pile everything into a single file.
5. **Determinism is non-negotiable per device.** Seed every RNG explicitly (CPU and, when CUDA is in play, also CUDA), sort every iterable that crosses the I/O boundary, pin every stochastic operation. Byte equality is guaranteed within the same resolved device — not across CPU↔CUDA. Phase 6 exists specifically to validate this.
6. **Phase 4 trains and predicts. Phase 5 measures.** The training manifest carries val MAE only. Test MAE never appears in this phase's outputs (TR-MAN-03).
7. **No checkpoints in v1.** Re-training is cheap at this scale; predictions are the deliverable.
8. **The specification is source of truth.** When the plan and the spec disagree, fix the plan or fix the spec — do not silently improvise.

---

## Proposed Repository Layout (additions)

Phase 4 adds the following files to the existing repo. Files outside this list are untouched (Phases 1–3's packages stay read-only from Phase 4's perspective).

```
NFLPredictor/
├── Data/
│   ├── raw/
│   │   └── training_config.yaml                    # NEW — v1 default config
│   └── processed/
│       ├── predictions/                            # NEW dir
│       │   ├── rung0_mean__none__s1.parquet
│       │   ├── rung0_mean__none__s3.parquet
│       │   ├── rung1_team_mean__none__s1.parquet
│       │   ├── rung1_team_mean__none__s3.parquet
│       │   ├── rung2_linear__flat__s1.parquet
│       │   ├── rung2_linear__flat__s3.parquet
│       │   ├── rung2_linear__pos__s1.parquet
│       │   ├── rung2_linear__pos__s3.parquet
│       │   ├── rung3_mlp__flat__s1.parquet
│       │   ├── rung3_mlp__flat__s3.parquet
│       │   ├── rung3_mlp__pos__s1.parquet
│       │   └── rung3_mlp__pos__s3.parquet
│       └── training_manifest.json                  # NEW — training-build provenance
├── src/
│   └── nflpredictor/
│       └── train/                                  # NEW package
│           ├── __init__.py
│           ├── __main__.py                         # entry: python -m nflpredictor.train
│           ├── pipeline.py                         # top-level orchestration
│           ├── config.py                           # YAML load + validation
│           ├── sources.py                          # Phase 2 + Phase 3 load + SHA verification
│           ├── encoders.py                         # categorical encoding policy
│           ├── models.py                           # rung 2 linear, rung 3 MLP
│           ├── trivial.py                          # rungs 0, 1 closed-form
│           ├── train_loop.py                       # learned-rung optimizer + early stop
│           ├── predict.py                          # prediction emission helpers
│           ├── outputs.py                          # parquet writers
│           └── manifest.py                         # training_manifest.json construction
└── tests/
    ├── test_train_smoke.py
    ├── test_train_config.py
    ├── test_train_sources.py
    ├── test_train_encoders.py
    ├── test_train_trivial_rungs.py
    ├── test_train_outputs.py
    ├── test_train_models.py
    ├── test_train_loop.py
    ├── test_train_manifest.py
    ├── test_train_integration.py
    ├── test_train_determinism.py
    ├── test_train_pipeline_run.py
    └── fixtures/
        └── train/                                  # NEW subdir
            ├── raw_phase2/
            │   ├── features_flat_2024.parquet      # sliced (~30 games)
            │   ├── features_pos_2024.parquet       # sliced (matching)
            │   ├── feature_vocab.json              # full or pruned to match
            │   └── feature_manifest.json           # regenerated to match slice SHAs
            ├── raw_phase3/
            │   ├── splits_2024.json                # built from sliced universe
            │   └── splits_manifest.json            # regenerated
            ├── raw/
            │   └── training_config.yaml            # mirrors v1 default (or trimmed for speed)
            └── expected/
                ├── predictions/
                │   └── *.parquet                   # 12 files (or fewer if config trimmed)
                └── training_manifest.json          # build_timestamp_utc blanked
```

Tests use a `test_train_*` prefix to keep them visually separate from earlier phases' test files. All existing tests stay green throughout.

---

## Phase 1: Scaffolding, Config, and Upstream Hash Gates

Establishes the package skeleton, ships the v1 reference `training_config.yaml`, and enforces both upstream input contracts (Phase 2 and Phase 3). Nothing functional yet beyond input validation — but every later phase trusts these invariants.

**Satisfies:** `TR-IN-01` through `TR-IN-08`, `TR-CFG-01` through `TR-CFG-09`, `TR-TEST-01`, `TR-TEST-09`.

### 1.1 Runtime Dependency

- [ ] Activate the venv: `source .venv/bin/activate`.
- [ ] On the dev machine (CPU only): install `torch` from the PyTorch CPU index URL to avoid pulling CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cpu`. On a CUDA training machine, install the matching CUDA wheel instead (e.g., `pip install torch --index-url https://download.pytorch.org/whl/cu124`) — the `torch` import name is identical, only the wheel differs.
- [ ] Record the installed version in the venv's `pip freeze`; the exact wheel becomes the pinned PyTorch wheel for the determinism contract (TR-NF-01). Byte determinism is per-device — the CPU dev machine and a CUDA training machine will have different `torch_version` strings recorded in the manifest, and that is expected.
- [ ] Update `pyproject.toml` to add `torch` as a runtime dep. Do not pin a specific version in source control beyond the major/minor — the exact wheel is captured by `pip freeze` in the venv and recorded in the training manifest at run time (TR-MAN-01 `torch_version`).

### 1.2 Package Skeleton

- [ ] Create `src/nflpredictor/train/__init__.py` (empty).
- [ ] Create `src/nflpredictor/train/__main__.py` with a stub `main()` that prints `"training build not implemented yet"` and exits `0`.
- [ ] Verify the entry point: `python -m nflpredictor.train` prints the stub message.
- [ ] Add a smoke test `tests/test_train_smoke.py` that imports `nflpredictor.train` and asserts the import succeeds. Run `pytest` to confirm.

### 1.3 Default Training Config

- [ ] Create `Data/raw/training_config.yaml` with the v1 contents shown verbatim in §4.1 of the spec (seed `1729`; full `rungs`/`shapes`/`strategies`; the per-rung hyperparameter blocks; the `embedding_dims` table covering every high-cardinality categorical in Phase 2's vocab).
- [ ] Confirm the file parses cleanly with `python -c "import yaml; print(yaml.safe_load(open('Data/raw/training_config.yaml')))"`.

### 1.4 Config Loader and Validator

- [ ] In `src/nflpredictor/train/config.py`, define a `TrainingConfig` frozen dataclass with fields mirroring the YAML schema: `training_version: str`, `seed: int`, `rungs: tuple[str, ...]`, `shapes: tuple[str, ...]`, `strategies: tuple[str, ...]`, `linear: LinearHyperparams`, `mlp: MlpHyperparams`, `embedding_dims: dict[str, int]`. Nested dataclasses `LinearHyperparams` and `MlpHyperparams` carry their respective fields.
- [ ] Add `device: str` to `TrainingConfig` (one of `"auto"`, `"cpu"`, `"cuda"`; default `"auto"`).
- [ ] Implement `load_training_config(path: pathlib.Path, vocab_keys: Iterable[str]) -> TrainingConfig` that:
  - Reads via `yaml.safe_load` (per `TR-SEC-04`).
  - Rejects unknown top-level keys (`TR-CFG-01`).
  - Validates `training_version` is a non-empty string (`TR-CFG-02`).
  - Validates `seed` is a non-negative integer (`TR-CFG-03`).
  - Validates `rungs` is non-empty, every entry is in `{mean, team_mean, linear, mlp}`, no duplicates (`TR-CFG-04`).
  - Validates `shapes` is non-empty, every entry is in `{flat, pos}`, no duplicates (`TR-CFG-05`).
  - Validates `strategies` is non-empty, every entry is in `{S1, S3}`, no duplicates (`TR-CFG-06`).
  - Validates `linear` and `mlp` carry their full required key set (`TR-CFG-07`); `mlp.activation` is one of `{"gelu", "relu"}` (`TR-RUNG-04`).
  - Validates `embedding_dims` has an entry for every high-cardinality categorical in `vocab_keys` (caller passes the set of categorical-column names whose vocab size > 8 from `feature_vocab.json` — wired in §1.5) and every dim is a positive integer (`TR-CFG-08`).
  - Validates `device` is one of `{"auto", "cpu", "cuda"}` (`TR-CFG-10`). Runtime resolution (auto→cpu/cuda and the explicit-cuda-but-not-available fail-fast) lives in §4.1, not here.

### 1.5 Upstream Source-Hash Gates and Strategy Availability

- [ ] In `src/nflpredictor/train/sources.py`, define module-level constants for the Phase 2 basenames (`features_flat_2024.parquet`, `features_pos_2024.parquet`, `feature_vocab.json`, `feature_manifest.json`) and Phase 3 basenames (`splits_2024.json`, `splits_manifest.json`).
- [ ] Reuse `compute_sha256` from `src/nflpredictor/databuild/manifest.py` (already imported by Phase 2 and Phase 3).
- [ ] Implement `verify_phase2_outputs(processed_dir: pathlib.Path) -> dict` that:
  - Loads `feature_manifest.json`.
  - Recomputes SHA-256 of each tracked Phase 2 output (`features_flat_2024.parquet`, `features_pos_2024.parquet`, `feature_vocab.json`) on disk.
  - Compares against the manifest's `output_sha256` map.
  - Raises `Phase2OutputMismatchError` naming the divergent file on any mismatch (`TR-IN-05`).
  - Returns the parsed manifest dict for downstream provenance.
- [ ] Implement `verify_phase3_outputs(processed_dir: pathlib.Path) -> dict` analogously for `splits_2024.json` against `splits_manifest.json` (`TR-IN-06`). Raises `Phase3OutputMismatchError` on mismatch.
- [ ] Implement `verify_strategy_availability(splits: dict, required_strategies: Iterable[str]) -> None` that confirms every requested strategy is present in the Phase 3 splits artifact; raises with a clear error if not (`TR-IN-08`).

### 1.6 Tests

- [ ] `tests/test_train_config.py` (`TR-TEST-01`): cover (a) shipped v1 load (must accept); (b) unknown top-level key (must reject); (c) `rungs = []` (must reject); (d) duplicate entry in `rungs`, `shapes`, or `strategies` (must reject); (e) `rungs` entry outside `{mean, team_mean, linear, mlp}` (must reject); (f) missing `embedding_dims` for a high-cardinality categorical (must reject); (g) negative `seed` (must reject); (h) unsupported `mlp.activation` (must reject); (i) `device` value outside `{"auto","cpu","cuda"}` (must reject). Use the real `feature_vocab.json` (or a small fixture) to provide `vocab_keys`.
- [ ] `tests/test_train_sources.py` (`TR-TEST-09`): cover (a) happy path against real Phase 2 + Phase 3 outputs; (b) tampered `features_flat_2024.parquet` → fail-fast naming the file; (c) tampered `features_pos_2024.parquet` → fail-fast; (d) tampered `feature_vocab.json` → fail-fast; (e) tampered `splits_2024.json` → fail-fast; (f) missing Phase 2 manifest → fail-fast; (g) requested strategy absent from splits artifact → fail-fast.

**Definition of done:** Config loading rejects every malformed case enumerated in §3.2; both upstream hash gates block runs against tampered inputs; strategy availability is enforced.

---

## Phase 2: Feature/Split Loading and Categorical Encoder

Loads the upstream artifacts into in-memory structures and implements the model's input layer (low-card one-hot + high-card `nn.Embedding`). Pure data plumbing — no training yet.

**Satisfies:** `TR-SHAPE-01`, `TR-SHAPE-02`, `TR-SHAPE-03`, `TR-SHAPE-04`, `TR-CAT-01` through `TR-CAT-06`, `TR-MODEL-05`, `TR-NF-02`, `TR-TEST-02`.

### 2.1 Source Loaders

- [ ] In `src/nflpredictor/train/sources.py`, implement `load_features(processed_dir, shape: str) -> pandas.DataFrame` that reads the appropriate parquet (`features_flat_2024.parquet` for `flat`; `features_pos_2024.parquet` for `pos`) via `pyarrow.parquet`. Preserves column dtypes; sorts rows by `GameId` for determinism.
- [ ] Implement `load_vocab(processed_dir) -> dict[str, list[str]]` that reads `feature_vocab.json` and returns the categorical-column → vocab map.
- [ ] Implement `load_splits(processed_dir) -> dict` that reads `splits_2024.json` and returns the parsed structure.
- [ ] Implement `assert_label_parity(flat: pd.DataFrame, pos: pd.DataFrame) -> None` that verifies the two frames share identical `(GameId, home_score, away_score)` triples (same set, same label values for each `GameId`); raises with a clear error on mismatch (`TR-NF-02`).

### 2.2 Column Classification

- [ ] In `src/nflpredictor/train/encoders.py`, implement `classify_columns(features: pd.DataFrame, vocab: dict[str, list[str]]) -> ColumnClassification` returning a frozen dataclass with fields:
  - `numeric: tuple[str, ...]` — columns not in `vocab` and not label columns (`home_score`, `away_score`).
  - `low_card_categorical: tuple[str, ...]` — vocab columns with `len(vocab[col]) ≤ 8` (`TR-CAT-01`).
  - `high_card_categorical: tuple[str, ...]` — vocab columns with `len(vocab[col]) > 8` (`TR-CAT-02`).
  - `labels: tuple[str, ...]` — `("home_score", "away_score")`.
  - All four tuples are sorted lexicographically by column name (deterministic concat order — `TR-CAT-05`).
- [ ] The boundary `≤ 8` is read from `feature_vocab.json` sizes, not from runtime data inspection (`TR-CAT-03`).

### 2.3 FeatureEncoder Module

- [ ] In `src/nflpredictor/train/encoders.py`, implement `class FeatureEncoder(nn.Module)`:
  - `__init__(classification, vocab_sizes, embedding_dims)`: constructs an `nn.ModuleDict` of `nn.Embedding` layers keyed by **vocab key** (shared across physical columns that point at the same vocab — e.g., `home_team_code` and `away_team_code` both use `team_codes`), with `num_embeddings = vocab_sizes[key] + 1` (the +1 reserves slot 0 for Phase 2's `NULL_SENTINEL = -1`, TR-CAT-07) and `embedding_dim = embedding_dims[key]`. Defaults from `nn.Embedding` initialization (`TR-MODEL-05`).
  - `d_in` property: returns the total flat-vector width = `len(numeric) + sum((vocab_size + 1) for col in low_card_categorical) + sum(embedding_dim for col in high_card_categorical)` (`TR-CAT-05`, `TR-CAT-06`).
  - `forward(numeric: Tensor, low_card_indices: dict[str, Tensor], high_card_indices: dict[str, Tensor]) -> Tensor`: bumps every categorical index by `+1` (so `NULL_SENTINEL = -1` lands in the reserved slot 0, TR-CAT-07), then builds one-hot tensors for each low-card column (using `F.one_hot(idx + 1, num_classes=vocab_size + 1).float()`) and embeddings for each high-card column. Concatenates in the order: numeric → low-card (sorted by name) → high-card (sorted by name). Returns shape `(batch, d_in)`.
- [ ] Add a `prepare_batch(df_rows, classification) -> dict[str, Tensor]` helper that converts a DataFrame slice into the three tensors the encoder expects. Use `torch.float32` for numeric and labels, `torch.long` for categorical indices.

### 2.4 Tests

- [ ] `tests/test_train_sources.py` (extension): label-parity check accepts a real Phase 2 pair and rejects a synthetic mismatched pair.
- [ ] `tests/test_train_encoders.py` (`TR-TEST-02`): build a synthetic vocab (one column with `len(vocab) == 4` for low-card; one with `len(vocab) == 30` for high-card) and a 5-row synthetic DataFrame; assert:
  - Column classification routes columns correctly per `len(vocab) ≤ 8` boundary.
  - `FeatureEncoder.d_in` equals `numeric_width + 4 + embedding_dim_for_high_card_col`.
  - Forward pass produces a `(5, d_in)` tensor with correct shape.
  - Sort order: forcing column-name shuffles in the DataFrame and vocab still produces a deterministic `d_in` and identical encoder output.
  - Re-instantiating with the same seed produces identical embedding weights.

**Definition of done:** Phase 2 and Phase 3 sources load cleanly with label parity verified; the encoder produces deterministic flat vectors of the correct width for any vocab/feature combination.

---

## Phase 3: Trivial Rungs and Prediction Parquet Writers

Implements the closed-form rungs (mean, team_mean) and the parquet output contract. After this phase, the pipeline can emit four of the twelve prediction parquets end-to-end (the trivial rungs × two strategies).

**Satisfies:** `TR-RUNG-01`, `TR-RUNG-02`, `TR-RUNG-05`, `TR-OUT-01` through `TR-OUT-07`, partial `TR-TEST-03`, partial `TR-TEST-05`.

### 3.1 Rung 0 — Mean Predictor

- [ ] In `src/nflpredictor/train/trivial.py`, implement `predict_mean(train_labels: pd.DataFrame, target_game_ids: list[str]) -> pd.DataFrame` that:
  - Computes `mean(home_score)` and `mean(away_score)` over `train_labels`.
  - Returns a frame with columns `(GameId, pred_home, pred_away)` where each row repeats the same `(mean_home, mean_away)` pair (`TR-RUNG-01`).
  - Is deterministic: no PyTorch, no RNG (`TR-RUNG-05`).

### 3.2 Rung 1 — Team-Mean Predictor

- [ ] Implement `predict_team_mean(train_features: pd.DataFrame, target_features: pd.DataFrame, home_team_col: str, away_team_col: str) -> pd.DataFrame`:
  - Computes per-team `home_mean = mean(home_score where home_team == T)` and `away_mean = mean(away_score where away_team == T)` from `train_features`.
  - For each row in `target_features`, looks up `home_mean[H]` and `away_mean[A]`.
  - Falls back to the global mean (per `TR-RUNG-01`) when a team is unseen in training (`TR-RUNG-02`).
  - Returns columns `(GameId, pred_home, pred_away)`.
- [ ] Identify the team-id column names: the encoded `team_codes` integer columns for home and away (likely `HomeTeamCode` and `AwayTeamCode` per Phase 2's encoding; verify by inspecting `features_flat_2024.parquet` columns during implementation).

### 3.3 Strategy Handler for Trivial Rungs

- [ ] In `src/nflpredictor/train/predict.py`, implement `run_trivial_combo(rung_id: str, strategy: str, features: pd.DataFrame, splits: dict, labels: pd.DataFrame) -> pd.DataFrame` that:
  - For `strategy == "S1"`: trains on `S1.train`, predicts on `S1.val ∪ S1.test`. Returns a long frame with `(slice, GameId, pred_home, pred_away)` (`TR-STRAT-01`).
  - For `strategy == "S3"`: iterates folds, trains on each `S3.folds[i].train`, predicts on `S3.folds[i].val`. Returns `(fold_index, GameId, pred_home, pred_away)` (`TR-STRAT-02`).
  - Dispatches by `rung_id` to `predict_mean` or `predict_team_mean`.
  - For trivial rungs only (`TR-STRAT-04`).

### 3.4 Parquet Writers

- [ ] In `src/nflpredictor/train/outputs.py`, implement `write_s1_predictions(df: pd.DataFrame, path: pathlib.Path) -> None`:
  - Sorts rows by `(slice, GameId)` with `slice` ordering `val` before `test` (`TR-OUT-02`).
  - Coerces dtypes: `slice` → string, `GameId` → string, `pred_home` / `pred_away` → float64.
  - Writes via `pyarrow.parquet.write_table` with `compression="snappy"`, `row_group_size=1024` (`TR-OUT-06`).
- [ ] Implement `write_s3_predictions(df: pd.DataFrame, path: pathlib.Path) -> None`:
  - Sorts rows by `(fold_index, GameId)` with `fold_index` ascending (`TR-OUT-03`).
  - Coerces dtypes: `fold_index` → int8, `GameId` → string, `pred_home` / `pred_away` → float64.
  - Same writer settings as S1.
- [ ] Implement `combination_filename(rung_id, shape, strategy) -> str` returning `f"{rung_id}__{shape}__{strategy.lower()}.parquet"` (`TR-OUT-01`).
- [ ] Implement `ensure_predictions_dir(processed_dir) -> pathlib.Path` that creates `Data/processed/predictions/` if absent and returns the path (`TR-OUT-07`). Overwriting existing matching files is acceptable; do not delete unrelated files.

### 3.5 Tests

- [ ] `tests/test_train_trivial_rungs.py` (partial `TR-TEST-03`):
  - Rung 0: on a 6-game synthetic label fixture, assert every prediction row equals `(mean(home_train), mean(away_train))`.
  - Rung 1: assert per-team lookups return the right means; assert global-mean fallback for an unseen team injected into the val set.
  - Both rungs: assert byte-identical pandas-frame output across two consecutive invocations (closed-form determinism — `TR-RUNG-05`).
- [ ] `tests/test_train_outputs.py` (partial `TR-TEST-05`):
  - Synthetic S1 frame round-trips through `write_s1_predictions` → `pq.read_table` and matches expected sort order, column set, dtype.
  - Synthetic S3 frame round-trips analogously.
  - GameId sets in the written file match the `splits_2024.json` fixture's `S1.val`/`S1.test`/`S3.folds[i].val` exactly (`TR-OUT-04`, `TR-OUT-05`).
  - Re-running the writer on the same input produces a byte-identical parquet (`pathlib.Path.read_bytes()` equality).

**Definition of done:** Rungs 0 and 1 produce correct predictions for any synthetic label fixture; parquet writers honor the schema, sort, and dtype contracts from §3.10 / §4.2.

---

## Phase 4: Learned Rungs and Training Loop

Implements the two PyTorch rungs (linear and MLP) and the training loop with early stopping. The bulk of the new code lands here.

**Satisfies:** `TR-RUNG-03`, `TR-RUNG-04`, `TR-RUNG-06`, `TR-RUNG-07`, `TR-MODEL-01` through `TR-MODEL-05`, `TR-TRAIN-01` through `TR-TRAIN-07`, `TR-STRAT-01`, `TR-STRAT-02`, `TR-STRAT-03`, `TR-TEST-04`.

### 4.1 Device Resolution and Determinism Setup

- [ ] In `src/nflpredictor/train/train_loop.py`, implement `resolve_device(requested: str) -> ResolvedDevice` (small dataclass with `name: str`, `torch_device: torch.device`, `cuda_device_name: str | None`, `cuda_version: str | None`):
  - `requested == "cpu"` → CPU regardless of CUDA availability.
  - `requested == "cuda"` → CUDA when `torch.cuda.is_available()`; otherwise raise a clear error (`TR-CFG-10`).
  - `requested == "auto"` → CUDA when available, else CPU.
  - When CUDA resolves: also capture `torch.cuda.get_device_name(0)` and `torch.version.cuda` for the manifest.
- [ ] Implement `seed_all(seed: int, device: ResolvedDevice) -> None` that sets `torch.manual_seed(seed)`, `numpy.random.seed(seed)`, `random.seed(seed)`, `os.environ["PYTHONHASHSEED"] = str(seed)`, and calls `torch.use_deterministic_algorithms(True)` (`TR-TRAIN-02`). When `device.name == "cuda"`, also call `torch.cuda.manual_seed_all(seed)`. The pipeline entry point (§5.2) sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment before any model construction on CUDA, so seed_all can assume it is in place.
- [ ] Document in a one-line comment why `PYTHONHASHSEED` must be set process-wide before any hashed structure is iterated; note that the training entry point sets it as early as possible.
- [ ] Single-device discipline (`TR-TRAIN-06`): never wrap models in `DataParallel` / `DistributedDataParallel`; never select MPS even when available on macOS. All tensors live on `device.torch_device`.

### 4.2 Learned Model Definitions

- [ ] In `src/nflpredictor/train/models.py`, implement `class LinearRung(nn.Module)`:
  - `__init__(encoder: FeatureEncoder)`: stores the encoder, constructs `self.head = nn.Linear(encoder.d_in, 2)` (`TR-RUNG-03`, `TR-MODEL-01`).
  - `forward(numeric, low_card_indices, high_card_indices)`: encodes and returns the head's output.
- [ ] Implement `class MlpRung(nn.Module)`:
  - `__init__(encoder, hidden_dim, activation: str, dropout: float)`: builds `encoder → nn.Linear(d_in, hidden_dim) → activation_layer → nn.Dropout(dropout) → nn.Linear(hidden_dim, 2)` (`TR-RUNG-04`, `TR-MODEL-01`).
  - `activation_layer` is `nn.GELU()` or `nn.ReLU()` per config.
- [ ] Confirm both models use PyTorch's default Linear/Embedding initialization (no custom init — `TR-MODEL-05`).

### 4.3 Training Loop with Early Stopping

- [ ] In `src/nflpredictor/train/train_loop.py`, implement `train_learned_rung(model, train_batches, val_batches, *, optimizer_cfg, max_epochs, early_stop_patience, seed) -> TrainingResult` that:
  - Constructs `torch.optim.Adam(model.parameters(), lr=optimizer_cfg.lr)` (`TR-MODEL-03`).
  - Constructs `torch.nn.L1Loss(reduction="mean")` (`TR-MODEL-02`).
  - Per epoch: shuffles training batches deterministically using `torch.Generator().manual_seed(seed + epoch)` (or an equivalent stable derivation); runs forward/backward/step per batch.
  - After each epoch: computes val MAE = `mean(|pred_home - true_home| + |pred_away - true_away|) / 2` over the entire val set (`TR-MAN-04`).
  - Tracks the best (lowest) val MAE seen and snapshots `state_dict` at that epoch.
  - Triggers early stop when no improvement over the running best for `early_stop_patience` consecutive epochs (`TR-TRAIN-04`).
  - Returns a `TrainingResult` dataclass with fields: `best_epoch: int`, `best_val_mae: float`, `epochs_trained: int`, `stopped_early: bool`, `best_state_dict: dict`.
- [ ] Implement `make_batches(features: pd.DataFrame, labels: pd.DataFrame, classification, batch_size, *, generator: torch.Generator | None) -> Iterator[Batch]` that produces deterministic minibatch tuples of `(numeric, low_card_indices, high_card_indices, labels)`. Without a generator (val/test), iterate in fixed GameId-sorted order with no shuffling.

### 4.4 Prediction Emission for Learned Rungs

- [ ] In `src/nflpredictor/train/predict.py`, implement `predict_learned(model, batches) -> pd.DataFrame`:
  - Sets `model.eval()`; iterates batches with `torch.no_grad()`; concatenates outputs.
  - Returns `(GameId, pred_home, pred_away)` (caller adds `slice` or `fold_index`).
- [ ] Implement `run_learned_combo(rung_id, shape, strategy, features, splits, labels, classification, encoder_factory, hyperparams, seed, device) -> pd.DataFrame`:
  - For `strategy == "S1"`: calls `seed_all(seed, device)`; builds a fresh encoder + model on `device.torch_device`; trains on `S1.train`; predicts on `S1.val` and `S1.test` using best-epoch params (`TR-TRAIN-05`, `TR-STRAT-01`). Returns long frame with `slice` column.
  - For `strategy == "S3"`: per fold, calls `seed_all(seed, device)` (fresh seed reset per fold for byte-identical fold outputs); builds a fresh encoder + model on `device.torch_device`; trains on `fold.train`; predicts on `fold.val`. No test predictions (`TR-STRAT-02`, `TR-STRAT-03`). Returns long frame with `fold_index` column.
  - Returns the same `TrainingResult` records (one per S1 or per-fold for S3) alongside the predictions for manifest aggregation.
- [ ] After predictions are produced and the val MAE captured, discard `state_dict` (no checkpoint persistence — `TR-TRAIN-07`).

### 4.5 Combination Enumeration

- [ ] In `src/nflpredictor/train/pipeline.py`, implement `enumerate_combinations(config: TrainingConfig) -> list[Combination]` returning the Cartesian product of `rungs × shapes × strategies`, filtered per `TR-RUNG-07`:
  - Trivial rungs (`mean`, `team_mean`) pair only with `shape = "none"`.
  - Learned rungs (`linear`, `mlp`) pair with each entry in `config.shapes`.
  - Skipped combinations are recorded in the manifest's `training_summaries` block (or silently dropped per the spec — read TR-RUNG-07 carefully and implement the silent-skip variant; document the skip in a top-level `skipped_combinations` field of the manifest for traceability).

### 4.6 Tests

- [ ] `tests/test_train_models.py`: instantiate both `LinearRung` and `MlpRung` with a 4-row synthetic batch; assert forward pass produces a `(4, 2)` tensor; assert parameter counts match expectations (linear: `(d_in + 1) * 2`; MLP: `(d_in + 1) * hidden + (hidden + 1) * 2`).
- [ ] `tests/test_train_loop.py` (`TR-TEST-04`):
  - Build a 5-game synthetic train + 3-game synthetic val fixture with deliberately trivial labels (e.g., score depends linearly on one numeric feature) so training converges quickly.
  - Run `train_learned_rung` for 3 epochs; assert val MAE is finite and decreased from epoch 0 to epoch 2.
  - Inject a synthetic "no improvement" plateau (constant val MAE for `patience + 1` epochs from the start); assert early stop fires and `stopped_early == True`.
  - Assert `best_epoch` corresponds to the epoch with the minimum val MAE (not the final epoch), and predictions from `best_state_dict` match what that epoch's parameters would produce.
- [ ] `tests/test_train_loop.py` (extension): determinism — seed_all + train twice from a clean module load; assert two `best_state_dict` outputs are byte-identical and val MAE values are equal.

**Definition of done:** Rungs 2 and 3 train and predict against synthetic data; early stopping behaves as specified; the training loop is deterministic under fixed seed within the pinned PyTorch wheel.

---

## Phase 5: Pipeline Orchestration, Manifest, and Real-Data Smoke Run

Wires everything together into a runnable build, emits the training manifest, and runs end-to-end against the real Phase 2 + Phase 3 outputs for the first time.

**Satisfies:** `TR-MAN-01` through `TR-MAN-08`, `TR-NF-05`, `TR-NF-06`, `TR-NF-07`, `TR-NF-08`, `TR-NF-09`, `TR-NF-10`, rest of `TR-TEST-05`, `TR-TEST-10`.

### 5.1 Manifest Construction

- [ ] In `src/nflpredictor/train/manifest.py`, reuse `compute_sha256` and `try_get_git_commit` from `databuild.manifest`.
- [ ] Implement `build_training_summary_s1(result: TrainingResult | None, val_mae: float) -> dict` returning the `TR-MAN-02` S1 shape: `{"val_mae", "epochs_trained", "best_epoch", "stopped_early"}`. For trivial rungs, the three epoch fields are `null`.
- [ ] Implement `build_training_summary_s3(per_fold_results: list[TrainingResult | None], per_fold_val_mae: list[float]) -> dict` returning the `TR-MAN-02` S3 shape: `{"fold_count", "mean_val_mae", "per_fold"}` where each per-fold entry mirrors the S1 structure.
- [ ] Implement `build_training_manifest(*, config, resolved_device, phase2_source_sha256, phase3_source_sha256, output_sha256, training_config_sha256, phase2_manifest_git_commit, phase3_manifest_git_commit, training_summaries, skipped_combinations, torch_version) -> dict` constructing the full manifest dict per `TR-MAN-01`. The `resolved_device` parameter carries `device_requested` (raw config value), `device_resolved` (`"cpu"` or `"cuda"`), `cuda_device_name`, and `cuda_version`; pass `null` for the two CUDA fields when CPU is resolved.
- [ ] Implement `write_training_manifest(manifest: dict, path: pathlib.Path) -> None` using `json.dump(..., sort_keys=True, indent=2)` with a trailing newline (`TR-MAN-07`).
- [ ] Add an assertion (or a unit test in §5.5) that no key in `training_summaries` mentions `test_mae` (`TR-MAN-03` enforcement).

### 5.2 Pipeline Orchestration

- [ ] In `src/nflpredictor/train/pipeline.py`, implement `run_training_build(raw_dir, processed_dir, *, repo_dir=None) -> None` executing the §5.1 stages from the spec:
  1. Set `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")` **before** importing torch-using modules at function scope, so it is in place if CUDA is later resolved (`TR-TRAIN-02`).
  2. Verify upstream outputs (`verify_phase2_outputs`, `verify_phase3_outputs`).
  3. Load `feature_vocab.json` to derive high-card vocab keys; load `training_config.yaml` via `load_training_config(path, high_card_vocab_keys)`.
  4. Resolve the device via `resolve_device(config.device)`; log the resolved device to stdout.
  5. Verify strategy availability (`verify_strategy_availability`).
  6. Load features for both shapes, vocab, splits; verify label parity (`assert_label_parity`).
  7. Enumerate combinations.
  8. For each combination: dispatch to `run_trivial_combo` or `run_learned_combo` (passing the resolved device for learned rungs); collect predictions and training results.
  9. Write each combination's parquet via the appropriate writer.
  10. Compute output SHAs against the on-disk parquets.
  11. Construct and write `training_manifest.json` **last** (`TR-MAN-06`), threading the resolved device through to `build_training_manifest`.
- [ ] Compute `training_config_sha256` from the raw `training_config.yaml` bytes (`TR-MAN-08`).
- [ ] Compute `phase2_source_sha256` and `phase3_source_sha256` from the on-disk artifacts; record under their repo-relative paths (mirrors Phase 3's pattern).
- [ ] Record `torch.__version__` in the manifest as `torch_version` (`TR-MAN-01`).
- [ ] Stdout logging of the per-combination ladder summary (`TR-NF-07`): one line per combination listing rung, shape, strategy, val MAE (mean over folds for S3), and epochs trained / early-stop flag for learned rungs. Format should be scannable — a small table or aligned columns.
- [ ] Update `src/nflpredictor/train/__main__.py` to call `run_training_build` with default paths and return exit 1 on any exception (`TR-NF-06`).

### 5.3 Smoke Run

- [ ] Run `python -m nflpredictor.train` against the real Phase 2 + Phase 3 outputs. Verify:
  - All twelve expected parquet filenames appear under `Data/processed/predictions/`.
  - Each parquet parses cleanly and exposes the expected columns.
  - S1 parquets cover exactly the `S1.val` and `S1.test` GameId sets.
  - S3 parquets cover exactly the per-fold val GameId sets across all 9 folds.
  - `Data/processed/training_manifest.json` carries every required key from `TR-MAN-01`.
  - Per-combination val MAE values are sane (rung 0 ~ baseline-floor MAE; learned rungs lower); the ladder summary on stdout matches the manifest.
  - No `test_mae` appears anywhere in the manifest (manual grep).
  - Wall-clock under `TR-NF-04`'s 5-minute limit.
  - Capture two successive runs' parquet SHAs and confirm identical bytes (informal smoke-test of determinism — the formal test lands in Phase 6).

### 5.4 Tests

- [ ] `tests/test_train_manifest.py`:
  - Round-trip `build_training_manifest` over a synthetic input and assert the returned dict contains all `TR-MAN-01` keys.
  - Assert `build_training_summary_s1` and `build_training_summary_s3` return the exact shape from `TR-MAN-02`.
  - Assert no key path inside `training_summaries` contains the substring `"test_mae"` for a sample manifest (`TR-TEST-10`).
- [ ] `tests/test_train_outputs.py` (extension, rest of `TR-TEST-05`): integration-style check that the combination-id → filename mapping in `combination_filename` matches the §3.10 pattern for every valid `(rung_id, shape, strategy)` triple.

**Definition of done:** A real training-build run emits twelve parquets and a manifest in `Data/processed/`; per-combination val MAE summary is scannable on stdout; the build runs end-to-end without exceptions in under 5 minutes; no test MAE appears in any output.

---

## Phase 6: Determinism Hardening, Integration Tests, and Documentation

Locks the training build against silent drift, ships the synthetic fixture and the formal byte-determinism test, pins the real-data identity test, and updates the project docs.

**Satisfies:** `TR-NF-01`, `TR-NF-03`, `TR-NF-04`, `TR-NF-10`, `TR-TEST-06`, `TR-TEST-07`, `TR-TEST-08`, closure on any straggling `TR-TEST-*` IDs.

### 6.1 Determinism Audit

- [ ] Review every `sort` and `sorted()` call in `src/nflpredictor/train/`. Each must specify a fully-tie-breaking sort key. Document inline with one-line comments referencing the satisfied spec ID where non-obvious.
- [ ] Confirm every `json.dump` invocation matches its required option set: manifest uses `sort_keys=True`, `indent=2`, trailing newline.
- [ ] Confirm every parquet writer uses `compression="snappy"`, `row_group_size=1024`, and `pyarrow` (no fallback to `fastparquet`).
- [ ] Confirm no `set()` iteration, dict iteration, or `glob()` order affects output. Where a set or dict is iterated, the output passes through `sorted(...)` at the boundary.
- [ ] Confirm all logging goes to `sys.stderr` for warnings/diagnostics and `sys.stdout` only for the ladder summary table; nothing else writes to stdout.
- [ ] Confirm `seed_all` is invoked exactly at the start of every learned-rung training run (per-combination for S1; per-fold for S3) and that `torch.use_deterministic_algorithms(True)` is set before any model construction.
- [ ] Confirm no network access at runtime (`TR-NF-10`) — grep for `urllib`, `requests`, `http`, `download`, `hub.load` in the package; manual confirmation only.

### 6.2 Synthetic Fixture

- [ ] Create `tests/fixtures/train/raw_phase2/features_flat_2024.parquet` — a sliced Phase 2 feature matrix small enough to be checked in (~30 games spanning at least Weeks 1–15 + a handful in 16–18 so all three slices have content). Same column structure as the real parquet.
- [ ] Create `tests/fixtures/train/raw_phase2/features_pos_2024.parquet` — the matching pos-shape slice for the same 30 games.
- [ ] Create `tests/fixtures/train/raw_phase2/feature_vocab.json` — either the full vocab from real Phase 2 or a pruned-to-fixture variant. Document the choice in the fixture's README/regenerator.
- [ ] Generate a matching `tests/fixtures/train/raw_phase2/feature_manifest.json` whose `output_sha256` entries equal the SHAs of the three sliced files (other manifest keys can be minimal stubs).
- [ ] Generate `tests/fixtures/train/raw_phase3/splits_2024.json` by running Phase 3's `run_split_build` against the sliced Phase 2 fixture (so the splits artifact is internally consistent). Save the matching `splits_manifest.json`.
- [ ] Create `tests/fixtures/train/raw/training_config.yaml` — either the full v1 default or a trimmed variant (e.g., `max_epochs: 10`, `early_stop_patience: 3`) to keep test runtime under ~30 seconds. Document the trim.
- [ ] Run the training build against the fixture once; manually inspect outputs; check them in under `tests/fixtures/train/expected/` (predictions parquets + `training_manifest.json` with `build_timestamp_utc` blanked).
- [ ] Add `tests/fixtures/train/_regenerate.py` (companion to earlier phases' regenerators) that rebuilds the fixture on demand. Doc-string notes the fixture is pinned to a specific PyTorch wheel — regenerating after a `torch` upgrade may legitimately change parquet bytes.

### 6.3 Integration Test

- [ ] `tests/test_train_integration.py` (`TR-TEST-06`):
  - Runs `run_training_build` against the synthetic fixture into a temp directory.
  - Asserts each emitted prediction parquet matches the checked-in expected file byte-for-byte (use `pathlib.Path.read_bytes()` equality).
  - Loads both training manifests, blanks `build_timestamp_utc`, and asserts byte-equality of the remainder.

### 6.4 Determinism Test

- [ ] `tests/test_train_determinism.py` (`TR-TEST-07`):
  - Runs `run_training_build` twice in succession against identical fixture inputs into two temp directories.
  - Asserts byte-equality of every prediction parquet across the two runs.
  - For `training_manifest.json`, blanks `build_timestamp_utc` in both and asserts byte-equality of the remainder.

### 6.5 Pinned Real-Data Identities

- [ ] `tests/test_train_pipeline_run.py` (`TR-TEST-08`):
  - Runs `run_training_build` against the actual Phase 2 + Phase 3 outputs into a temp directory.
  - Asserts all 12 expected parquet filenames exist.
  - For each S1 parquet: asserts the `slice == "val"` rows' GameIds equal `splits_2024.json → S1.val` exactly; `slice == "test"` rows' GameIds equal `S1.test` exactly.
  - For each S3 parquet: asserts each `fold_index = i` slice's GameIds equal `splits_2024.json → S3.folds[i].val` exactly across all 9 folds.
  - Asserts the manifest's `training_summaries` block carries a val MAE for every combination and (via `TR-TEST-10`'s helper) no `test_mae` key appears anywhere.
  - Optionally pin the actual real-data val MAE values once the first run reveals them — useful regression guard, but tighten thresholds carefully to avoid flakiness across PyTorch patch versions.

### 6.6 Performance Sanity Check

- [ ] Run `time python -m nflpredictor.train` against the real Phase 2 + Phase 3 outputs. Confirm wall-clock is comfortably under `TR-NF-04`'s 5-minute limit. Record the time in the plan or a comment.

### 6.7 Documentation Updates

- [ ] Update `CLAUDE.md`'s "Project status" section to note that Phase 4 implementation is complete, the training build is invoked via `python -m nflpredictor.train`, the twelve prediction parquets land in `Data/processed/predictions/`, and the training manifest at `Data/processed/training_manifest.json`. Note the device-aware behavior (`device: "auto"` resolves to CUDA when available, else CPU; explicit `"cpu"` / `"cuda"` honored) and the per-device determinism caveat (byte-identical within the same resolved device + pinned PyTorch wheel; cross-device differences expected and not asserted).
- [ ] Add a "Status: Implementation complete (YYYY-MM-DD)" header note to `Docs/Idea.md`'s Phase 4 section, mirroring the format used for Phases 1–3. Include the real-data run summary (per-rung val MAE for the v1 ladder).
- [ ] Optionally update `README.md` to reference the new build step in the project's run sequence.

**Definition of done:** Every `TR-TEST-*` requirement in the spec has a corresponding passing test; re-running the training build produces byte-identical outputs within the pinned PyTorch wheel; documentation is updated.

---

## After Phase 6

Phase 4 is complete when:

- All six phases above are checked off.
- `pytest` passes with the venv activated.
- `python -m nflpredictor.train` produces the twelve expected prediction parquets and the training manifest in `Data/processed/`.
- Re-running the training build within the same pinned PyTorch wheel produces byte-identical predictions (modulo the manifest timestamp).
- The `CLAUDE.md` and `Idea.md` doc updates are in place.

At that point, Phase 5 (Evaluation) can begin scoping against the *actual* Phase 4 predictions on disk — per-rung val MAE values, per-fold S3 deltas, and the S1 test predictions waiting to be measured — rather than imagined data, which is the same discipline that paced Phases 1–3 against their successors.

---

## References

- [Spec-Phase4-BaselineLadder.md](./Spec-Phase4-BaselineLadder.md) — Formal specification for Phase 4.
- [Spec-Phase3-Splits.md](./Spec-Phase3-Splits.md) — Phase 3 specification; Phase 4 reads its outputs.
- [Spec-Phase2-FeatureEngineering.md](./Spec-Phase2-FeatureEngineering.md) — Phase 2 specification; Phase 4 reads its outputs.
- [Plan-Phase3-Splits.md](./Plan-Phase3-Splits.md) — Stylistic precedent for this document.
- [Plan-Phase2-FeatureEngineering.md](./Plan-Phase2-FeatureEngineering.md) — Stylistic precedent for this document.
- [Plan-Phase1-DataBuild.md](./Plan-Phase1-DataBuild.md) — Stylistic precedent for this document.
- [Idea.md](./Idea.md) — Source idea document; §"Phase 4: Baseline & Model Ladder" lists the open questions the spec resolved.
- [CLAUDE.md](../CLAUDE.md) — Repository-level notes; will be updated during Phase 6.7.
