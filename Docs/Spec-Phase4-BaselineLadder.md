# Phase 4: Baseline & Model Ladder Specification

## 1. Introduction

### 1.1 Purpose

This specification defines the **Baseline & Model Ladder** phase of the NFL Predictor project. Phase 4 trains a fixed ladder of regression models against Phase 3's splits and emits per-rung prediction artifacts that Phase 5 will consume to compute evaluation metrics.

Phases 1, 2, and 3 are implemented and frozen; this spec assumes their outputs as given. Phase 5 (Evaluation) and beyond remain exploratory in `Docs/Idea.md` and are intentionally not specified yet — Phase 4 closes the contract one layer up the stack by producing the prediction surface that an evaluation phase can bind to.

### 1.2 Scope

In scope:

- A fourth deterministic build step (separate from Phases 1, 2, and 3) that reads Phase 2's feature matrices and vocab, Phase 3's split artifact, and a hand-edited `Data/raw/training_config.yaml`, and writes one prediction parquet per `(rung, feature_shape, strategy)` combination plus a training manifest into `Data/processed/`.
- A fixed v1 ladder of four rungs: rung 0 (mean predictor), rung 1 (team-mean predictor), rung 2 (single-layer `nn.Linear`), rung 3 (small MLP).
- Two feature-shape variants per learned rung: B-flat (slot-indexed) and B-pos (position-indexed), trained independently and emitted as separate prediction files.
- Two split-strategy regimes: **S1** (single fold; produces val + test predictions) and **S3** (expanding-window CV; produces per-fold val predictions only).
- A single multi-output regression head (2 outputs: home and away scores) per learned model, with MAE (`nn.L1Loss`) as the training loss.
- A model-layer categorical encoding policy: one-hot for low-cardinality fields, learned `nn.Embedding` for high-cardinality fields.
- A training manifest recording source SHAs, config SHA, output SHAs, per-combination val MAE, and PyTorch version.
- Testing requirements that protect determinism, source-hash pinning, and prediction-coverage contracts.

Out of scope:

- **Test-slice metric computation.** Phase 4 emits S1 test predictions to disk but does not compute test MAE; Phase 5 owns that.
- **Rung selection.** Phase 4 reports per-rung val MAE; the human (or Phase 5) decides which rung is the "chosen" model.
- **Re-training on `S1.train ∪ S1.val` before test scoring.** The S1 test predictions Phase 4 emits come from a model trained on `S1.train` only. A "train on all pre-test data" recipe is a Phase 5 or follow-on amendment.
- **Rung 4 (attention over starter slots).** Deferred to a Phase 4 amendment after Phase 5/6 error analysis indicates it is justified.
- **Checkpoint persistence.** Trained weights are discarded after predictions are emitted. Re-training is cheap at this scale.
- **Multi-GPU orchestration.** Single-device training (CPU or one CUDA device) is in scope; multi-GPU / distributed training is not.
- **Cross-device byte determinism.** Byte-identical outputs are guaranteed only within the same resolved device (CPU↔CPU or CUDA↔CUDA on identical hardware). CPU and CUDA runs of the same config are expected to differ at the float-bit level.
- **Hyperparameter search (Optuna, grid sweeps, learning-rate finders).** The training config carries fixed hyperparameters; tuning is a follow-on amendment.
- **Stratified, matchup-aware, or non-temporal evaluation strategies.** Phase 4 binds to whatever Phase 3 emits.

### 1.3 Definitions and Acronyms

| Term | Definition |
|------|------------|
| Rung | One model in the ladder, indexed `0..3`. Higher rungs are progressively more expressive. |
| Learned rung | A rung whose forward pass involves trainable parameters (rungs 2 and 3). |
| Trivial rung | A rung whose predictions are closed-form summary statistics of the training labels (rungs 0 and 1). No PyTorch optimizer; deterministic by construction. |
| Feature shape | One of `flat` (Phase 2's `features_flat_2024.parquet`), `pos` (Phase 2's `features_pos_2024.parquet`), or `none` (no feature inputs — rungs 0 and 1 only). |
| Combination | A `(rung, feature_shape, strategy)` triple. Each combination is trained and predicted independently and corresponds to one output parquet. |
| Strategy | One of `S1` or `S3`, named identically to Phase 3's split-strategy contract. |
| Slice | One of `train`, `val`, `test`. Train is read-only; val and test receive predictions. |
| Fold | An S3 `(train, val)` pair, indexed `0..8` for the v1 default. |
| Training config | The hand-edited `Data/raw/training_config.yaml` declaring rungs, shapes, strategies, seed, per-rung hyperparameters, and per-categorical embedding dims. |
| Predictions artifact | A parquet at `Data/processed/predictions/<combo>.parquet` holding one prediction row per `(slice, GameId)` for S1, or `(fold_index, GameId)` for S3. |
| Training manifest | `Data/processed/training_manifest.json` — provenance for the training run. |
| Headline metric | Val MAE on the S1.val slice. Used for ladder gating; computed once per `(rung, shape)` for S1 and once per fold for S3. |
| Tiebreaker | When two `(rung, shape)` combinations are within a small margin on S1.val MAE, the S3 mean val MAE across all 9 folds is the tiebreaker. Phase 4 emits the inputs; the comparison is a human or Phase 5 act. |

### 1.4 Design Principles

- **Contract first.** The predictions parquet schemas and the manifest are stable, documented contracts. Phase 5 binds to them, not to the training internals.
- **Determinism (per-device).** Identical inputs (Phase 2 outputs + Phase 3 outputs + `training_config.yaml` + a fixed seed + a pinned PyTorch wheel) shall produce byte-identical prediction parquets when re-run on the same resolved device (same CPU machine, or same CUDA hardware + driver + CUDA version). The only permitted source of run-to-run output drift in that regime is the manifest's `build_timestamp_utc`. Cross-device runs (CPU vs CUDA, or different CUDA hardware) may differ at the float-bit level and are not gated.
- **Source-hash pinning.** The build refuses to run if any upstream artifact's on-disk SHA-256 does not match the SHA recorded in the corresponding upstream manifest. Drift fails fast; it never silently propagates into predictions.
- **One responsibility per phase.** Phase 4 trains and predicts. It does not compute test metrics, does not select the "winning" rung, and does not produce model-comparison narrative. Those are Phase 5 concerns.
- **Device-aware, single-device.** Training runs on whatever single device the resolved config selects: CPU by default on a CPU-only machine; CUDA when available and the config permits. Multi-GPU is out of scope; the model has no `DataParallel` / `DistributedDataParallel` wrapping.
- **No checkpoints in v1.** Re-training is cheap at 272 games × 4 rungs × 2 shapes × 2 strategies. Predictions are the deliverable; weights are throwaway.
- **Configuration over code edits.** Changing hyperparameters, the seed, or per-categorical embedding dims is a YAML edit. Adding or removing a rung is a code change accompanied by a config-schema change and a `training_version` bump.
- **Test slice is read-only-with-care.** Phase 4 *predicts* on S1.test (so the artifact exists for Phase 5) but never *measures* it. The manifest carries val MAE only.

### 1.5 Relationship to Phases 1, 2, 3

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
```

The Phase 4 build refuses to run if any of Phase 2's tracked outputs (`features_flat_2024.parquet`, `features_pos_2024.parquet`, `feature_vocab.json`) or Phase 3's output (`splits_2024.json`) is missing or has an on-disk SHA-256 that disagrees with its upstream manifest. This guarantees that prediction files are always pinned to a specific Phase 2 build and a specific Phase 3 split.

---

## 2. Technology Additions

| Layer | Technology | Purpose |
|-------|------------|---------|
| Training | `torch` (CPU or CUDA build, version pinned per machine in the project venv) | Defines, trains, and evaluates the learned rungs. |
| Optimizer | `torch.optim.Adam` | Sole optimizer for v1. Constant learning rate per rung. |
| Loss | `torch.nn.L1Loss` (MAE) | Sole training loss for v1. |
| Determinism | `torch.use_deterministic_algorithms(True)` + `torch.manual_seed` + `torch.cuda.manual_seed_all` (when CUDA) + `numpy.random.seed` + `random.seed` + `PYTHONHASHSEED` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` (when CUDA) | Pins all stochastic operations to the seed in `training_config.yaml` for the resolved device. |
| Device selection | `torch.cuda.is_available()` gated by `device` config field | Auto-detects CUDA when `device: "auto"` (default); falls back to CPU; honors explicit `"cpu"` / `"cuda"`. |
| Parquet I/O | `pyarrow` | Reads Phase 2 features; writes per-combination prediction parquets. |
| JSON I/O | stdlib `json` | Reads Phase 3 splits; writes the training manifest. |
| YAML I/O | `pyyaml` (`safe_load`) | Reads `Data/raw/training_config.yaml`. |
| Data wrangling | `pandas` (or `polars`, implementation choice) | Joins features against splits; constructs per-slice tensors. |

The build is executable from the repository root via `python -m nflpredictor.train` and requires no interactive input.

---

## 3. Functional Requirements

### 3.1 Input Contract

| ID | Requirement |
|----|-------------|
| TR-IN-01 | The build shall read `Data/processed/features_flat_2024.parquet`, `Data/processed/features_pos_2024.parquet`, `Data/processed/feature_vocab.json`, and `Data/processed/feature_manifest.json` as Phase 2 inputs. |
| TR-IN-02 | The build shall read `Data/processed/splits_2024.json` and `Data/processed/splits_manifest.json` as Phase 3 inputs. |
| TR-IN-03 | The build shall read `Data/raw/training_config.yaml` as a secondary input. The file is required; the build shall fail fast if it is missing. |
| TR-IN-04 | The build shall not modify any file under `Data/raw/` or any Phase 1, Phase 2, or Phase 3 output under `Data/processed/`. |
| TR-IN-05 | The build shall recompute the SHA-256 of each Phase 2 tracked output (`features_flat_2024.parquet`, `features_pos_2024.parquet`, `feature_vocab.json`) and compare each to the corresponding `output_sha256` entry in `feature_manifest.json`. On any mismatch the build shall fail fast with a clear error naming the divergent file. |
| TR-IN-06 | The build shall recompute the SHA-256 of `splits_2024.json` and compare it to the corresponding `output_sha256` entry in `splits_manifest.json`. On mismatch the build shall fail fast. |
| TR-IN-07 | The build shall reject (fail-fast) any `training_config.yaml` whose schema does not satisfy §3.2. |
| TR-IN-08 | The build shall verify that `splits_2024.json` contains both `S1` and `S3` strategies. If a required strategy is absent for any combination the config requests, the build shall fail fast. |

### 3.2 Training Config Schema

| ID | Requirement |
|----|-------------|
| TR-CFG-01 | The `training_config.yaml` schema shall include the top-level keys defined in §4.1. Unknown top-level keys shall cause the build to fail fast. |
| TR-CFG-02 | `training_version` (string) is recorded in the manifest. Any change to ladder membership, output layout, or determinism policy requires bumping this version. |
| TR-CFG-03 | `seed` shall be a non-negative integer. The same seed value shall be applied to `torch.manual_seed`, `numpy.random.seed`, and `random.seed` at the start of every learned-rung training run. |
| TR-CFG-04 | `rungs` is a non-empty list. Each entry shall be one of `mean`, `team_mean`, `linear`, `mlp`. Order is preserved in the artifact filenames and manifest. Duplicate entries shall cause the build to fail fast. |
| TR-CFG-05 | `shapes` is a non-empty list. Each entry shall be one of `flat`, `pos`. Order is preserved. Duplicate entries shall cause the build to fail fast. |
| TR-CFG-06 | `strategies` is a non-empty list. Each entry shall be one of `S1`, `S3`. Order is preserved. Duplicate entries shall cause the build to fail fast. |
| TR-CFG-07 | `linear` and `mlp` shall each be objects carrying the hyperparameters defined in §4.1. Missing required keys shall cause the build to fail fast. |
| TR-CFG-08 | `embedding_dims` shall be an object mapping high-cardinality categorical column names (per §3.6) to positive integers. The build shall fail fast if any high-cardinality column from Phase 2's vocab lacks an `embedding_dims` entry. |
| TR-CFG-09 | A reference default config shall ship in the repository (`Data/raw/training_config.yaml`) and reproduce the v1 defaults declared in §3.3. |
| TR-CFG-10 | `device` (string) shall be one of `"auto"`, `"cpu"`, or `"cuda"`. `"auto"` resolves to `"cuda"` when `torch.cuda.is_available()` is true at runtime and `"cpu"` otherwise. `"cuda"` shall fail fast at startup with a clear error if no CUDA device is available; `"cpu"` always resolves to CPU regardless of CUDA availability. The resolved device is recorded in the manifest (TR-MAN-01). |

### 3.3 v1 Default Training Config

The default `training_config.yaml` shipped in the repo declares the following. Editing this file (and only this file) is how hyperparameters, seed, or which combinations get trained change.

| Field | v1 Default |
|-------|-----------|
| `training_version` | `"v1"` |
| `seed` | `1729` |
| `device` | `"auto"` |
| `rungs` | `["mean", "team_mean", "linear", "mlp"]` |
| `shapes` | `["flat", "pos"]` |
| `strategies` | `["S1", "S3"]` |
| `linear.lr` | `0.001` |
| `linear.batch_size` | `32` |
| `linear.max_epochs` | `200` |
| `linear.early_stop_patience` | `20` |
| `mlp.lr` | `0.001` |
| `mlp.batch_size` | `32` |
| `mlp.max_epochs` | `200` |
| `mlp.early_stop_patience` | `20` |
| `mlp.hidden_dim` | `256` |
| `mlp.activation` | `"gelu"` |
| `mlp.dropout` | `0.1` |
| `embedding_dims.Archetype` | `16` |
| `embedding_dims.team_codes` | `8` |
| `embedding_dims.coaches` | `8` |
| `embedding_dims.officials` | `8` |
| `embedding_dims.positions` | `8` |
| `embedding_dims.stadium` | `8` |

### 3.4 Ladder Definition

| ID | Requirement |
|----|-------------|
| TR-RUNG-01 | **Rung 0 (`mean`).** For each strategy's per-slice training partition, the prediction shall be the pair `(mean(home_score over training rows), mean(away_score over training rows))`. The same pair is returned for every prediction row in that strategy's val and test slices. Rung 0 uses `feature_shape = none`. |
| TR-RUNG-02 | **Rung 1 (`team_mean`).** For each strategy's per-slice training partition, compute per-team `home_mean = mean(home_score over training rows where home team = T)` and `away_mean = mean(away_score over training rows where away team = T)`. The prediction for a game with home team H and away team A shall be `(home_mean[H], away_mean[A])`. If H or A is absent from the training partition, fall back to the corresponding global mean (TR-RUNG-01). Rung 1 uses `feature_shape = none`. |
| TR-RUNG-03 | **Rung 2 (`linear`).** A single `nn.Linear(d_in, 2)` over a flat input vector constructed per §3.6 from the chosen `feature_shape` (`flat` or `pos`). No nonlinearity, no hidden layer. |
| TR-RUNG-04 | **Rung 3 (`mlp`).** A two-layer MLP: `nn.Linear(d_in, hidden_dim) → activation → nn.Dropout(dropout) → nn.Linear(hidden_dim, 2)`. `hidden_dim`, `activation`, and `dropout` are taken from `training_config.yaml`. The activation shall be one of `"gelu"` or `"relu"`. |
| TR-RUNG-05 | Rungs 0 and 1 are *trivial rungs*: their implementations shall be closed-form over the training labels and shall not invoke any PyTorch optimizer. Their predictions shall be byte-identical across runs without depending on the seed. |
| TR-RUNG-06 | Rungs 2 and 3 are *learned rungs*: their training shall use the loss, optimizer, batch size, and epoch policy defined in §3.8. |
| TR-RUNG-07 | When a `(rung, feature_shape)` pair is invalid (e.g., a trivial rung with a non-`none` shape, or a learned rung with `none` shape), the build shall skip the combination silently and document the skip in the manifest. Trivial rungs shall always be run with `feature_shape = none`; learned rungs shall be run once per shape in `shapes`. |

### 3.5 Feature Shape Variants

| ID | Requirement |
|----|-------------|
| TR-SHAPE-01 | Each learned rung shall be trained independently for every entry in `shapes` (`flat`, `pos`). The two trained models share no state. |
| TR-SHAPE-02 | The `flat` shape shall read `features_flat_2024.parquet`. The `pos` shape shall read `features_pos_2024.parquet`. The build shall not synthesize features across the two files. |
| TR-SHAPE-03 | Label columns (`home_score`, `away_score`) are read from the chosen feature parquet. They are identical across `flat` and `pos`; this is verified once at startup (TR-NF-02). |
| TR-SHAPE-04 | Per-slot `_matched` flags are passed to the model as ordinary 0/1 numeric inputs. They are not masked out and do not gate the contribution of unmatched players. |

### 3.6 Categorical Encoding Policy (Model Layer)

| ID | Requirement |
|----|-------------|
| TR-CAT-01 | **Low-cardinality categoricals** (vocab size ≤ 8) shall be expanded to one-hot vectors at the model's input layer. For the v1 vocab, these are `roof`, `surface`, `day_of_week`. |
| TR-CAT-02 | **High-cardinality categoricals** (vocab size > 8) shall be passed through a per-column `nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dims[col])`. For the v1 vocab, these are `Archetype`, `team_codes`, `coaches`, `officials`, `positions`, `stadium`. |
| TR-CAT-03 | The cardinality boundary in TR-CAT-01 / TR-CAT-02 is determined by the **vocab sizes recorded in `feature_vocab.json`**, not by the categorical column's runtime distribution. Any column whose vocab grows above 8 in a future Phase 2 build automatically becomes an embedding column without a Phase 4 code change, provided `embedding_dims` carries an entry for it (TR-CFG-08). |
| TR-CAT-04 | Numeric columns (Madden numerics, game-level numerics, per-slot `_matched`, the labels) shall pass through unchanged to the concatenation step. |
| TR-CAT-05 | The model's input layer shall concatenate, in a stable column-name-sorted order: (a) all numeric columns, (b) all one-hot expansions of low-cardinality categoricals, (c) all `nn.Embedding` outputs for high-cardinality categoricals. The resulting flat vector has dimension `d_in`. |
| TR-CAT-06 | `d_in` shall be derived from Phase 2's feature parquet column set plus `feature_vocab.json`, not hard-coded. A column-set change in Phase 2 propagates automatically to the model's input layer. The low-card one-hot width per column is `vocab_size + 1` (the extra slot is the null indicator, TR-CAT-07); high-card embedding contribution per column is the configured `embedding_dim` regardless of vocab size. |
| TR-CAT-07 | The encoder shall handle Phase 2's `NULL_SENTINEL` (the integer `-1`, defined in Phase 2's `vocab.py` for legitimately-missing categorical cells — e.g., a roster slot a team doesn't fill in `features_pos`). Every categorical index shall be bumped by `+1` at the model boundary, mapping `-1 → 0` and `0..N-1 → 1..N`. Embedding tables shall have `num_embeddings = vocab_size + 1` and one-hot widths shall be `num_classes = vocab_size + 1`. The reserved index `0` ("null") gets a learnable embedding (no `padding_idx`), and contributes a fixed one-hot column for low-card categoricals. |

### 3.7 Per-Rung Model Architecture

| ID | Requirement |
|----|-------------|
| TR-MODEL-01 | All learned rungs shall produce two outputs corresponding to `(pred_home, pred_away)`. The output layer is a single `nn.Linear` with `out_features = 2`. |
| TR-MODEL-02 | All learned rungs shall be trained with `torch.nn.L1Loss` (MAE) summed over the two outputs. |
| TR-MODEL-03 | All learned rungs shall be trained with `torch.optim.Adam` at the learning rate declared per-rung in `training_config.yaml`. |
| TR-MODEL-04 | No batch normalization, layer normalization, or weight decay is used in v1. Dropout is used only in rung 3 at the rate declared in `training_config.yaml`. |
| TR-MODEL-05 | Embedding-layer weights shall be initialized by PyTorch's default `nn.Embedding` initialization (normal). Linear-layer weights shall use PyTorch's default `nn.Linear` initialization (Kaiming-uniform on weights, zero on bias). v1 does not customize initialization. |

### 3.8 Training Loop

| ID | Requirement |
|----|-------------|
| TR-TRAIN-01 | For each `(rung, shape, strategy)` combination involving a learned rung, the build shall run a training loop with the partitioning declared in §3.9. |
| TR-TRAIN-02 | The training loop shall, at the start of every learned-rung training run, set `torch.manual_seed(seed)`, `numpy.random.seed(seed)`, `random.seed(seed)`, and enable `torch.use_deterministic_algorithms(True)`. When the resolved device is CUDA, the loop shall additionally set `torch.cuda.manual_seed_all(seed)` and ensure `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set in the environment (the build process sets it before any CUDA tensor is allocated). The `seed` value is taken from `training_config.yaml`. |
| TR-TRAIN-03 | Training shall use minibatch SGD with batch size from `training_config.yaml`. Batch composition shall be deterministic across runs (e.g., shuffled by a `torch.Generator` seeded from `seed`, or processed in fixed GameId-sorted order). |
| TR-TRAIN-04 | Training shall run for at most `max_epochs` epochs. Early stopping triggers when val MAE has not improved over the running best for `early_stop_patience` consecutive epochs. The best epoch's parameters (lowest val MAE seen so far) shall be used for prediction. |
| TR-TRAIN-05 | After training a learned rung, the build shall use the best-epoch parameters to produce predictions on the appropriate val and (for S1 only) test slices, per §3.9. |
| TR-TRAIN-06 | Training shall use the single device resolved from `config.device` per TR-CFG-10. The model, optimizer state, and all per-batch tensors shall live on that device for the duration of a learned-rung training run. Multi-GPU constructs (`DataParallel`, `DistributedDataParallel`) and MPS shall not be used in v1. |
| TR-TRAIN-07 | All trained weights are discarded after predictions are written. The build shall not persist any `state_dict`, checkpoint, or serialized model object. |

### 3.9 Strategy Handling

| ID | Requirement |
|----|-------------|
| TR-STRAT-01 | **S1 strategy.** For each combination, the build shall train on `S1.train` from `splits_2024.json` and predict on both `S1.val` and `S1.test`. Both prediction sets are written into the same `<combo>__s1.parquet` file, distinguished by the `slice` column (§4.2). |
| TR-STRAT-02 | **S3 strategy.** For each combination, the build shall train one model per fold using fold `i`'s `train` partition, and predict on fold `i`'s `val` partition. Predictions across all folds are written into a single `<combo>__s3.parquet` file, distinguished by the `fold_index` column (§4.2). |
| TR-STRAT-03 | The S3 build shall not produce predictions for the S3 test slice. Phase 5 uses S1's test predictions for any test-slice evaluation. |
| TR-STRAT-04 | For trivial rungs (mean, team_mean), the same training-then-prediction protocol applies: rung 0 and rung 1 compute their summary statistics from each strategy's per-fold (or single S1) training partition. Trivial rungs produce val and (for S1) test predictions analogous to learned rungs. |

### 3.10 Output Encoding

| ID | Requirement |
|----|-------------|
| TR-OUT-01 | The build shall write one parquet file per `(rung, shape, strategy)` combination into `Data/processed/predictions/`. The filename shall be `<rung_id>__<shape>__<strategy_lower>.parquet`, where `<rung_id>` is one of `rung0_mean`, `rung1_team_mean`, `rung2_linear`, `rung3_mlp`; `<shape>` is one of `none`, `flat`, `pos`; `<strategy_lower>` is one of `s1`, `s3`. |
| TR-OUT-02 | For S1 combinations, the parquet shall have columns: `slice` (string, one of `"val"` or `"test"`), `GameId` (string), `pred_home` (float64), `pred_away` (float64). Rows shall be sorted by `(slice, GameId)` with `slice` ordering `val` before `test`. |
| TR-OUT-03 | For S3 combinations, the parquet shall have columns: `fold_index` (int8), `GameId` (string), `pred_home` (float64), `pred_away` (float64). Rows shall be sorted by `(fold_index, GameId)` with `fold_index` ascending. |
| TR-OUT-04 | The set of `GameId` values in each S1 parquet's `slice = "val"` rows shall equal `splits_2024.json → S1.val`. The set of `GameId` values in each S1 parquet's `slice = "test"` rows shall equal `splits_2024.json → S1.test`. No duplicates within a slice. |
| TR-OUT-05 | For each S3 parquet, the set of `GameId` values whose `fold_index = i` shall equal `splits_2024.json → S3.folds[i].val`. No duplicates within a fold. |
| TR-OUT-06 | Parquet output shall use compression `snappy`, row-group size `1024`, and `pyarrow` writer (the same writer profile as Phase 2 features). |
| TR-OUT-07 | The build shall create `Data/processed/predictions/` if it does not exist. Existing files in that directory whose names match the §3.10 pattern shall be overwritten; other files shall not be touched. |

### 3.11 Training Manifest

| ID | Requirement |
|----|-------------|
| TR-MAN-01 | The build shall emit `Data/processed/training_manifest.json` containing at minimum the keys: `build_timestamp_utc`, `training_version`, `seed`, `torch_version`, `device_requested` (the raw `config.device` value), `device_resolved` (`"cpu"` or `"cuda"`), `cuda_device_name` (string or `null`), `cuda_version` (string or `null`), `training_config_sha256`, `phase2_source_sha256` (object: input filename → SHA), `phase3_source_sha256` (object: input filename → SHA), `output_sha256` (object: output filename relative to `Data/processed/` → SHA), `git_commit` (or `null`), `phase2_manifest_git_commit`, `phase3_manifest_git_commit`, `training_summaries`. When `device_resolved == "cpu"`, the two `cuda_*` fields shall be `null`. |
| TR-MAN-02 | `training_summaries` shall be an object keyed by combination id (the parquet filename without extension). For S1 entries the value is `{"val_mae": <float>, "epochs_trained": <int or null>, "best_epoch": <int or null>, "stopped_early": <bool or null>}`. For S3 entries the value is `{"fold_count": <int>, "mean_val_mae": <float>, "per_fold": [{"fold_index": i, "val_mae": <float>, "epochs_trained": <int or null>, "best_epoch": <int or null>, "stopped_early": <bool or null>}, …]}`. For trivial rungs, `epochs_trained`, `best_epoch`, and `stopped_early` shall be `null`. |
| TR-MAN-03 | **No test-slice metrics shall appear in the manifest.** `training_summaries` shall expose val MAE only. Computing test MAE is a Phase 5 responsibility. |
| TR-MAN-04 | `val_mae` shall be computed as `mean(|pred_home - true_home| + |pred_away - true_away|) / 2`, with the mean taken over the games in the relevant val slice and using the rung's best-epoch parameters (for learned rungs) or its closed-form output (for trivial rungs). |
| TR-MAN-05 | The manifest's `build_timestamp_utc` shall be in ISO 8601 UTC format. |
| TR-MAN-06 | The manifest shall be written **last** — after every prediction parquet — so each output SHA can be computed against the on-disk file. |
| TR-MAN-07 | The manifest shall be written with sorted keys and stable two-space indentation. Byte-identical inputs and a pinned PyTorch wheel shall produce byte-identical manifests modulo the timestamp. |
| TR-MAN-08 | `training_config_sha256` shall be the SHA-256 of the raw `Data/raw/training_config.yaml` file bytes (not the parsed/normalized form). This mirrors Phases 1–3's source-hash discipline. |

---

## 4. Data Model

### 4.1 `Data/raw/training_config.yaml`

```yaml
training_version: "v1"

seed: 1729
device: "auto"   # one of "auto" | "cpu" | "cuda"

rungs:      [mean, team_mean, linear, mlp]
shapes:     [flat, pos]
strategies: [S1, S3]

linear:
  lr: 0.001
  batch_size: 32
  max_epochs: 200
  early_stop_patience: 20

mlp:
  lr: 0.001
  batch_size: 32
  max_epochs: 200
  early_stop_patience: 20
  hidden_dim: 256
  activation: "gelu"
  dropout: 0.1

embedding_dims:
  Archetype:   16
  team_codes:   8
  coaches:      8
  officials:    8
  positions:    8
  stadium:      8
```

### 4.2 `Data/processed/predictions/<combo>.parquet` Structures

**S1 schema** (filename ends with `__s1.parquet`):

| Column | Type | Description |
|--------|------|-------------|
| `slice` | string | One of `"val"`, `"test"`. |
| `GameId` | string | Phase 1 / Phase 2 `GameId`. |
| `pred_home` | float64 | Predicted home score. |
| `pred_away` | float64 | Predicted away score. |

Row order: `slice` ascending (`val` first, then `test`); within each slice, `GameId` ascending.

**S3 schema** (filename ends with `__s3.parquet`):

| Column | Type | Description |
|--------|------|-------------|
| `fold_index` | int8 | Phase 3 fold index, `0..fold_count-1`. |
| `GameId` | string | Phase 1 / Phase 2 `GameId`. |
| `pred_home` | float64 | Predicted home score. |
| `pred_away` | float64 | Predicted away score. |

Row order: `fold_index` ascending; within each fold, `GameId` ascending.

### 4.3 `Data/processed/training_manifest.json` Structure

```json
{
  "build_timestamp_utc": "2026-05-20T13:45:00Z",
  "training_version": "v1",
  "seed": 1729,
  "torch_version": "2.12.0+cpu",
  "device_requested": "auto",
  "device_resolved": "cpu",
  "cuda_device_name": null,
  "cuda_version": null,
  "training_config_sha256": "…",
  "phase2_source_sha256": {
    "features_flat_2024.parquet": "…",
    "features_pos_2024.parquet": "…",
    "feature_vocab.json": "…"
  },
  "phase3_source_sha256": {
    "splits_2024.json": "…"
  },
  "output_sha256": {
    "predictions/rung0_mean__none__s1.parquet": "…",
    "predictions/rung0_mean__none__s3.parquet": "…",
    "…": "…"
  },
  "git_commit": "…",
  "phase2_manifest_git_commit": "…",
  "phase3_manifest_git_commit": "…",
  "training_summaries": {
    "rung0_mean__none__s1": {
      "val_mae": 9.87,
      "epochs_trained": null, "best_epoch": null, "stopped_early": null
    },
    "rung2_linear__flat__s1": {
      "val_mae": 8.93,
      "epochs_trained": 87, "best_epoch": 67, "stopped_early": true
    },
    "rung3_mlp__pos__s3": {
      "fold_count": 9,
      "mean_val_mae": 8.71,
      "per_fold": [
        {"fold_index": 0, "val_mae": 9.12, "epochs_trained": 73, "best_epoch": 53, "stopped_early": true},
        "…"
      ]
    }
  }
}
```

### 4.4 Output Files

| File | Type | Description |
|------|------|-------------|
| `Data/processed/predictions/<combo>__s1.parquet` × 6 | Parquet | One per `(rung, shape)` combination with `strategy = S1`. Holds val + test predictions. |
| `Data/processed/predictions/<combo>__s3.parquet` × 6 | Parquet | One per `(rung, shape)` combination with `strategy = S3`. Holds per-fold val predictions. |
| `Data/processed/training_manifest.json` | JSON | Build provenance + per-combination val MAE summary. |

The 12 prediction parquets are enumerated as: trivial rungs `{rung0_mean, rung1_team_mean}` × `{none}` × `{s1, s3}` = 4 files; learned rungs `{rung2_linear, rung3_mlp}` × `{flat, pos}` × `{s1, s3}` = 8 files. The exact count is a function of the config; trimming `rungs`, `shapes`, or `strategies` in `training_config.yaml` reduces the output set proportionally without a code change.

---

## 5. Build Pipeline Design

The pipeline is described conceptually; implementation may organize it differently as long as §3's contract is honored. Recommended module layout, mirroring Phase 1's `databuild/`, Phase 2's `features/`, and Phase 3's `splits/`:

```
src/nflpredictor/train/
    __init__.py
    __main__.py        # entry point: python -m nflpredictor.train
    pipeline.py        # top-level orchestration
    config.py          # training_config.yaml load + validation
    sources.py         # Phase 2 + Phase 3 input load + SHA verification
    encoders.py        # categorical encoding policy (TR-CAT-01..06)
    models.py          # rung definitions (mean, team_mean, linear, mlp)
    train_loop.py      # learned-rung optimizer + early stopping
    predict.py         # closed-form + best-epoch prediction emission
    outputs.py         # parquet writers
    manifest.py        # training_manifest.json construction
```

### 5.1 Conceptual Stages

1. **Load and validate inputs.** Read `training_config.yaml`; validate per §3.2 and §4.1. Read `feature_manifest.json` and `splits_manifest.json`; verify Phase 2 and Phase 3 output SHAs against on-disk files (TR-IN-05, TR-IN-06).
2. **Load features and splits into memory.** Read both Phase 2 parquets and `splits_2024.json`. Verify labels are identical across `flat` and `pos` parquets (TR-NF-02).
3. **Build the encoder.** From `feature_vocab.json` + the chosen feature parquet's column set + `embedding_dims`, construct the input-layer encoder per §3.6 and derive `d_in`.
4. **Enumerate combinations.** Cartesian product `rungs × shapes × strategies`, filtered per TR-RUNG-07 (trivial rungs only with `shape = none`; learned rungs only with `shape ∈ {flat, pos}`).
5. **Per combination, train and predict.**
   - For trivial rungs: compute closed-form summary stats from the strategy's training partition(s); emit predictions for val (and test, for S1).
   - For learned rungs: seed RNGs (TR-TRAIN-02); for each fold (S3) or once (S1), instantiate the model, run the early-stopping training loop, predict using best-epoch params.
6. **Compute val MAE.** Per combination (S1) or per fold (S3), record the val MAE that goes into the manifest.
7. **Emit prediction parquets.** Per §3.10 and §4.2.
8. **Emit manifest.** Compute output SHAs and per-combination summaries; write `training_manifest.json` last.

### 5.2 Determinism Boundaries

- The only non-deterministic input is the wall-clock timestamp written to `training_manifest.json`.
- Trivial rungs (0, 1) are deterministic by construction and device-independent (no PyTorch involved).
- Learned rungs (2, 3) are deterministic given (a) the same `training_config.yaml` (including `seed` and resolved `device`), (b) the same Phase 2 and Phase 3 inputs, (c) the same pinned PyTorch wheel, (d) the same resolved device — same CPU machine, or same CUDA hardware + driver + CUDA version — with `torch.use_deterministic_algorithms(True)` enabled. Across these conditions, predictions and the manifest (modulo timestamp) shall be byte-identical.
- Cross-device runs (CPU vs CUDA, or different CUDA hardware) may differ at the float-bit level. The manifest records `device_resolved`, `cuda_device_name`, and `cuda_version` so a downstream consumer can tell whether two artifact sets are comparable byte-for-byte.
- Any drift in Phase 2 or Phase 3 produces a TR-IN-05 / TR-IN-06 hash mismatch and a hard failure rather than silent prediction drift.
- PyTorch version drift is **not** masked: `torch_version` is recorded in the manifest, and changing the wheel may legitimately change byte output. Such a change is treated as a deliberate environment update and should accompany a project-level note (CLAUDE.md or similar).

---

## 6. Integration / Endpoint / Tooling Design

Not applicable. The training build is a single-shot offline script with no network surface, no API, and no UI.

---

## 7. Changes to Existing Requirements

`Docs/Idea.md` §"Phase 4: Baseline & Model Ladder" enumerated open questions. The decisions encoded in this spec are:

- **Rung set in v1**: rungs 0–3 (mean, team-mean, `nn.Linear`, small MLP). Rung 4 (attention over starter slots) is deferred to a Phase 4 amendment; see §12.
- **Feature shape**: both `flat` and `pos` ship in v1, trained as independent combinations per learned rung. The choice between them is an empirical comparison Phase 5 will surface, not a Phase 4 commitment.
- **Regression head**: single multi-output `nn.Linear(_, 2)` predicting `(home_score, away_score)` jointly. Two-headed-independent models are out of scope for v1.
- **Loss function**: MAE (`nn.L1Loss`). MSE and Huber are not used in v1.
- **Split-strategy consumption**: both `S1` and `S3` are consumed. S1's val MAE is the headline. S3's per-fold mean val MAE is the tiebreaker for close `(rung, shape)` comparisons. The test slice is touched once at project end (in Phase 5), via S1's emitted test predictions.
- **Model-layer categorical encoding**: one-hot for low-cardinality (`roof`, `surface`, `day_of_week`), learned `nn.Embedding` for high-cardinality (`Archetype`, `team_codes`, `coaches`, `officials`, `positions`, `stadium`). Cardinality boundary `≤ 8` driven by `feature_vocab.json`, not by source-data inspection.
- **Output artifacts**: predictions only; no checkpoints in v1. Twelve parquets covering every `(rung, shape, strategy)` combination, plus a single `training_manifest.json`.

The Phase 1, 2, and 3 specs are unchanged. `Docs/Idea.md` §"Phase 4" requires a follow-on edit to mark the open questions as resolved; this spec is the source of truth in the interim.

---

## 8. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| TR-NF-01 | The build shall be deterministic per device: identical inputs (Phase 2 outputs, Phase 3 outputs, `training_config.yaml`), a pinned PyTorch wheel, and the same resolved device (same CPU machine, or same CUDA hardware + driver + CUDA version) shall produce byte-identical prediction parquets and a byte-identical manifest (excluding `build_timestamp_utc`) across runs. Cross-device byte equality is not required. |
| TR-NF-02 | At startup, the build shall verify that `features_flat_2024.parquet` and `features_pos_2024.parquet` carry identical `(GameId, home_score, away_score)` triples (same set, same label values). On mismatch the build shall fail fast. |
| TR-NF-03 | The build shall be re-runnable: it shall produce correct output regardless of whether the prediction parquets or manifest exist, are stale, or are absent. Stale outputs in `Data/processed/predictions/` matching the §3.10 filename pattern shall be overwritten. |
| TR-NF-04 | The build shall complete in under 5 minutes for the v1 ladder against the 272-game 2024 universe on a CUDA training machine (single GPU, modern PyTorch). CPU runs of the full v1 config are expected to take ~30–60 minutes on a modern 8-core laptop and are not gated by this requirement; CPU is the development/testing path, not the routine training path. |
| TR-NF-05 | The `build_timestamp_utc` field in `training_manifest.json` is the only permitted source of run-to-run output drift under fixed inputs and PyTorch wheel. |
| TR-NF-06 | The build shall emit a non-zero exit code on any fatal error. |
| TR-NF-07 | The build shall log per-combination val MAE (and for S3, per-fold val MAE and mean val MAE) to stdout in a stable, scannable format so the ladder summary is visible without opening the manifest. |
| TR-NF-08 | All JSON outputs shall use UTF-8 encoding with Unix line endings (`\n`). |
| TR-NF-09 | All Parquet outputs shall use the `pyarrow` writer with `snappy` compression and the row-group profile declared in TR-OUT-06. |
| TR-NF-10 | The build shall not download any model weights, dataset, or other resource at runtime. All inputs are local files. |

---

## 9. UI Requirements

Not applicable.

---

## 10. Testing Requirements

| ID | Requirement |
|----|-------------|
| TR-TEST-01 | A unit test shall verify `training_config.yaml` validation against: (a) the default v1 config (must accept); (b) an unknown top-level key (must reject); (c) `rungs = []` (must reject); (d) a duplicate entry in `rungs`, `shapes`, or `strategies` (must reject); (e) a `rungs` entry not in `{mean, team_mean, linear, mlp}` (must reject); (f) a missing `embedding_dims` entry for a high-cardinality categorical from `feature_vocab.json` (must reject); (g) a negative `seed` (must reject); (h) an unsupported `mlp.activation` value (must reject); (i) a `device` value outside `{"auto", "cpu", "cuda"}` (must reject). |
| TR-TEST-02 | A unit test shall verify the categorical encoding policy (§3.6) on a synthetic vocab: columns with `len(vocab) ≤ 8` go through one-hot; columns with `len(vocab) > 8` go through `nn.Embedding` with the configured dim; the resulting `d_in` matches the sum of numeric width + one-hot widths + embedding dims. |
| TR-TEST-03 | A unit test shall verify the trivial-rung implementations on a small synthetic label fixture: rung 0 returns `(mean(home_train), mean(away_train))` for every prediction row; rung 1 returns per-team means with global-mean fallback for unseen teams; both rungs are exactly byte-identical across two consecutive invocations. |
| TR-TEST-04 | A unit test shall verify the learned-rung training loop: a 5-game synthetic dataset run for 3 epochs produces a finite val MAE; early stopping kicks in when val MAE plateaus; the best-epoch parameters (not the final-epoch parameters) are used for prediction. |
| TR-TEST-05 | A unit test shall verify the prediction parquet schemas (§4.2): S1 files have columns `(slice, GameId, pred_home, pred_away)` and contain exactly the expected `GameId` sets per slice (TR-OUT-04); S3 files have columns `(fold_index, GameId, pred_home, pred_away)` and exactly cover the expected per-fold sets (TR-OUT-05); rows are sorted per TR-OUT-02 / TR-OUT-03. |
| TR-TEST-06 | An integration test shall run the full Phase 4 build against a small synthetic Phase 2 + Phase 3 stand-in (~30 games) and assert that the prediction parquets and manifest match a checked-in expected snapshot (excluding `build_timestamp_utc`). |
| TR-TEST-07 | A determinism test shall run `run_training_build` twice in succession against identical Phase 2, Phase 3, and config inputs and assert byte-equality of all twelve prediction parquets and the manifest (excluding `build_timestamp_utc`) in the same pinned PyTorch wheel on CPU. |
| TR-TEST-08 | A pinned-identity test shall run the real Phase 4 build against the actual Phase 2 and Phase 3 outputs and assert: (a) all 12 expected parquet filenames exist; (b) each S1 parquet covers exactly the `S1.val` and `S1.test` GameId sets; (c) each S3 parquet covers exactly the expected per-fold val GameId sets across all 9 folds; (d) the manifest carries a val MAE for every combination and no test MAE for any combination. |
| TR-TEST-09 | A source-pinning test shall verify that a TR-IN-05 hash mismatch (e.g., a hand-edited `features_flat_2024.parquet`) and a TR-IN-06 hash mismatch (e.g., a hand-edited `splits_2024.json`) each cause the build to fail fast with a clear error naming the divergent file. |
| TR-TEST-10 | A test shall verify that the manifest's `training_summaries` block contains no key whose value mentions `test_mae`, enforcing TR-MAN-03. |

---

## 11. Security Considerations

| ID | Consideration |
|----|---------------|
| TR-SEC-01 | The build operates exclusively on local files under the repository root. It shall not make any network requests, including PyTorch Hub downloads or telemetry. |
| TR-SEC-02 | The build shall not write to any path outside `Data/processed/`. |
| TR-SEC-03 | No input or output of this build contains credentials, PII beyond publicly available game schedules and player names, or other sensitive data. |
| TR-SEC-04 | `training_config.yaml` is user-editable; the build validates its schema (§3.2) but is not required to defend against adversarial input — the only consumer is the build script in the same repository. YAML loading shall use `yaml.safe_load`. |
| TR-SEC-05 | The build shall not deserialize any pickle, joblib, or `torch.load` checkpoint at runtime. v1 reads only parquet, JSON, and YAML. |

---

## 12. Future Considerations

The following are explicitly out of scope for Phase 4 v1 and recorded so they are not lost:

- **Rung 4 (attention over starter slots).** Adding a permutation-invariant set encoder over the 22 home + 22 away starter vectors becomes a Phase 4 amendment if Phase 5/6 error analysis indicates that rung 3 has plateaued and starter-set structure is the binding constraint. Requires either a third Phase 2 shape (`set`) or a model-side reshape from `flat`/`pos`.
- **Re-training on `S1.train ∪ S1.val` before test scoring.** Currently the S1 test predictions Phase 4 emits come from a model trained only on `S1.train`. A follow-on amendment may add an optional re-train step that consumes `S1.train ∪ S1.val` and re-emits the test parquet rows.
- **Hyperparameter search.** Optuna, grid sweeps, or a learning-rate finder would slot in as a follow-on amendment with a `training_version` bump. v1 carries fixed hyperparameters in the config.
- **Checkpoint persistence.** If a downstream phase (Phase 7 error analysis, or external sharing) needs the trained weights, persisting `state_dict` per learned combination is a follow-on amendment. v1 deliberately omits this; re-training is cheap.
- **Multi-GPU / distributed training.** Out of scope for v1. Single-device (CPU or one CUDA device) is sufficient at 272 games; `DataParallel` / `DistributedDataParallel` would be follow-on amendments if the dataset grows enough to justify them.
- **MPS (Apple Silicon GPU) backend.** Out of scope for v1. Determinism guarantees on MPS are weaker than CPU or CUDA; revisit if a contributor specifically asks for it.
- **Per-rung loss functions.** v1 standardizes on MAE. Mixing MSE/Huber across rungs to compare loss-shape effects would be a Phase 7-flavored experiment, not a Phase 4 v1 requirement.
- **Two-headed-independent models** (one model per side instead of a single multi-output model). Considered and declined for v1; shared representation matches the standard pattern and is cheaper to maintain.
- **Cross-PyTorch-wheel byte determinism.** Not promised. The manifest's `torch_version` records the wheel used; changing it is a deliberate environment update.
- **Multi-season training.** When Phase 2 outputs cover more than 2024, the build will need a `--season` flag or per-season config wiring. v1 stays 2024-only.

---

## 13. References

- [Idea.md](./Idea.md) — Source idea document; §"Phase 4: Baseline & Model Ladder" enumerates the open questions this spec resolves.
- [Spec-Phase1-DataBuild.md](./Spec-Phase1-DataBuild.md) — Phase 1 specification.
- [Spec-Phase2-FeatureEngineering.md](./Spec-Phase2-FeatureEngineering.md) — Phase 2 specification. Phase 4 reads its outputs and binds to its `feature_manifest.json`.
- [Spec-Phase3-Splits.md](./Spec-Phase3-Splits.md) — Phase 3 specification. Phase 4 reads its splits artifact and binds to its `splits_manifest.json`.
- [Plan-Phase1-DataBuild.md](./Plan-Phase1-DataBuild.md), [Plan-Phase2-FeatureEngineering.md](./Plan-Phase2-FeatureEngineering.md), [Plan-Phase3-Splits.md](./Plan-Phase3-Splits.md) — Stylistic precedent for the Phase 4 implementation plan to follow.
- `Data/processed/features_flat_2024.parquet`, `Data/processed/features_pos_2024.parquet`, `Data/processed/feature_vocab.json` — Phase 2 outputs; primary inputs to Phase 4.
- `Data/processed/splits_2024.json` — Phase 3 output; secondary input to Phase 4.
- `Data/processed/feature_manifest.json`, `Data/processed/splits_manifest.json` — Upstream manifests; consulted for source-hash pinning per TR-IN-05 and TR-IN-06.
- [CLAUDE.md](../CLAUDE.md) — Repository-level notes on venv, dataset shape, and the Phase 1–3 build conventions Phase 4 mirrors.
