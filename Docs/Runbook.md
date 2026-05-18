# NFLPredictor Runbook — End-to-End Build on a Fresh Machine

This document walks through running every build phase (1 → 2 → 3 → 4) and the full test suite on a fresh checkout. It's the canonical test-on-another-machine recipe — useful when validating a CUDA training machine, an additional dev machine, or a CI environment.

If you only want to run a single phase, see [README.md](../README.md). If you want to understand *what* each phase does, see the per-phase spec/plan in `Docs/`.

---

## 1. Prerequisites

- **Python 3.11+** (3.12 is what the dev machine uses; either works).
- **git**.
- **~2 GB free disk** for the venv (mostly PyTorch + pyarrow).
- Internet access for the initial `pip install` only — every build step is fully offline after that.
- **For CUDA training**: an NVIDIA GPU with a current driver, plus the CUDA toolkit version that matches the PyTorch wheel you install in §2.3. The CPU dev machine and the CUDA machine will hold *different* PyTorch wheels in their respective venvs; this is expected. (Per-device determinism, not cross-device.)

---

## 2. First-time setup

### 2.1 Clone and create the venv

```bash
git clone https://github.com/mufaka/NFLPredictor.git
cd NFLPredictor

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
```

### 2.2 Install dependencies

For everything *except* PyTorch:

```bash
pip install -e ".[dev]"
```

This installs `pandas`, `pyarrow`, `pyyaml`, `rapidfuzz`, `pytest`, plus the project itself in editable mode. PyTorch comes from `pyproject.toml`'s `torch>=2.4` dep and will pull the default wheel from PyPI — which on Linux/macOS is a **CUDA-bundled wheel** that's big and may not match your CUDA toolkit. Step §2.3 replaces it with the right wheel for your machine.

### 2.3 Install the correct PyTorch wheel for this machine

The dev machine ran on CPU. The CUDA training machine should install a CUDA wheel matching its driver/CUDA version.

**CPU-only machine:**

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 12.4 machine (adjust the URL for your CUDA version):**

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
```

Other CUDA versions: see <https://pytorch.org/get-started/locally/> for the right index URL (`cu121`, `cu118`, `cu126`, etc.).

Verify the install matched your hardware:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"
```

- CPU machine: `cuda available: False`.
- CUDA machine: `cuda available: True` (and `torch.__version__` ends in `+cuXXX`).

If a CUDA machine reports `cuda available: False`, the wheel doesn't match the driver — re-run §2.3 with a different `cuXXX` URL.

---

## 3. Sanity-check the test suite first

Before kicking off the build, confirm the imports and unit tests pass:

```bash
pytest -q
```

Expected on a fresh checkout (with `Data/processed/predictions/` still empty):

```
436 passed, 4 skipped in ~110s
```

The 4 skipped tests are `tests/test_train_pipeline_run.py` — they activate once Phase 4 has produced real-data predictions.

Two notes on the test suite:
- The integration + determinism tests for Phase 4 (`test_train_integration.py`, `test_train_determinism.py`) force `device: "cpu"` in their fixture config, so they run identically on CUDA and CPU machines. They should pass on any machine with the same major PyTorch version as the dev machine wheels. If they fail with a byte-mismatch on a different machine, regenerate the fixture once (§7).
- The full suite takes ~2 minutes because the integration test runs the full 12-combination pipeline against a 36-game synthetic fixture twice (once for byte-equality vs expected, once for determinism vs a second run).

---

## 4. Run each build phase in order

Each phase reads checked-in inputs from `Data/raw/` and writes to `Data/processed/`. Every phase is deterministic and re-runnable; running a phase twice produces byte-identical outputs (modulo each manifest's `build_timestamp_utc`).

Run from the repo root with the venv active.

### 4.1 Phase 1 — Data Build

```bash
python -m nflpredictor.databuild
```

**Inputs:** `Data/raw/{box_scores_2024.csv, maddennfl24fullplayerratings.csv, player_overrides.csv}`.

**Outputs:** `Data/processed/{madden_2024.csv, box_scores_2024.csv, player_id_mapping.csv, build_manifest.json}`.

**Expected wall-clock:** < 30 seconds on any modern machine.

**Quick verify:**

```bash
ls -la Data/processed/{madden_2024.csv,box_scores_2024.csv,player_id_mapping.csv,build_manifest.json}
python -c "import json; m=json.load(open('Data/processed/build_manifest.json')); print('matched tiers:', m['counts'])"
```

### 4.2 Phase 2 — Feature Engineering

```bash
python -m nflpredictor.features
```

Refuses to run unless Phase 1's outputs match the SHAs recorded in `build_manifest.json` (which `python -m nflpredictor.databuild` just produced).

**Outputs:** `Data/processed/{features_flat_2024.parquet, features_pos_2024.parquet, feature_vocab.json, feature_manifest.json}`.

**Expected wall-clock:** ~1–2 seconds.

**Quick verify:**

```bash
python -c "
import pyarrow.parquet as pq
f = pq.ParquetFile('Data/processed/features_flat_2024.parquet')
p = pq.ParquetFile('Data/processed/features_pos_2024.parquet')
print('flat:', f.metadata.num_rows, 'rows x', f.metadata.num_columns, 'cols')
print('pos:', p.metadata.num_rows, 'rows x', p.metadata.num_columns, 'cols')
"
```

Expected: `flat: 272 rows x 202 cols`, `pos: 272 rows x 258 cols`.

### 4.3 Phase 3 — Splits

```bash
python -m nflpredictor.splits
```

Refuses to run unless `features_flat_2024.parquet` matches `feature_manifest.json`.

**Outputs:** `Data/processed/{splits_2024.json, splits_manifest.json}`.

**Expected wall-clock:** < 1 second.

**Quick verify:**

```bash
python -c "
import json
s = json.load(open('Data/processed/splits_2024.json'))
print('S1: train=%d val=%d test=%d' % (len(s['S1']['train']), len(s['S1']['val']), len(s['S1']['test'])))
print('S3: %d folds, test=%d' % (len(s['S3']['folds']), len(s['S3']['test'])))
"
```

Expected: `S1: train=179 val=45 test=48`, `S3: 9 folds, test=48`.

### 4.4 Phase 4 — Baseline & Model Ladder

```bash
python -m nflpredictor.train
```

This is the heavy phase. Refuses to run unless every Phase 2 output and `splits_2024.json` match their upstream manifest hashes.

**Outputs:** `Data/processed/predictions/<rung>__<shape>__<strategy>.parquet` × 12, plus `Data/processed/training_manifest.json`.

**Expected wall-clock:**
- **CUDA (target path):** < 5 minutes (TR-NF-04 target).
- **CPU:** ~30–60 minutes — works, but not the routine training path.

**Watch the stdout log** for the device line and per-combination summary:

```
device: requested=auto resolved=cuda cuda=NVIDIA RTX 4090 cuda_version=12.4
enumerated 12 combinations:
  rung0_mean__none__s1                      S1 val_mae=...  (trivial)
  rung0_mean__none__s3                      S3 folds=9  mean_val_mae=...
  rung1_team_mean__none__s1                 S1 val_mae=...  (trivial)
  ...
  rung3_mlp__pos__s3                        S3 folds=9  mean_val_mae=...
wrote manifest: .../Data/processed/training_manifest.json
```

**Quick verify after the run:**

```bash
ls Data/processed/predictions/ | wc -l       # expect: 12
python -c "
import json
m = json.load(open('Data/processed/training_manifest.json'))
print('device_resolved:', m['device_resolved'])
print('torch_version:', m['torch_version'])
print('cuda_device_name:', m.get('cuda_device_name'))
print('cuda_version:', m.get('cuda_version'))
print('combos in manifest:', len(m['training_summaries']))
assert 'test_mae' not in json.dumps(m), 'TR-MAN-03 violated'
print('TR-MAN-03 (no test_mae anywhere) OK')
"
```

Expected: `combos in manifest: 12`, `TR-MAN-03 ... OK`.

---

## 5. Activate the pinned-identity tests against real data

Once `Data/processed/predictions/` exists, the 4 previously-skipped `test_train_pipeline_run.py` tests activate automatically. Re-run the suite:

```bash
pytest tests/test_train_pipeline_run.py -v
```

Expected:

```
tests/test_train_pipeline_run.py::test_all_twelve_expected_parquets_exist PASSED
tests/test_train_pipeline_run.py::test_s1_parquets_cover_exact_val_and_test_gameid_sets PASSED
tests/test_train_pipeline_run.py::test_s3_parquets_cover_exact_per_fold_val_gameid_sets PASSED
tests/test_train_pipeline_run.py::test_manifest_carries_val_mae_for_every_combination PASSED
```

These assert structural contracts only (file presence, exact GameId coverage per slice, val_mae present per combo, no test_mae anywhere). They do not pin specific `val_mae` numeric values — those will differ between CPU and CUDA runs by design.

Then re-run the full suite:

```bash
pytest -q
```

Expected: `440 passed in ~110s` (4 previously-skipped tests now passing).

---

## 6. Determinism re-run

To convince yourself the build is deterministic on this machine, run Phase 4 twice and compare the prediction parquets:

```bash
# Snapshot the first run.
mkdir -p /tmp/preds_a
cp Data/processed/predictions/*.parquet /tmp/preds_a/
cp Data/processed/training_manifest.json /tmp/manifest_a.json

# Re-run.
python -m nflpredictor.train

# Compare.
diff -rq /tmp/preds_a Data/processed/predictions/   # should print nothing
python -c "
import json
a = json.load(open('/tmp/manifest_a.json'))
b = json.load(open('Data/processed/training_manifest.json'))
a.pop('build_timestamp_utc')
b.pop('build_timestamp_utc')
print('manifest equal modulo timestamp:', a == b)
"
```

Expected: `diff` prints nothing; the manifest comparison prints `True`. If either fails, the build has a hidden non-determinism source — file an issue.

---

## 7. (Optional) Regenerate the synthetic test fixture

The integration + determinism tests run against `tests/fixtures/train/expected/`, which was generated on the dev CPU machine. If they fail on this machine with byte-mismatch errors, regenerate the fixture so it matches this machine's PyTorch wheel:

```bash
python -m tests.fixtures.train._regenerate
pytest tests/test_train_integration.py tests/test_train_determinism.py -v
```

The regen script forces `device: "cpu"` in the fixture's training config so the fixture is portable across CPU/CUDA machines, but **CPU computation is only bit-exact across machines if the same PyTorch wheel + CPU architecture combination is used**. Different wheels (CPU vs CUDA), different CPU vendors (Intel vs AMD vs ARM), or different SIMD/BLAS backends can introduce float-level diffs.

If you regenerated the fixture, commit the updated `tests/fixtures/train/expected/` so the next person doesn't see a mismatch.

---

## 8. Wall-clock budget summary

| Phase | Expected runtime (CPU) | Expected runtime (CUDA) |
|-------|------------------------|--------------------------|
| 1. databuild | < 30 s | n/a (no GPU code) |
| 2. features | ~1–2 s | n/a |
| 3. splits | < 1 s | n/a |
| 4. train | ~30–60 min | < 5 min |
| Full pytest suite | ~110 s + Phase 4 inputs | ~110 s + Phase 4 inputs |

The 5-min CUDA budget (TR-NF-04) is the spec target. If Phase 4 on CUDA takes much longer than that, double-check `device_resolved` in the training manifest — `cpu` would mean the auto-detect fell back.

---

## 9. Troubleshooting

**`Phase 2 manifest not found` or `Phase 3 manifest not found`**: you skipped a phase. Run them in order (1 → 2 → 3 → 4).

**`hash mismatch` on phase startup**: you modified a `Data/processed/` file by hand (or a previous build crashed mid-write). Re-run the offending upstream phase to regenerate.

**`device='cuda' but no CUDA device is available`**: explicitly set `device: "auto"` in `Data/raw/training_config.yaml` (and don't pass an env override) so the build falls back to CPU automatically, OR fix the CUDA install per §2.3.

**Integration test byte-mismatch on a non-dev machine**: see §7.

**`torch.use_deterministic_algorithms(True)` errors at runtime**: the version of PyTorch installed lacks deterministic implementations of some op the model uses on your hardware. Upgrade the PyTorch wheel or open an issue with the exact op name from the traceback.

**Stale `Data/processed/predictions/` from a previous run**: Phase 4 overwrites every file in `predictions/` that matches its naming pattern (`<rung>__<shape>__<strategy>.parquet`) but does not delete unrelated files. If you've renamed rungs/shapes/strategies in the config and want a clean slate: `rm -rf Data/processed/predictions/ Data/processed/training_manifest.json` and re-run Phase 4.

---

## 10. What to send back from the test run

If the CUDA test passes cleanly, the things worth sharing back to the project are:

- The full stdout log from `python -m nflpredictor.train` (per-combination val MAE summary + the resolved device line).
- `Data/processed/training_manifest.json` (small JSON; carries the device + CUDA metadata + per-combination summaries).
- Confirmation that `pytest -q` reports `440 passed`.

The 12 `Data/processed/predictions/*.parquet` files are the canonical training output — commit them on the CUDA machine if that's where the project's official Phase 4 artifacts will live, or send them back to the dev machine for a single commit.

---

## References

- [README.md](../README.md) — per-phase summary
- [Spec-Phase1-DataBuild.md](./Spec-Phase1-DataBuild.md) / [Plan-Phase1-DataBuild.md](./Plan-Phase1-DataBuild.md)
- [Spec-Phase2-FeatureEngineering.md](./Spec-Phase2-FeatureEngineering.md) / [Plan-Phase2-FeatureEngineering.md](./Plan-Phase2-FeatureEngineering.md)
- [Spec-Phase3-Splits.md](./Spec-Phase3-Splits.md) / [Plan-Phase3-Splits.md](./Plan-Phase3-Splits.md)
- [Spec-Phase4-BaselineLadder.md](./Spec-Phase4-BaselineLadder.md) / [Plan-Phase4-BaselineLadder.md](./Plan-Phase4-BaselineLadder.md)
- [CLAUDE.md](../CLAUDE.md) — Agent-facing project notes
