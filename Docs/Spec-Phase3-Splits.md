# Phase 3: Splits Specification

## 1. Introduction

### 1.1 Purpose

This specification defines the **Splits** phase of the NFL Predictor project. Phase 3 partitions the 2024 game universe into deterministic train, validation, and test slices and emits a small split-assignment artifact that every downstream phase (baselines, model ladder, evaluation, error analysis) binds to.

Phases 1 and 2 are implemented and frozen; this spec assumes their outputs as given. Phases 4–6 remain exploratory in `Docs/Idea.md` and are intentionally not specified yet — Phase 3 closes the contract one layer up the stack.

### 1.2 Scope

In scope:

- A third deterministic build step (separate from Phases 1 and 2) that reads Phase 2's feature matrices and a hand-edited `Data/raw/splits_config.yaml` and writes a split-assignment artifact + manifest into `Data/processed/`.
- Two split strategies emitted in parallel: **S1** (single fixed train/val/test partition by week boundary) and **S3** (expanding-window cross-validation, with the test slice held fixed).
- Configurable week boundaries; the v1 default is Train Weeks 1–12 / Val Weeks 13–15 / Test Weeks 16–18.
- A splits manifest recording source SHAs, config SHA, output SHAs, and per-strategy game counts.
- Testing requirements that protect determinism and split disjointness.

Out of scope:

- Any model code, training loop, or metric computation. (Phase 4 and beyond.)
- Splits on data other than the 2024 season. (Multi-season is a future amendment.)
- Re-running Phase 1 or Phase 2. The split build assumes Phase 2's outputs are present and trusts their `feature_manifest.json`.
- Stratified, matchup-aware, or non-temporal split strategies.

### 1.3 Definitions and Acronyms

| Term | Definition |
|------|------------|
| Game universe | The 272 distinct `GameId` values present in `features_flat_2024.parquet`. Phase 3 partitions this universe; it does not invent or filter games. |
| Split role | One of `train`, `val`, `test`. Under S1, every game has exactly one role. |
| **S1** | Single-fold strategy. One fixed train/val/test partition, defined by week boundaries in the config. |
| **S3** | Expanding-window cross-validation strategy. The test slice is identical to S1's. Train and val rotate across folds within the early-season region. |
| Fold | A single `(train, val)` pair within S3. The number of folds is a deterministic function of the config's boundaries. |
| Week | The integer NFL week (1–18) for a game, read from the `week` column of the Phase 2 feature matrix. Phase 3 does not re-derive week from `GameDate`. |
| Splits config | The hand-edited `Data/raw/splits_config.yaml` declaring boundaries and which strategies to emit. |
| Splits artifact | `Data/processed/splits_2024.json` — the file Phase 4 reads to learn which games belong to which slice. |
| Splits manifest | `Data/processed/splits_manifest.json` — provenance for a split build (input SHAs, config SHA, output SHAs, per-strategy summaries, build timestamp). |

### 1.4 Design Principles

- **Contract first.** The splits artifact is a stable, documented contract. Phase 4 binds to that contract, not to the build's internals.
- **Determinism above all.** Identical inputs (Phase 2 outputs + `splits_config.yaml`) must produce byte-identical `splits_2024.json` and manifest across runs. The only permitted source of drift is the manifest's build timestamp.
- **Auditable partition.** Every `GameId` in the universe appears in exactly one role under S1. No game is silently dropped; the manifest's per-strategy summary states the counts.
- **Loss-less preservation of upstream layers.** Phase 1 and Phase 2 outputs are read-only. The Phase 3 build never modifies them.
- **Configuration over code edits.** Changing week boundaries or toggling a strategy is a YAML edit. Adding a *new strategy* is a code change accompanied by a config-schema change and a `splits_version` bump.

### 1.5 Relationship to Phases 1 and 2

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
```

The Phase 3 build refuses to run if Phase 2's outputs are missing or if their on-disk SHA-256 hashes disagree with `feature_manifest.json`'s recorded output hashes. This guarantees that a split file is always pinned to a specific Phase 2 build.

---

## 2. Technology Additions

| Layer | Technology | Purpose |
|-------|------------|---------|
| Build step | Python (same venv as Phases 1 and 2) | Reads Phase 2 outputs + `splits_config.yaml`; emits JSON. |
| Parquet read | `pyarrow` | Reads `features_flat_2024.parquet` to obtain GameIds and weeks. |
| YAML I/O | `pyyaml` (`safe_load`) | Reads `Data/raw/splits_config.yaml`. |
| Storage format | JSON (two-space indent, trailing newline) | Both artifact and manifest. |

The build is executable from the repository root via `python -m nflpredictor.splits` and requires no interactive input.

---

## 3. Functional Requirements

### 3.1 Input Contract

| ID | Requirement |
|----|-------------|
| SP-IN-01 | The build shall read `Data/processed/features_flat_2024.parquet` and `Data/processed/feature_manifest.json` as primary inputs. |
| SP-IN-02 | The build shall read `Data/raw/splits_config.yaml` as a secondary input. The file is required; the build shall fail fast if it is missing. |
| SP-IN-03 | The build shall not modify any file under `Data/raw/` or any Phase 1 or Phase 2 output under `Data/processed/`. |
| SP-IN-04 | The build shall recompute the SHA-256 of `features_flat_2024.parquet` and compare it to the corresponding `output_sha256` entry in `feature_manifest.json`. On mismatch the build shall fail fast with a clear error naming the divergent file. |
| SP-IN-05 | The feature matrix shall contain a `week` column with integer values in `[1, 18]`. The build shall fail fast if the column is absent or contains values outside that range. |
| SP-IN-06 | The build shall reject (fail-fast) any `splits_config.yaml` whose schema does not satisfy §3.2. |

### 3.2 Splits Config Schema

| ID | Requirement |
|----|-------------|
| SP-CFG-01 | The `splits_config.yaml` schema shall include the top-level keys defined in §4.1. Unknown top-level keys shall cause the build to fail fast. |
| SP-CFG-02 | `strategies` is a non-empty list. Each entry shall be one of `S1` or `S3`. Order is preserved in the artifact. Duplicate entries shall cause the build to fail fast. |
| SP-CFG-03 | `train_weeks`, `val_weeks`, `test_weeks` are each a closed-range `[start, end]` of integer NFL weeks. Each range satisfies `1 ≤ start ≤ end ≤ 18`. The three ranges shall be contiguous, non-overlapping, and together a subset of `[1, 18]`. |
| SP-CFG-04 | The ranges shall be strictly ordered in time: `train_weeks[1] < val_weeks[0]` and `val_weeks[1] < test_weeks[0]`. |
| SP-CFG-05 | When `S3` is listed in `strategies`, `s3.k_start` shall be a positive integer satisfying `1 ≤ k_start < val_weeks[1]`. (At minimum, one fold must be constructible with `k = k_start` and val week `k + 1 ≤ val_weeks[1]`.) |
| SP-CFG-06 | `splits_version` (string) is recorded in the manifest. Any change to boundary semantics, fold-numbering, or artifact layout requires bumping this version. |
| SP-CFG-07 | A reference default config shall ship in the repository (`Data/raw/splits_config.yaml`) and reproduce the v1 boundaries declared in §3.3. |

### 3.3 v1 Default Splits

The default `splits_config.yaml` shipped in the repo declares the following. Editing this file (and only this file) is how boundaries move or strategies toggle.

| Field | v1 Default |
|-------|-----------|
| `strategies` | `["S1", "S3"]` |
| `train_weeks` | `[1, 12]` |
| `val_weeks` | `[13, 15]` |
| `test_weeks` | `[16, 18]` |
| `s3.k_start` | `6` |
| `splits_version` | `"v1"` |

### 3.4 S1 Strategy

| ID | Requirement |
|----|-------------|
| SP-S1-01 | When `S1` is in `strategies`, the build shall assign every `GameId` in the universe to exactly one of `train`, `val`, or `test` based on the game's `week` value: `train` if `week ∈ [train_weeks[0], train_weeks[1]]`, `val` if `week ∈ [val_weeks[0], val_weeks[1]]`, `test` if `week ∈ [test_weeks[0], test_weeks[1]]`. |
| SP-S1-02 | The three S1 role lists shall partition the game universe: every `GameId` appears in exactly one list, and the union of the three lists equals the universe. |
| SP-S1-03 | Within each role list, `GameId` values shall be sorted lexicographically ascending. |

### 3.5 S3 Strategy

| ID | Requirement |
|----|-------------|
| SP-S3-01 | When `S3` is in `strategies`, the build shall emit a single `test` list identical to S1's `test` list (games in Weeks `[test_weeks[0], test_weeks[1]]`). The S3 test slice does not rotate across folds. |
| SP-S3-02 | The build shall emit a sequence of expanding-window folds, indexed from `0`. For fold `i`: `k = k_start + i`; `train` is the set of games in Weeks `[1, k]`; `val` is the set of games in Week `k + 1`. |
| SP-S3-03 | The fold sequence shall terminate at the largest `i` for which `k + 1 ≤ val_weeks[1]`. For the v1 default (`k_start = 6`, `val_weeks = [13, 15]`), folds run `k ∈ {6, 7, …, 14}` with val weeks `{7, 8, …, 15}` — a total of 9 folds. |
| SP-S3-04 | Within each fold, `GameId` values in `train` and `val` shall each be sorted lexicographically ascending. |
| SP-S3-05 | For every fold, `train ∩ val = ∅` and `(train ∪ val) ∩ test = ∅`. |
| SP-S3-06 | S3 folds shall be listed in ascending `fold_index` order in the artifact. |

### 3.6 Output Encoding

| ID | Requirement |
|----|-------------|
| SP-OUT-01 | The build shall emit `Data/processed/splits_2024.json` with the structure declared in §4.2. |
| SP-OUT-02 | The artifact shall be written with `json.dump(..., indent=2)` and a trailing newline. Top-level and per-strategy key order is fixed by §4.2; the build constructs the structure deterministically so identical inputs produce byte-identical output without relying on `sort_keys`. |
| SP-OUT-03 | The set of `GameId` values across S1's three role lists shall equal the set of `GameId` values in `features_flat_2024.parquet`. |
| SP-OUT-04 | The union of `(train ∪ val)` across all S3 folds shall equal the set of `GameId` values whose `week ≤ val_weeks[1]`. (S3's `test` lists those whose `week ≥ test_weeks[0]` once, not per fold.) |

### 3.7 Splits Manifest

| ID | Requirement |
|----|-------------|
| SP-MAN-01 | The build shall emit `Data/processed/splits_manifest.json` containing at minimum the keys: `build_timestamp_utc`, `splits_version`, `splits_config_sha256`, `phase2_source_sha256` (object: input filename → SHA), `output_sha256` (object: output filename → SHA), `git_commit` (or `null`), `phase2_manifest_git_commit` (the `git_commit` from `feature_manifest.json`, or `null`), `strategy_summaries`. |
| SP-MAN-02 | `strategy_summaries` is an object keyed by strategy name. For `S1`, the value is `{"train_n": N, "val_n": N, "test_n": N}`. For `S3`, the value is `{"test_n": N, "fold_count": N, "folds": [{"fold_index": i, "k": k, "train_n": N, "val_n": N}, …]}`. |
| SP-MAN-03 | The manifest's `build_timestamp_utc` shall be in ISO 8601 UTC format. |
| SP-MAN-04 | The manifest shall be written with sorted keys and stable two-space indentation; byte-identical inputs shall produce byte-identical manifests modulo the timestamp. |
| SP-MAN-05 | The manifest shall be written **last** — after the splits artifact — so the artifact's SHA can be computed against the on-disk file. |
| SP-MAN-06 | `splits_config_sha256` shall be the SHA-256 of the raw `Data/raw/splits_config.yaml` file bytes (not the parsed/normalized form). Comments, key ordering, and whitespace edits do change the hash. This mirrors Phase 1 and Phase 2's source-hash discipline. |

---

## 4. Data Model

### 4.1 `Data/raw/splits_config.yaml`

```yaml
splits_version: "v1"

strategies: [S1, S3]

train_weeks: [1, 12]
val_weeks:   [13, 15]
test_weeks:  [16, 18]

s3:
  k_start: 6
```

### 4.2 `Data/processed/splits_2024.json` Structure

```json
{
  "splits_version": "v1",
  "S1": {
    "train": ["...", "..."],
    "val":   ["...", "..."],
    "test":  ["...", "..."]
  },
  "S3": {
    "test": ["...", "..."],
    "folds": [
      {"fold_index": 0, "k": 6, "train": ["..."], "val": ["..."]},
      {"fold_index": 1, "k": 7, "train": ["..."], "val": ["..."]}
    ]
  }
}
```

- Top-level key order: `splits_version`, then each strategy in the order declared in `strategies`.
- Within S1: roles ordered `train`, `val`, `test`.
- Within S3: `test` first, then `folds` (an ordered list).
- Per-fold key order: `fold_index`, `k`, `train`, `val`.
- Every `GameId` list is lexicographically sorted ascending.

### 4.3 Output Files

| File | Type | Description |
|------|------|-------------|
| `Data/processed/splits_2024.json` | JSON | The split assignment artifact. |
| `Data/processed/splits_manifest.json` | JSON | Build provenance. |

---

## 5. Build Pipeline Design

The pipeline is described conceptually; implementation may organize it differently as long as §3's contract is honored. Recommended module layout, mirroring Phase 1's `databuild/` and Phase 2's `features/`:

```
src/nflpredictor/splits/
    __init__.py
    __main__.py      # entry point: python -m nflpredictor.splits
    pipeline.py      # top-level orchestration
    config.py        # splits_config.yaml load + validation
    s1.py            # S1 partition logic
    s3.py            # S3 expanding-window folds
    outputs.py       # JSON writers
    manifest.py      # splits_manifest.json construction
```

### 5.1 Conceptual Stages

1. **Load and validate inputs.** Read `splits_config.yaml`; validate per §3.2 and §4.1. Read `features_flat_2024.parquet` for `GameId` + `week`; verify the parquet's SHA against `feature_manifest.json` (SP-IN-04).
2. **Build the `(GameId → week)` map.** This is the universe Phase 3 partitions.
3. **Compute S1 partition.** Bucket each game by week range.
4. **Compute S3 folds.** Reuse S1's test list as S3's test; construct each fold's `(train, val)` from the `(GameId → week)` map.
5. **Sort.** Sort every `GameId` list lexicographically ascending.
6. **Emit artifact.** Write `splits_2024.json` with the §4.2 structure.
7. **Emit manifest.** Compute output SHAs and per-strategy summaries; write `splits_manifest.json` last.

### 5.2 Determinism Boundaries

The only non-deterministic input is the wall-clock timestamp written to `splits_manifest.json`. The splits artifact is fully deterministic given the inputs. Phase 3's determinism story chains from Phase 2's: any drift in Phase 2 produces an SP-IN-04 hash mismatch and a hard failure rather than silent split drift.

---

## 6. Integration / Endpoint / Tooling Design

Not applicable. The split build is a single-shot offline script with no network surface, no API, and no UI.

---

## 7. Changes to Existing Requirements

`Docs/Idea.md` §"Phase 3: Splits" enumerated open questions for Phase 3. The decisions encoded in this spec are:

- **Boundaries**: train / val / test = Weeks 1–12 / 13–15 / 16–18.
- **Strategies**: both S1 and S3 ship in v1. S3's test slice is identical to S1's and does not rotate; folds expand across Weeks 1–15 to produce 9 `(train, val)` pairs.
- **Final-final 2024 holdout**: not separately reserved. The test slice (Weeks 16–18) is the unconditional held-out partition for evaluation within this project.
- **Test universe**: 2024 only. The Idea.md framing of "2025 is the unconditional test set" has been superseded — 2024 is the entire evaluation universe for this iteration. 2025 use is a future, out-of-scope aspiration. Idea.md §"Phase 3: Splits" and §"Phase 7" require a follow-on edit to reflect this; this spec is the source of truth in the interim.

The Phase 1 and Phase 2 specs are unchanged.

---

## 8. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| SP-NF-01 | The build shall be deterministic: identical Phase 2 outputs and `splits_config.yaml` shall produce byte-identical artifact and manifest (excluding timestamp) across runs. |
| SP-NF-02 | The build shall be re-runnable: it shall produce correct output regardless of whether the two split artifacts exist, are stale, or are absent. Stale outputs shall be overwritten. |
| SP-NF-03 | The build shall complete in under 5 seconds on a modern laptop for the current data scale (272 games × 18 weeks). |
| SP-NF-04 | The `build_timestamp_utc` field in `splits_manifest.json` is the only permitted source of run-to-run output drift. |
| SP-NF-05 | The build shall emit a non-zero exit code on any fatal error. |
| SP-NF-06 | The build shall log per-strategy counts (S1: train / val / test sizes; S3: fold count + each fold's train/val sizes) to stdout so the partition is visible without opening the manifest. |
| SP-NF-07 | All JSON outputs shall use UTF-8 encoding with Unix line endings (`\n`). |

---

## 9. UI Requirements

Not applicable.

---

## 10. Testing Requirements

| ID | Requirement |
|----|-------------|
| SP-TEST-01 | A unit test shall verify `splits_config.yaml` validation against (a) the default v1 config (must accept); (b) a config with overlapping `train_weeks` and `val_weeks` (must reject); (c) a config with a gap between `train_weeks[1]` and `val_weeks[0]` (must reject); (d) a config with `s3.k_start ≥ val_weeks[1]` (must reject); (e) a config with an unknown top-level key (must reject); (f) `strategies = []` (must reject); (g) duplicate entries in `strategies` (must reject). |
| SP-TEST-02 | A unit test shall verify the S1 partition against a synthetic `(GameId, week)` fixture: every game appears in exactly one role; the three role lists are disjoint and union to the full universe; week assignments match the boundary rules. |
| SP-TEST-03 | A unit test shall verify S3 fold construction against the same fixture: the S3 test list equals S1's test list element-for-element; folds `0..N-1` are present with the expected `k` values; per-fold `train ∪ val` is disjoint from `test`; each fold's val is exactly one week; each list is sorted. |
| SP-TEST-04 | An integration test shall run the full Phase 3 build against a small synthetic Phase 2 stand-in (~30 games) and assert that the splits artifact and manifest match a checked-in expected snapshot. |
| SP-TEST-05 | A determinism test shall run `run_split_build` twice in succession against identical Phase 2 outputs and identical `splits_config.yaml` and assert byte-equality of both the artifact and the manifest (excluding `build_timestamp_utc`). |
| SP-TEST-06 | A pinned-identity test shall run the real Phase 3 build against the actual Phase 2 outputs and assert: (a) S1's three lists union to exactly 272 GameIds with no duplicates; (b) S1's train, val, test counts match the expected counts for the v1 boundaries on real 2024 data; (c) S3's `fold_count` is 9; (d) S3's test list equals S1's test list element-for-element. |
| SP-TEST-07 | A test shall verify that an SP-IN-04 hash mismatch (e.g., a hand-edited `features_flat_2024.parquet`) causes the build to fail fast with a clear error. |
| SP-TEST-08 | A test shall verify that every `GameId` list in the artifact is sorted lexicographically ascending. |

---

## 11. Security Considerations

| ID | Consideration |
|----|---------------|
| SP-SEC-01 | The build operates exclusively on local files under the repository root. It shall not make any network requests. |
| SP-SEC-02 | The build shall not write to any path outside `Data/processed/`. |
| SP-SEC-03 | No input or output of this build contains credentials, PII beyond publicly available game schedules, or other sensitive data. |
| SP-SEC-04 | `splits_config.yaml` is user-editable; the build validates its schema (§3.2) but is not required to defend against adversarial input — the only consumer is the build script in the same repository. YAML loading shall use `yaml.safe_load`. |

---

## 12. Future Considerations

The following are explicitly out of scope for Phase 3 v1 and recorded so they are not lost:

- **Multi-season splits.** When additional seasons exist as Phase 2 outputs, a `--season YYYY` flag or per-season config would extend the build. v1 stays 2024-only.
- **Stratified or matchup-aware strategies.** Current strategies are pure-temporal. If error analysis surfaces team or matchup imbalance issues across folds, a follow-on strategy (e.g., enforce that each team appears in val at least once across S3 folds) becomes a Phase 3 amendment with a `splits_version` bump.
- **Final-final backstop slice.** Currently not reserved. If model freezing (Phase 6) shows sensitivity to "validation contamination from heavy iteration," a small backstop slice could be carved out of train as a follow-on amendment.
- **Alternative S3 regimes.** Sliding-window CV (fixed train length, advancing window) and growing-fold CV (train and val both grow each fold) are alternative regimes. Out of scope for v1.
- **2025 evaluation.** A future iteration may evaluate the frozen Phase 4 model against 2025 data if and when that dataset becomes available. That is a project-level future amendment, not a Phase 3 concern.

---

## 13. References

- [Idea.md](./Idea.md) — Source idea document; §"Phase 3: Splits" enumerates the open questions this spec resolves.
- [Spec-Phase1-DataBuild.md](./Spec-Phase1-DataBuild.md) — Phase 1 specification.
- [Spec-Phase2-FeatureEngineering.md](./Spec-Phase2-FeatureEngineering.md) — Phase 2 specification. Phase 3 reads its outputs and binds to its `feature_manifest.json`.
- [Plan-Phase1-DataBuild.md](./Plan-Phase1-DataBuild.md) — Phase 1 implementation plan (stylistic precedent for the Phase 3 plan to follow).
- [Plan-Phase2-FeatureEngineering.md](./Plan-Phase2-FeatureEngineering.md) — Phase 2 implementation plan (stylistic precedent for the Phase 3 plan to follow).
- `Data/processed/features_flat_2024.parquet` — Phase 2 output; primary input to Phase 3.
- `Data/processed/feature_manifest.json` — Phase 2 manifest; consulted for source-hash pinning per SP-IN-04.
- [CLAUDE.md](../CLAUDE.md) — Repository-level notes on venv, dataset shape, and join gotchas.
