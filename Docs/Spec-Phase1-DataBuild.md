# Phase 1: Data Build Specification

## 1. Introduction

### 1.1 Purpose

This specification defines the **Data Build** phase of the NFL Predictor project. The Data Build phase converts two disparate raw data sources — a season of NFL box scores and a snapshot of Madden NFL 24 player ratings — into a small set of joinable, deterministic processed artifacts that constitute the training data for all later phases of the project.

This is a Phase 1 specification. Phases 2 through 7 (feature engineering, splits, model ladder, evaluation, error analysis, 2025 test) remain exploratory in `Docs/Idea.md` and are intentionally not specified yet. The reasoning behind that scoping choice is in §1.5.

### 1.2 Scope

In scope:

- A deterministic, re-runnable build step that reads from `Data/raw/` and writes to `Data/processed/`.
- Player identity assignment, name and team-name normalization, tiered matching with manual overrides, and unmatched-player handling.
- Three processed data artifacts (`madden_2024.csv`, `box_scores_2024.csv`, `player_id_mapping.csv`) plus a `build_manifest.json` provenance file.
- A position-consistency warning check between box-score and Madden position labels.
- Testing requirements that protect the contract from silent regression.

Out of scope:

- Feature selection, encoding, or model input shaping. (Phase 2)
- Train/validation/test split logic. (Phase 3)
- Any model code or training loop. (Phase 4)
- Metrics, error analysis, or 2025-season inference. (Phases 5–7)
- Replacement of the Madden source with a different provider.
- Choice of build-step programming language or libraries.

### 1.3 Definitions and Acronyms

| Term | Definition |
|------|------------|
| Box score | A single row in `box_scores_2024.csv` representing one NFL game, including 22 home + 22 away starters identified by name, position, and a Pro-Football-Reference player ID. |
| PFR ID | The Pro-Football-Reference player ID present in the raw box-scores file's `_ID` columns (e.g., `MahoPa00`). May occasionally be blank. |
| `madden_id` | The build-generated stable player identifier of the form `YYYY-NNNNN` (e.g. `2024-00001`). The canonical join key after the build runs. |
| `matched` | A flag column on the processed Madden file. `1` for rows present in the raw Madden source; `0` for unmatched-player rows appended by the build. |
| Starter | A player named in one of the 44 lineup slots (22 home + 22 away) of a box-score row. |
| Manual override | A row in the hand-edited `player_overrides.csv` that pins a specific box-score name+team to a specific Madden row, bypassing automated matching. |
| Null-fill | The final step of the Madden output that replaces null rating values on `matched=0` rows with averages (numeric) or modes (categorical) computed over `matched=1` rows. |
| Build manifest | A small JSON artifact that records the provenance of a build run (source hashes, timestamp, normalization version, summary counts). |
| Tier-1 / Tier-2 / Tier-3 / Tier-4 match | The four levels of the matching pipeline: exact normalized team+name; exact normalized name league-wide; fuzzy match above threshold; manual override. |

### 1.4 Design Principles

- **Contract first.** The build emits a stable, documented contract. Downstream phases bind to that contract, not to the build's internals.
- **Determinism above all.** Identical inputs must produce byte-identical outputs across runs. This is the single hardest invariant the build must hold; every other decision bends to it.
- **Auditable identity.** Every player identity assignment is traceable to its source via the mapping file's `note` column, including the resolution mechanism used.
- **Loss-less preservation of the raw layer.** The raw files in `Data/raw/` are never modified. The processed layer is the only thing that changes.
- **Soft failures preferred to hard failures.** Quality issues that don't break the contract (e.g., position mismatches) are logged and recorded in the mapping file, not raised as build errors.
- **Language-agnostic specification.** The build step can be implemented in any language. The spec defines the contract; the implementation chooses its own tooling.

### 1.5 Why Phase 1 Is Specified First

The project deliberately defines Phase 1 in full before drafting specifications for Phases 2 through 7. This is a teaching-relevant decision and is recorded here so the reasoning survives later iterations:

1. **Phase 1 produces a contract that every later phase depends on.** File shapes, the `madden_id` system, the `matched` flag, and the null-fill semantics are referenced by every Phase 2+ decision. Changing the contract after Phase 2 has bound to it invalidates downstream work. Pin it down first; let the rest follow.

2. **The contract is the least-ambiguous part of the project right now.** Inputs are concrete files on disk. The mapping strategy, ID format, and null-fill rule are already decided in `Docs/Idea.md`. There is nothing speculative to specify. By contrast, Phase 2's "which Madden columns matter" is *intentionally* something that should be informed by empirical inspection of the assembled data, not pre-decided.

3. **Later phases benefit from being shaped against real assembled data, not imagined data.** A Phase 2 specification written today would have to guess how many `matched=0` players exist in 2024, whether the weather strings parse cleanly, and whether `Archetype` cardinality justifies an embedding vs. a one-hot. Each of those becomes a five-minute empirical answer once Phase 1 runs. Specifying them now would force premature commitment.

4. **This mirrors how production ML pipelines are organized.** The data contract is owned, versioned, and frozen *before* the modeling team writes code against it. Inverting that order is a common source of mid-project pain — features get specified against assumed schemas, then break when the data lands differently. The project is structured this way on purpose, as a worked example of the pattern.

5. **For a single-developer learning project, smaller specs are more actionable.** A single end-to-end specification covering all seven phases would either be vague (giving the developer no useful constraints) or speculatively over-detailed (forcing premature commitment as above). A short chain of focused specs — data contract → feature spec → model spec → evaluation spec — is more useful in practice and reads as a teaching artifact.

The decision is therefore: this specification stops at Phase 1's contract. Phase 2 will be specified after the build is implemented and 2024 data has been inspected.

---

## 2. Technology Additions

The build step is language-agnostic. No specific language or libraries are mandated by this specification.

| Layer | Technology | Purpose |
|-------|------------|---------|
| Build step | A scripted, non-notebook implementation in any language | Reads `Data/raw/`, emits `Data/processed/`, runs deterministically. The script is the primary executable artifact of Phase 1. |
| Storage format | CSV (UTF-8, RFC 4180) | All processed data files. Chosen for inspectability; column counts are modest enough that CSV's lack of typing is acceptable. |
| Manifest format | JSON | The build manifest. Small, structured, machine-readable. |
| Hashing | SHA-256 | Source-file fingerprints recorded in the manifest. |

The build is expected to be executable from the repository root via a single command and to require no interactive input.

---

## 3. Functional Requirements

### 3.1 Input Contract

| ID | Requirement |
|----|-------------|
| DB-IN-01 | The build shall read `Data/raw/box_scores_2024.csv` and `Data/raw/maddennfl24fullplayerratings.csv` as its primary inputs. |
| DB-IN-02 | The build shall read `Data/raw/player_overrides.csv` as a secondary input. The file may be empty (header only) for the first build; the build shall not require non-empty overrides to succeed. |
| DB-IN-03 | The build shall not modify any file under `Data/raw/`. |
| DB-IN-04 | The build shall reject (fail-fast with a clear error) any raw input whose header row does not match the expected schema declared in this specification. |
| DB-IN-05 | The build shall tolerate the leading and trailing whitespace present in the raw Madden file's ` Total Salary ` and ` Signing Bonus ` headers by stripping it on read while preserving the column data as-is in the processed file. |

### 3.2 Normalization

| ID | Requirement |
|----|-------------|
| DB-NORM-01 | The build shall maintain a team-name normalization table that maps each PFR three-letter team code (e.g., `kan`, `rav`, `sfo`) to its corresponding Madden franchise nickname (e.g., `Chiefs`, `Ravens`, `49ers`). |
| DB-NORM-02 | The team-name normalization table shall be defined as code or a versioned data file inside the build; it shall not be silently inferred at runtime. |
| DB-NORM-03 | The build shall produce a normalized form of every player name for comparison purposes by: (a) stripping trailing suffix tokens `Jr.`, `Sr.`, `II`, `III`, `IV` (case-insensitive, with or without trailing period); (b) folding accented characters to their ASCII equivalents; (c) collapsing runs of whitespace to a single space; (d) trimming leading/trailing whitespace; (e) lowercasing the result. |
| DB-NORM-04 | The build shall use normalized names only for comparison. The original case-preserving name shall remain unchanged in all output files. |
| DB-NORM-05 | The normalization rules shall be assigned a `normalization_version` constant (a short string, e.g., `"v1"`) recorded in the build manifest. Any change to normalization rules requires bumping this version. |

### 3.3 Madden ID Assignment

| ID | Requirement |
|----|-------------|
| DB-ID-01 | The build shall assign every row in the processed Madden file a `madden_id` of the form `YYYY-NNNNN`, where `YYYY` is the Madden source vintage (`2024` for the current source) and `NNNNN` is a zero-padded sequential integer beginning at `00001`. |
| DB-ID-02 | The build shall assign `madden_id` values deterministically. The sequence shall be assigned in a stable sort order over the raw Madden rows, defined as `(Team, Position, Full Name, Jersey Number)` ascending. |
| DB-ID-03 | The build shall append unmatched-player rows to the Madden file in a stable sort order over `(box_score_team_code, normalized_name, first_game_id_seen)` and assign `madden_id` values continuing the sequence from the last raw-Madden assignment. |
| DB-ID-04 | The build shall ensure that re-running the build on identical raw inputs produces identical `madden_id` assignments. |
| DB-ID-05 | Once assigned, a `madden_id` shall not be reused for a different player in the same vintage. |

### 3.4 Tiered Matching

| ID | Requirement |
|----|-------------|
| DB-MATCH-01 | The build shall attempt to match each box-score starter to a Madden row using a four-tier pipeline, evaluated in order. The first tier to produce a match wins; later tiers are not consulted for that starter. |
| DB-MATCH-02 | **Tier 1 (Manual override)**: If `player_overrides.csv` contains an entry for the starter, the build shall use the override's specified `madden_id`. Manual overrides take precedence over all automated tiers. |
| DB-MATCH-03 | **Tier 2 (Exact normalized team+name)**: If a unique Madden row exists with the same normalized name and the team mapped by `DB-NORM-01`, the build shall match the starter to that row. |
| DB-MATCH-04 | **Tier 3 (Exact normalized name league-wide)**: If Tier 2 fails, the build shall search Madden for an exact normalized-name match league-wide. If exactly one match exists, the build shall use it (this handles in-season trades where Madden's team is stale). If multiple matches exist, Tier 3 fails for that starter and the build advances to Tier 4. |
| DB-MATCH-05 | **Tier 4 (Fuzzy match)**: If Tier 3 fails, the build shall compute a token-set similarity score between the starter's normalized name and each Madden row's normalized name within the starter's team (per `DB-NORM-01`). If the highest score meets or exceeds a fixed threshold (default: `0.85`) and is at least `0.10` higher than the second-highest score, the build shall match the starter to that row. Otherwise, Tier 4 fails. |
| DB-MATCH-06 | When all four tiers fail, the build shall mark the starter as **unmatched** and proceed per §3.6. |
| DB-MATCH-07 | The build shall record the tier that produced each successful match in the mapping file's `note` column (see §3.8). |

### 3.5 Manual Overrides

| ID | Requirement |
|----|-------------|
| DB-OVR-01 | The `player_overrides.csv` file shall have the header: `box_score_name,box_score_team_code,box_score_id,madden_id,reason`. |
| DB-OVR-02 | An override row's `box_score_id` may be blank; when blank, the override matches by `(box_score_name, box_score_team_code)` only. When present, the override matches by `box_score_id` alone. |
| DB-OVR-03 | An override row's `reason` is human-readable free text; the build shall preserve it verbatim in the mapping file's `note` column for affected starters. |
| DB-OVR-04 | If an override references a `madden_id` that does not exist in the raw Madden file (i.e., points at a not-yet-appended unmatched row), the build shall fail with a clear error message identifying the offending override. Overrides may only point at IDs the build has already assigned during the same run. |
| DB-OVR-05 | The build shall fail with a clear error if two override rows match the same starter; ambiguous overrides shall not be silently resolved. |

### 3.6 Unmatched Player Handling

| ID | Requirement |
|----|-------------|
| DB-UNM-01 | For each starter that all four matching tiers fail to resolve, the build shall append a new row to the processed Madden file with: (a) `Team` set to the Madden franchise nickname corresponding to the starter's team code; (b) `Position` set to the starter's box-score position; (c) `Full Name` set to the starter's box-score name (case-preserving); (d) `matched` set to `0`; (e) all other rating and metadata columns left null; (f) `madden_id` assigned per §3.3. |
| DB-UNM-02 | All raw Madden rows shall have `matched` set to `1` in the processed file. |
| DB-UNM-03 | If the same unmatched player appears in multiple games, the build shall create exactly one appended Madden row for that player and reuse its `madden_id` for every occurrence. Deduplication uses the normalized name and team code. |

### 3.7 Null-Fill

| ID | Requirement |
|----|-------------|
| DB-FILL-01 | After all unmatched-player rows have been appended (per §3.6), the build shall fill null values in numeric rating columns on `matched=0` rows with the arithmetic mean of that column over `matched=1` rows. |
| DB-FILL-02 | The build shall fill null values in categorical columns (including `Archetype`, `Running Style`, `College`, `Player Handness`) on `matched=0` rows with the mode of that column over `matched=1` rows. If multiple modes exist, the build shall choose the lexicographically smallest. |
| DB-FILL-03 | The `matched` column itself shall never be filled. Its values are exactly `1` for raw-Madden rows and `0` for build-appended rows. |
| DB-FILL-04 | Columns that are null in the raw Madden source on `matched=1` rows shall remain null in the processed file. Only nulls on `matched=0` rows are filled. |
| DB-FILL-05 | The build shall fill columns derived from raw data (`Height`, `Weight`, `Age`, `Birthdate`, `Years Pro`, `Jersey Number`, ` Total Salary `, ` Signing Bonus `) using the same mean (numeric) or mode (categorical) rule defined above. The intent is that no `matched=0` row contains a null value after the null-fill step. |

### 3.8 Mapping File

| ID | Requirement |
|----|-------------|
| DB-MAP-01 | The build shall emit `Data/processed/player_id_mapping.csv` with the header: `box_score_id,madden_id,note`. |
| DB-MAP-02 | The mapping file shall contain **one row per unique `(box_score_id, madden_id)` pair**. Per-occurrence rows (one per game/slot) are out of scope; the deduplicated shape was chosen because the mapping file's purpose is audit and troubleshooting of identity assignment, not per-game tracing. |
| DB-MAP-03 | The `box_score_id` column shall hold the original PFR ID as it appeared in the raw box-scores file. When the raw `_ID` was blank, this column shall be blank in the mapping file. |
| DB-MAP-04 | The `note` column shall record the resolution mechanism in a controlled vocabulary, optionally augmented with detail. Permitted values and patterns: `tier1: manual override (<reason>)`, `tier2: deterministic team+name match`, `tier3: deterministic name match league-wide`, `tier4: fuzzy match score=<0.00–1.00>`, `unmatched: appended with null-fill`, `blank source _id; resolved by name`. |
| DB-MAP-05 | If a single `(box_score_id, madden_id)` pair has multiple distinct notes across the season (rare; e.g., a tier-2 hit in week 1 and a tier-3 hit in week 17 for the same player), the build shall record the most informative note (preferring later tiers, as they encode more nuance). |
| DB-MAP-06 | The mapping file shall be sorted ascending by `(madden_id, box_score_id)` for deterministic output. |

### 3.9 Processed Box-Scores File

| ID | Requirement |
|----|-------------|
| DB-OUT-01 | The build shall emit `Data/processed/box_scores_2024.csv` with the same column shape and order as the raw `Data/raw/box_scores_2024.csv`. |
| DB-OUT-02 | Every per-slot `_ID` column (`HomeOff01_ID` … `AwayDef11_ID`) in the processed box-scores file shall contain a `madden_id` value, not a PFR ID. |
| DB-OUT-03 | No per-slot `_ID` column in the processed box-scores file shall be blank. Blanks in the raw file are resolved to `madden_id` values via the matching pipeline (per §3.4). |
| DB-OUT-04 | Non-`_ID` columns in the processed box-scores file shall be copied verbatim from the raw file. Header whitespace, value casing, and string formatting are preserved. |
| DB-OUT-05 | The processed box-scores file shall be sorted by `GameId` ascending. |

### 3.10 Processed Madden File

| ID | Requirement |
|----|-------------|
| DB-OUT-10 | The build shall emit `Data/processed/madden_2024.csv` with the original 69 columns of the raw Madden file (with header whitespace stripped per `DB-IN-05`) plus two appended columns: `madden_id` (first column) and `matched` (last column). |
| DB-OUT-11 | The processed Madden file shall contain `2368 + N` rows, where `N` is the number of unique unmatched starters appended (see §3.6). |
| DB-OUT-12 | The processed Madden file shall be sorted by `madden_id` ascending. |
| DB-OUT-13 | After null-fill (§3.7), no cell in any `matched=0` row shall be null. Cells in `matched=1` rows may remain null if they were null in the raw source. |

### 3.11 Build Manifest

| ID | Requirement |
|----|-------------|
| DB-MAN-01 | The build shall emit `Data/processed/build_manifest.json` containing at minimum the keys: `build_timestamp_utc`, `normalization_version`, `source_sha256` (an object mapping each raw input file path to its SHA-256 hex digest), `output_sha256` (an object mapping each processed output file path to its SHA-256 hex digest), `git_commit` (the current git commit hash, or `null` if not in a git repository), `counts`. |
| DB-MAN-02 | The `counts` object shall contain: `raw_madden_rows`, `unmatched_appended_rows`, `total_madden_rows_processed`, `box_score_games`, `total_starter_slots`, `tier1_matches`, `tier2_matches`, `tier3_matches`, `tier4_matches`, `unmatched_players_unique`, `position_mismatches_logged`. |
| DB-MAN-03 | The manifest's `build_timestamp_utc` shall be in ISO 8601 UTC format (e.g., `2026-05-17T14:32:09Z`). |
| DB-MAN-04 | The manifest shall be written with sorted keys and stable two-space indentation so byte-identical inputs produce byte-identical manifests (modulo the timestamp; see `DB-NF-04`). |

### 3.12 Position Consistency Check

| ID | Requirement |
|----|-------------|
| DB-POS-01 | When a starter is matched to a Madden row (Tier 2, 3, or 4), the build shall compare the starter's box-score `Position` to the matched Madden row's `Position`. |
| DB-POS-02 | If the positions disagree under a documented equivalence map (e.g., `OL` ↔ `T`/`G`/`C`, `DB` ↔ `CB`/`S`), the build shall log a warning to stderr and append a suffix `; position mismatch box=<X> madden=<Y>` to the corresponding mapping file `note`. |
| DB-POS-03 | A position mismatch shall not cause the build to fail. The match still stands; the warning exists only for human auditing. |
| DB-POS-04 | The position equivalence map shall be defined as code or a versioned data file inside the build; it shall not be silently inferred at runtime. |

---

## 4. Data Model

### 4.1 Raw Inputs (Read-Only)

#### `Data/raw/box_scores_2024.csv`

The schema of this file is given by its existing 164-column header and is not redefined here. The build treats every non-`_ID` column as opaque pass-through data. The `_ID` columns (`HomeOff01_ID` through `AwayDef11_ID`) are the only columns transformed by the build.

#### `Data/raw/maddennfl24fullplayerratings.csv`

The schema of this file is given by its existing 69-column header and is not redefined here. Columns ` Total Salary ` and ` Signing Bonus ` carry leading and trailing whitespace in their headers; the build strips this on read.

#### `Data/raw/player_overrides.csv`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `box_score_name` | string | Required | The player's name as it appears in the box-scores file (case-preserving). |
| `box_score_team_code` | string | Required | The PFR three-letter team code as it appears in the box-scores file. |
| `box_score_id` | string | Optional | The PFR ID, if present in the raw box-scores file. May be blank. |
| `madden_id` | string | Required, must exist in the raw Madden source after ID assignment | The `madden_id` to pin this player to. |
| `reason` | string | Optional | Free-text rationale recorded in the mapping file. |

### 4.2 Processed Outputs

#### `Data/processed/madden_2024.csv`

The 69 original Madden columns (with stripped header whitespace) plus:

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `madden_id` | string | First column; unique; format `YYYY-NNNNN` | Build-assigned stable player identifier. |
| `matched` | integer | Last column; `0` or `1` | `1` for raw-Madden rows; `0` for build-appended unmatched rows. |

**Sort order:** ascending by `madden_id`.

#### `Data/processed/box_scores_2024.csv`

Same 164-column shape as the raw file. The 44 per-slot `_ID` columns now contain `madden_id` values; all other columns are verbatim copies of the raw values.

**Sort order:** ascending by `GameId`.

#### `Data/processed/player_id_mapping.csv`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `box_score_id` | string | May be blank | The PFR ID from the raw box-scores file. Blank when the raw `_ID` cell was blank. |
| `madden_id` | string | Required; foreign key to `madden_2024.csv` | The resolved player identifier. |
| `note` | string | Required; controlled vocabulary per `DB-MAP-04` | Human-readable resolution provenance. |

**Sort order:** ascending by `(madden_id, box_score_id)`.

#### `Data/processed/build_manifest.json`

```
{
  "build_timestamp_utc": "<ISO 8601 UTC>",
  "git_commit": "<40-char hex or null>",
  "normalization_version": "<string>",
  "source_sha256": {
    "Data/raw/box_scores_2024.csv": "<64-char hex>",
    "Data/raw/maddennfl24fullplayerratings.csv": "<64-char hex>",
    "Data/raw/player_overrides.csv": "<64-char hex>"
  },
  "output_sha256": {
    "Data/processed/madden_2024.csv": "<64-char hex>",
    "Data/processed/box_scores_2024.csv": "<64-char hex>",
    "Data/processed/player_id_mapping.csv": "<64-char hex>"
  },
  "counts": {
    "raw_madden_rows": <int>,
    "unmatched_appended_rows": <int>,
    "total_madden_rows_processed": <int>,
    "box_score_games": <int>,
    "total_starter_slots": <int>,
    "tier1_matches": <int>,
    "tier2_matches": <int>,
    "tier3_matches": <int>,
    "tier4_matches": <int>,
    "unmatched_players_unique": <int>,
    "position_mismatches_logged": <int>
  }
}
```

---

## 5. Build Pipeline Design

This specification does not mandate a particular language or library. The pipeline is described conceptually below; implementation may organize it differently as long as the contract in §3 is honored.

### 5.1 Conceptual Stages

1. **Load and validate inputs.** Read all three raw files. Verify headers match expected schemas (DB-IN-04). Fail fast on mismatches.
2. **Build the team-name lookup.** Read or construct the PFR-code-to-Madden-nickname table.
3. **Assign `madden_id` to raw Madden rows.** Sort per DB-ID-02; assign sequential IDs.
4. **Match starters.** For every starter slot in every game, run the four-tier pipeline (§3.4). Collect a mapping table in memory keyed by the slot's identifying tuple.
5. **Resolve unmatched players.** Deduplicate by `(normalized_name, team_code)`; append to the in-memory Madden table per §3.6.
6. **Run null-fill.** Compute means and modes over `matched=1` rows; fill `matched=0` cells per §3.7.
7. **Rewrite box-score `_ID` columns.** Replace each raw PFR ID with its resolved `madden_id`. Track the resolution mechanism for the mapping file.
8. **Run position-consistency checks.** Compare box-score and Madden positions; log warnings and annotate mapping notes per §3.12.
9. **Emit outputs.** Write the three processed CSVs and the manifest JSON in the order: `madden_2024.csv` → `box_scores_2024.csv` → `player_id_mapping.csv` → `build_manifest.json`. The manifest is written last because it includes the output files' SHA-256 hashes.

### 5.2 Determinism Boundaries

The only non-deterministic input to the build is the wall-clock timestamp recorded in `build_manifest.json` (DB-MAN-03). All other outputs must be byte-identical across re-runs on identical inputs.

---

## 6. Integration / Endpoint / Tooling Design

Not applicable. The build is a single-shot offline script with no network surface, no API, and no UI.

---

## 7. Changes to Existing Requirements

`Docs/Overview.md` previously listed Linear / Random Forest / Gradient Boosted Trees as the candidate model ladder. That guidance is superseded by `Docs/Idea.md` and is out of scope for this Phase 1 specification. The Overview is annotated as superseded.

No other prior specifications exist; this is the first formal spec in the project.

---

## 8. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| DB-NF-01 | The build shall be deterministic: identical raw inputs and overrides shall produce byte-identical output CSV files across runs. |
| DB-NF-02 | The build shall be re-runnable: it shall produce correct output regardless of whether `Data/processed/` exists, is empty, or contains stale files. Stale outputs shall be overwritten. |
| DB-NF-03 | The build shall complete in under 60 seconds on a modern laptop for the current data scale (272 games × 44 starters; 2,368 Madden rows). |
| DB-NF-04 | The `build_timestamp_utc` field in the manifest is the only permitted source of run-to-run output drift. All other output bytes shall be identical across re-runs on identical inputs. |
| DB-NF-05 | The build shall emit a non-zero exit code on any fatal error. Warnings (e.g., position mismatches) shall not affect the exit code. |
| DB-NF-06 | The build shall produce sufficient stderr/stdout logging to make tier-distribution and unmatched-player counts visible without opening the manifest. |
| DB-NF-07 | All output files shall use UTF-8 encoding with Unix line endings (`\n`). |

---

## 9. UI Requirements

Not applicable.

---

## 10. Testing Requirements

| ID | Requirement |
|----|-------------|
| DB-TEST-01 | A fixture-based unit test shall verify name normalization (DB-NORM-03) against at least the following cases: `"Patrick Mahomes"`, `"Odafe Oweh"` (no suffix), `"Roquan Smith"` (no suffix; bonus: matches `SmitRo07`-style ID), `"Lamar Jackson"`, `"A.J. Brown"` (punctuated initials), `"D'Andre Swift"` (apostrophe), `"DJ Moore"` (no spaces in initials), `"Patrick Mahomes II"` (suffix), `"Marvin Harrison Jr."` (suffix), and an accented-character case. |
| DB-TEST-02 | A fixture-based unit test shall verify team-code normalization (DB-NORM-01) for the full 32-team mapping. |
| DB-TEST-03 | An integration test shall run the full build against a synthetic 2-game, 4-Madden-row fixture and assert that the three output CSVs and the manifest match a checked-in expected snapshot. |
| DB-TEST-04 | A regression test fixture shall pin specific high-stakes player identities — at minimum: `Patrick Mahomes` (Tier 2), `T.J. Watt` (blank source `_ID`, name-lookup resolution), and one fuzzy-match case — and shall fail if the resolved `madden_id` for any pinned identity changes. |
| DB-TEST-05 | A determinism test shall run the build twice in succession against identical inputs and assert byte-identical output files (excluding `build_timestamp_utc` in the manifest). |
| DB-TEST-06 | A test shall verify that an empty `player_overrides.csv` (header only) is accepted by the build. |
| DB-TEST-07 | A test shall verify that an override pointing to a non-existent `madden_id` causes the build to fail with the expected error class (DB-OVR-04). |
| DB-TEST-08 | A test shall verify that null-fill (§3.7) leaves no null cells in any `matched=0` row of the processed Madden file. |

---

## 11. Security Considerations

| ID | Consideration |
|----|---------------|
| DB-SEC-01 | The build operates exclusively on local files under the repository root. It shall not make any network requests. |
| DB-SEC-02 | The build shall not write to any path outside `Data/processed/`. |
| DB-SEC-03 | No input or output of this build contains credentials, personally identifying information beyond publicly available player names and statistics, or other sensitive data. |
| DB-SEC-04 | The `player_overrides.csv` file is user-editable; the build shall validate its schema (DB-IN-04) but is not required to defend against adversarial input — the only consumer is the build script in the same repository. |

---

## 12. Future Considerations

The following are explicitly out of scope for Phase 1 and are recorded here so they are not lost:

- **Madden 25 ingestion**: When the user transitions to a Madden 25 source, the build will need a vintage selector and a stability rule for `madden_id` continuity across vintages. The `YYYY-` prefix already encodes vintage; cross-vintage identity is a Phase 7 concern.
- **2025 box-scores ingestion**: The Phase 1 build will be invoked against a `Data/raw/box_scores_2025.csv` input when that data arrives from the user's separate application. The processed file would be emitted as `Data/processed/box_scores_2025.csv`. Multi-season support is a minor extension of the existing spec, not a fundamental change.
- **Alternative ratings providers**: If Madden is replaced by another provider with the same 69-column shape, only Section 2's input file changes; the rest of the contract is provider-agnostic.
- **Provenance for the fuzzy threshold**: The fuzzy-match threshold of `0.85` (DB-MATCH-05) is a chosen default. A future iteration may want to tune this against an empirical ROC-style analysis of the 2024 data; doing so would be a spec amendment.
- **Phase 2+ specifications**: Feature engineering, splits, model ladder, evaluation, error analysis, and 2025-test specifications will be drafted after Phase 1 is implemented and the assembled data has been empirically inspected (see §1.5).

---

## 13. References

- [Idea.md](./Idea.md) — Source idea document; defines Phases 1–7 and records all upstream decisions feeding this spec.
- [Overview.md](./Overview.md) — Historical project overview; superseded by `Idea.md`.
- `Data/raw/box_scores_2024.csv` — Raw box-scores input.
- `Data/raw/maddennfl24fullplayerratings.csv` — Raw Madden ratings input.
- [CLAUDE.md](../CLAUDE.md) — Repository-level notes on dataset shape and join gotchas.
