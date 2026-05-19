# Phase 6 — Pipeline Walkthrough: one game, raw → predictions → plots

**Walkthrough game**: `202411280dal` (Dallas Cowboys' Thanksgiving home opener, 2024 Week 13). This game ID is the DD-WT-01 commit — it lives at the top of this notebook so the trace is reproducible; changing it is a content edit, not a contract change.

It was picked because the fixture’s split assignment puts it in **S1.val** *and* **S3 fold k=12 val**, so every one of the 12 Phase 4 prediction parquets contains it. A single game traces the full output surface of the model ladder.

> **Data source note.** This notebook is authored against the train+evaluate fixture (`tests/fixtures/evaluate/`) so cells render real outputs on the CPU-only dev machine. After the CUDA machine has run the real Phase 4 + Phase 5 pipeline (populating `Data/processed/predictions/` and `Data/processed/evaluation/`), re-point `PROCESSED_DIR` at `Data/processed/` and re-execute the notebook. The regen command lives in `CLAUDE.md` (DD-WT-05). DD-WT-04: `Docs/Phase6-Walkthrough.md` is the `jupyter nbconvert --to markdown` export of this notebook; do not hand-edit it.


```python
import json
import pathlib
import shutil
import sys
import tempfile

import pandas as pd

# Locate the repo root regardless of whether the kernel was started from
# the repo root or from `notebooks/`.
_here = pathlib.Path.cwd()
if (_here / "src" / "nflpredictor").is_dir():
    REPO_ROOT = _here
elif (_here.parent / "src" / "nflpredictor").is_dir():
    REPO_ROOT = _here.parent
else:
    raise RuntimeError(f"Cannot locate repo root from {_here}")
sys.path.insert(0, str(REPO_ROOT / "src"))

from nflpredictor.diagnostics import encoding, trace

GAME_ID = "202411280dal"
RAW_DIR = REPO_ROOT / "Data" / "raw"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "evaluate"

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 140)
```


```python
# The eval fixture has Phase 2 / 3 / 4 / 5 outputs split across raw_phase2,
# raw_phase3, raw_phase4, and expected. The diagnostics helpers expect them
# co-located in one `processed_dir`. We stage a tmp dir once per run.
STAGE = pathlib.Path(tempfile.mkdtemp(prefix="phase6_walkthrough_"))

for src_dir in (FIXTURE / "raw_phase2", FIXTURE / "raw_phase3"):
    for p in src_dir.iterdir():
        shutil.copy2(p, STAGE / p.name)

shutil.copy2(FIXTURE / "raw_phase4" / "training_manifest.json", STAGE / "training_manifest.json")

pred_dst = STAGE / "predictions"
pred_dst.mkdir()
for p in sorted((FIXTURE / "raw_phase4" / "predictions").glob("*.parquet")):
    shutil.copy2(p, pred_dst / p.name)

# Use the real player_id_mapping.csv from Data/processed/ — the eval fixture
# doesn’t ship one, and the real mapping is what readers will use on actual data.
shutil.copy2(
    REPO_ROOT / "Data" / "processed" / "player_id_mapping.csv",
    STAGE / "player_id_mapping.csv",
)

EVAL_BREAKDOWNS = FIXTURE / "expected" / "breakdowns"

PROCESSED_DIR = STAGE
print(f"Staged processed_dir: {PROCESSED_DIR}")
print("Contents:", sorted(p.name for p in PROCESSED_DIR.iterdir()))
```

    Staged processed_dir: /tmp/phase6_walkthrough_65vzudyr
    Contents: ['feature_manifest.json', 'feature_vocab.json', 'features_flat_2024.parquet', 'features_pos_2024.parquet', 'player_id_mapping.csv', 'predictions', 'splits_2024.json', 'splits_manifest.json', 'training_manifest.json']


## 1. Game selection and rationale

We chose `202411280dal` because it lands in *every* combination’s output: the **S1** val slice (so all 6 S1 prediction parquets emit a row for it) and **S3** fold `k=12`’s val slice (so all 6 S3 prediction parquets emit a row for it via that fold). That gives one game a full row across the 12 prediction parquets, which keeps Section 5 short and exhaustive.

Membership is read straight from `splits_2024.json` via `trace.lookup_split_membership`.


```python
splits_info = trace.lookup_split_membership(GAME_ID, processed_dir=PROCESSED_DIR)
print("Split membership for", GAME_ID, ":")
print(json.dumps(splits_info, indent=2))
```

    Split membership for 202411280dal :
    {
      "S1": "val",
      "S3_folds": [
        12
      ],
      "S3_test": false
    }


## 2. Phase 1 — raw box-score row + Madden join

`trace.load_raw_game` returns the single row of `box_scores_2024.csv` matching `GameId`. `trace.resolve_starters` flattens the 44 starter slots (`HomeOff01..HomeDef11`, `AwayOff01..AwayDef11`) and joins each PFR-style `_ID` to its resolved `madden_id` via `player_id_mapping.csv`. The `note` column records which Phase 1 tier resolved the join (deterministic, fuzzy, override, or appended-unmatched).


```python
raw = trace.load_raw_game(GAME_ID, raw_dir=RAW_DIR)
header_cols = [
    "GameId", "GameDate", "DayOfWeek", "HomeTeam", "AwayTeam",
    "HomeTeamCode", "AwayTeamCode", "HomeScore", "AwayScore",
    "Stadium", "Roof", "Surface", "Weather",
]
print("Game header (raw box-score row, abridged):")
for col in header_cols:
    print(f"  {col:>15}: {raw[col]!r}")
```

    Game header (raw box-score row, abridged):
               GameId: '202411280dal'
             GameDate: '2024-11-28'
            DayOfWeek: 'Thursday'
             HomeTeam: 'Dallas Cowboys'
             AwayTeam: 'New York Giants'
         HomeTeamCode: 'dal'
         AwayTeamCode: 'nyg'
            HomeScore: np.int64(27)
            AwayScore: np.int64(20)
              Stadium: 'AT&T Stadium'
                 Roof: 'retractable roof (closed)'
              Surface: 'matrixturf'
              Weather: nan



```python
starters = trace.resolve_starters(GAME_ID, raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR)
print(f"Starter slots: {len(starters)} (expected 44)")
print(f"Unmapped slots (no madden_id resolved): {starters['madden_id'].isna().sum()}\n")

# Show the home offense in full, then a short look at home defense.
cols = ["slot", "position", "name", "box_score_id", "madden_id", "note"]
print("Home offense (11 slots):")
print(starters[starters['slot'].str.startswith('HomeOff')][cols].to_string(index=False))
print("\nHome defense (11 slots):")
print(starters[starters['slot'].str.startswith('HomeDef')][cols].to_string(index=False))
```

    Starter slots: 44 (expected 44)
    Unmapped slots (no madden_id resolved): 0
    
    Home offense (11 slots):
         slot position             name box_score_id  madden_id                                 note
    HomeOff01       QB      Cooper Rush     RushCo00 2024-00930 tier2: deterministic team+name match
    HomeOff02       RB      Rico Dowdle     DowdRi01 2024-00911 tier2: deterministic team+name match
    HomeOff03       WR    Jalen Tolbert     TolbJa00 2024-00959 tier2: deterministic team+name match
    HomeOff04       WR    Brandin Cooks     CookBr00 2024-00954 tier2: deterministic team+name match
    HomeOff05       WR      CeeDee Lamb     LambCe00 2024-00955 tier2: deterministic team+name match
    HomeOff06       TE Luke Schoonmaker     SchoLu00 2024-00950 tier2: deterministic team+name match
    HomeOff07       OL      Tyler Smith     SmitTy02 2024-00924 tier2: deterministic team+name match
    HomeOff08       OT     Tyler Guyton     GuytTy00 2024-02452   unmatched: appended with null-fill
    HomeOff09        T   Terence Steele     SteeTe01 2024-00943 tier2: deterministic team+name match
    HomeOff10       OG     Cooper Beebe     BeebCo00 2024-02445   unmatched: appended with null-fill
    HomeOff11        C    Brock Hoffman     HoffBr00 2024-00890 tier2: deterministic team+name match
    
    Home defense (11 slots):
         slot position                name box_score_id  madden_id                                                                     note
    HomeDef01       DE    Chauncey Golston     GolsCh00 2024-00915                                     tier2: deterministic team+name match
    HomeDef02       DT          Mazi Smith     SmitMa06 2024-00901                                     tier2: deterministic team+name match
    HomeDef03       DT      Osa Odighizuwa     OdigOs00 2024-00903                                     tier2: deterministic team+name match
    HomeDef04      MLB      Eric Kendricks     KendEr00 2024-00631                              tier3: deterministic name match league-wide
    HomeDef05       LB DeMarvion Overshown     OverDe00 2024-00926                                     tier2: deterministic team+name match
    HomeDef06       LB       Micah Parsons     ParsMi00 2024-00936 tier2: deterministic team+name match; position mismatch box=LB madden=RE
    HomeDef07       CB         DaRon Bland     BlanDa00 2024-00893                                     tier2: deterministic team+name match
    HomeDef08       CB       Jourdan Lewis     LewiJo01 2024-00895                                     tier2: deterministic team+name match
    HomeDef09       CB         Josh Butler     ButlJo00 2024-02447                                       unmatched: appended with null-fill
    HomeDef10        S        Malik Hooker     HookMa00 2024-00906                                     tier2: deterministic team+name match
    HomeDef11        S      Donovan Wilson     WilsDo01 2024-00944                                     tier2: deterministic team+name match


## 3. Phase 2 — feature encoding

`encoding.encode_one_game_flat` returns the row of `features_flat_2024.parquet` for this game (202 columns in the v1 default config). `encoding.encode_one_game_pos` returns the corresponding row of `features_pos_2024.parquet` (258 columns).

`encoding.explain_categorical` walks one column through the encoder pipeline: raw value → vocab key → integer code (with Phase 4’s `NULL_BUMP = 1`) → routing (one-hot for vocab size ≤ 8, embedding for > 8). High-card vocabs (`team_codes`, `Archetype`, etc.) feed into a single shared `nn.Embedding` per vocab key (TR-CAT-05).


```python
flat = encoding.encode_one_game_flat(GAME_ID, processed_dir=PROCESSED_DIR)
print(f"features_flat row: {len(flat)} columns")
show_cols = [
    "GameId", "week", "day_of_week", "roof", "surface",
    "home_team_code", "away_team_code",
    "HomeOff01_position", "HomeOff01_madden_overall_rating", "HomeOff01_madden_archetype",
    "home_score", "away_score",
]
print("\nSelected columns (game-level + one offensive slot + labels):")
for col in show_cols:
    print(f"  {col:>35}: {flat[col]!r}")
```

    features_flat row: 202 columns
    
    Selected columns (game-level + one offensive slot + labels):
                                   GameId: '202411280dal'
                                     week: np.int64(13)
                              day_of_week: np.int32(4)
                                     roof: np.int32(2)
                                  surface: np.int32(4)
                           home_team_code: np.int32(8)
                           away_team_code: np.int32(19)
                       HomeOff01_position: np.int32(16)
          HomeOff01_madden_overall_rating: np.float64(65.0)
               HomeOff01_madden_archetype: np.int32(32)
                               home_score: np.float64(27.0)
                               away_score: np.float64(20.0)



```python
pos = encoding.encode_one_game_pos(GAME_ID, processed_dir=PROCESSED_DIR)
print(f"features_pos row: {len(pos)} columns")
# The pos shape groups Madden columns under canonical position slots:
# HomeQB1 / HomeRB1..2 / HomeWR1..4 / HomeTE1..3 / HomeOL1..5 / HomeDL1..5 /
# HomeLB1..4 / HomeCB1..4 / HomeS1..2 — 29 home slots — plus the same 29 on
# the away side. Each slot has a _madden_overall_rating and a _madden_archetype.
show_cols = [
    "HomeQB1_madden_overall_rating", "HomeQB1_madden_archetype",
    "HomeRB1_madden_overall_rating", "HomeRB1_madden_archetype",
    "AwayQB1_madden_overall_rating", "AwayQB1_madden_archetype",
    "home_score", "away_score",
]
print("\nSelected pos columns (Home QB + Home RB1 + Away QB + labels):")
for col in show_cols:
    print(f"  {col:>40}: {pos[col]!r}")
```

    features_pos row: 258 columns
    
    Selected pos columns (Home QB + Home RB1 + Away QB + labels):
                 HomeQB1_madden_overall_rating: np.float64(65.0)
                      HomeQB1_madden_archetype: np.int32(32)
                 HomeRB1_madden_overall_rating: np.float64(67.0)
                      HomeRB1_madden_archetype: np.int32(17)
                 AwayQB1_madden_overall_rating: np.float64(64.0)
                      AwayQB1_madden_archetype: np.int32(35)
                                    home_score: np.float64(27.0)
                                    away_score: np.float64(20.0)



```python
# Two illustrative columns:
#   (a) home_team_code -> team_codes vocab -> embedding routing (high-card, size 32).
#   (b) roof -> roof vocab -> one-hot routing (low-card, size 4).
#
# Note: explain_categorical walks the *raw* string through the encoder. The
# *encoded* feature row stores the Phase-2 vocab index (no bump); Phase 4's
# encoder bumps that index by NULL_BUMP=1 to reserve slot 0 for the null
# sentinel (TR-CAT-07). So integer_code from the helper == flat value + 1.
home_team_raw = str(raw["HomeTeamCode"])  # PFR 3-letter code
roof_raw = str(raw["Roof"])

team_explain = encoding.explain_categorical("home_team_code", home_team_raw, processed_dir=PROCESSED_DIR)
roof_explain = encoding.explain_categorical("roof", roof_raw, processed_dir=PROCESSED_DIR)

print("home_team_code →")
print(json.dumps(team_explain, indent=2))
print(
    f"  Phase 2 vocab index in features_flat['home_team_code']: {int(flat['home_team_code'])}\n"
    f"  Phase 4 integer_code (= vocab_index + NULL_BUMP):       {team_explain['integer_code']}"
)
print("\nroof →")
print(json.dumps(roof_explain, indent=2))
print(
    f"  Phase 2 vocab index in features_flat['roof']: {int(flat['roof'])}\n"
    f"  Phase 4 integer_code (= vocab_index + NULL_BUMP): {roof_explain['integer_code']}"
)
```

    home_team_code →
    {
      "raw": "dal",
      "vocab_key": "team_codes",
      "vocab_size": 32,
      "integer_code": 9,
      "routing": "embedding",
      "embedding_table": "team_codes"
    }
      Phase 2 vocab index in features_flat['home_team_code']: 8
      Phase 4 integer_code (= vocab_index + NULL_BUMP):       9
    
    roof →
    {
      "raw": "retractable roof (closed)",
      "vocab_key": "roof",
      "vocab_size": 4,
      "integer_code": 3,
      "routing": "one_hot",
      "embedding_table": "roof"
    }
      Phase 2 vocab index in features_flat['roof']: 2
      Phase 4 integer_code (= vocab_index + NULL_BUMP): 3


## 4. Phase 3 — split assignment

Phase 3 produces two coexisting split strategies:
* **S1** — a single train/val/test partition fixed by week boundaries (train Weeks 1–12 / val 13–15 / test 16–18).
* **S3** — nine expanding-window `(train, val)` folds plus a shared held-out test slice identical to S1’s.

`trace.lookup_split_membership` (re-run here for the per-section narrative) reports both. The `S3_folds` list is the set of `k` values whose val slice contains the game; `S3_test` flags membership in S3’s held-out test slice.


```python
splits_info = trace.lookup_split_membership(GAME_ID, processed_dir=PROCESSED_DIR)
print(f"S1 bucket            : {splits_info['S1']!r}")
print(f"S3 folds (k where in val): {splits_info['S3_folds']}")
print(f"S3 held-out test?    : {splits_info['S3_test']}")

# Decode for the reader: this combination means every S1 combination
# emits one val prediction, and every S3 combination emits one prediction
# from the fold whose val slice happens to include this game. 6 S1 + 6 S3 = 12 rows.
expected_rows = (1 if splits_info["S1"] in ("val", "test") else 0) * 6 \
              + len(splits_info["S3_folds"]) * 6
print(f"\nExpected predictions for this game: {expected_rows} (6 S1 + 6 S3)")
```

    S1 bucket            : 'val'
    S3 folds (k where in val): [12]
    S3 held-out test?    : False
    
    Expected predictions for this game: 12 (6 S1 + 6 S3)


## 5. Phase 4 — predictions per learned combination

`trace.lookup_predictions` walks every `<rung>__<shape>__<strategy>.parquet` in `predictions/` and emits one row per `(combination_id, slice)` that produced a prediction for the game. For S1 parquets the slice is `"val"` or `"test"`; for S3 parquets the slice is `"fold_<k>"` (using the fold’s `k` value, not its zero-based index). Residuals are computed against `home_score` / `away_score` from `features_flat`.


```python
preds = trace.lookup_predictions(GAME_ID, processed_dir=PROCESSED_DIR)
print(f"Prediction rows for {GAME_ID}: {len(preds)}\n")
print(preds.to_string(index=False))
```

    Prediction rows for 202411280dal: 12
    
               combination_id   slice  pred_home  pred_away  true_home  true_away  residual_home  residual_away
         rung0_mean__none__s1     val  23.041667  24.416667       27.0       20.0      -3.958333       4.416667
         rung0_mean__none__s3 fold_12  23.041667  24.416667       27.0       20.0      -3.958333       4.416667
    rung1_team_mean__none__s1     val  23.041667  17.000000       27.0       20.0      -3.958333      -3.000000
    rung1_team_mean__none__s3 fold_12  23.041667  17.000000       27.0       20.0      -3.958333      -3.000000
       rung2_linear__flat__s1     val  26.752432  21.401213       27.0       20.0      -0.247568       1.401213
       rung2_linear__flat__s3 fold_12  28.832504  24.525169       27.0       20.0       1.832504       4.525169
        rung2_linear__pos__s1     val  20.433571  27.587416       27.0       20.0      -6.566429       7.587416
        rung2_linear__pos__s3 fold_12  20.433571  27.587416       27.0       20.0      -6.566429       7.587416
          rung3_mlp__flat__s1     val  20.594995  19.101183       27.0       20.0      -6.405005      -0.898817
          rung3_mlp__flat__s3 fold_12  20.594995  19.101183       27.0       20.0      -6.405005      -0.898817
           rung3_mlp__pos__s1     val  22.062527  20.721458       27.0       20.0      -4.937473       0.721458
           rung3_mlp__pos__s3 fold_12  22.062527  20.721458       27.0       20.0      -4.937473       0.721458


## 6. Phase 5 — locate the game on plots and breakdowns

Phase 5 emits scatter / residual / by-week PNGs per `(combination_id, slice)` plus five breakdown parquets (`by_team`, `by_week`, `by_home_away`, `by_surface`, `by_roof`). The plots live at `Data/processed/evaluation/plots/` on the CUDA machine after a real run; the eval fixture intentionally omits them, so this section shows the breakdown rows the game contributes to and leaves the plot-locating step as prose.

Reading the breakdowns: each parquet is universe-complete — every cell of the breakdown’s key space appears with `n_games` and per-metric columns, with `null` metrics when `n_games == 0`. The rows below are the *cells the game is bucketed into* (e.g., its team rows in `by_team`, its week in `by_week`).


```python
# Pull the per-game keys used to locate this game across breakdowns.
# Surface/roof in the breakdown parquets are integer-coded (matching the
# vocab); the corresponding label column is human-readable.
week = int(flat["week"])
home_code = str(raw["HomeTeamCode"])
away_code = str(raw["AwayTeamCode"])
roof_code = int(flat["roof"])
surface_code = int(flat["surface"])
print(
    f"Game keys — week={week}, home={home_code}, away={away_code}, "
    f"roof={roof_code} ({raw['Roof']!r}), surface={surface_code} ({raw['Surface']!r})\n"
)

# To keep the printout readable, focus on one representative combination/slice
# pairing (rung 2 linear flat, S1 val). The full universe is in the parquet.
FOCUS_COMBO = "rung2_linear__flat__s1"
FOCUS_SLICE = "val"

for name in ("by_team", "by_week", "by_home_away", "by_surface", "by_roof"):
    df = pd.read_parquet(EVAL_BREAKDOWNS / f"{name}.parquet")
    full = df[(df["combination_id"] == FOCUS_COMBO) & (df["slice"] == FOCUS_SLICE)]
    if name == "by_team":
        cells = full[full["team_code"].isin([home_code, away_code])]
        show = ["team_code", "home_or_away", "n_games", "mae", "mae_home", "mae_away"]
    elif name == "by_week":
        cells = full[full["week"] == week]
        show = ["week", "n_games", "mae", "mae_home", "mae_away"]
    elif name == "by_home_away":
        cells = full
        show = ["home_or_away", "n_games", "mae", "mae_home", "mae_away"]
    elif name == "by_surface":
        cells = full[full["surface_code"] == surface_code]
        show = ["surface_code", "surface_label", "n_games", "mae", "mae_home", "mae_away"]
    elif name == "by_roof":
        cells = full[full["roof_code"] == roof_code]
        show = ["roof_code", "roof_label", "n_games", "mae", "mae_home", "mae_away"]
    print(
        f"--- {name}.parquet — cells the game contributes to for "
        f"({FOCUS_COMBO}, {FOCUS_SLICE}) — {len(cells)}/{len(full)} rows ---"
    )
    print(cells[show].to_string(index=False))
    print()
```

    Game keys — week=13, home=dal, away=nyg, roof=2 ('retractable roof (closed)'), surface=4 ('matrixturf')
    
    --- by_team.parquet — cells the game contributes to for (rung2_linear__flat__s1, val) — 4/64 rows ---
    team_code home_or_away  n_games       mae  mae_home  mae_away
          dal         away        1 10.348879 15.207222  5.490536
          dal         home        1  0.824390  0.247568  1.401213
          nyg         away        1  0.824390  0.247568  1.401213
          nyg         home        0       NaN       NaN       NaN
    
    --- by_week.parquet — cells the game contributes to for (rung2_linear__flat__s1, val) — 1/18 rows ---
     week  n_games      mae  mae_home  mae_away
       13        2 1.302568  1.520551  1.084586
    
    --- by_home_away.parquet — cells the game contributes to for (rung2_linear__flat__s1, val) — 2/2 rows ---
    home_or_away  n_games      mae  mae_home  mae_away
            away        6 8.201387  9.734038  6.668736
            home        6 8.201387  9.734038  6.668736
    
    --- by_surface.parquet — cells the game contributes to for (rung2_linear__flat__s1, val) — 1/6 rows ---
     surface_code surface_label  n_games     mae  mae_home  mae_away
                4    matrixturf        1 0.82439  0.247568  1.401213
    
    --- by_roof.parquet — cells the game contributes to for (rung2_linear__flat__s1, val) — 1/4 rows ---
     roof_code                roof_label  n_games     mae  mae_home  mae_away
             2 retractable roof (closed)        1 0.82439  0.247568  1.401213
    

