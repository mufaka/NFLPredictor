# Phase 6 — Pipeline Walkthrough: one game, raw → predictions → plots

**Walkthrough game**: `202409050kan` (Kansas City Chiefs' home opener vs the Baltimore Ravens, 2024 Week 1). This game ID is the DD-WT-01 commit — it lives at the top of this notebook so the trace is reproducible; changing it is a content edit, not a contract change.

It was picked because it is a 2024-season game, and the default season-holdout split places the whole 2024 season in **`season_holdout.val`**. So every one of the 6 Phase 4 prediction parquets emits a `val` row for it — a single game traces the full output surface of the model ladder.

> **Data source note.** This notebook is authored against the train+evaluate fixture (`tests/fixtures/evaluate/`) so cells render real outputs on the CPU-only dev machine. After a full real Phase 4 + Phase 5 run (populating `Data/processed/predictions/` and `Data/processed/evaluation/`), re-point `PROCESSED_DIR` at `Data/processed/` and re-execute. DD-WT-04: `Docs/Phase6-Walkthrough.md` is the `jupyter nbconvert --to markdown` export of this notebook; do not hand-edit it.


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

GAME_ID = "202409050kan"
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

    Staged processed_dir: /tmp/phase6_walkthrough__e0f69aj
    Contents: ['feature_manifest.json', 'feature_vocab.json', 'features_flat_all.parquet', 'features_pos_all.parquet', 'player_id_mapping.csv', 'predictions', 'splits_all.json', 'splits_manifest.json', 'training_manifest.json']


## 1. Game selection and rationale

We chose `202409050kan` because it lands in every combination's output. The default split strategy is **season-holdout**: the whole 2024 season is the `val` slice, so all 6 `season_holdout` prediction parquets emit a `val` row for this game. That gives one game a full row across the prediction surface, which keeps Section 5 short and exhaustive.

Membership is read straight from `splits_all.json` via `trace.lookup_split_membership`.


```python
splits_info = trace.lookup_split_membership(GAME_ID, processed_dir=PROCESSED_DIR)
print("Split membership for", GAME_ID, ":")
print(json.dumps(splits_info, indent=2))
```

    Split membership for 202409050kan :
    {
      "season_holdout": "val",
      "loso_cv_val_seasons": [],
      "loso_cv_in_test": false
    }


## 2. Phase 1 — raw box-score row + Madden join

`trace.load_raw_game` derives the season from the `GameId` date prefix and returns the single matching row of that season's `box_scores_<YYYY>.csv`. `trace.resolve_starters` flattens the 44 starter slots (`HomeOff01..HomeDef11`, `AwayOff01..AwayDef11`) and joins each PFR-style `_ID` to its resolved `madden_id` via `player_id_mapping.csv`. The `note` column records which Phase 1 tier resolved the join (deterministic, fuzzy, override, or appended-unmatched).


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
               GameId: '202409050kan'
             GameDate: '2024-09-05'
            DayOfWeek: 'Thursday'
             HomeTeam: 'Kansas City Chiefs'
             AwayTeam: 'Baltimore Ravens'
         HomeTeamCode: 'kan'
         AwayTeamCode: 'rav'
            HomeScore: np.int64(27)
            AwayScore: np.int64(20)
              Stadium: 'GEHA Field at Arrowhead Stadium'
                 Roof: 'outdoors'
              Surface: 'grass'
              Weather: '67 degrees, relative humidity 53%, wind 8 mph'



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
         slot position                name box_score_id  madden_id                                                                    note
    HomeOff01       QB     Patrick Mahomes     MahoPa00 2020-01118                                    tier2: deterministic team+name match
    HomeOff02       RB       Isiah Pacheco     PachIs00 2022-01134                                    tier2: deterministic team+name match
    HomeOff03       WR       Xavier Worthy     WortXa00 2024-01161                                    tier2: deterministic team+name match
    HomeOff04       WR JuJu Smith-Schuster     SmitJu00 2020-01932                                    tier2: deterministic team+name match
    HomeOff05       WR         Rashee Rice     RiceRa01 2023-01182                                    tier2: deterministic team+name match
    HomeOff06       TE        Travis Kelce     KelcTr00 2020-01136                                    tier2: deterministic team+name match
    HomeOff07        T       Jawaan Taylor     TaylJa02 2020-01053                                    tier2: deterministic team+name match
    HomeOff08        T  Kingsley Suamataia     SuamKi00 2024-01122                                    tier2: deterministic team+name match
    HomeOff09        G          Joe Thuney     ThunJo00 2020-01531 tier2: deterministic team+name match; position mismatch box=T madden=LG
    HomeOff10        G          Trey Smith     SmitTr05 2021-01005 tier2: deterministic team+name match; position mismatch box=T madden=LG
    HomeOff11        C      Creed Humphrey     HumpCr00 2021-00977                                    tier2: deterministic team+name match
    
    Home defense (11 slots):
         slot position                 name box_score_id  madden_id                                 note
    HomeDef01       DE        Michael Danna     DannMi00 2020-01120 tier2: deterministic team+name match
    HomeDef02       DE George Karlaftis III     KarlGe00 2022-01138 tier2: deterministic team+name match
    HomeDef03       DT          Mike Pennel     PennMi00 2020-01088 tier2: deterministic team+name match
    HomeDef04       DT          Chris Jones     JoneCh09 2020-01084 tier2: deterministic team+name match
    HomeDef05       LB          Nick Bolton     BoltNi00 2021-01012 tier2: deterministic team+name match
    HomeDef06       LB       Drue Tranquill     TranDr00 2020-01179 tier2: deterministic team+name match
    HomeDef07       LB           Leo Chenal     ChenLe00 2022-01148 tier2: deterministic team+name match
    HomeDef08       CB       Trent McDuffie     McDuTr00 2022-01121 tier2: deterministic team+name match
    HomeDef09       CB        Jaylen Watson     WatsJa02 2022-01116 tier2: deterministic team+name match
    HomeDef10        S          Justin Reid     ReidJu00 2020-00873 tier2: deterministic team+name match
    HomeDef11        S           Bryan Cook     CookBr02 2022-01165 tier2: deterministic team+name match


## 3. Phase 2 — feature encoding

`encoding.encode_one_game_flat` returns the row of `features_flat_all.parquet` for this game (203 columns in the default config). `encoding.encode_one_game_pos` returns the corresponding row of `features_pos_all.parquet` (259 columns). Both begin with the `GameId` and `season` identifier columns.

`encoding.explain_categorical` walks one column through the encoder pipeline: raw value → vocab key → integer code (with Phase 4's `NULL_BUMP = 1`) → routing (one-hot for vocab size ≤ 8, embedding for > 8). High-card vocabs (`team_codes`, `archetype`, etc.) feed into a single shared `nn.Embedding` per vocab key (TR-CAT-05).


```python
flat = encoding.encode_one_game_flat(GAME_ID, processed_dir=PROCESSED_DIR)
print(f"features_flat row: {len(flat)} columns")
show_cols = [
    "GameId", "season", "week", "day_of_week", "roof", "surface",
    "home_team_code", "away_team_code",
    "HomeOff01_position", "HomeOff01_madden_overallrating", "HomeOff01_madden_archetype",
    "home_score", "away_score",
]
print("\nSelected columns (identifiers + game-level + one offensive slot + labels):")
for col in show_cols:
    print(f"  {col:>35}: {flat[col]!r}")
```

    features_flat row: 203 columns
    
    Selected columns (identifiers + game-level + one offensive slot + labels):
                                   GameId: '202409050kan'
                                   season: np.int32(2024)
                                     week: np.int64(1)
                              day_of_week: np.int32(4)
                                     roof: np.int32(1)
                                  surface: np.int32(3)
                           home_team_code: np.int32(14)
                           away_team_code: np.int32(26)
                       HomeOff01_position: np.int32(29)
           HomeOff01_madden_overallrating: np.float64(99.0)
               HomeOff01_madden_archetype: np.int32(31)
                               home_score: np.float64(27.0)
                               away_score: np.float64(20.0)



```python
pos = encoding.encode_one_game_pos(GAME_ID, processed_dir=PROCESSED_DIR)
print(f"features_pos row: {len(pos)} columns")
# The pos shape groups Madden columns under canonical position slots:
# HomeQB1 / HomeRB1..2 / HomeWR1..4 / HomeTE1..3 / HomeOL1..5 / HomeDL1..5 /
# HomeLB1..4 / HomeDB1..5 — 29 home slots — plus the same 29 on the away side.
# Each slot has a _madden_overallrating and a _madden_archetype.
show_cols = [
    "HomeQB1_madden_overallrating", "HomeQB1_madden_archetype",
    "HomeRB1_madden_overallrating", "HomeRB1_madden_archetype",
    "AwayQB1_madden_overallrating", "AwayQB1_madden_archetype",
    "home_score", "away_score",
]
print("\nSelected pos columns (Home QB + Home RB1 + Away QB + labels):")
for col in show_cols:
    print(f"  {col:>40}: {pos[col]!r}")
```

    features_pos row: 259 columns
    
    Selected pos columns (Home QB + Home RB1 + Away QB + labels):
                  HomeQB1_madden_overallrating: np.float64(99.0)
                      HomeQB1_madden_archetype: np.int32(31)
                  HomeRB1_madden_overallrating: np.float64(87.0)
                      HomeRB1_madden_archetype: np.int32(19)
                  AwayQB1_madden_overallrating: np.float64(98.0)
                      AwayQB1_madden_archetype: np.int32(31)
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
      "raw": "kan",
      "vocab_key": "team_codes",
      "vocab_size": 32,
      "integer_code": 15,
      "routing": "embedding",
      "embedding_table": "team_codes"
    }
      Phase 2 vocab index in features_flat['home_team_code']: 14
      Phase 4 integer_code (= vocab_index + NULL_BUMP):       15
    
    roof →
    {
      "raw": "outdoors",
      "vocab_key": "roof",
      "vocab_size": 4,
      "integer_code": 2,
      "routing": "one_hot",
      "embedding_table": "roof"
    }
      Phase 2 vocab index in features_flat['roof']: 1
      Phase 4 integer_code (= vocab_index + NULL_BUMP): 2


## 4. Phase 3 — split assignment

Phase 3 splits by **whole season**. The default config uses one strategy:
* **`season_holdout`** — a single partition: train = seasons 2020–2023, val = 2024, test = 2025.

A second strategy, **`loso_cv`** (leave-one-season-out cross-validation), is available but opt-in. `trace.lookup_split_membership` reports both: `season_holdout` is the train/val/test bucket; `loso_cv_val_seasons` lists the held-out seasons of any loso_cv fold whose val slice contains the game (empty when loso_cv is not enabled); `loso_cv_in_test` flags membership in loso_cv's fixed test slice.


```python
splits_info = trace.lookup_split_membership(GAME_ID, processed_dir=PROCESSED_DIR)
print(f"season_holdout bucket    : {splits_info['season_holdout']!r}")
print(f"loso_cv val seasons      : {splits_info['loso_cv_val_seasons']}")
print(f"loso_cv held-out test?   : {splits_info['loso_cv_in_test']}")

# Decode for the reader: a season_holdout val game means every season_holdout
# combination emits exactly one val prediction for it — 6 combinations in the
# default config, so 6 prediction rows.
expected_rows = (
    6 if splits_info["season_holdout"] in ("val", "test") else 0
)
print(f"\nExpected predictions for this game: {expected_rows} (6 season_holdout combinations)")
```

    season_holdout bucket    : 'val'
    loso_cv val seasons      : []
    loso_cv held-out test?   : False
    
    Expected predictions for this game: 6 (6 season_holdout combinations)


## 5. Phase 4 — predictions per learned combination

`trace.lookup_predictions` walks every `<rung>__<shape>__<strategy>.parquet` in `predictions/` and emits one row per `(combination_id, slice)` that produced a prediction for the game. For `season_holdout` parquets the slice is `"val"` or `"test"`; for `loso_cv` parquets the slice is `"fold_<val_season>"`. Residuals are computed against `home_score` / `away_score` from `features_flat`.


```python
preds = trace.lookup_predictions(GAME_ID, processed_dir=PROCESSED_DIR)
print(f"Prediction rows for {GAME_ID}: {len(preds)}\n")
print(preds.to_string(index=False))
```

    Prediction rows for 202409050kan: 6
    
                           combination_id slice  pred_home  pred_away  true_home  true_away  residual_home  residual_away
         rung0_mean__none__season_holdout   val  22.187500  23.312500       27.0       20.0      -4.812500       3.312500
    rung1_team_mean__none__season_holdout   val  27.000000  23.312500       27.0       20.0       0.000000       3.312500
       rung2_linear__flat__season_holdout   val  30.981251  20.981707       27.0       20.0       3.981251       0.981707
        rung2_linear__pos__season_holdout   val  33.782223  26.643696       27.0       20.0       6.782223       6.643696
          rung3_mlp__flat__season_holdout   val  28.942469  30.320202       27.0       20.0       1.942469      10.320202
           rung3_mlp__pos__season_holdout   val  26.201117  28.517235       27.0       20.0      -0.798883       8.517235


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
# pairing (rung 2 linear flat, season_holdout val). The full universe is in the parquet.
FOCUS_COMBO = "rung2_linear__flat__season_holdout"
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

    Game keys — week=1, home=kan, away=rav, roof=1 ('outdoors'), surface=3 ('grass')
    
    --- by_team.parquet — cells the game contributes to for (rung2_linear__flat__season_holdout, val) — 4/64 rows ---
    team_code home_or_away  n_games      mae  mae_home  mae_away
          kan         away        0      NaN       NaN       NaN
          kan         home        1 2.481479  3.981251  0.981707
          rav         away        1 2.481479  3.981251  0.981707
          rav         home        0      NaN       NaN       NaN
    


    --- by_week.parquet — cells the game contributes to for (rung2_linear__flat__season_holdout, val) — 1/18 rows ---
     week  n_games      mae  mae_home  mae_away
        1        4 6.509743  8.170909  4.848578
    


    --- by_home_away.parquet — cells the game contributes to for (rung2_linear__flat__season_holdout, val) — 2/2 rows ---
    home_or_away  n_games      mae  mae_home  mae_away
            away        4 6.509743  8.170909  4.848578
            home        4 6.509743  8.170909  4.848578
    
    --- by_surface.parquet — cells the game contributes to for (rung2_linear__flat__season_holdout, val) — 1/6 rows ---
     surface_code surface_label  n_games      mae  mae_home  mae_away
                3         grass        2 4.509819  4.697774  4.321864
    
    --- by_roof.parquet — cells the game contributes to for (rung2_linear__flat__season_holdout, val) — 1/4 rows ---
     roof_code roof_label  n_games      mae  mae_home  mae_away
             1   outdoors        3 5.465963  5.346752  5.585175
    

