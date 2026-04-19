# NBA Player Prop Forecasting App

This repository contains a production-style Streamlit app for NBA player points prop forecasting.
It predicts uncertainty-aware outcomes with a transformer quantile model and includes a betting-lines workflow with automatic result grading.

## Current Project State

### What is implemented now
1. Transformer-only modeling pipeline for player points (`PTS`) quantile regression.
2. Quantile outputs from the loaded artifact (default artifact is configured for `q10`, `q50`, `q90`).
3. Auto data loading from 2020 through the current season window.
4. Fixed train/test split date at `2024-06-18` for artifact consistency.
5. Three app pages:
1. `Predict Matchup`
2. `Betting Lines`
3. `Test Stats`
6. Betting page grades picks using completed game stats fetched automatically from `nba_api`.

### What is no longer the current setup
1. Legacy rolling-feature-only schema.
2. Legacy MLP-only model path.
3. Manual CSV updates for completed actual points (now automatic via API lookup, with optional manual fallback fields).

## Architecture

### Data layer (`src/data.py`)
1. Pulls multi-season player and team game logs from `nba_api` (`LeagueGameLog`).
2. Builds leakage-safe player rows with context features.
3. Adds opponent last-game context features for inference and training.
4. Returns:
1. `final_df` for training/evaluation.
2. `current_players` for live matchup inference.
3. `current_teams` for team metadata and opponent context.

### Model layer (`src/model.py`)
1. `PlayerPropTransformer`: sequence model over per-player game history.
2. `PinballLoss`: multi-quantile loss function.
3. Default quantiles in code: `(0.10, 0.50, 0.90)`.

### Service layer (`src/service.py`)
1. `train_model(...)`: trains transformer artifacts with temporal validation and early stopping.
2. `predict_matchup(...)`: predicts home/away player quantiles for selected matchup.
3. `get_matchup_rosters(...)`: resolves official roster previews.
4. `evaluate_test_set(...)`: computes test diagnostics and calibration-oriented metrics.
5. `model_summary(...)`: artifact metadata for app display.

## Active Feature Schema

The app is currently wired to artifact `models/player_prop_artifacts_opp28.pt` and uses the opponent-context sequence setup.

Target:
1. `PTS`

Feature groups used for training/inference:
1. Context (3):
1. `is_playoff`
2. `home`
3. `days_of_rest`
2. Raw player sequence features (16):
1. `MIN`
2. `FGM`
3. `FGA`
4. `FG3M`
5. `FG3A`
6. `FTM`
7. `FTA`
8. `AST`
9. `REB`
10. `OREB`
11. `DREB`
12. `TOV`
13. `STL`
14. `BLK`
15. `PF`
16. `PLUS_MINUS`
3. Opponent last-game context features (9):
1. `Opp_LastGame_PTS`
2. `Opp_LastGame_AST`
3. `Opp_LastGame_REB`
4. `Opp_LastGame_FGA`
5. `Opp_LastGame_FG3A`
6. `Opp_LastGame_TOV`
7. `Opp_LastGame_STL`
8. `Opp_LastGame_BLK`
9. `Opp_LastGame_PLUS_MINUS`

Total active features: 28.

## App Pages

### 1) Predict Matchup
1. Select home/away teams and playoff toggle.
2. Preview official rosters for each team.
3. Run model inference for both rosters.
4. View full team forecasts side by side (not truncated to top-N).

### 2) Betting Lines
1. Reads `prize_picks_lines.csv` (required columns: `game_date`, `player_name`, `team`, `opponent`, `line`).
2. Builds model recommendations (`over`, `under`, `push`) from `q50` vs line.
3. Fetches completed game `PTS` automatically from `nba_api` (`LeagueGameFinder`) by date range.
4. Scores picks as `correct`, `incorrect`, `pending`, or `push`.
5. Displays:
1. Pick cards ordered by biggest edge.
2. Interval on cards using available quantile bounds.
3. Edge and status charts.
4. Cumulative accuracy chart.
5. Detailed table in an expander.

Notes:
1. If API data is not available yet for a game, status remains `pending`.
2. Optional CSV columns `actual_points` or `actual` can still serve as fallback values.

### 3) Test Stats
1. Artifact metadata and split details.
2. Headline error metrics on test rows.
3. Calibration and interval diagnostics.
4. Player-volume and outlier analyses.

## Caching and Performance

The app uses multiple cache layers:
1. Dataset cache (`st.cache_data`) for historical pulls and processed frames.
2. Artifact cache (`st.cache_resource`) for loaded model artifacts.
3. Betting prediction cache in session state to avoid recomputing repeated slate matchups.
4. Cached completed-game stat pulls for date ranges.

This means first load is the slowest; subsequent reruns are substantially faster.

## Setup

### Prerequisites
1. Python 3.10+
2. Internet access for NBA API calls

### Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

## Train or Refresh Artifact

Use:

```bash
python scripts/train_artifact.py
```

Current defaults in `scripts/train_artifact.py`:
1. `--start-year 2020`
2. `--end-year` rolls with current season
3. `--split-date 2024-06-18`
4. `--output models/player_prop_artifacts_opp28.pt`
5. `--sequence-length 20`

Example override:

```bash
python scripts/train_artifact.py --start-year 2020 --end-year 2026 --split-date 2024-06-18 --output models/player_prop_artifacts_opp28.pt --sequence-length 20
```

## Repository Layout

```text
NBA-Prop-Forecasting-App/
|- app.py
|- prize_picks_lines.csv
|- requirements.txt
|- scripts/
|  |- train_artifact.py
|- src/
|  |- data.py
|  |- model.py
|  |- service.py
|- models/
|  |- player_prop_artifacts_opp28.pt   # expected at runtime
```

## Known Limitations

1. Name matching between CSV lines and NBA API box scores is normalization-based and can still miss rare aliases.
2. Team/opponent abbreviations in CSV must map to NBA abbreviations present in `current_teams`.
3. NBA API latency/rate limits can delay updates or produce temporary fetch failures.
4. Model quality depends on artifact freshness and selected feature schema.

## License

See `LICENSE`.
