"""Streamlit app for NBA player prop quantile predictions."""

from __future__ import annotations

import os
import re
import string
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from nba_api.stats.endpoints import leaguegamefinder

from src.betting import (
    append_line_history_snapshot,
    attach_recommendation_columns,
    build_player_team_abbreviation_map,
    fetch_the_odds_api_player_points_lines,
    normalize_player_name,
)
from src.data import get_nba_data
from src.service import (
    evaluate_test_set,
    get_matchup_rosters,
    model_summary,
    predict_matchup,
    team_lookup,
)

st.set_page_config(page_title="NBA Player Prop Predictor", layout="wide")

st.markdown(
    """
<style>
:root {
    --sport-bg: #0b1220;
    --sport-surface: #111b2f;
    --sport-card: #16233d;
    --sport-accent: #f97316;
    --sport-accent-2: #22c55e;
    --sport-text: #e5e7eb;
    --sport-muted: #94a3b8;
    --sport-border: #223250;
}

@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700;800&family=Teko:wght@500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(circle at 10% 15%, rgba(249, 115, 22, 0.18), transparent 35%),
        radial-gradient(circle at 90% 10%, rgba(34, 197, 94, 0.15), transparent 35%),
        linear-gradient(180deg, #0a101c 0%, var(--sport-bg) 100%);
    color: var(--sport-text);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1627 0%, #0d1a2f 100%);
    border-right: 1px solid var(--sport-border);
}

h1, h2, h3 {
    font-family: 'Teko', sans-serif !important;
    letter-spacing: 0.03em;
    color: #f8fafc;
}

p, label, div, span {
    font-family: 'Barlow', sans-serif;
}

.sport-hero {
    border: 1px solid var(--sport-border);
    background: linear-gradient(135deg, rgba(249, 115, 22, 0.14), rgba(22, 35, 61, 0.9));
    border-radius: 14px;
    padding: 0.9rem 1rem 0.8rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(3, 10, 23, 0.35);
}

.sport-hero h2 {
    margin: 0;
    line-height: 1;
    font-size: 2rem;
}

.sport-hero p {
    margin: 0.2rem 0 0;
    color: #cbd5e1;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(17, 27, 47, 0.96), rgba(14, 23, 40, 0.96));
    border: 1px solid var(--sport-border);
    border-radius: 12px;
    padding: 0.65rem 0.8rem;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
}

[data-testid="stMetricLabel"] {
    color: var(--sport-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #f8fafc;
    font-family: 'Teko', sans-serif;
    letter-spacing: 0.02em;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid #fb923c;
    background: linear-gradient(180deg, #fb923c, #f97316);
    color: #0f172a;
    font-weight: 700;
    letter-spacing: 0.01em;
}

.stButton > button:hover {
    border-color: #fdba74;
    background: linear-gradient(180deg, #fdba74, #fb923c);
}

.stButton > button:focus {
    box-shadow: 0 0 0 0.2rem rgba(249, 115, 22, 0.35);
}

[data-baseweb="select"] > div,
.stDateInput > div,
.stNumberInput > div,
.stTextInput > div,
[data-testid="stSlider"] {
    background: var(--sport-surface);
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--sport-border);
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stExpander"] {
    border: 1px solid var(--sport-border);
    border-radius: 10px;
    background: rgba(17, 27, 47, 0.8);
}

.stCaption {
    color: var(--sport-muted) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="sport-hero">
  <h2>NBA Player Prop Predictor</h2>
  <p>Quantile-driven projections for matchup-focused prop research.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.caption("Data and model load automatically, so you can jump straight to matchup quantile predictions.")

DATA_START_YEAR = 2020
TRAIN_TEST_SPLIT_DATE = pd.Timestamp("2024-06-18")
MODEL_ARTIFACT_PATH = Path("models/player_prop_artifacts_opp28.pt")
BETTING_LINES_PATH = Path("prize_picks_lines.csv")


def _rolling_end_year_exclusive(today: date | None = None) -> int:
    """Return end_year (exclusive) so current season is included automatically."""
    today = today or date.today()
    return today.year + (1 if today.month >= 9 else 0)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Load and cache rolling processed datasets from 2020 to present."""
    end_year = _rolling_end_year_exclusive()
    train_df, current_players, current_teams = get_nba_data(start_year=DATA_START_YEAR, end_year=end_year)
    return train_df, current_players, current_teams, end_year


@st.cache_resource(show_spinner=False)
def load_pretrained_artifacts(artifact_path: str):
    """Load pre-trained model artifacts from disk."""
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model artifact at '{path}'. Run scripts/train_artifact.py to generate it."
        )
    try:
        # Artifact files are generated locally by this project and include Python objects.
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Backward compatibility for older torch versions without weights_only.
        return torch.load(path, map_location="cpu")


def _team_label(row: pd.Series) -> str:
    """Build readable team labels for selectors."""
    parts: list[str] = []
    if "TEAM_ABBREVIATION" in row and pd.notna(row["TEAM_ABBREVIATION"]):
        parts.append(str(row["TEAM_ABBREVIATION"]))
    if "TEAM_NAME" in row and pd.notna(row["TEAM_NAME"]):
        parts.append(str(row["TEAM_NAME"]))

    return " - ".join(parts) if parts else "Unknown Team"


def _build_team_name_map(teams_df: pd.DataFrame) -> dict[int, str]:
    """Create TEAM_ID -> team label map for display tables."""
    name_map: dict[int, str] = {}
    for _, row in teams_df.iterrows():
        try:
            team_id = int(row["TEAM_ID"])
        except (TypeError, ValueError):
            continue
        name_map[team_id] = _team_label(row)
    return name_map


def _normalize_player_name(name: object) -> str:
    """Normalize player names for robust joins."""
    return normalize_player_name(name)


@st.cache_data(show_spinner=False, ttl=60 * 10)
def load_betting_lines_csv(csv_path: str) -> pd.DataFrame:
    """Load sampled betting lines CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing betting lines file at '{path}'.")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False, ttl=60 * 10)
def load_live_betting_lines(
    api_key: str,
    player_team_map_items: tuple[tuple[str, str], ...],
    regions: str = "us",
    market: str = "player_points",
    commence_days: int = 3,
) -> pd.DataFrame:
    """Fetch live player-points lines from The Odds API and cache the result."""
    player_team_map = dict(player_team_map_items)
    return fetch_the_odds_api_player_points_lines(
        api_key=api_key,
        player_team_map=player_team_map,
        regions=regions,
        market=market,
        commence_days=commence_days,
    )


def _build_team_abbreviation_map(teams_df: pd.DataFrame) -> dict[str, int]:
    """Build TEAM_ABBREVIATION -> TEAM_ID map."""
    if "TEAM_ABBREVIATION" not in teams_df.columns:
        return {}
    mapping: dict[str, int] = {}
    rows = teams_df[["TEAM_ID", "TEAM_ABBREVIATION"]].dropna().drop_duplicates()
    for _, row in rows.iterrows():
        try:
            mapping[str(row["TEAM_ABBREVIATION"]).strip().upper()] = int(row["TEAM_ID"])
        except (TypeError, ValueError):
            continue
    return mapping


def _sorted_quantile_columns(df: pd.DataFrame) -> list[str]:
    """Return quantile columns like q10/q50/q90 sorted by numeric suffix."""
    quant_cols = []
    for col in df.columns:
        if not isinstance(col, str) or not col.startswith("q"):
            continue
        try:
            quantile_value = float(col[1:])
        except (TypeError, ValueError):
            continue
        quant_cols.append((quantile_value, col))
    quant_cols.sort(key=lambda x: x[0])
    return [col for _, col in quant_cols]


@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_completed_player_stats_for_range(date_from_iso: str, date_to_iso: str) -> pd.DataFrame:
    """Fetch completed NBA player stats for a date range via nba_api."""
    empty = pd.DataFrame(columns=["game_day", "team", "player_key", "actual_value", "GAME_ID"])

    finder = leaguegamefinder.LeagueGameFinder(
        date_from_nullable=date_from_iso,
        date_to_nullable=date_to_iso,
        player_or_team_abbreviation="P",
    )
    frames = finder.get_data_frames()
    if not frames:
        return empty

    df = frames[0].copy()
    required = {"GAME_DATE", "TEAM_ABBREVIATION", "PLAYER_NAME", "PTS"}
    if not required.issubset(df.columns):
        return empty

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["game_day"] = df["GAME_DATE"].dt.normalize()
    df["team"] = df["TEAM_ABBREVIATION"].astype(str).str.strip().str.upper()
    df["player_key"] = df["PLAYER_NAME"].map(_normalize_player_name)
    df["actual_value"] = pd.to_numeric(df["PTS"], errors="coerce")

    result = df[["game_day", "team", "player_key", "actual_value", "GAME_ID"]].copy()
    result = result.dropna(subset=["game_day", "actual_value"])
    result = result[result["player_key"].ne("")]
    result = result.drop_duplicates(subset=["game_day", "team", "player_key"], keep="first")
    return result


def _ensure_state_defaults() -> None:
    """Initialize required session state keys."""
    st.session_state.setdefault("data_loaded", False)
    st.session_state.setdefault("train_df", None)
    st.session_state.setdefault("current_players", None)
    st.session_state.setdefault("current_teams", None)
    st.session_state.setdefault("artifacts", None)
    st.session_state.setdefault("data_end_year", None)
    st.session_state.setdefault("init_error", None)
    st.session_state.setdefault("test_eval", None)
    st.session_state.setdefault("test_eval_error", None)
    st.session_state.setdefault("latest_predictions", None)
    st.session_state.setdefault("last_matchup_key", None)
    st.session_state.setdefault("betting_preds_cache", {})
    st.session_state.setdefault("logged_live_line_snapshots", set())


_ensure_state_defaults()

with st.sidebar:
    st.header("Setup")
    st.caption("Data and model are prepared automatically.")
    st.caption(f"Data window: {DATA_START_YEAR} to present")
    st.caption(f"Fixed split date: {TRAIN_TEST_SPLIT_DATE.date()}")
    page = st.radio("Page", options=["Predict Matchup", "Betting Lines", "Test Stats"], index=0)

if not st.session_state.data_loaded or st.session_state.artifacts is None:
    try:
        with st.spinner("Loading data and model artifacts..."):
            train_df, current_players, current_teams, data_end_year = load_datasets()
            artifacts = load_pretrained_artifacts(str(MODEL_ARTIFACT_PATH))

        st.session_state.train_df = train_df
        st.session_state.current_players = current_players
        st.session_state.current_teams = current_teams
        st.session_state.artifacts = artifacts
        st.session_state.data_end_year = int(data_end_year)
        st.session_state.latest_predictions = None
        st.session_state.last_matchup_key = None
        st.session_state.data_loaded = True
        st.session_state.init_error = None

        try:
            st.session_state.test_eval = evaluate_test_set(df=train_df, artifacts=artifacts)
            st.session_state.test_eval_error = None
        except Exception as eval_exc:
            st.session_state.test_eval = None
            st.session_state.test_eval_error = str(eval_exc)
    except Exception as exc:
        st.session_state.init_error = str(exc)
        st.session_state.data_loaded = False

if not st.session_state.data_loaded:
    st.error(
        f"Startup initialization failed: {st.session_state.init_error}"
    )
    st.info("Generate the artifact with: python scripts/train_artifact.py")
    st.stop()

train_df: pd.DataFrame = st.session_state.train_df
current_players: pd.DataFrame = st.session_state.current_players
current_teams: pd.DataFrame = st.session_state.current_teams
artifacts = st.session_state.artifacts
player_team_map_items = tuple(
    sorted(build_player_team_abbreviation_map(current_players, current_teams).items())
)
odds_api_key = os.getenv("ODDS_API_KEY", "").strip()

summary = model_summary(artifacts)

if page == "Betting Lines":
    st.divider()
    st.subheader("Betting Lines Command Center")
    st.caption(
        "Use a CSV snapshot or a live Odds API feed for baseline lines. Completed-game actuals are fetched automatically from nba_api."
    )

    source_options = ["CSV snapshot"]
    default_source_index = 0
    if odds_api_key:
        source_options.insert(0, "Live Odds API")
    source_mode = st.radio("Line Source", options=source_options, horizontal=True, index=default_source_index)

    if source_mode == "Live Odds API":
        try:
            with st.spinner("Fetching live player-points lines..."):
                lines_df = load_live_betting_lines(
                    api_key=odds_api_key,
                    player_team_map_items=player_team_map_items,
                ).copy()
        except Exception as exc:
            st.error(f"Could not fetch live Odds API lines: {exc}")
            st.stop()
        if lines_df.empty:
            st.warning("The live Odds API request returned no upcoming player-points lines in the current window.")
            st.stop()
        live_snapshot_cols = [col for col in ["game_date", "player_name", "team", "opponent", "line", "market"] if col in lines_df.columns]
        live_snapshot_key = "||".join(
            lines_df[live_snapshot_cols]
            .sort_values(live_snapshot_cols)
            .astype(str)
            .agg("|".join, axis=1)
            .tolist()
        )
        if live_snapshot_key and live_snapshot_key not in st.session_state.logged_live_line_snapshots:
            history_path = append_line_history_snapshot(lines_df)
            st.session_state.logged_live_line_snapshots.add(live_snapshot_key)
            if history_path is not None:
                st.caption(f"Logged this live snapshot to `{history_path}` for future calibration.")
    else:
        try:
            lines_df = load_betting_lines_csv(str(BETTING_LINES_PATH)).copy()
        except Exception as exc:
            st.error(f"Could not load betting lines CSV: {exc}")
            st.stop()

    required_cols = {"game_date", "player_name", "team", "opponent", "line"}
    missing_cols = sorted(required_cols - set(lines_df.columns))
    if missing_cols:
        st.error(f"Betting lines CSV is missing required columns: {', '.join(missing_cols)}")
        st.stop()

    lines_df["game_date"] = pd.to_datetime(lines_df["game_date"], errors="coerce")
    lines_df["line"] = pd.to_numeric(lines_df["line"], errors="coerce")
    lines_df["team"] = lines_df["team"].astype(str).str.strip().str.upper()
    lines_df["opponent"] = lines_df["opponent"].astype(str).str.strip().str.upper()
    lines_df["player_key"] = lines_df["player_name"].map(_normalize_player_name)
    lines_df["game_day"] = lines_df["game_date"].dt.normalize()

    if "is_home" in lines_df.columns:
        is_home_str = lines_df["is_home"].astype(str).str.strip().str.lower()
        lines_df["is_home"] = is_home_str.isin({"1", "true", "t", "yes", "y", "home"})
    else:
        st.warning("Column is_home not found; defaulting all rows to home-team perspective.")
        lines_df["is_home"] = True

    abbr_map = _build_team_abbreviation_map(current_teams)
    lines_df["team_id"] = lines_df["team"].map(abbr_map)
    lines_df["opponent_team_id"] = lines_df["opponent"].map(abbr_map)
    lines_df["home_team_id"] = lines_df["team_id"].where(lines_df["is_home"], lines_df["opponent_team_id"])
    lines_df["away_team_id"] = lines_df["opponent_team_id"].where(lines_df["is_home"], lines_df["team_id"])

    invalid_rows = lines_df[
        lines_df["game_day"].isna()
        | lines_df["line"].isna()
        | lines_df["team_id"].isna()
        | lines_df["opponent_team_id"].isna()
        | lines_df["home_team_id"].isna()
        | lines_df["away_team_id"].isna()
        | lines_df["player_key"].eq("")
    ]
    if not invalid_rows.empty:
        st.warning(f"Skipping {len(invalid_rows)} invalid row(s) from betting lines CSV.")

    lines_df = lines_df.drop(invalid_rows.index).copy()
    if lines_df.empty:
        st.error("No valid betting lines to score.")
        st.stop()

    for col in ("team_id", "opponent_team_id", "home_team_id", "away_team_id"):
        lines_df[col] = lines_df[col].astype(int)

    unique_games = (
        lines_df[["game_day", "home_team_id", "away_team_id"]]
        .drop_duplicates()
        .sort_values(["game_day", "home_team_id", "away_team_id"])
        .reset_index(drop=True)
    )

    cache_seed = (
        unique_games["game_day"].dt.strftime("%Y-%m-%d")
        + "|"
        + unique_games["home_team_id"].astype(str)
        + "|"
        + unique_games["away_team_id"].astype(str)
    )
    matchup_cache_key = "v2||" + "||".join(cache_seed.tolist())

    if matchup_cache_key in st.session_state.betting_preds_cache:
        cached_preds = st.session_state.betting_preds_cache[matchup_cache_key].copy()
        cached_quant_cols = _sorted_quantile_columns(cached_preds)
        if "q50" in cached_preds.columns and len(cached_quant_cols) >= 2:
            preds_df = cached_preds
        else:
            preds_df = pd.DataFrame()
    else:
        preds_df = pd.DataFrame()

    if preds_df.empty:
        prediction_frames: list[pd.DataFrame] = []
        prediction_errors: list[str] = []

        with st.spinner("Generating model recommendations..."):
            for game in unique_games.itertuples(index=False):
                try:
                    preds = predict_matchup(
                        artifacts=artifacts,
                        current_players=current_players,
                        current_teams=current_teams,
                        history_df=train_df,
                        home_team_id=int(game.home_team_id),
                        away_team_id=int(game.away_team_id),
                        is_playoff=False,
                        enforce_official_roster=False,
                    ).copy()
                except Exception as exc:
                    prediction_errors.append(
                        f"{str(pd.Timestamp(game.game_day).date())} {int(game.home_team_id)} vs {int(game.away_team_id)}: {exc}"
                    )
                    continue

                preds["game_day"] = pd.Timestamp(game.game_day)
                preds["team_id"] = pd.to_numeric(preds["TEAM_ID"], errors="coerce").astype("Int64")
                preds["player_key"] = preds["PLAYER_NAME"].map(_normalize_player_name)
                quantile_cols = _sorted_quantile_columns(preds)
                keep_quant_cols = []
                if "q50" in preds.columns:
                    keep_quant_cols.append("q50")
                if quantile_cols:
                    keep_quant_cols.append(quantile_cols[0])
                    keep_quant_cols.append(quantile_cols[-1])
                keep_quant_cols = list(dict.fromkeys(keep_quant_cols))
                prediction_frames.append(preds[["game_day", "team_id", "player_key", *keep_quant_cols]])

        if prediction_errors:
            st.warning(f"{len(prediction_errors)} matchup prediction batch(es) failed and were skipped.")

        if prediction_frames:
            preds_df = pd.concat(prediction_frames, ignore_index=True)
            preds_df = preds_df.drop_duplicates(subset=["game_day", "team_id", "player_key"], keep="first")
        else:
            preds_df = pd.DataFrame(columns=["game_day", "team_id", "player_key", "q50"])

        st.session_state.betting_preds_cache[matchup_cache_key] = preds_df.copy()

    scored = lines_df.merge(
        preds_df,
        how="left",
        on=["game_day", "team_id", "player_key"],
    )
    scored_quant_cols = _sorted_quantile_columns(scored)
    interval_low_col = scored_quant_cols[0] if len(scored_quant_cols) >= 2 else None
    interval_high_col = scored_quant_cols[-1] if len(scored_quant_cols) >= 2 else None
    scored = attach_recommendation_columns(
        scored,
        q50_col="q50",
        line_col="line",
        q_low_col=interval_low_col,
        q_high_col=interval_high_col,
    )
    scored["model_recommendation"] = scored["bet_side"]

    # Pull official completed-game points from nba_api for each date represented in the lines file.
    unique_days = sorted(scored["game_day"].dropna().dt.strftime("%Y-%m-%d").unique().tolist())
    if unique_days:
        min_day = unique_days[0]
        max_day = unique_days[-1]
        try:
            with st.spinner("Fetching completed game stats from nba_api..."):
                api_actuals = load_completed_player_stats_for_range(min_day, max_day)
            api_actuals = api_actuals[api_actuals["game_day"].dt.strftime("%Y-%m-%d").isin(unique_days)].copy()
            api_actuals = api_actuals.drop_duplicates(subset=["game_day", "team", "player_key"], keep="first")
        except Exception as exc:
            st.warning(f"nba_api fetch failed for range {min_day} to {max_day}: {exc}")
            api_actuals = pd.DataFrame(columns=["game_day", "team", "player_key", "actual_value", "GAME_ID"])
    else:
        api_actuals = pd.DataFrame(columns=["game_day", "team", "player_key", "actual_value", "GAME_ID"])

    scored = scored.merge(
        api_actuals,
        how="left",
        on=["game_day", "team", "player_key"],
        suffixes=("", "_api"),
    )

    manual_actual_points = (
        pd.to_numeric(scored["actual_points"], errors="coerce")
        if "actual_points" in scored.columns
        else pd.Series(index=scored.index, dtype="float64")
    )
    manual_actual_generic = (
        pd.to_numeric(scored["actual"], errors="coerce")
        if "actual" in scored.columns
        else pd.Series(index=scored.index, dtype="float64")
    )
    manual_actual = manual_actual_points.combine_first(manual_actual_generic)
    scored["actual_value"] = scored["actual_value"].combine_first(manual_actual)

    scored["actual_side"] = "pending"
    scored.loc[scored["actual_value"].notna() & (scored["actual_value"] > scored["line"]), "actual_side"] = "over"
    scored.loc[scored["actual_value"].notna() & (scored["actual_value"] < scored["line"]), "actual_side"] = "under"
    scored.loc[scored["actual_value"].notna() & (scored["actual_value"] == scored["line"]), "actual_side"] = "push"

    graded_mask = (
        scored["model_recommendation"].isin(["over", "under"])
        & scored["actual_side"].isin(["over", "under"])
    )
    scored["correct"] = pd.NA
    scored.loc[graded_mask, "correct"] = (
        scored.loc[graded_mask, "model_recommendation"] == scored.loc[graded_mask, "actual_side"]
    )

    scored["status"] = "pending"
    scored.loc[scored["model_recommendation"] == "push", "status"] = "push"
    scored.loc[scored["actual_side"] == "push", "status"] = "push"
    scored.loc[graded_mask & (scored["correct"] == True), "status"] = "correct"
    scored.loc[graded_mask & (scored["correct"] == False), "status"] = "incorrect"

    total_lines = int(len(scored))
    graded_picks = int(graded_mask.sum())
    correct_picks = int((scored["status"] == "correct").sum())
    accuracy_pct = (100.0 * correct_picks / graded_picks) if graded_picks > 0 else None
    scored["confidence"] = scored["confidence_score"].fillna(0.0)
    if interval_low_col and interval_high_col:
        scored["interval"] = (
            scored[interval_low_col].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
            + " - "
            + scored[interval_high_col].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        )
    else:
        scored["interval"] = "N/A"

    # Visual pick cards
    card_df = scored.sort_values(["confidence", "game_date"], ascending=[False, True]).head(12).reset_index(drop=True)

    st.markdown("#### Pick Board")
    if card_df.empty:
        st.info("No lines available to display.")
    else:
        cols = st.columns(3)
        for i, row in card_df.iterrows():
            pick = str(row.get("model_recommendation", "pending")).upper()
            label_txt = str(row.get("bet_label", "lose")).upper()
            status = str(row.get("status", "pending")).upper()
            if row.get("status") == "correct":
                border = "#16a34a"
                badge_bg = "rgba(22, 163, 74, 0.2)"
            elif row.get("status") == "incorrect":
                border = "#dc2626"
                badge_bg = "rgba(220, 38, 38, 0.2)"
            elif row.get("bet_label") == "solid":
                border = "#22c55e"
                badge_bg = "rgba(34, 197, 94, 0.18)"
            elif row.get("bet_label") == "moderate":
                border = "#f97316"
                badge_bg = "rgba(249, 115, 22, 0.2)"
            elif row.get("bet_label") == "fair":
                border = "#eab308"
                badge_bg = "rgba(234, 179, 8, 0.18)"
            elif row.get("model_recommendation") in {"over", "under"}:
                border = "#f97316"
                badge_bg = "rgba(249, 115, 22, 0.2)"
            else:
                border = "#475569"
                badge_bg = "rgba(71, 85, 105, 0.2)"

            game_dt = row.get("game_date")
            game_label = pd.to_datetime(game_dt).strftime("%b %d") if pd.notna(game_dt) else "Unknown date"
            player = str(row.get("player_name", "Unknown Player"))
            team = str(row.get("team", "?"))
            opp = str(row.get("opponent", "?"))
            line_val = row.get("line")
            q50_val = row.get("q50")
            q_low_val = row.get(interval_low_col) if interval_low_col else None
            q_high_val = row.get(interval_high_col) if interval_high_col else None
            edge_val = row.get("edge")
            actual_val = row.get("actual_value")
            confidence_pct = row.get("confidence_pct")

            line_txt = f"{float(line_val):.1f}" if pd.notna(line_val) else "N/A"
            q50_txt = f"{float(q50_val):.1f}" if pd.notna(q50_val) else "N/A"
            interval_txt = (
                f"{float(q_low_val):.1f} - {float(q_high_val):.1f}"
                if pd.notna(q_low_val) and pd.notna(q_high_val)
                else "N/A"
            )
            edge_txt = f"{float(edge_val):+.1f}" if pd.notna(edge_val) else "N/A"
            actual_txt = f"{float(actual_val):.1f}" if pd.notna(actual_val) else "pending"
            confidence_txt = f"{float(confidence_pct):.1f}%" if pd.notna(confidence_pct) else "N/A"

            cols[i % 3].markdown(
                f"""
<div style="
    border: 1px solid {border};
    border-left: 5px solid {border};
    border-radius: 12px;
    padding: 0.75rem 0.85rem;
    margin-bottom: 0.8rem;
    background: linear-gradient(180deg, rgba(15,23,42,0.9), rgba(17,27,47,0.92));
">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:0.6rem;">
    <div style="font-weight:700;color:#f8fafc;">{player}</div>
    <div style="font-size:0.72rem;padding:0.12rem 0.4rem;border-radius:999px;background:{badge_bg};color:#e2e8f0;">{status}</div>
  </div>
  <div style="color:#94a3b8;font-size:0.82rem;">{game_label} | {team} vs {opp}</div>
  <div style="margin-top:0.4rem;display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.35rem;font-size:0.82rem;">
    <div><span style="color:#94a3b8;">Line</span><br><b style="color:#f1f5f9;">{line_txt}</b></div>
    <div><span style="color:#94a3b8;">Model q50</span><br><b style="color:#f1f5f9;">{q50_txt}</b></div>
    <div><span style="color:#94a3b8;">Edge</span><br><b style="color:#f1f5f9;">{edge_txt}</b></div>
  </div>
    <div style="margin-top:0.35rem;font-size:0.78rem;color:#cbd5e1;">Interval ({interval_low_col or 'N/A'}-{interval_high_col or 'N/A'}): <b>{interval_txt}</b></div>
  <div style="margin-top:0.45rem;display:flex;justify-content:space-between;align-items:center;">
    <div style="font-size:0.76rem;color:#cbd5e1;">Pick: <b>{pick}</b> | Label: <b>{label_txt}</b></div>
    <div style="font-size:0.76rem;color:#cbd5e1;">Confidence: <b>{confidence_txt}</b></div>
  </div>
  <div style="margin-top:0.2rem;display:flex;justify-content:space-between;align-items:center;">
    <div style="font-size:0.76rem;color:#cbd5e1;">Actual PTS: <b>{actual_txt}</b></div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Lines", f"{total_lines:,}")
    k2.metric("Graded Picks", f"{graded_picks:,}")
    k3.metric("Accuracy", f"{accuracy_pct:.1f}%" if accuracy_pct is not None else "N/A")

    chart_left, chart_right = st.columns([3, 2])
    with chart_left:
        edge_plot_df = scored[scored["model_recommendation"].isin(["over", "under"])].copy()
        if not edge_plot_df.empty:
            edge_plot_df = edge_plot_df.sort_values("confidence", ascending=False).head(25)
            hover_cols = [col for col in ["team", "opponent", "line", interval_low_col, "q50", interval_high_col, "actual_value"] if col in edge_plot_df.columns]
            fig_edge = px.bar(
                edge_plot_df,
                x="player_name",
                y="edge",
                color="status",
                hover_data=hover_cols,
                title="Top Model Edges (q50 - line)",
                color_discrete_map={
                    "correct": "#16a34a",
                    "incorrect": "#dc2626",
                    "pending": "#f97316",
                    "push": "#64748b",
                },
            )
            fig_edge.update_layout(margin={"l": 10, "r": 10, "t": 40, "b": 10}, xaxis_title="", yaxis_title="Edge")
            st.plotly_chart(fig_edge, use_container_width=True)
        else:
            st.info("No model edges to visualize yet.")

    with chart_right:
        status_counts = scored["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count")
        if not status_counts.empty:
            fig_status = px.pie(
                status_counts,
                names="status",
                values="count",
                hole=0.58,
                title="Pick Status Mix",
                color="status",
                color_discrete_map={
                    "correct": "#16a34a",
                    "incorrect": "#dc2626",
                    "pending": "#f97316",
                    "push": "#64748b",
                },
            )
            fig_status.update_layout(margin={"l": 10, "r": 10, "t": 40, "b": 10})
            st.plotly_chart(fig_status, use_container_width=True)

    graded = scored.loc[graded_mask, ["game_day", "correct"]].copy()
    if not graded.empty:
        graded["correct_int"] = graded["correct"].astype(bool).astype(int)
        daily = (
            graded.groupby("game_day", as_index=False)
            .agg(correct=("correct_int", "sum"), graded=("correct_int", "size"))
            .sort_values("game_day")
            .reset_index(drop=True)
        )
        daily["cum_correct"] = daily["correct"].cumsum()
        daily["cum_graded"] = daily["graded"].cumsum()
        daily["cum_accuracy_pct"] = 100.0 * daily["cum_correct"] / daily["cum_graded"]

        fig_acc = px.line(
            daily,
            x="game_day",
            y="cum_accuracy_pct",
            markers=True,
            labels={"game_day": "Date", "cum_accuracy_pct": "Cumulative correct (%)"},
            title="Model Pick Accuracy Over Time",
        )
        fig_acc.update_layout(margin={"l": 10, "r": 10, "t": 40, "b": 10})
        st.plotly_chart(fig_acc, use_container_width=True)
    else:
        st.info("No graded picks yet. Completed game stats will appear automatically after games finalize.")

    table_cols = [
        "game_date",
        "player_name",
        "team",
        "opponent",
        "line",
        interval_low_col if interval_low_col else "q10",
        "q50",
        interval_high_col if interval_high_col else "q90",
        "interval",
        "model_recommendation",
        "bet_label",
        "confidence_pct",
        "actual_value",
        "actual_side",
        "status",
        "sportsbook_count",
        "sportsbooks",
        "source",
    ]
    sort_cols = [col for col in ["confidence", "game_date", "team", "player_name"] if col in scored.columns]
    sorted_scored = scored.sort_values(sort_cols, ascending=[False, True, True, True][: len(sort_cols)]) if sort_cols else scored
    with st.expander("Detailed Lines Table", expanded=False):
        st.dataframe(
            sorted_scored[[col for col in table_cols if col in sorted_scored.columns]].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
    st.stop()

if page == "Test Stats":
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Training Rows", f"{len(train_df):,}")
    col_b.metric("Current Players", f"{len(current_players):,}")
    col_c.metric("Current Teams", f"{len(current_teams):,}")

    st.divider()
    st.subheader("Model Setup")
    info_1, info_2, info_3 = st.columns(3)
    info_1.metric("Artifact Split Date", summary["train_end_date"])
    info_2.metric("Artifact Train Rows", f"{int(summary['train_rows']):,}")
    info_3.metric("Artifact Test Rows", f"{int(summary['test_rows']):,}")

    st.caption(
        f"Data seasons fetched: {DATA_START_YEAR}-{st.session_state.data_end_year - 1} | "
        f"Artifact split date: {TRAIN_TEST_SPLIT_DATE.date()}"
    )

    if summary["train_end_date"] != str(TRAIN_TEST_SPLIT_DATE.date()):
        st.warning(
            "Loaded artifact was trained with a different split date. "
            f"Expected {TRAIN_TEST_SPLIT_DATE.date()}, got {summary['train_end_date']}."
        )

    if st.session_state.test_eval is None and st.session_state.test_eval_error is None:
        try:
            with st.spinner("Computing test diagnostics..."):
                st.session_state.test_eval = evaluate_test_set(df=train_df, artifacts=artifacts)
        except Exception as eval_exc:
            st.session_state.test_eval_error = str(eval_exc)

    st.divider()
    st.subheader("Model Evaluation Diagnostics")

    if st.session_state.test_eval_error:
        st.warning(f"Could not compute test diagnostics: {st.session_state.test_eval_error}")

    if st.session_state.test_eval is not None:
        test_eval = st.session_state.test_eval

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE (q50)", f"{test_eval.summary['mae_q50']:.3f}")
        m2.metric("RMSE (q50)", f"{test_eval.summary['rmse_q50']:.3f}")
        m3.metric("R2 (q50)", f"{test_eval.summary['r2_q50']:.3f}")

        m4, m5, m6 = st.columns(3)
        m4.metric("Test Rows", f"{int(test_eval.summary['test_rows']):,}")
        m5.metric("Avg Width (q10-q90)", f"{test_eval.summary['interval_width_q10_q90']:.3f}")
        m6.metric("Coverage (q10-q90)", f"{test_eval.summary['interval_coverage_q10_q90']:.3f}")

        st.markdown("#### Empirical Coverage Calibration")
        calibration_df = test_eval.quantile_metrics[["nominal_quantile", "empirical_coverage"]].copy()
        calibration_df = calibration_df.sort_values("nominal_quantile").reset_index(drop=True)

        fig_calibration = go.Figure()
        fig_calibration.add_trace(
            go.Scatter(
                x=calibration_df["nominal_quantile"],
                y=calibration_df["empirical_coverage"],
                mode="lines+markers",
                name="Empirical",
            )
        )
        fig_calibration.add_trace(
            go.Scatter(
                x=calibration_df["nominal_quantile"],
                y=calibration_df["nominal_quantile"],
                mode="lines",
                name="Ideal",
                line={"dash": "dash"},
            )
        )
        fig_calibration.update_layout(
            xaxis_title="Nominal quantile",
            yaxis_title="Empirical coverage",
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
        )
        st.plotly_chart(fig_calibration, use_container_width=True)

        st.markdown("#### Interval Width vs Total Games In Dataset")
        player_profile = test_eval.player_interval_profile.copy()
        if not player_profile.empty:
            fig_profile = px.scatter(
                player_profile,
                x="total_games_in_dataset",
                y="mean_interval_width_q10_q90",
                hover_data=["PLAYER_NAME", "outlier_rate", "test_rows"],
                labels={
                    "total_games_in_dataset": "Total games in dataset",
                    "mean_interval_width_q10_q90": "Mean interval width (q10-q90)",
                },
            )
            fig_profile.update_traces(marker={"size": 10, "opacity": 0.7})
            fig_profile.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10})
            st.plotly_chart(fig_profile, use_container_width=True)
        else:
            st.info("Player interval profile is unavailable.")

        st.markdown("#### Performance by Player Data Volume")
        st.dataframe(test_eval.games_bucket_metrics, use_container_width=True, hide_index=True)

    st.stop()

st.divider()
st.subheader("Predict Matchup")
st.caption("Pick teams, run the model, and scan projections with visual summaries.")

teams_df = team_lookup(current_teams)
if teams_df.empty:
    st.error("No teams available for selection.")
    st.stop()

teams_df = teams_df.copy()
teams_df["label"] = teams_df.apply(_team_label, axis=1)
team_name_map = _build_team_name_map(teams_df)
team_ids = teams_df["TEAM_ID"].astype(int).tolist()

col_home, col_away, col_game = st.columns([2, 2, 1])
with col_home:
    home_team_id = st.selectbox(
        "Team Name (Home)",
        options=team_ids,
        index=0,
        format_func=lambda tid: team_name_map.get(int(tid), "Unknown Team"),
    )
with col_away:
    away_candidates = [tid for tid in team_ids if tid != int(home_team_id)]
    away_team_id = st.selectbox(
        "Team Name (Away)",
        options=away_candidates,
        index=0,
        format_func=lambda tid: team_name_map.get(int(tid), "Unknown Team"),
    )
with col_game:
    is_playoff = st.toggle("Playoffs", value=False)

home_label = team_name_map.get(int(home_team_id), "Home Team")
away_label = team_name_map.get(int(away_team_id), "Away Team")
playoff_label = "PLAYOFF GAME" if is_playoff else "REGULAR SEASON"
st.markdown(
    f"""
<div style="
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    margin: 0.35rem 0 0.8rem;
    background: linear-gradient(120deg, rgba(14,23,40,0.95), rgba(15,23,42,0.86));
">
  <div style="font-size:0.76rem;color:#94a3b8;letter-spacing:0.06em;">MATCHUP PREVIEW</div>
  <div style="display:flex;justify-content:space-between;align-items:center;gap:0.8rem;">
    <div style="font-size:1.35rem;font-family:'Teko',sans-serif;color:#f8fafc;">{home_label} vs {away_label}</div>
    <div style="font-size:0.75rem;padding:0.2rem 0.5rem;border-radius:999px;background:rgba(249,115,22,0.2);color:#fdba74;">{playoff_label}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

current_matchup_key = (int(home_team_id), int(away_team_id), bool(is_playoff))
if st.session_state.last_matchup_key != current_matchup_key:
    # Prevent stale prediction tables from previous team selections.
    st.session_state.latest_predictions = None
    st.session_state.last_matchup_key = current_matchup_key

try:
    home_roster, away_roster = get_matchup_rosters(
        current_players=current_players,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        enforce_official_roster=True,
    )

    st.markdown("#### Official Roster Preview")
    rp1, rp2 = st.columns(2)
    with rp1:
        st.markdown(
            f"""
<div style="border:1px solid #334155;border-radius:12px;padding:0.7rem;background:rgba(15,23,42,0.65);">
  <div style="font-weight:700;color:#f8fafc;">{home_label}</div>
  <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:0.4rem;">{len(home_roster)} active players</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.dataframe(home_roster[["PLAYER_NAME"]], use_container_width=True, hide_index=True, height=320)
    with rp2:
        st.markdown(
            f"""
<div style="border:1px solid #334155;border-radius:12px;padding:0.7rem;background:rgba(15,23,42,0.65);">
  <div style="font-weight:700;color:#f8fafc;">{away_label}</div>
  <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:0.4rem;">{len(away_roster)} active players</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.dataframe(away_roster[["PLAYER_NAME"]], use_container_width=True, hide_index=True, height=320)
except Exception as exc:
    st.warning(f"Could not load roster preview: {exc}")

if st.button("Run Matchup Model", type="primary"):
    try:
        with st.spinner("Calculating player predictions..."):
            output = predict_matchup(
                artifacts=artifacts,
                current_players=current_players,
                current_teams=current_teams,
                history_df=train_df,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                is_playoff=is_playoff,
                enforce_official_roster=True,
            )
        st.session_state.latest_predictions = output
    except Exception as exc:
        st.session_state.latest_predictions = None
        st.error(f"Prediction failed: {exc}")

if st.session_state.latest_predictions is not None:
    prediction_df = st.session_state.latest_predictions.copy()

    home_preds = prediction_df[prediction_df["TEAM_ID"] == int(home_team_id)].copy()
    away_preds = prediction_df[prediction_df["TEAM_ID"] == int(away_team_id)].copy()

    quantile_cols = [col for col in prediction_df.columns if col.startswith("q")]
    q50_col = "q50" if "q50" in quantile_cols else (quantile_cols[0] if quantile_cols else None)
    q10_col = "q10" if "q10" in quantile_cols else None
    q90_col = "q90" if "q90" in quantile_cols else None

    if q10_col and q90_col:
        home_preds["interval_width"] = home_preds[q90_col] - home_preds[q10_col]
        away_preds["interval_width"] = away_preds[q90_col] - away_preds[q10_col]
    else:
        home_preds["interval_width"] = pd.NA
        away_preds["interval_width"] = pd.NA

    if q50_col:
        home_preds = home_preds.sort_values(q50_col, ascending=False)
        away_preds = away_preds.sort_values(q50_col, ascending=False)

    show_cols = [
        "PLAYER_NAME",
        *quantile_cols,
        "interval_width",
        "history_games",
        "sequence_history_games",
        "interval_scale",
    ]

    st.markdown("#### Full Team Forecasts")
    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown(f"##### {home_label}")
        st.dataframe(home_preds[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True, height=520)
    with fc2:
        st.markdown(f"##### {away_label}")
        st.dataframe(away_preds[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True, height=520)

    st.markdown("#### Single-Player Bet Recommendation")
    selector_team_options = [int(home_team_id), int(away_team_id)]
    rc1, rc2, rc3 = st.columns([1.2, 1.6, 1.2])
    with rc1:
        selected_team_id = st.selectbox(
            "Recommendation Team",
            options=selector_team_options,
            format_func=lambda tid: home_label if int(tid) == int(home_team_id) else away_label,
        )

    selected_team_preds = prediction_df[prediction_df["TEAM_ID"] == int(selected_team_id)].copy()
    selected_team_preds = selected_team_preds.sort_values(q50_col, ascending=False) if q50_col else selected_team_preds
    player_options = selected_team_preds["PLAYER_NAME"].dropna().tolist()

    with rc2:
        selected_player = st.selectbox("Player", options=player_options)

    live_line_default = None
    live_line_context = ""
    line_source_options = ["Manual line"]
    if odds_api_key:
        line_source_options.insert(0, "Live Odds API line")

    with rc3:
        line_source = st.selectbox("Baseline Source", options=line_source_options)

    if line_source == "Live Odds API line":
        try:
            live_lines_df = load_live_betting_lines(
                api_key=odds_api_key,
                player_team_map_items=player_team_map_items,
            ).copy()
        except Exception as exc:
            live_lines_df = pd.DataFrame()
            st.warning(f"Live odds lookup failed: {exc}")

        if not live_lines_df.empty:
            selected_team_abbr = current_teams.loc[
                current_teams["TEAM_ID"] == int(selected_team_id), "TEAM_ABBREVIATION"
            ].astype(str).str.upper().tolist()
            selected_opponent_abbr = away_preds["TEAM_ID"].iloc[0] if int(selected_team_id) == int(home_team_id) else home_preds["TEAM_ID"].iloc[0]
            opponent_abbr_lookup = current_teams.loc[
                current_teams["TEAM_ID"] == int(selected_opponent_abbr), "TEAM_ABBREVIATION"
            ].astype(str).str.upper().tolist()
            player_key = _normalize_player_name(selected_player)
            live_match = live_lines_df[
                live_lines_df["player_key"].eq(player_key)
                & live_lines_df["team"].isin(selected_team_abbr)
                & live_lines_df["opponent"].isin(opponent_abbr_lookup)
            ].sort_values("game_date")
            if not live_match.empty:
                live_line_default = float(live_match.iloc[0]["line"])
                live_line_context = (
                    f"Live baseline from {int(live_match.iloc[0]['sportsbook_count'])} sportsbook(s): "
                    f"{live_match.iloc[0]['sportsbooks']}"
                )
            else:
                st.info("No live line was found for that player in the current matchup window, so you can enter one manually.")

    selected_player_row = selected_team_preds[selected_team_preds["PLAYER_NAME"] == selected_player].head(1).copy()
    fallback_line = float(selected_player_row[q50_col].iloc[0]) if q50_col and not selected_player_row.empty else 0.0
    baseline_line = st.number_input(
        "Sportsbook line",
        min_value=0.0,
        value=float(live_line_default if live_line_default is not None else fallback_line),
        step=0.5,
    )

    recommendation_df = selected_player_row.copy()
    recommendation_df["line"] = float(baseline_line)
    recommendation_df = attach_recommendation_columns(
        recommendation_df,
        q50_col=q50_col or "q50",
        line_col="line",
        q_low_col=q10_col,
        q_high_col=q90_col,
    )

    if not recommendation_df.empty:
        rec_row = recommendation_df.iloc[0]
        interval_txt = "N/A"
        if q10_col and q90_col and pd.notna(rec_row.get(q10_col)) and pd.notna(rec_row.get(q90_col)):
            interval_txt = f"{float(rec_row[q10_col]):.1f} - {float(rec_row[q90_col]):.1f}"

        if live_line_context:
            st.caption(live_line_context)

        st.markdown(
            f"""
<div style="
    border: 1px solid #334155;
    border-left: 6px solid #22c55e;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-top: 0.3rem;
    background: linear-gradient(180deg, rgba(15,23,42,0.92), rgba(17,27,47,0.94));
">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:0.75rem;">
    <div>
      <div style="font-size:0.78rem;color:#94a3b8;letter-spacing:0.06em;">USER-ORIENTED RECOMMENDATION</div>
      <div style="font-size:1.35rem;font-family:'Teko',sans-serif;color:#f8fafc;">{selected_player}</div>
    </div>
    <div style="font-size:0.82rem;padding:0.2rem 0.55rem;border-radius:999px;background:rgba(34,197,94,0.16);color:#dcfce7;">
      {str(rec_row.get('bet_label', 'lose')).upper()}
    </div>
  </div>
  <div style="margin-top:0.5rem;display:grid;grid-template-columns:repeat(5,1fr);gap:0.5rem;font-size:0.82rem;">
    <div><span style="color:#94a3b8;">Recommended side</span><br><b style="color:#f8fafc;">{str(rec_row.get('bet_side', 'pending')).upper()}</b></div>
    <div><span style="color:#94a3b8;">Sportsbook line</span><br><b style="color:#f8fafc;">{float(rec_row['line']):.1f}</b></div>
    <div><span style="color:#94a3b8;">Model q50</span><br><b style="color:#f8fafc;">{float(rec_row[q50_col]):.1f}</b></div>
    <div><span style="color:#94a3b8;">Edge</span><br><b style="color:#f8fafc;">{float(rec_row.get('edge', 0.0)):+.1f}</b></div>
    <div><span style="color:#94a3b8;">Confidence</span><br><b style="color:#f8fafc;">{float(rec_row.get('confidence_pct', 0.0)):.1f}%</b></div>
  </div>
  <div style="margin-top:0.45rem;font-size:0.8rem;color:#cbd5e1;">Model interval: <b>{interval_txt}</b></div>
  <div style="margin-top:0.2rem;font-size:0.78rem;color:#cbd5e1;">History games: <b>{int(rec_row.get('history_games', 0))}</b> | Sequence games used: <b>{int(rec_row.get('sequence_history_games', 0))}</b> | Interval scale: <b>{float(rec_row.get('interval_scale', 1.0)):.2f}x</b></div>
</div>
""",
            unsafe_allow_html=True,
        )
