"""Data fetching and feature engineering for NBA player prop modeling.

This module provides a single public pipeline function, ``get_nba_data``, that
returns:
1. Historical training data with leakage-safe shifted player features.
2. Current player feature state for inference.
3. Current team metadata for inference/UI.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

DEFAULT_START_YEAR = 2020
DEFAULT_END_YEAR = 2026
MIN_MODEL_DATE = pd.Timestamp("2020-01-01")

RAW_PLAYER_SEQUENCE_COLS: tuple[str, ...] = (
    "MIN",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "AST",
    "REB",
    "OREB",
    "DREB",
    "TOV",
    "STL",
    "BLK",
    "PF",
    "PLUS_MINUS",
)

PLAYER_INFERENCE_COLS: tuple[str, ...] = ()

TEAM_INFERENCE_COLS: tuple[str, ...] = ()


def _season_strings(start_year: int, end_year: int) -> list[str]:
    """Build season labels like ``2015-16`` from year bounds."""
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(start_year, end_year)]


def fetch_nba_api_data(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch regular season + playoff game logs for one season."""
    print(f"  -> Fetching {season}...")

    rs_players = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="P",
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]
    rs_teams = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="T",
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]
    rs_players["is_playoff"] = 0

    time.sleep(2)

    po_players = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="P",
        season_type_all_star="Playoffs",
    ).get_data_frames()[0]
    po_teams = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="T",
        season_type_all_star="Playoffs",
    ).get_data_frames()[0]
    po_players["is_playoff"] = 1

    time.sleep(2)

    player_frames = [frame for frame in (rs_players, po_players) if not frame.empty]
    team_frames = [frame for frame in (rs_teams, po_teams) if not frame.empty]

    players_df = pd.concat(player_frames, ignore_index=True) if player_frames else rs_players.copy()
    teams_df = pd.concat(team_frames, ignore_index=True) if team_frames else rs_teams.copy()
    return players_df, teams_df


def fetch_multiple_seasons(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and combine multi-season player/team logs from NBA API."""
    seasons = _season_strings(start_year=start_year, end_year=end_year)
    all_players_data: list[pd.DataFrame] = []
    all_teams_data: list[pd.DataFrame] = []

    print(f"Starting historical data fetch from {start_year} to {end_year}...")
    for season in seasons:
        try:
            players_df, teams_df = fetch_nba_api_data(season)
            all_players_data.append(players_df)
            all_teams_data.append(teams_df)
        except Exception as exc:
            print(f"Failed to fetch data for {season}. Error: {exc}")
            time.sleep(5)

    players_raw = pd.concat(all_players_data, ignore_index=True)
    teams_raw = pd.concat(all_teams_data, ignore_index=True)

    players_raw["GAME_DATE"] = pd.to_datetime(players_raw["GAME_DATE"])
    teams_raw["GAME_DATE"] = pd.to_datetime(teams_raw["GAME_DATE"])

    print(f"Success! Loaded {len(players_raw)} total player game logs.")
    return players_raw, teams_raw


def get_nba_data(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch and transform raw logs into training and inference datasets."""
    players_raw, teams_raw = fetch_multiple_seasons(start_year=start_year, end_year=end_year)

    df = players_raw.copy()
    df["home"] = df["MATCHUP"].str.contains("vs.").astype(int)
    df["MIN"] = df["MIN"].astype(float)
    df = df[df["MIN"] > 0].dropna(subset=["MIN", "PTS", "GAME_DATE"]).reset_index(drop=True)
    df = df.sort_values(by=["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    df["days_of_rest"] = df.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days
    df["days_of_rest"] = df["days_of_rest"].fillna(10).clip(upper=10)

    final_df = df.copy()

    columns_of_interest = [
        "GAME_ID",
        "GAME_DATE",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "PTS",
        "is_playoff",
        "home",
        "days_of_rest",
        *RAW_PLAYER_SEQUENCE_COLS,
        *PLAYER_INFERENCE_COLS,
        *TEAM_INFERENCE_COLS,
    ]

    final_df = final_df[columns_of_interest].copy()
    final_df = final_df[final_df["GAME_DATE"] >= MIN_MODEL_DATE].copy()

    final_df = final_df.reset_index(drop=True)

    current_players = df[df["GAME_DATE"] >= MIN_MODEL_DATE].groupby("PLAYER_ID").tail(1).copy()
    current_teams = teams_raw[teams_raw["GAME_DATE"] >= MIN_MODEL_DATE].groupby("TEAM_ID").tail(1).copy()

    player_context_cols = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "days_of_rest",
        *RAW_PLAYER_SEQUENCE_COLS,
        *PLAYER_INFERENCE_COLS,
    ]
    current_players = current_players[player_context_cols].copy()

    team_meta_cols = ["TEAM_ID"]
    for optional_col in ("TEAM_ABBREVIATION", "TEAM_NAME"):
        if optional_col in current_teams.columns:
            team_meta_cols.append(optional_col)
    current_teams = current_teams[team_meta_cols].copy()

    return final_df, current_players.reset_index(drop=True), current_teams.reset_index(drop=True)


__all__ = [
    "DEFAULT_START_YEAR",
    "DEFAULT_END_YEAR",
    "RAW_PLAYER_SEQUENCE_COLS",
    "PLAYER_INFERENCE_COLS",
    "TEAM_INFERENCE_COLS",
    "fetch_nba_api_data",
    "fetch_multiple_seasons",
    "get_nba_data",
]
