"""Betting line utilities and recommendation helpers for the NBA prop app."""

from __future__ import annotations

import json
import math
import re
import string
import unicodedata
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
NBA_SPORT_KEY = "basketball_nba"
NBA_PLAYER_POINTS_MARKET = "player_points"
NBA_PROP_TO_APP_TARGET = {
    "player_points": "PTS",
}
DEFAULT_LINE_HISTORY_PATH = Path("data/betting_line_history.csv")

NBA_TEAM_NAME_TO_ABBR: dict[str, str] = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def normalize_player_name(name: object) -> str:
    """Normalize player names for joins across data sources."""
    if pd.isna(name):
        return ""
    value = unicodedata.normalize("NFKD", str(name))
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\s+", " ", value).strip()
    return value


def build_player_team_abbreviation_map(
    current_players: pd.DataFrame,
    current_teams: pd.DataFrame,
) -> dict[str, str]:
    """Build normalized player-name -> team abbreviation map from app datasets."""
    if current_players.empty or current_teams.empty:
        return {}

    team_abbr_map: dict[int, str] = {}
    if {"TEAM_ID", "TEAM_ABBREVIATION"}.issubset(current_teams.columns):
        teams = current_teams[["TEAM_ID", "TEAM_ABBREVIATION"]].dropna().drop_duplicates()
        for _, row in teams.iterrows():
            try:
                team_abbr_map[int(row["TEAM_ID"])] = str(row["TEAM_ABBREVIATION"]).strip().upper()
            except (TypeError, ValueError):
                continue

    player_map: dict[str, str] = {}
    for _, row in current_players[["PLAYER_NAME", "TEAM_ID"]].dropna().drop_duplicates().iterrows():
        player_key = normalize_player_name(row["PLAYER_NAME"])
        if not player_key:
            continue
        try:
            team_abbr = team_abbr_map[int(row["TEAM_ID"])]
        except (KeyError, TypeError, ValueError):
            continue
        player_map[player_key] = team_abbr
    return player_map


def _http_get_json(url: str, params: dict[str, Any], timeout: int = 30) -> tuple[Any, dict[str, str]]:
    """Execute a small JSON GET request with standard-library networking only."""
    query = urlencode(params, doseq=True)
    full_url = f"{url}?{query}" if query else url
    with urlopen(full_url, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        headers = {str(k): str(v) for k, v in response.headers.items()}
    return json.loads(body), headers


def fetch_the_odds_api_player_points_lines(
    api_key: str,
    player_team_map: dict[str, str],
    regions: str = "us",
    market: str = NBA_PLAYER_POINTS_MARKET,
    bookmakers: str | None = None,
    commence_days: int = 3,
) -> pd.DataFrame:
    """Fetch and aggregate upcoming NBA player-points lines from The Odds API."""
    if not api_key:
        raise ValueError("Missing The Odds API key.")

    commence_from = datetime.now(UTC) - timedelta(hours=6)
    commence_to = commence_from + timedelta(days=max(commence_days, 1))

    events, _ = _http_get_json(
        f"{ODDS_API_BASE_URL}/sports/{NBA_SPORT_KEY}/events",
        {
            "apiKey": api_key,
            "dateFormat": "iso",
            "commenceTimeFrom": commence_from.isoformat().replace("+00:00", "Z"),
            "commenceTimeTo": commence_to.isoformat().replace("+00:00", "Z"),
        },
    )
    if not isinstance(events, list) or not events:
        return pd.DataFrame(
            columns=[
                "game_date",
                "player_name",
                "is_home",
                "team",
                "opponent",
                "line",
                "market",
                "sportsbook_count",
                "sportsbooks",
                "source",
            ]
        )

    records: list[dict[str, Any]] = []
    for event in events:
        event_id = str(event.get("id", "")).strip()
        if not event_id:
            continue

        home_name = str(event.get("home_team", "")).strip()
        away_name = str(event.get("away_team", "")).strip()
        home_abbr = NBA_TEAM_NAME_TO_ABBR.get(home_name, "")
        away_abbr = NBA_TEAM_NAME_TO_ABBR.get(away_name, "")
        commence_time = event.get("commence_time")

        event_params: dict[str, Any] = {
            "apiKey": api_key,
            "regions": regions,
            "markets": market,
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        if bookmakers:
            event_params["bookmakers"] = bookmakers

        odds_payload, _ = _http_get_json(
            f"{ODDS_API_BASE_URL}/sports/{NBA_SPORT_KEY}/events/{event_id}/odds",
            event_params,
        )

        bookmakers_payload = odds_payload.get("bookmakers", []) if isinstance(odds_payload, dict) else []
        for bookmaker in bookmakers_payload:
            book_key = str(bookmaker.get("key", "")).strip()
            book_title = str(bookmaker.get("title", "")).strip() or book_key
            for market_payload in bookmaker.get("markets", []) or []:
                if str(market_payload.get("key", "")).strip() != market:
                    continue

                paired: dict[tuple[str, float], dict[str, Any]] = {}
                for outcome in market_payload.get("outcomes", []) or []:
                    player_name = str(
                        outcome.get("description")
                        or outcome.get("participant")
                        or outcome.get("player_name")
                        or ""
                    ).strip()
                    outcome_side = str(outcome.get("name", "")).strip().lower()
                    point = outcome.get("point")
                    if not player_name or point is None:
                        continue

                    try:
                        line_value = float(point)
                    except (TypeError, ValueError):
                        continue

                    pair_key = (player_name, line_value)
                    paired.setdefault(
                        pair_key,
                        {
                            "over_price": None,
                            "under_price": None,
                        },
                    )
                    if outcome_side == "over":
                        paired[pair_key]["over_price"] = outcome.get("price")
                    elif outcome_side == "under":
                        paired[pair_key]["under_price"] = outcome.get("price")

                for (player_name, line_value), price_info in paired.items():
                    player_key = normalize_player_name(player_name)
                    team_abbr = player_team_map.get(player_key, "")
                    if not team_abbr:
                        continue

                    if team_abbr == home_abbr:
                        is_home = True
                        opponent = away_abbr
                    elif team_abbr == away_abbr:
                        is_home = False
                        opponent = home_abbr
                    else:
                        continue

                    records.append(
                        {
                            "game_date": commence_time,
                            "player_name": player_name,
                            "player_key": player_key,
                            "is_home": is_home,
                            "team": team_abbr,
                            "opponent": opponent,
                            "line": line_value,
                            "market": market,
                            "sportsbook": book_title,
                            "over_price": price_info["over_price"],
                            "under_price": price_info["under_price"],
                            "source": "the_odds_api",
                        }
                    )

    raw_df = pd.DataFrame.from_records(records)
    if raw_df.empty:
        return raw_df

    raw_df["game_date"] = pd.to_datetime(raw_df["game_date"], errors="coerce")
    raw_df = raw_df.dropna(subset=["game_date", "line"])
    if raw_df.empty:
        return raw_df

    grouped = (
        raw_df.groupby(
            ["game_date", "player_name", "player_key", "is_home", "team", "opponent", "market", "source"],
            as_index=False,
        )
        .agg(
            line=("line", "median"),
            sportsbook_count=("sportsbook", "nunique"),
            sportsbooks=("sportsbook", lambda x: ", ".join(sorted({str(v) for v in x if str(v).strip()}))),
            over_price_avg=("over_price", "mean"),
            under_price_avg=("under_price", "mean"),
        )
        .sort_values(["game_date", "team", "player_name"])
        .reset_index(drop=True)
    )
    return grouped


def normal_cdf(value: float) -> float:
    """Standard normal cumulative density."""
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def recommendation_from_quantiles(
    q50: float | int | None,
    line: float | int | None,
    q_low: float | int | None = None,
    q_high: float | int | None = None,
) -> dict[str, Any]:
    """Convert model quantiles vs line into a side, confidence, and user label."""
    if q50 is None or line is None or pd.isna(q50) or pd.isna(line):
        return {
            "bet_side": "pending",
            "bet_label": "lose",
            "confidence_score": float("nan"),
            "confidence_pct": float("nan"),
            "edge": float("nan"),
            "edge_abs": float("nan"),
        }

    median = float(q50)
    threshold = float(line)
    edge = median - threshold
    side = "over" if edge > 0 else "under" if edge < 0 else "push"

    confidence = 0.5
    if q_low is not None and q_high is not None and pd.notna(q_low) and pd.notna(q_high):
        width = max(float(q_high) - float(q_low), 0.0)
        if width > 0:
            sigma = width / 2.5631031311  # q90 - q10 under a normal approximation
            over_prob = 1.0 - normal_cdf((threshold - median) / sigma)
            under_prob = 1.0 - over_prob
            confidence = max(over_prob, under_prob)

    if side == "push":
        label = "lose"
    elif confidence >= 0.67 or abs(edge) >= 4.0:
        label = "solid"
    elif confidence >= 0.60 or abs(edge) >= 2.5:
        label = "moderate"
    elif confidence >= 0.53 or abs(edge) >= 1.0:
        label = "fair"
    else:
        label = "lose"

    return {
        "bet_side": side,
        "bet_label": label,
        "confidence_score": confidence,
        "confidence_pct": confidence * 100.0,
        "edge": edge,
        "edge_abs": abs(edge),
    }


def attach_recommendation_columns(
    df: pd.DataFrame,
    q50_col: str = "q50",
    line_col: str = "line",
    q_low_col: str | None = None,
    q_high_col: str | None = None,
) -> pd.DataFrame:
    """Attach recommendation fields to a frame containing model quantiles and lines."""
    enriched = df.copy()

    recs = [
        recommendation_from_quantiles(
            q50=row.get(q50_col),
            line=row.get(line_col),
            q_low=row.get(q_low_col) if q_low_col else None,
            q_high=row.get(q_high_col) if q_high_col else None,
        )
        for _, row in enriched.iterrows()
    ]
    rec_df = pd.DataFrame(recs, index=enriched.index)
    return pd.concat([enriched, rec_df], axis=1)


def default_export_path(today: date | None = None) -> str:
    """Return a dated default filename for exported line snapshots."""
    today = today or date.today()
    return f"betting_lines_{today.isoformat()}.csv"


def append_line_history_snapshot(
    lines_df: pd.DataFrame,
    output_path: str | Path = DEFAULT_LINE_HISTORY_PATH,
    fetched_at: datetime | None = None,
    run_label: str | None = None,
) -> Path | None:
    """Append a fetched line snapshot to the local historical store."""
    if lines_df.empty:
        return None

    fetched_at = fetched_at or datetime.now(UTC)
    run_label = run_label or fetched_at.strftime("%Y%m%dT%H%M%SZ")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = lines_df.copy()
    snapshot["fetched_at"] = fetched_at.isoformat().replace("+00:00", "Z")
    snapshot["snapshot_date"] = fetched_at.date().isoformat()
    snapshot["source_run_id"] = run_label

    ordered_cols = [
        "fetched_at",
        "snapshot_date",
        "source_run_id",
        "game_date",
        "player_name",
        "player_key",
        "is_home",
        "team",
        "opponent",
        "line",
        "market",
        "sportsbook_count",
        "sportsbooks",
        "over_price_avg",
        "under_price_avg",
        "source",
    ]
    remaining_cols = [col for col in snapshot.columns if col not in ordered_cols]
    snapshot = snapshot[[col for col in ordered_cols if col in snapshot.columns] + remaining_cols]
    snapshot.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path
