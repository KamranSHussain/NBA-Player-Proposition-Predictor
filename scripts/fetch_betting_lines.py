"""Export upcoming NBA player-points betting lines into the app CSV shape."""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from src.betting import (
    append_line_history_snapshot,
    build_player_team_abbreviation_map,
    default_export_path,
    fetch_the_odds_api_player_points_lines,
)
from src.data import get_nba_data


def _rolling_end_year_exclusive(today: date | None = None) -> int:
    """Return the exclusive end year so the current season is included."""
    today = today or date.today()
    return today.year + (1 if today.month >= 9 else 0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch NBA player-points lines from The Odds API and save them as CSV.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("ODDS_API_KEY", ""),
        help="The Odds API key. Defaults to ODDS_API_KEY env var.",
    )
    parser.add_argument(
        "--output",
        default=default_export_path(),
        help="Where to save the exported CSV.",
    )
    parser.add_argument(
        "--regions",
        default="us",
        help="Comma-delimited Odds API regions to query. Default: us",
    )
    parser.add_argument(
        "--bookmakers",
        default="",
        help="Optional comma-delimited bookmaker keys to restrict the feed.",
    )
    parser.add_argument(
        "--commence-days",
        type=int,
        default=3,
        help="Number of upcoming days to scan for games. Default: 3",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="Historical start year for the NBA roster context used in name/team mapping.",
    )
    parser.add_argument(
        "--skip-history-log",
        action="store_true",
        help="Skip appending this fetch to data/betting_line_history.csv.",
    )
    return parser.parse_args()


def main() -> None:
    """Fetch lines, normalize them, and write a CSV for the app."""
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set ODDS_API_KEY.")

    end_year = _rolling_end_year_exclusive()
    _, current_players, current_teams = get_nba_data(start_year=args.start_year, end_year=end_year)
    player_team_map = build_player_team_abbreviation_map(current_players, current_teams)

    lines_df = fetch_the_odds_api_player_points_lines(
        api_key=args.api_key,
        player_team_map=player_team_map,
        regions=args.regions,
        bookmakers=args.bookmakers or None,
        commence_days=args.commence_days,
    )

    if lines_df.empty:
        raise SystemExit("No player-points lines were returned for the requested window.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines_df.to_csv(output_path, index=False)
    print(f"Saved {len(lines_df):,} rows to {output_path}")
    if not args.skip_history_log:
        history_path = append_line_history_snapshot(lines_df)
        if history_path is not None:
            print(f"Appended snapshot history to {history_path}")


if __name__ == "__main__":
    main()
