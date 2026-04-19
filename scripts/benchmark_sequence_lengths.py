"""Benchmark transformer sequence lengths on the held-out NBA test split."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import get_nba_data
from src.service import evaluate_test_set, train_model

DEFAULT_START_YEAR = 2020
DEFAULT_SPLIT_DATE = "2024-06-18"
DEFAULT_SEQUENCE_LENGTHS = (10, 20, 30, 40)


def _rolling_end_year_exclusive(today: date | None = None) -> int:
    """Return end_year (exclusive) so current season is included automatically."""
    today = today or date.today()
    return today.year + (1 if today.month >= 9 else 0)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for sequence-length benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark multiple sequence lengths.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=_rolling_end_year_exclusive())
    parser.add_argument("--split-date", type=str, default=DEFAULT_SPLIT_DATE)
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEQUENCE_LENGTHS),
        help="One or more sequence lengths to benchmark.",
    )
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("data/sequence_length_benchmark.csv"))
    return parser.parse_args()


def main() -> None:
    """Train/evaluate one artifact per sequence length and summarize results."""
    args = parse_args()
    if args.end_year <= args.start_year:
        raise ValueError("end-year must be greater than start-year.")

    print(f"Fetching data from {args.start_year} to {args.end_year} (exclusive)...")
    train_df, _, _ = get_nba_data(start_year=args.start_year, end_year=args.end_year)

    results: list[dict[str, float | int | str]] = []
    for seq_len in args.sequence_lengths:
        print(f"\nBenchmarking sequence_length={seq_len}...")
        artifacts = train_model(
            df=train_df,
            split_date=args.split_date,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            early_stopping_patience=args.patience,
            sequence_length=seq_len,
        )
        test_eval = evaluate_test_set(df=train_df, artifacts=artifacts)
        results.append(
            {
                "sequence_length": int(seq_len),
                "epochs_trained": int(artifacts.epochs_trained),
                "train_loss": float(artifacts.train_loss),
                "val_loss": float(artifacts.val_loss),
                "test_loss": float(artifacts.test_loss),
                "mae_q50": float(test_eval.summary["mae_q50"]),
                "rmse_q50": float(test_eval.summary["rmse_q50"]),
                "r2_q50": float(test_eval.summary["r2_q50"]),
                "interval_width_q10_q90": float(test_eval.summary["interval_width_q10_q90"]),
                "interval_coverage_q10_q90": float(test_eval.summary["interval_coverage_q10_q90"]),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["mae_q50", "rmse_q50", "test_loss"],
        ascending=[True, True, True],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)

    print("\nBenchmark summary:")
    print(results_df.to_string(index=False))
    print(f"\nSaved benchmark results to {args.output}")


if __name__ == "__main__":
    main()
