
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.risk.var_backtesting import backtest_historical_var


LOOKBACK = 252
CONFIDENCE_LEVELS = (0.95, 0.99)

TICKERS = (
    "AAPL",
    "IWM",
    "JNJ",
    "JPM",
    "MSFT",
    "QQQ",
    "SPY",
    "XOM",
)

OBSERVATIONS_PER_TICKER = 2518
MISSING_RATE = 0.005
REPEATS = 3
SEED = 20261119


def load_reference_function():
    """Load the exact slow reference helper from the existing test file."""
    test_file = ROOT / "tests" / "test_var_backtesting_rolling_queue.py"

    if not test_file.exists():
        raise FileNotFoundError(
            f"Reference test file not found: {test_file}"
        )

    namespace = runpy.run_path(str(test_file))

    if "slow_reference_backtest" not in namespace:
        raise RuntimeError(
            "slow_reference_backtest was not found in the test file."
        )

    return namespace["slow_reference_backtest"]


def build_synthetic_returns():
    """
    Generate reproducible daily return observations for eight tickers.

    Missing observations are retained as NaN rather than forward-filled.
    """
    rng = np.random.default_rng(SEED)

    dates = pd.bdate_range(
        start="2016-01-04",
        periods=OBSERVATIONS_PER_TICKER,
    )

    frames = []

    for ticker in TICKERS:
        values = rng.normal(
            loc=0.0003,
            scale=0.015,
            size=OBSERVATIONS_PER_TICKER,
        )

        missing_mask = (
            rng.random(OBSERVATIONS_PER_TICKER) < MISSING_RATE
        )

        values[missing_mask] = np.nan

        frames.append(
            pd.DataFrame({
                "date": dates,
                "ticker": ticker,
                "simple_return": values,
                "source_row_missing": missing_mask,
            })
        )

    return pd.concat(frames, ignore_index=True)


def time_function(function, returns):
    start = time.perf_counter()

    result = function(
        returns,
        lookback=LOOKBACK,
        confidence_levels=CONFIDENCE_LEVELS,
    )

    elapsed = time.perf_counter() - start

    return result, elapsed


def main():
    print("HISTORICAL VaR BACKTEST PERFORMANCE BENCHMARK")
    print("=" * 55)

    reference_function = load_reference_function()
    returns = build_synthetic_returns()

    print(f"Tickers: {len(TICKERS)}")
    print(f"Observations per ticker: {OBSERVATIONS_PER_TICKER:,}")
    print(f"Total input rows: {len(returns):,}")
    print(f"Lookback: {LOOKBACK}")
    print(f"Confidence levels: {CONFIDENCE_LEVELS}")
    print(f"Missing-return rate: {MISSING_RATE:.2%}")
    print(f"Repetitions: {REPEATS}")
    print(f"Random seed: {SEED}")

    old_times = []
    new_times = []

    reference_output = None
    optimized_output = None

    # Alternate execution order to reduce systematic ordering effects.
    for repetition in range(REPEATS):
        print(f"\nRepetition {repetition + 1}/{REPEATS}")

        if repetition % 2 == 0:
            reference_output, old_elapsed = time_function(
                reference_function,
                returns,
            )

            optimized_output, new_elapsed = time_function(
                backtest_historical_var,
                returns,
            )
        else:
            optimized_output, new_elapsed = time_function(
                backtest_historical_var,
                returns,
            )

            reference_output, old_elapsed = time_function(
                reference_function,
                returns,
            )

        old_times.append(old_elapsed)
        new_times.append(new_elapsed)

        print(f"Previous implementation: {old_elapsed:.3f} seconds")
        print(f"Optimized implementation: {new_elapsed:.3f} seconds")

    print("\nOUTPUT EQUIVALENCE")
    print("-" * 55)

    pd.testing.assert_frame_equal(
        reference_output,
        optimized_output,
        check_dtype=True,
        check_exact=True,
    )

    print("PASS: outputs match exactly.")
    print(f"Forecast rows: {len(optimized_output):,}")

    print("\nPERFORMANCE SUMMARY")
    print("-" * 55)

    old_median = float(np.median(old_times))
    new_median = float(np.median(new_times))

    speedup = old_median / new_median

    print(f"Previous median runtime: {old_median:.3f} seconds")
    print(f"Optimized median runtime: {new_median:.3f} seconds")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Runtime reduction: {(1 - new_median / old_median):.2%}")

    print("\nDATA QUALITY SUMMARY")
    print("-" * 55)

    print(f"Input missing returns: {returns['simple_return'].isna().sum():,}")
    print(
        "Output missing realized returns: "
        f"{optimized_output['return_missing'].sum():,}"
    )
    print(
        "Output source-row-missing flags: "
        f"{optimized_output['source_row_missing'].sum():,}"
    )

    print("\nBENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
