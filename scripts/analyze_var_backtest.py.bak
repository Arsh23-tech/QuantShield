
import pandas as pd

from src.risk.var_backtesting import backtest_historical_var
from src.risk.var_backtest_diagnostics import (
    conditional_coverage_test,
    christoffersen_independence_test,
    kupiec_coverage_test,
)


def holm_adjust(p_values, alpha=0.05):
    """Holm-Bonferroni adjustment without an additional dependency."""
    p_values = [float(p) for p in p_values]
    m = len(p_values)

    if m == 0:
        return [], []

    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0

    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max

    rejected = [p < alpha for p in adjusted]
    return adjusted, rejected


def main():
    prices = pd.read_parquet("data/processed/market_prices.parquet")
    prices = prices.sort_values(["ticker", "date"])
    prices["simple_return"] = (
        prices.groupby("ticker")["adjusted_close"].pct_change()
    )

    returns = prices[["date", "ticker", "simple_return"]]

    forecasts = backtest_historical_var(
        returns,
        lookback=252,
        confidence_levels=(0.95, 0.99),
    )

    rows = []

    for ticker, group in forecasts.groupby("ticker", sort=True):
        group = group.sort_values("date")

        for confidence in (0.95, 0.99):
            suffix = int(confidence * 100)
            exception_column = f"exception_{suffix}"
            exceptions = group[exception_column]

            coverage = kupiec_coverage_test(
                exceptions, 1 - confidence
            )
            independence = christoffersen_independence_test(
                exceptions
            )
            conditional = conditional_coverage_test(
                exceptions, 1 - confidence
            )

            rows.append({
                "ticker": ticker,
                "confidence": confidence,
                "exceptions": coverage["exceptions"],
                "observed_rate": coverage["observed_exception_rate"],
                "kupiec_p": coverage["p_value"],
                "independence_p": independence["p_value"],
                "conditional_p": conditional["p_value"],
            })

    results = pd.DataFrame(rows)

    # Initialize Holm-adjusted columns with the correct data types.
    for diagnostic in ("kupiec", "independence", "conditional"):
        results[f"{diagnostic}_holm_p"] = float("nan")
        results[f"{diagnostic}_holm_reject"] = pd.Series(
            False, index=results.index, dtype="bool"
        )

    # Apply Holm correction separately for each confidence level
    # and diagnostic family.
    for confidence in (0.95, 0.99):
        mask = results["confidence"] == confidence

        for diagnostic in ("kupiec", "independence", "conditional"):
            p_column = f"{diagnostic}_p"
            adjusted_column = f"{diagnostic}_holm_p"
            reject_column = f"{diagnostic}_holm_reject"

            adjusted, rejected = holm_adjust(
                results.loc[mask, p_column].tolist()
            )

            results.loc[mask, adjusted_column] = adjusted
            results.loc[mask, reject_column] = rejected

    columns = [
        "ticker",
        "confidence",
        "exceptions",
        "observed_rate",
        "kupiec_p",
        "kupiec_holm_p",
        "kupiec_holm_reject",
        "independence_p",
        "independence_holm_p",
        "independence_holm_reject",
        "conditional_p",
        "conditional_holm_p",
        "conditional_holm_reject",
    ]

    print("\nVA R BACKTEST DIAGNOSTICS — HOLM-ADJUSTED")
    print(results[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nRejections after Holm correction:")
    for confidence in (0.95, 0.99):
        subset = results[results["confidence"] == confidence]
        print(f"\n{confidence:.0%} VaR")
        for diagnostic in ("kupiec", "independence", "conditional"):
            count = int(subset[f"{diagnostic}_holm_reject"].sum())
            print(f"  {diagnostic}: {count}/{len(subset)} assets")


if __name__ == "__main__":
    main()
