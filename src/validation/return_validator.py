
import numpy as np
import pandas as pd


def validate_returns(
    df: pd.DataFrame,
    extreme_return_threshold: float = 0.20,
) -> dict:
    """
    Diagnose invalid or potentially suspicious calculated returns.

    Extreme returns are flagged for review, not automatically rejected.
    This function does not modify or remove observations.

    The threshold is a configurable review trigger, not a universal
    definition of an erroneous market return.
    """

    required_columns = {"ticker", "date", "simple_return"}
    missing_columns = sorted(required_columns - set(df.columns))

    if missing_columns:
        return {
            "status": "FAIL",
            "missing_columns": missing_columns,
            "rows_checked": len(df),
            "issues": {},
        }

    if (
        not isinstance(extreme_return_threshold, (int, float))
        or isinstance(extreme_return_threshold, bool)
        or not np.isfinite(extreme_return_threshold)
        or extreme_return_threshold <= 0
        or extreme_return_threshold >= 1
    ):
        raise ValueError(
            "extreme_return_threshold must be finite and between 0 and 1."
        )

    issues = {}

    # Parse dates consistently and report invalid date values.
    parsed_dates = pd.to_datetime(
        df["date"],
        errors="coerce",
        format="mixed",
        utc=True,
    )

    invalid_dates = parsed_dates.isna()

    if invalid_dates.any():
        issues["invalid_date_rows"] = int(invalid_dates.sum())

    # Require a non-empty string ticker identifier.
    valid_tickers = df["ticker"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )

    invalid_tickers = ~valid_tickers

    if invalid_tickers.any():
        issues["invalid_ticker_rows"] = int(invalid_tickers.sum())

    returns = pd.to_numeric(
        df["simple_return"],
        errors="coerce",
    )

    non_finite = ~np.isfinite(returns)

    # A first observation is defined chronologically for each valid ticker.
    # Invalid dates and tickers cannot establish a valid first observation.
    first_observation = pd.Series(
        False,
        index=df.index,
        dtype=bool,
    )

    valid_rows = parsed_dates.notna() & valid_tickers

    # Daily return data must contain at most one observation per ticker/date.
    duplicate_keys = pd.DataFrame(
        {
            "date": parsed_dates.dt.normalize(),
            "ticker": df["ticker"],
        },
        index=df.index,
    )

    valid_duplicate_keys = duplicate_keys.loc[valid_rows]

    duplicate_rows = valid_duplicate_keys.duplicated(
        subset=["date", "ticker"]
    ).sum()

    if duplicate_rows:
        issues["duplicate_ticker_date_rows"] = int(duplicate_rows)

    if valid_rows.any():
        chronological = pd.DataFrame(
            {
                "ticker": df.loc[valid_rows, "ticker"],
                "date": parsed_dates.loc[valid_rows],
            }
        )

        earliest_indices = (
            chronological.groupby("ticker")["date"].idxmin()
        )

        first_observation.loc[earliest_indices] = True

    unexpected_non_finite = non_finite & ~first_observation

    if unexpected_non_finite.any():
        issues["non_finite_returns_after_first_observation"] = int(
            unexpected_non_finite.sum()
        )

    # Simple returns below -100% are mathematically invalid.
    invalid_below_minus_100 = returns < -1

    if invalid_below_minus_100.any():
        issues["returns_below_minus_100_percent"] = int(
            invalid_below_minus_100.sum()
        )

    # Extreme returns are warnings for review, not automatic failures.
    extreme_returns = (
        returns.abs() > extreme_return_threshold
    ) & np.isfinite(returns)

    if extreme_returns.any():
        issues["extreme_return_rows_for_review"] = int(
            extreme_returns.sum()
        )

    blocking_issues = {
        "invalid_date_rows",
        "invalid_ticker_rows",
        "duplicate_ticker_date_rows",
        "non_finite_returns_after_first_observation",
        "returns_below_minus_100_percent",
    }

    return {
        "status": (
            "FAIL"
            if blocking_issues.intersection(issues)
            else "PASS"
        ),
        "rows_checked": len(df),
        "extreme_return_threshold": float(
            extreme_return_threshold
        ),
        "issues": issues,
    }
