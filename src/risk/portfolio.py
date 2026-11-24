
import numpy as np
import pandas as pd


def calculate_portfolio_returns(
    returns: pd.DataFrame,
    weights: dict[str, float],
    return_column: str = "simple_return",
) -> pd.DataFrame:
    """
    Calculate daily returns for a target-weight portfolio.

    Assumption:
        The portfolio is rebalanced to the specified weights daily.

    Missing-data policy:
        Allows dates where all asset returns are NaN, such as the
        first return date. Raises an error for partially missing
        returns rather than silently changing portfolio exposures.
    """

    if not weights:
        raise ValueError("Portfolio weights cannot be empty.")

    weight_series = pd.Series(weights, dtype="float64")

    if not np.isfinite(weight_series.to_numpy()).all():
        raise ValueError("Portfolio weights must be finite.")

    if (weight_series < 0).any():
        raise ValueError(
            "Negative weights are not supported in this long-only version."
        )

    if not np.isclose(weight_series.sum(), 1.0, atol=1e-8):
        raise ValueError(
            f"Portfolio weights must sum to 1.0; "
            f"received {weight_series.sum():.8f}."
        )

    required = {"date", "ticker", return_column}
    missing = required - set(returns.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = returns[["date", "ticker", return_column]].copy()
    df["date"] = pd.to_datetime(df["date"])

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker observations found.")

    df[return_column] = pd.to_numeric(
        df[return_column], errors="raise"
    )

    available_returns = df[return_column].dropna()

    if not np.isfinite(available_returns).all():
        raise ValueError("Asset returns must be finite or NaN.")

    df = df[df["ticker"].isin(weight_series.index)]

    observed_tickers = set(df["ticker"].unique())
    missing_tickers = set(weight_series.index) - observed_tickers

    if missing_tickers:
        raise ValueError(
            f"No return data found for tickers: "
            f"{sorted(missing_tickers)}"
        )

    # Align all selected assets on the same dates.
    aligned = df.pivot(
        index="date",
        columns="ticker",
        values=return_column,
    )

    aligned = aligned.reindex(columns=weight_series.index)
    aligned = aligned.sort_index()

    # Reject partially missing rows so portfolio exposure is not
    # silently changed by renormalizing the remaining weights.
    missing_by_date = aligned.isna().any(axis=1)
    all_missing_by_date = aligned.isna().all(axis=1)

    partial_missing = missing_by_date & ~all_missing_by_date

    if partial_missing.any():
        bad_dates = aligned.index[partial_missing].strftime(
            "%Y-%m-%d"
        ).tolist()

        raise ValueError(
            "Partially missing asset returns found on dates: "
            f"{bad_dates[:10]}. Resolve these before calculating "
            "portfolio returns."
        )

    # The first return observation is NaN for every asset.
    aligned = aligned.dropna(how="all")

    if aligned.empty:
        raise ValueError("No complete portfolio return dates found.")

    # Weighted daily return under the daily-rebalancing assumption.
    portfolio_return = aligned.dot(weight_series)

    result = aligned.copy()
    result["portfolio_return"] = portfolio_return

    return result.reset_index()
