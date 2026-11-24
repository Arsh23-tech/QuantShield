
import numpy as np
import pandas as pd


def calculate_historical_var(
    returns: pd.DataFrame,
    return_column: str = "simple_return",
    lookback: int = 252,
    confidence_levels: tuple = (0.95, 0.99),
) -> pd.DataFrame:
    """
    Calculate historical VaR for individual assets.

    Uses the most recent `lookback` valid daily returns
    for each ticker.

    VaR is reported as a positive loss threshold when
    the corresponding lower-tail return quantile is negative.
    """

    if lookback < 1:
        raise ValueError("lookback must be at least 1.")

    if not confidence_levels:
        raise ValueError("At least one confidence level is required.")

    for confidence in confidence_levels:
        if not 0 < confidence < 1:
            raise ValueError(
                "Confidence levels must be between 0 and 1."
            )

    required = {"date", "ticker", return_column}
    missing = required - set(returns.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = returns.copy()
    df["date"] = pd.to_datetime(df["date"])

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker observations found.")

    df[return_column] = pd.to_numeric(
        df[return_column], errors="raise"
    )

    valid_returns = df[return_column].dropna()

    if not np.isfinite(valid_returns).all():
        raise ValueError("Returns must be finite or NaN.")

    df = df.sort_values(["ticker", "date"])

    results = []

    for ticker, group in df.groupby("ticker"):
        asset_returns = (
            group[return_column]
            .dropna()
            .tail(lookback)
        )

        if len(asset_returns) < lookback:
            raise ValueError(
                f"{ticker} has only {len(asset_returns)} valid "
                f"returns; {lookback} are required."
            )

        result = {
            "ticker": ticker,
            "observations": len(asset_returns),
            "start_date": group.loc[
                asset_returns.index, "date"
            ].min(),
            "end_date": group.loc[
                asset_returns.index, "date"
            ].max(),
        }

        for confidence in confidence_levels:
            quantile = asset_returns.quantile(
                1 - confidence
            )

            result[f"var_{int(confidence * 100)}"] = max(
                0.0, -float(quantile)
            )

        results.append(result)

    return pd.DataFrame(results).sort_values(
        "var_95", ascending=False
    ).reset_index(drop=True)
