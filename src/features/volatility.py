
import numpy as np
import pandas as pd


def calculate_rolling_volatility(
    returns: pd.DataFrame,
    return_column: str = "simple_return",
    window: int = 21,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Calculate rolling daily and annualized volatility
    separately for each ticker.
    """

    if window < 2:
        raise ValueError("window must be at least 2")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

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
        raise ValueError(
            f"{return_column} must contain only finite values or NaN."
        )

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df["daily_volatility"] = (
        df.groupby("ticker")[return_column]
        .transform(
            lambda series: series.rolling(
                window=window,
                min_periods=window,
            ).std()
        )
    )

    df["annualized_volatility"] = (
        df["daily_volatility"] * np.sqrt(periods_per_year)
    )

    return df
