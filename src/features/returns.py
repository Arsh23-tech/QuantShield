
import numpy as np
import pandas as pd


def calculate_returns(
    prices: pd.DataFrame,
    price_column: str = "adjusted_close",
) -> pd.DataFrame:
    """
    Calculate simple and log returns for each ticker.

    Expects one row per ticker/date observation.
    Returns a copy with two additional columns.
    """

    required_columns = {"date", "ticker", price_column}
    missing = required_columns - set(prices.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = prices.copy()

    df["date"] = pd.to_datetime(df["date"])

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError(
            "Duplicate date/ticker observations found."
        )

    df[price_column] = pd.to_numeric(
        df[price_column], errors="raise"
    )

    if (
        df[price_column].isna().any()
        or not np.isfinite(df[price_column]).all()
        or (df[price_column] <= 0).any()
    ):
        raise ValueError(
            f"{price_column} must contain finite, positive prices."
        )

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Previous available observation for the same ticker.
    previous_price = df.groupby("ticker")[price_column].shift(1)

    df["simple_return"] = (
        df[price_column] / previous_price
    ) - 1

    df["log_return"] = np.log(
        df[price_column] / previous_price
    )

    return df
