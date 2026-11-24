import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
]


def validate_market_data(df: pd.DataFrame) -> dict:
    """Validate market data and return a quality report."""

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        return {
            "status": "FAIL",
            "missing_columns": missing_columns,
        }

    issues = {}

    # Missing values in required fields
    missing_values = df[REQUIRED_COLUMNS].isna().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        issues["missing_values"] = missing_values.to_dict()

    # Validate date values.
    parsed_dates = pd.to_datetime(
        df["date"],
        errors="coerce",
        utc=True,
        format="mixed",
    )

    invalid_dates = df["date"].notna() & parsed_dates.isna()

    if invalid_dates.any():
        issues["invalid_date_rows"] = int(invalid_dates.sum())

    # Validate ticker identifiers.
    valid_tickers = df["ticker"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )

    invalid_tickers = df["ticker"].notna() & ~valid_tickers

    if invalid_tickers.any():
        issues["invalid_ticker_rows"] = int(invalid_tickers.sum())

    # Check chronological ordering within each ticker.
    ordering_data = pd.DataFrame(
        {
            "ticker": df["ticker"],
            "date": parsed_dates,
        },
        index=df.index,
    )

    valid_ordering_rows = (
        ordering_data["ticker"].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        & ordering_data["date"].notna()
    )

    ordered = ordering_data.loc[valid_ordering_rows].copy()
    date_differences = ordered.groupby("ticker")["date"].diff()

    out_of_order = date_differences.lt(pd.Timedelta(0))

    if out_of_order.any():
        issues["unsorted_ticker_dates"] = int(out_of_order.sum())

    # Duplicate ticker/date observations
    duplicate_count = int(
        df.duplicated(subset=["date", "ticker"]).sum()
    )

    if duplicate_count:
        issues["duplicate_rows"] = duplicate_count

    
    # Numeric validation for price and volume fields.
    # Use temporary numeric views so the input DataFrame is not modified.
    numeric_columns = [
        "open", "high", "low", "close", "adjusted_close", "volume"
    ]

    numeric_data = pd.DataFrame(
        {
            col: pd.to_numeric(df[col], errors="coerce")
            for col in numeric_columns
        },
        index=df.index,
    )

    invalid_numeric_values = {}

    for col in numeric_columns:
        # Count non-missing original values that cannot be parsed as numbers.
        non_numeric = df[col].notna() & numeric_data[col].isna()

        # Infinite values are numeric in pandas but invalid market data.
        non_finite = (
            numeric_data[col].notna()
            & ~np.isfinite(numeric_data[col])
        )

        invalid_count = int((non_numeric | non_finite).sum())

        if invalid_count:
            invalid_numeric_values[col] = invalid_count

    if invalid_numeric_values:
        issues["invalid_numeric_values"] = invalid_numeric_values

    # Invalid or non-positive prices.
    price_columns = [
        "open", "high", "low", "close", "adjusted_close"
    ]
    numeric_prices = numeric_data[price_columns]

    invalid_prices = (
        numeric_prices.notna()
        & (
            ~np.isfinite(numeric_prices)
            | numeric_prices.le(0)
        )
    ).any(axis=1)

    if invalid_prices.any():
        issues["invalid_price_rows"] = int(invalid_prices.sum())

    # Negative volume.
    numeric_volume = numeric_data["volume"]
    negative_volume = numeric_volume.lt(0).fillna(False)

    if negative_volume.any():
        issues["negative_volume_rows"] = int(negative_volume.sum())

    # OHLC consistency. Evaluate only rows with four valid finite OHLC values.
    ohlc_columns = ["open", "high", "low", "close"]
    ohlc = numeric_data[ohlc_columns]

    valid_ohlc_values = (
        ohlc.notna().all(axis=1)
        & np.isfinite(ohlc).all(axis=1)
    )

    invalid_ohlc = valid_ohlc_values & (
        (ohlc["high"] < ohlc["low"])
        | (ohlc["high"] < ohlc["open"])
        | (ohlc["high"] < ohlc["close"])
        | (ohlc["low"] > ohlc["open"])
        | (ohlc["low"] > ohlc["close"])
    )

    if invalid_ohlc.any():
        issues["invalid_ohlc_rows"] = int(invalid_ohlc.sum())

    return {
        "status": "PASS" if not issues else "FAIL",
        "rows_checked": len(df),
        "issues": issues,
    }