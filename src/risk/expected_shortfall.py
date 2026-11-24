
import numpy as np
import pandas as pd


def calculate_historical_es(
    returns: pd.DataFrame,
    return_column: str = "simple_return",
    lookback: int = 252,
    confidence_levels: tuple = (0.95, 0.99),
) -> pd.DataFrame:
    """
    Calculate historical Expected Shortfall (ES) for individual assets.

    ES is reported as a positive loss magnitude and calculated as
    the average of the worst (1 - confidence) fraction of returns.
    A fractional weight is applied to the boundary observation when
    the requested tail mass is not a whole number of observations.

    Uses the most recent `lookback` valid daily returns per ticker.
    """

    if isinstance(lookback, bool) or not isinstance(
        lookback, (int, np.integer)
    ) or lookback < 1:
        raise ValueError("lookback must be a positive integer.")

    if not confidence_levels:
        raise ValueError("At least one confidence level is required.")

    for confidence in confidence_levels:
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float, np.integer, np.floating))
            or not np.isfinite(confidence)
            or not 0 < confidence < 1
        ):
            raise ValueError(
                "Confidence levels must be finite numbers between 0 and 1."
            )

    required = {"date", "ticker", return_column}
    missing = required - set(returns.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = returns.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    if df[["date", "ticker"]].isna().any().any():
        raise ValueError("Date and ticker values must not be missing.")

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

        # Sort worst returns first. Each confidence level uses a
        # fractional tail mass when lookback * (1 - confidence)
        # is not an integer.
        ordered_returns = np.sort(asset_returns.to_numpy())

        for confidence in confidence_levels:
            tail_mass = len(ordered_returns) * (1 - confidence)
            whole_count = int(np.floor(tail_mass))
            fractional_count = tail_mass - whole_count

            weighted_return_sum = float(
                ordered_returns[:whole_count].sum()
            )

            if fractional_count > 0 and whole_count < len(ordered_returns):
                weighted_return_sum += (
                    fractional_count * ordered_returns[whole_count]
                )

            tail_average_return = weighted_return_sum / tail_mass
            es = max(0.0, -tail_average_return)

            suffix = int(confidence * 100)
            result[f"es_{suffix}"] = es
            result[f"tail_observations_{suffix}"] = tail_mass

        results.append(result)

    sort_column = "es_95" if "es_95" in results[0] else (
        f"es_{int(confidence_levels[0] * 100)}"
    )

    return (
        pd.DataFrame(results)
        .sort_values(sort_column, ascending=False)
        .reset_index(drop=True)
    )
