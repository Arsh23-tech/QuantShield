
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from src.risk.tail_statistics import empirical_lower_tail_mean
from src.validation.trading_sessions import align_returns_to_trading_sessions


def backtest_historical_var(
    returns: pd.DataFrame,
    return_column: str = "simple_return",
    lookback: int = 252,
    confidence_levels: tuple = (0.95, 0.99),
    calendar_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Backtest rolling historical VaR and Expected Shortfall for individual
    assets.

    Each forecast uses the most recent `lookback` valid returns strictly
    before its evaluation date. Missing evaluation returns are retained
    and receive an unknown exception classification.

    VaR uses the existing linear empirical quantile convention.
    ES integrates the piecewise-linear empirical quantile function over
    the lower tail, using the same historical observations as VaR.

    If calendar_name is provided, each ticker is first aligned to the
    expected sessions of that exchange calendar.
    """
    if (
        not isinstance(lookback, int)
        or isinstance(lookback, bool)
        or lookback < 1
    ):
        raise ValueError("lookback must be a positive integer.")

    if not isinstance(confidence_levels, (tuple, list)) or not confidence_levels:
        raise ValueError("confidence_levels must be a non-empty tuple or list.")

    levels = []
    for confidence in confidence_levels:
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            raise ValueError(
                "Confidence levels must be finite and between 0 and 1."
            )

        if (
            isinstance(confidence, bool)
            or not np.isfinite(value)
            or not 0 < value < 1
        ):
            raise ValueError(
                "Confidence levels must be finite and between 0 and 1."
            )

        levels.append(value)

    if len(set(levels)) != len(levels):
        raise ValueError("confidence_levels must not contain duplicates.")

    suffixes = [int(round(level * 100)) for level in levels]

    if len(set(suffixes)) != len(suffixes):
        raise ValueError(
            "Confidence levels must not map to duplicate integer suffixes."
        )

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    required = {"date", "ticker", return_column}
    missing = required - set(returns.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Preserve optional metadata columns needed to track missing source rows.
    optional_columns = [
        column
        for column in (
            "source_row_missing",
            "source_price_row_missing",
            "return_missing",
        )
        if column in returns.columns
    ]

    df = returns[
        ["date", "ticker", return_column] + optional_columns
    ].copy()

    df["date"] = pd.to_datetime(df["date"], errors="raise")

    if df["date"].isna().any() or df["ticker"].isna().any():
        raise ValueError("Dates and tickers must not be missing.")

    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    df["date"] = df["date"].dt.normalize()

    invalid_tickers = ~df["ticker"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )

    if invalid_tickers.any():
        raise ValueError("Tickers must be non-empty strings.")

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker observations found.")

    df[return_column] = pd.to_numeric(
        df[return_column],
        errors="raise",
    )

    observed = df[return_column].dropna().to_numpy(dtype=float)

    if not np.isfinite(observed).all():
        raise ValueError("Returns must be finite or NaN.")

    if (observed < -1).any():
        raise ValueError("Returns below -100% are invalid.")

    df = df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    rows = []

    for ticker, original_group in df.groupby("ticker", sort=True):
        group = original_group.copy()

        if calendar_name is not None:
            group = align_returns_to_trading_sessions(
                df,
                ticker=ticker,
                start_date=group["date"].min(),
                end_date=group["date"].max(),
                return_column=return_column,
                calendar_name=calendar_name,
            )

        group = group.sort_values("date").reset_index(drop=True)

        values = group[return_column].to_numpy(dtype=float)
        dates = group["date"].to_numpy()

        # Use the trading-session metadata convention when present;
        # otherwise fall back to the price-level metadata convention.
        if "source_row_missing" in group.columns:
            source_missing_values = group[
                "source_row_missing"
            ].to_numpy()
        elif "source_price_row_missing" in group.columns:
            source_missing_values = group[
                "source_price_row_missing"
            ].to_numpy()
        else:
            source_missing_values = np.zeros(len(group), dtype=bool)

        # The queue contains positions of valid observations strictly
        # before the current evaluation position. Its size never exceeds
        # `lookback`.
        history_positions = deque(maxlen=lookback)
        ticker_forecast_count = 0

        for i, realized_return in enumerate(values):
            # Forecast before adding today's observation. This preserves
            # the one-step-ahead/no-lookahead property.
            if len(history_positions) == lookback:
                history_indices = np.fromiter(
                    history_positions,
                    dtype=np.intp,
                    count=lookback,
                )
                historical = values[history_indices]

                forecast = {
                    "ticker": ticker,
                    "date": dates[i],
                    "forecast_start_date": dates[history_indices[0]],
                    "forecast_end_date": dates[history_indices[-1]],
                    "realized_return": realized_return,
                    "source_row_missing": bool(source_missing_values[i]),
                    "return_missing": bool(pd.isna(realized_return)),
                }

                for confidence, suffix in zip(levels, suffixes):
                    tail_probability = 1.0 - confidence

                    # Preserve the existing historical VaR calculation.
                    var_loss = max(
                        0.0,
                        -float(
                            np.quantile(
                                historical,
                                tail_probability,
                            )
                        ),
                    )

                    # Integrate the empirical lower-tail quantile function.
                    lower_tail_mean = empirical_lower_tail_mean(
                        historical,
                        tail_probability,
                    )

                    es_loss = max(0.0, -lower_tail_mean)

                    # Enforce coherent ordering under the positive-loss
                    # convention without changing the VaR estimate.
                    es_loss = max(var_loss, es_loss)

                    forecast[f"var_{suffix}"] = var_loss
                    forecast[f"es_{suffix}"] = es_loss

                    if pd.isna(realized_return):
                        forecast[f"exception_{suffix}"] = pd.NA
                    else:
                        forecast[f"exception_{suffix}"] = bool(
                            realized_return < -var_loss
                        )

                rows.append(forecast)
                ticker_forecast_count += 1

            # Add the current observation only after the forecast.
            # Missing returns are not inserted into the estimation history.
            if np.isfinite(realized_return):
                history_positions.append(i)

        if ticker_forecast_count == 0:
            raise ValueError(
                f"Ticker {ticker} requires at least {lookback + 1} "
                "sessions with sufficient prior valid returns "
                "to generate a forecast."
            )

    if not rows:
        raise ValueError(
            f"At least {lookback + 1} observations are required "
            "to generate forecasts."
        )

    result = pd.DataFrame(rows)

    for suffix in suffixes:
        result[f"exception_{suffix}"] = (
            result[f"exception_{suffix}"].astype("boolean")
        )

    return result.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


def summarize_var_backtest(
    backtest: pd.DataFrame,
    confidence_levels: tuple = (0.95, 0.99),
) -> pd.DataFrame:
    """
    Summarize exception rates using only evaluable forecasts.

    Forecasts with missing realized returns are counted separately and
    excluded from the exception-rate denominator.
    """
    required = {"ticker"}
    missing = required - set(backtest.columns)

    if missing:
        raise ValueError(
            f"Backtest missing required columns: {sorted(missing)}"
        )

    if not isinstance(confidence_levels, (tuple, list)) or not confidence_levels:
        raise ValueError(
            "confidence_levels must be a non-empty tuple or list."
        )

    rows = []

    for ticker, group in backtest.groupby("ticker", sort=True):
        for confidence in confidence_levels:
            confidence = float(confidence)

            if not np.isfinite(confidence) or not 0 < confidence < 1:
                raise ValueError(
                    "Confidence levels must be finite and between 0 and 1."
                )

            suffix = int(round(confidence * 100))
            column = f"exception_{suffix}"

            if column not in group.columns:
                raise ValueError(
                    f"Backtest missing required column: {column}"
                )

            classifications = group[column].astype("boolean")
            evaluable = classifications.notna()

            observations = int(evaluable.sum())
            exceptions = int(classifications[evaluable].sum())
            unevaluable = int((~evaluable).sum())

            rows.append({
                "ticker": ticker,
                "confidence": confidence,
                "forecasts": int(len(group)),
                "observations": observations,
                "exceptions": exceptions,
                "exception_rate": (
                    exceptions / observations
                    if observations > 0
                    else np.nan
                ),
                "expected_exception_rate": 1.0 - confidence,
                "unevaluable_forecasts": unevaluable,
            })

    return pd.DataFrame(rows)
