import numpy as np
import pandas as pd

from collections import deque

from src.risk.tail_statistics import empirical_lower_tail_mean


def calculate_ewma_fhs_var(
    returns: pd.DataFrame,
    return_column: str = "simple_return",
    decay_factor: float = 0.94,
    min_periods: int = 252,
    residual_lookback: int = 1000,
    min_residuals: int | None = None,
    confidence_levels: tuple = (0.95, 0.99),
) -> pd.DataFrame:
    """
    Calculate one-step-ahead EWMA volatility, FHS VaR, and FHS ES.

    Each standardized residual is calculated using the EWMA volatility
    forecast made before observing that residual's return.

    At date t, only residuals from dates strictly before t are used
    to calculate VaR and ES.

    A missing return resets the EWMA warm-up sequence but does not
    erase previously generated standardized residuals.
    """

    if (
        isinstance(decay_factor, bool)
        or not isinstance(
            decay_factor,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(decay_factor)
        or not 0 < decay_factor < 1
    ):
        raise ValueError("decay_factor must be between 0 and 1.")

    if (
        isinstance(min_periods, bool)
        or not isinstance(min_periods, (int, np.integer))
        or min_periods < 2
    ):
        raise ValueError("min_periods must be an integer of at least 2.")

    if (
        isinstance(residual_lookback, bool)
        or not isinstance(residual_lookback, (int, np.integer))
        or residual_lookback < 1
    ):
        raise ValueError("residual_lookback must be a positive integer.")

    if min_residuals is None:
        min_residuals = residual_lookback

    if (
        isinstance(min_residuals, bool)
        or not isinstance(min_residuals, (int, np.integer))
        or min_residuals < 1
        or min_residuals > residual_lookback
    ):
        raise ValueError(
            "min_residuals must be a positive integer no greater "
            "than residual_lookback."
        )

    if (
        not isinstance(confidence_levels, (tuple, list))
        or not confidence_levels
    ):
        raise ValueError(
            "confidence_levels must be a non-empty tuple or list."
        )

    suffixes = []

    for confidence in confidence_levels:
        if (
            isinstance(confidence, bool)
            or not isinstance(
                confidence,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(confidence)
            or not 0 < confidence < 1
        ):
            raise ValueError(
                "Confidence levels must be finite numbers between 0 and 1."
            )

        suffixes.append(int(confidence * 100))

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

    df = returns.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    if df[["date", "ticker"]].isna().any().any():
        raise ValueError("Date and ticker values must not be missing.")

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

    metadata_columns = [
        column
        for column in (
            "source_row_missing",
            "source_price_row_missing",
            "adjusted_close_missing",
            "return_missing",
        )
        if column in df.columns
    ]

    results = []

    for ticker, group in df.groupby("ticker", sort=True):
        group = group.reset_index(drop=True)
        values = group[return_column].to_numpy(dtype=float)

        variance = np.nan
        consecutive_valid = 0
        residual_history = deque(maxlen=residual_lookback)

        for i, realized_return in enumerate(values):
            # Forecast using information available before today's return.
            forecast_variance = (
                variance
                if consecutive_valid >= min_periods
                else np.nan
            )

            forecast_volatility = (
                np.sqrt(forecast_variance)
                if np.isfinite(forecast_variance)
                else np.nan
            )

            available_residuals = list(residual_history)

            row = {
                "date": group.loc[i, "date"],
                "ticker": ticker,
                "realized_return": realized_return,
                "ewma_volatility": forecast_volatility,
                "standardized_residual": np.nan,
                "residual_observations": len(available_residuals),
            }

            for column in metadata_columns:
                row[column] = group.loc[i, column]

            for confidence, suffix in zip(
                confidence_levels,
                suffixes,
            ):
                var_column = f"var_{suffix}"
                es_column = f"es_{suffix}"
                exception_column = f"exception_{suffix}"

                row[var_column] = np.nan
                row[es_column] = np.nan
                row[exception_column] = pd.NA

                if (
                    np.isfinite(forecast_volatility)
                    and len(available_residuals) >= min_residuals
                ):
                    tail_probability = 1 - confidence

                    lower_tail_quantile = np.quantile(
                        available_residuals,
                        tail_probability,
                        method="linear",
                    )

                    # Existing VaR calculation, unchanged.
                    var_loss = max(
                        0.0,
                        -forecast_volatility * lower_tail_quantile,
                    )

                    lower_tail_mean = empirical_lower_tail_mean(
                        available_residuals,
                        tail_probability,
                    )

                    es_loss = max(
                        0.0,
                        -forecast_volatility * lower_tail_mean,
                    )

                    row[var_column] = var_loss
                    row[es_column] = max(var_loss, es_loss)

                    if np.isfinite(realized_return):
                        row[exception_column] = bool(
                            realized_return < -var_loss
                        )

            # Update EWMA and residual history only after today's
            # forecast has been recorded.
            if np.isfinite(realized_return):
                if (
                    np.isfinite(forecast_volatility)
                    and forecast_volatility > 0
                ):
                    residual = (
                        realized_return / forecast_volatility
                    )

                    if np.isfinite(residual):
                        row["standardized_residual"] = float(residual)
                        residual_history.append(float(residual))

                if consecutive_valid == 0:
                    variance = realized_return ** 2
                else:
                    variance = (
                        decay_factor * variance
                        + (1 - decay_factor) * realized_return ** 2
                    )

                consecutive_valid += 1

            else:
                variance = np.nan
                consecutive_valid = 0

            results.append(row)

    output = pd.DataFrame(results)

    for suffix in suffixes:
        output[f"exception_{suffix}"] = output[
            f"exception_{suffix}"
        ].astype("boolean")

    return output.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)