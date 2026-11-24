import numpy as np
import pandas as pd
from scipy.stats import norm

from src.risk.tail_statistics import empirical_lower_tail_mean


def calculate_ewma_var(
    returns: pd.DataFrame,
    return_column: str = "simple_return",
    decay_factor: float = 0.94,
    min_periods: int = 252,
    confidence_levels: tuple = (0.95, 0.99),
) -> pd.DataFrame:
    """
    Calculate one-step-ahead EWMA volatility, parametric VaR,
    and parametric Expected Shortfall for individual assets.

    Forecasts for date t use only returns available before date t.

    Variance is initialized using the first valid return squared.
    At least min_periods consecutive valid historical returns are
    required before a forecast is issued.

    VaR and ES are positive loss measures, assuming zero expected
    return and normally distributed returns.
    """

    if (
        isinstance(decay_factor, bool)
        or not isinstance(
            decay_factor, (int, float, np.integer, np.floating)
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
                confidence, (int, float, np.integer, np.floating)
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

    observed = df[return_column].dropna()

    if not np.isfinite(observed).all():
        raise ValueError("Returns must be finite or NaN.")

    if (observed < -1).any():
        raise ValueError("Returns below -100% are invalid.")

    df = df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    results = []

    for ticker, group in df.groupby("ticker", sort=True):
        group = group.reset_index(drop=True)
        values = group[return_column].to_numpy(dtype=float)

        variance = np.nan
        consecutive_valid = 0

        for i, realized_return in enumerate(values):
            # Forecast before observing the current return.
            forecast_variance = (
                variance
                if consecutive_valid >= min_periods
                else np.nan
            )

            volatility = (
                np.sqrt(forecast_variance)
                if np.isfinite(forecast_variance)
                else np.nan
            )

            row = {
                "date": group.loc[i, "date"],
                "ticker": ticker,
                "realized_return": realized_return,
                "ewma_volatility": volatility,
            }

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

                if np.isfinite(forecast_variance):
                    z_score = norm.ppf(1 - confidence)

                    # Existing VaR formula, unchanged.
                    var_loss = max(
                        0.0,
                        -volatility * z_score,
                    )

                    # Gaussian Expected Shortfall.
                    tail_probability = 1 - confidence
                    density = norm.pdf(z_score)

                    es_loss = (
                        volatility
                        * density
                        / tail_probability
                    )

                    row[var_column] = var_loss
                    row[es_column] = max(var_loss, es_loss)

                    if np.isfinite(realized_return):
                        row[exception_column] = bool(
                            realized_return < -var_loss
                        )

            results.append(row)

            # Update variance only after recording today's forecast.
            if not np.isfinite(realized_return):
                variance = np.nan
                consecutive_valid = 0
                continue

            if consecutive_valid == 0:
                variance = realized_return ** 2
            else:
                variance = (
                    decay_factor * variance
                    + (1 - decay_factor) * realized_return ** 2
                )

            consecutive_valid += 1

    output = pd.DataFrame(results)

    for suffix in suffixes:
        output[f"exception_{suffix}"] = output[
            f"exception_{suffix}"
        ].astype("boolean")

    return output.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)