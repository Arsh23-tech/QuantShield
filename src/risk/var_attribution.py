
import numpy as np
import pandas as pd


def calculate_var_attribution(
    returns: pd.DataFrame,
    weights: dict[str, float],
    return_column: str = "simple_return",
    lookback: int = 252,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate historical Component VaR and Marginal VaR.

    Method:
    - Build portfolio returns from aligned asset returns and target weights.
    - Calculate historical VaR using a linearly interpolated quantile.
    - Interpolate the asset returns at the same portfolio VaR boundary.
    - Component VaR = weight * Marginal VaR.
    - Component VaR reconciles to portfolio VaR when the quantile is negative.

    Assumptions:
    - Long-only portfolio with weights summing to 1.
    - Daily target weights; no transaction costs.
    - Complete observations only; incomplete dates are excluded.
    - The VaR metric has a zero floor, consistent with QuantShield's
      current historical VaR implementation.
    - Marginal VaR is a local boundary-scenario estimate, not a guarantee
      of the exact change in VaR after a large weight change.
    """
    # 1. Validate parameters
    if not weights:
        raise ValueError("Weights cannot be empty.")

    if not isinstance(lookback, int) or lookback < 1:
        raise ValueError("lookback must be a positive integer.")

    if not confidence_levels:
        raise ValueError("At least one confidence level is required.")

    if any(
        not np.isfinite(level) or not 0 < level < 1
        for level in confidence_levels
    ):
        raise ValueError("Confidence levels must be between 0 and 1.")

    weight_series = pd.Series(weights, dtype="float64")

    if not np.isfinite(weight_series.to_numpy()).all():
        raise ValueError("Weights must be finite.")

    if (weight_series < 0).any():
        raise ValueError("Only long-only portfolios are supported.")

    if not np.isclose(weight_series.sum(), 1.0, atol=1e-8):
        raise ValueError("Portfolio weights must sum to 1.")

    # 2. Validate and prepare input data
    required = {"date", "ticker", return_column}
    missing_columns = required - set(returns.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df = returns[["date", "ticker", return_column]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df[return_column] = pd.to_numeric(
        df[return_column], errors="raise"
    )

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker observations found.")

    observed_values = df[return_column].dropna().to_numpy()

    if not np.isfinite(observed_values).all():
        raise ValueError("Returns must be finite when present.")

    df = df[df["ticker"].isin(weight_series.index)]

    observed_tickers = set(df["ticker"].unique())
    missing_tickers = set(weight_series.index) - observed_tickers

    if missing_tickers:
        raise ValueError(
            f"Missing returns for tickers: {sorted(missing_tickers)}"
        )

    aligned = (
        df.pivot(index="date", columns="ticker", values=return_column)
        .reindex(columns=weight_series.index)
        .sort_index()
    )

    # Use only dates with returns for every portfolio asset.
    aligned = aligned.dropna(how="any")

    if len(aligned) < lookback:
        raise ValueError(
            f"Need at least {lookback} complete observations; "
            f"found {len(aligned)}."
        )

    aligned = aligned.tail(lookback)

    # 3. Calculate portfolio returns
    portfolio_returns = aligned.dot(weight_series)

    if not np.isfinite(portfolio_returns.to_numpy()).all():
        raise ValueError("Portfolio returns contain non-finite values.")

    # Sort portfolio outcomes from worst to best, retaining asset returns
    # for the same historical scenarios.
    sorted_positions = np.argsort(portfolio_returns.to_numpy())
    sorted_portfolio = portfolio_returns.to_numpy()[sorted_positions]
    sorted_assets = aligned.iloc[sorted_positions].to_numpy()

    attribution_rows = []
    summary_rows = []

    # 4. Calculate VaR attribution for each confidence level
    for confidence in confidence_levels:
        tail_probability = 1.0 - confidence

        # Match NumPy's linear quantile interpolation convention.
        quantile_return = float(
            np.quantile(
                sorted_portfolio,
                tail_probability,
                method="linear",
            )
        )

        position = (len(sorted_portfolio) - 1) * tail_probability
        lower_index = int(np.floor(position))
        upper_index = int(np.ceil(position))
        fraction = position - lower_index

        lower_asset_returns = sorted_assets[lower_index]
        upper_asset_returns = sorted_assets[upper_index]

        boundary_asset_returns = (
            (1.0 - fraction) * lower_asset_returns
            + fraction * upper_asset_returns
        )

        portfolio_var = max(0.0, -quantile_return)

        if quantile_return < 0:
            marginal_var = -boundary_asset_returns
            component_var = weight_series.to_numpy() * marginal_var
            component_pct = component_var / portfolio_var * 100.0
        else:
            # The existing VaR convention floors positive quantiles at zero.
            marginal_var = np.zeros(len(weight_series))
            component_var = np.zeros(len(weight_series))
            component_pct = np.full(len(weight_series), np.nan)

        # Check that the boundary scenario reproduces the quantile.
        reconstructed_return = float(
            np.dot(weight_series.to_numpy(), boundary_asset_returns)
        )

        if not np.isclose(
            reconstructed_return, quantile_return, atol=1e-10
        ):
            raise ArithmeticError(
                "Interpolated asset returns do not reproduce "
                "the portfolio VaR quantile."
            )

        for i, ticker in enumerate(weight_series.index):
            attribution_rows.append(
                {
                    "ticker": ticker,
                    "weight": weight_series.iloc[i],
                    "confidence_level": confidence,
                    "marginal_var": marginal_var[i],
                    "component_var": component_var[i],
                    "component_var_pct": component_pct[i],
                    "boundary_asset_return": boundary_asset_returns[i],
                }
            )

        summary_rows.append(
            {
                "confidence_level": confidence,
                "observations": len(aligned),
                "start_date": aligned.index.min(),
                "end_date": aligned.index.max(),
                "portfolio_quantile_return": quantile_return,
                "portfolio_var": portfolio_var,
                "component_var_sum": component_var.sum(),
            }
        )

    attribution_df = pd.DataFrame(attribution_rows)
    summary_df = pd.DataFrame(summary_rows)

    return attribution_df, summary_df
