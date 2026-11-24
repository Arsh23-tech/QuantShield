
import numpy as np
import pandas as pd


def calculate_volatility_attribution(
    returns: pd.DataFrame,
    weights: dict[str, float],
    return_column: str = "simple_return",
    lookback: int = 252,
    periods_per_year: int = 252,
) -> tuple[pd.DataFrame, dict]:
    """
    Calculate covariance-based volatility attribution.

    Assumes a long-only portfolio with weights that sum to 1.
    Uses the most recent `lookback` complete daily observations.

    Component contributions sum to portfolio volatility.
    Annualized figures assume `periods_per_year` trading days.
    """

    if not weights:
        raise ValueError("Portfolio weights cannot be empty.")

    if lookback < 2:
        raise ValueError("lookback must be at least 2.")

    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive.")

    weight_series = pd.Series(weights, dtype="float64")

    if not np.isfinite(weight_series.to_numpy()).all():
        raise ValueError("Weights must be finite.")

    if (weight_series < 0).any():
        raise ValueError("Negative weights are not supported.")

    if not np.isclose(weight_series.sum(), 1.0, atol=1e-8):
        raise ValueError("Portfolio weights must sum to 1.")

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

    valid = df[return_column].dropna()
    if not np.isfinite(valid).all():
        raise ValueError("Returns must be finite or NaN.")

    missing_tickers = set(weight_series.index) - set(df["ticker"])
    if missing_tickers:
        raise ValueError(
            f"Missing return data for: {sorted(missing_tickers)}"
        )

    aligned = (
        df[df["ticker"].isin(weight_series.index)]
        .pivot(index="date", columns="ticker", values=return_column)
        .reindex(columns=weight_series.index)
        .sort_index()
    )

    # Use only dates with returns for every portfolio asset.
    aligned = aligned.dropna(how="any").tail(lookback)

    if len(aligned) < lookback:
        raise ValueError(
            f"Only {len(aligned)} complete observations available; "
            f"{lookback} required."
        )

    # Sample covariance matrix of daily returns.
    covariance = aligned.cov().to_numpy()
    w = weight_series.to_numpy()

    portfolio_variance = float(w @ covariance @ w)

    if not np.isfinite(portfolio_variance) or portfolio_variance <= 0:
        raise ValueError("Portfolio variance must be positive and finite.")

    portfolio_volatility = np.sqrt(portfolio_variance)

    # Derivative of portfolio volatility with respect to each weight.
    marginal_contribution = (covariance @ w) / portfolio_volatility

    # Each component's contribution to total portfolio volatility.
    component_contribution = w * marginal_contribution

    standalone_volatility = np.sqrt(
        np.diag(covariance)
    )

    attribution = pd.DataFrame({
        "ticker": weight_series.index,
        "weight": w,
        "standalone_daily_volatility": standalone_volatility,
        "marginal_daily_volatility": marginal_contribution,
        "component_daily_volatility": component_contribution,
    })

    attribution["risk_contribution_pct"] = (
        attribution["component_daily_volatility"]
        / portfolio_volatility
        * 100
    )

    annualization_factor = np.sqrt(periods_per_year)

    attribution["component_annualized_volatility"] = (
        attribution["component_daily_volatility"]
        * annualization_factor
    )

    attribution = attribution.sort_values(
        "risk_contribution_pct",
        ascending=False,
    ).reset_index(drop=True)

    summary = {
        "observations": len(aligned),
        "start_date": aligned.index.min(),
        "end_date": aligned.index.max(),
        "portfolio_daily_volatility": portfolio_volatility,
        "portfolio_annualized_volatility": (
            portfolio_volatility * annualization_factor
        ),
        "sum_component_daily_volatility": float(
            component_contribution.sum()
        ),
        "sum_risk_contribution_pct": float(
            attribution["risk_contribution_pct"].sum()
        ),
    }

    return attribution, summary
