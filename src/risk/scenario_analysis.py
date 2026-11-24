
import numpy as np
import pandas as pd


def calculate_scenario_impacts(
    weights: dict[str, float],
    scenarios: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate one-period portfolio impacts under hypothetical
    asset-level return shocks.

    Scenario values are assumed asset returns expressed as decimals.
    For example, -0.20 represents a 20% loss.

    Assumes fixed starting weights, no rebalancing, no transaction
    costs, and no nonlinear derivatives or leverage.
    """
    if not weights:
        raise ValueError("Weights cannot be empty.")

    if not scenarios:
        raise ValueError("At least one scenario is required.")

    weight_series = pd.Series(weights, dtype="float64")

    if not np.isfinite(weight_series.to_numpy()).all():
        raise ValueError("Weights must be finite.")

    if (weight_series < 0).any():
        raise ValueError("Only long-only portfolios are supported.")

    if not np.isclose(weight_series.sum(), 1.0, atol=1e-8):
        raise ValueError("Portfolio weights must sum to 1.")

    summary_rows = []
    detail_rows = []

    for scenario_name, shocks in scenarios.items():
        if not isinstance(shocks, dict):
            raise ValueError(
                f"Scenario '{scenario_name}' must map tickers to returns."
            )

        if set(shocks) != set(weight_series.index):
            missing = set(weight_series.index) - set(shocks)
            extra = set(shocks) - set(weight_series.index)

            raise ValueError(
                f"Scenario '{scenario_name}' must specify exactly "
                f"the portfolio tickers. Missing: {sorted(missing)}; "
                f"extra: {sorted(extra)}"
            )

        shock_series = pd.Series(shocks, dtype="float64").reindex(
            weight_series.index
        )

        if not np.isfinite(shock_series.to_numpy()).all():
            raise ValueError(
                f"Scenario '{scenario_name}' contains non-finite shocks."
            )

        if (shock_series < -1).any():
            raise ValueError(
                "A simple asset return cannot be less than -100%."
            )

        contributions = weight_series * shock_series
        portfolio_return = float(contributions.sum())

        summary_rows.append(
            {
                "scenario": scenario_name,
                "portfolio_return": portfolio_return,
                "portfolio_impact_pct": portfolio_return * 100,
                "portfolio_loss_pct": max(0.0, -portfolio_return * 100),
            }
        )

        for ticker in weight_series.index:
            detail_rows.append(
                {
                    "scenario": scenario_name,
                    "ticker": ticker,
                    "weight": weight_series[ticker],
                    "assumed_asset_return": shock_series[ticker],
                    "weighted_return_contribution": contributions[ticker],
                    "impact_contribution_pct": contributions[ticker] * 100,
                }
            )

        if not np.isclose(
            contributions.sum(), portfolio_return, atol=1e-12
        ):
            raise ArithmeticError(
                f"Contributions do not reconcile for '{scenario_name}'."
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)
