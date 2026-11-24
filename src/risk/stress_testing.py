
import numpy as np
import pandas as pd


def replay_historical_scenarios(
    returns: pd.DataFrame,
    weights: dict[str, float],
    return_column: str = "simple_return",
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Replay a portfolio through its worst historical daily returns.

    Args:
        returns:
            Long-format DataFrame containing date, ticker, and
            the selected simple-return column.
        weights:
            Mapping of ticker symbols to long-only portfolio weights.
            Weights must be finite and sum to 1.
        return_column:
            Column containing simple asset returns.
        top_n:
            Maximum number of worst portfolio-loss days to return.

    Returns:
        A tuple of (scenarios, contributions).

        scenarios:
            One row per selected loss day, including portfolio return
            and portfolio loss percentage.

        contributions:
            One row per asset per selected loss day, including its
            weighted return contribution and loss contribution.

    Behavior:
        - Only dates with valid returns for every portfolio asset
        are included.
        - Non-loss days are excluded from the stress scenarios.
        - If complete portfolio observations exist but there are no
        negative portfolio returns, return empty DataFrames with
        the expected output columns.
        - If no complete portfolio dates exist, raise ValueError.
        - Loss contributions are expressed in percentage points.
        - Current weights are applied to historical returns; this
        does not reconstruct historical portfolio weights.

    Limitations:
        - Assumes simple returns, long-only weights, and fixed daily
        target weights.
        - Excludes transaction costs, slippage, and market impact.
        - Historical replay is not a forecast of future losses.
    """

    # 1. Validate parameters.
    if not weights:
        raise ValueError("Weights cannot be empty.")

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    weight_series = pd.Series(weights, dtype="float64")

    if not np.isfinite(weight_series.to_numpy()).all():
        raise ValueError("Weights must be finite.")

    if (weight_series < 0).any():
        raise ValueError("Only long-only portfolios are supported.")

    if not np.isclose(weight_series.sum(), 1.0, atol=1e-8):
        raise ValueError("Portfolio weights must sum to 1.")

    # 2. Validate input data.
    required = {"date", "ticker", return_column}
    missing_columns = required - set(returns.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df = returns[["date", "ticker", return_column]].copy()

    # Validate only portfolio assets. Unrelated tickers should not
    # cause a portfolio calculation to fail.
    df = df[df["ticker"].isin(weight_series.index)].copy()

    # Dates must be valid for every portfolio observation.
    try:
        parsed_dates = pd.to_datetime(
            df["date"],
            errors="raise",
            format="mixed",
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Portfolio observation dates must be valid dates."
        ) from exc

    df["date"] = parsed_dates

    if df["date"].isna().any():
        raise ValueError(
            "Portfolio observation dates must not be missing."
        )

    # Convert and validate returns for the selected portfolio assets.
    try:
        df[return_column] = pd.to_numeric(
            df[return_column], errors="raise"
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Portfolio returns must be numeric."
        ) from exc

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker observations found.")

    observed_values = df[return_column].to_numpy()

    if pd.isna(observed_values).any():
        raise ValueError("Returns must not contain missing values.")

    if not np.isfinite(observed_values).all():
        raise ValueError("Returns must contain only finite values.")

    if (observed_values < -1.0).any():
        raise ValueError(
            "Simple returns cannot be less than -100%."
        )

    observed_tickers = set(df["ticker"].unique())
    missing_tickers = set(weight_series.index) - observed_tickers

    if missing_tickers:
        raise ValueError(
            f"Missing returns for tickers: {sorted(missing_tickers)}"
        )

    # 3. Align assets by date; require complete observations.
    aligned = (
        df.pivot(index="date", columns="ticker", values=return_column)
        .reindex(columns=weight_series.index)
        .sort_index()
    )

    aligned = aligned.dropna(how="any")

    if aligned.empty:
        raise ValueError("No complete portfolio return dates available.")

    # 4. Replay the portfolio through each historical day.
    weighted_contributions = aligned.mul(weight_series, axis="columns")
    portfolio_returns = weighted_contributions.sum(axis=1)

    # Independent reconciliation check.
    if not np.allclose(
        weighted_contributions.sum(axis=1).to_numpy(),
        portfolio_returns.to_numpy(),
        atol=1e-12,
    ):
        raise ArithmeticError(
            "Asset contributions do not reconcile to portfolio returns."
        )

    # Select the worst loss days, excluding non-loss days.
    loss_days = portfolio_returns[portfolio_returns < 0].sort_values()

    if loss_days.empty:
        return (
            pd.DataFrame(
                columns=[
                    "scenario_rank",
                    "date",
                    "portfolio_return",
                    "portfolio_loss_pct",
                ]
            ),
            pd.DataFrame(
                columns=[
                    "date",
                    "ticker",
                    "weight",
                    "asset_return",
                    "weighted_return_contribution",
                    "loss_contribution_pct",
                    "loss_contribution_share_pct",
                ]
            ),
        )
    
    selected_dates = loss_days.head(top_n).index

    scenarios = pd.DataFrame(
        {
            "date": selected_dates,
            "portfolio_return": portfolio_returns.loc[selected_dates].values,
        }
    )

    scenarios["portfolio_loss_pct"] = (
        -scenarios["portfolio_return"] * 100
    )
    scenarios["scenario_rank"] = np.arange(1, len(scenarios) + 1)

    scenarios = scenarios[
        [
            "scenario_rank",
            "date",
            "portfolio_return",
            "portfolio_loss_pct",
        ]
    ]

    # 5. Calculate each asset's contribution on every selected day.
    contribution_rows = []

    for date in selected_dates:
        portfolio_loss_pct = -portfolio_returns.loc[date] * 100

        for ticker in weight_series.index:
            asset_return = aligned.loc[date, ticker]
            weighted_return = weighted_contributions.loc[date, ticker]

            loss_contribution_pct = -weighted_return * 100

            # A negative share means this asset offset portfolio losses.
            loss_contribution_share_pct = (
                loss_contribution_pct / portfolio_loss_pct
            ) * 100

            contribution_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "weight": weight_series[ticker],
                    "asset_return": asset_return,
                    "weighted_return_contribution": weighted_return,
                    "loss_contribution_pct": loss_contribution_pct,
                    "loss_contribution_share_pct": (
                        loss_contribution_share_pct
                    ),
                }
            )

    contributions = pd.DataFrame(contribution_rows)

    # 6. Verify contribution reconciliation for each scenario.
    contribution_sums = contributions.groupby("date")[
        "weighted_return_contribution"
    ].sum()

    expected_returns = portfolio_returns.loc[selected_dates]

    if not np.allclose(
        contribution_sums.reindex(expected_returns.index).to_numpy(),
        expected_returns.to_numpy(),
        atol=1e-12,
    ):
        raise ArithmeticError(
            "Scenario loss contributions do not reconcile."
        )

    return scenarios, contributions
