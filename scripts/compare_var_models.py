
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.validation.trading_sessions import get_expected_trading_sessions
from src.risk.ewma_var import calculate_ewma_var
from src.risk.ewma_fhs_var import calculate_ewma_fhs_var
from src.risk.var_backtesting import backtest_historical_var
from src.risk.var_backtest_diagnostics import (
    conditional_coverage_test,
    christoffersen_independence_test,
    kupiec_coverage_test,
)


LOOKBACK = 252
DECAY_FACTOR = 0.94
FHS_RESIDUAL_LOOKBACK = 1000
FHS_MIN_RESIDUALS = 1000

CONFIDENCE_LEVELS = (0.95, 0.99)
ALPHA = 0.05
CALENDAR_NAME = "XNYS"

DATA_PATH = Path("data/processed/market_prices.parquet")
OUTPUT_DIR = Path("results/var_model_comparison")

MODEL_NAMES = ("historical", "ewma", "ewma_fhs")

RISK_COLUMNS = ("var_95", "var_99", "es_95", "es_99")
ORDERING_PAIRS = (
    ("var_95", "var_99"),
    ("var_95", "es_95"),
    ("var_99", "es_99"),
    ("es_95", "es_99"),
)


def holm_adjust(p_values, alpha=ALPHA):
    """Return Holm-adjusted p-values and rejection decisions."""
    p_values = [float(p) for p in p_values]

    if not p_values:
        return [], []

    if not np.isfinite(p_values).all():
        raise ValueError("Holm adjustment requires finite p-values.")

    if any(p < 0 or p > 1 for p in p_values):
        raise ValueError("p-values must be between 0 and 1.")

    order = sorted(range(len(p_values)), key=lambda i: p_values[i])

    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    m = len(p_values)

    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max

    return adjusted, [p < alpha for p in adjusted]


def load_returns():
    """
    Align prices to XNYS sessions before calculating returns.

    No forward-filling is performed. A missing session price remains
    missing, and pct_change does not bridge across missing prices.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Price dataset not found: {DATA_PATH}")

    prices = pd.read_parquet(DATA_PATH)

    required = {"date", "ticker", "adjusted_close"}
    missing = required - set(prices.columns)

    if missing:
        raise ValueError(
            f"Price data missing columns: {sorted(missing)}"
        )

    prices = prices[["date", "ticker", "adjusted_close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="raise")
    prices["adjusted_close"] = pd.to_numeric(
        prices["adjusted_close"],
        errors="raise",
    )

    if prices[["date", "ticker"]].isna().any().any():
        raise ValueError("Price dates and tickers cannot be missing.")

    invalid_tickers = ~prices["ticker"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )
    if invalid_tickers.any():
        raise ValueError("Tickers must be non-empty strings.")

    if prices.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker price observations found.")

    if prices["date"].dt.tz is not None:
        prices["date"] = prices["date"].dt.tz_localize(None)

    prices["date"] = prices["date"].dt.normalize()

    observed_prices = prices["adjusted_close"].dropna().to_numpy(
        dtype=float
    )
    if (
        not np.isfinite(observed_prices).all()
        or (observed_prices <= 0).any()
    ):
        raise ValueError(
            "Observed adjusted closing prices must be positive and finite."
        )

    aligned_groups = []

    for ticker, group in prices.groupby("ticker", sort=True):
        start_date = group["date"].min()
        end_date = group["date"].max()

        sessions = get_expected_trading_sessions(
            start_date,
            end_date,
            calendar_name=CALENDAR_NAME,
        )

        indexed = group.set_index("date").sort_index()
        unexpected = indexed.index.difference(sessions)

        if len(unexpected):
            raise ValueError(
                f"Unexpected non-trading dates for {ticker}: "
                f"{unexpected.strftime('%Y-%m-%d').tolist()}"
            )

        aligned = indexed.reindex(sessions)
        aligned.index.name = "date"
        aligned["ticker"] = ticker
        aligned["source_price_row_missing"] = (
            ~aligned.index.isin(indexed.index)
        )
        aligned["adjusted_close_missing"] = (
            aligned["adjusted_close"].isna()
        )

        aligned_groups.append(aligned.reset_index())

    if not aligned_groups:
        raise ValueError("No price data available.")

    aligned_prices = pd.concat(aligned_groups, ignore_index=True)
    aligned_prices = aligned_prices.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    aligned_prices["simple_return"] = (
        aligned_prices.groupby("ticker", sort=False)[
            "adjusted_close"
        ].pct_change(fill_method=None)
    )

    returns = aligned_prices[
        [
            "date",
            "ticker",
            "simple_return",
            "source_price_row_missing",
            "adjusted_close_missing",
        ]
    ].copy()

    returns["return_missing"] = returns["simple_return"].isna()

    observed_returns = returns["simple_return"].dropna().to_numpy(
        dtype=float
    )
    if (
        not np.isfinite(observed_returns).all()
        or (observed_returns < -1).any()
    ):
        raise ValueError(
            "Observed returns must be finite and at least -100%."
        )

    return returns


def _validate_risk_forecasts(frame, model_name):
    """
    Validate individual risk estimates and their internal ordering.

    NaN values represent unavailable forecasts and are allowed here.
    Infinities, negative risk estimates, and inconsistent ordering
    are rejected before common-date filtering.
    """
    for column in RISK_COLUMNS:
        values = pd.to_numeric(frame[column], errors="raise")
        observed = values.dropna().to_numpy(dtype=float)

        if not np.isfinite(observed).all():
            raise ValueError(
                f"{model_name} {column} forecasts must be finite or missing."
            )

        if (observed < 0).any():
            raise ValueError(
                f"{model_name} {column} forecasts cannot be negative."
            )

        frame[column] = values

    for lower_column, upper_column in ORDERING_PAIRS:
        comparable = frame[
            [lower_column, upper_column]
        ].dropna()

        if (
            comparable[upper_column] + 1e-12
            < comparable[lower_column]
        ).any():
            raise ValueError(
                f"{model_name} forecasts violate risk ordering: "
                f"{upper_column} must be greater than or equal to "
                f"{lower_column}."
            )

    return frame


def calculate_diagnostics(forecasts):
    """
    Calculate per-model, per-ticker VaR exception diagnostics.

    ES is deliberately not used in these tests. Missing exception
    classifications are excluded and counted as unevaluable forecasts.
    """
    rows = []

    for (model, ticker), group in forecasts.groupby(
        ["model", "ticker"],
        sort=True,
    ):
        group = group.sort_values("date").reset_index(drop=True)

        for confidence in CONFIDENCE_LEVELS:
            suffix = int(confidence * 100)
            exception_column = f"exception_{suffix}"

            classifications = group[exception_column].astype("boolean")
            evaluable = classifications.notna()
            exceptions = classifications[evaluable].astype(bool)

            if exceptions.empty:
                raise ValueError(
                    f"No evaluable exception classifications for "
                    f"{model}, {ticker}, {confidence:.0%}."
                )

            expected_rate = 1 - confidence

            coverage = kupiec_coverage_test(
                exceptions,
                expected_rate,
            )
            independence = christoffersen_independence_test(exceptions)
            conditional = conditional_coverage_test(
                exceptions,
                expected_rate,
            )

            rows.append({
                "model": model,
                "ticker": ticker,
                "confidence": confidence,
                "forecasts": int(len(group)),
                "observations": int(evaluable.sum()),
                "unevaluable_forecasts": int((~evaluable).sum()),
                "exceptions": coverage["exceptions"],
                "observed_exception_rate": (
                    coverage["observed_exception_rate"]
                ),
                "expected_exception_rate": expected_rate,
                "kupiec_p": coverage["p_value"],
                "independence_p": independence["p_value"],
                "conditional_p": conditional["p_value"],
            })

    results = pd.DataFrame(rows)

    for diagnostic in ("kupiec", "independence", "conditional"):
        results[f"{diagnostic}_holm_p"] = np.nan
        results[f"{diagnostic}_holm_reject"] = False

    # Holm correction across model/ticker combinations within each
    # confidence level and diagnostic family.
    for confidence in CONFIDENCE_LEVELS:
        mask = results["confidence"] == confidence

        for diagnostic in ("kupiec", "independence", "conditional"):
            p_column = f"{diagnostic}_p"
            adjusted, rejected = holm_adjust(
                results.loc[mask, p_column].tolist()
            )

            results.loc[mask, f"{diagnostic}_holm_p"] = adjusted
            results.loc[
                mask,
                f"{diagnostic}_holm_reject",
            ] = rejected

    return results


def align_model_forecasts(
    historical,
    ewma_forecasts,
    fhs_forecasts=None,
):
    """
    Align two or three models by ticker/date.

    The optional FHS argument preserves compatibility with two-model
    callers. All models must agree on realized-return missingness and
    observed realized-return values.

    A retained row must have complete VaR and ES forecasts at both
    confidence levels for every included model.
    """
    model_frames = {
        "historical": historical,
        "ewma": ewma_forecasts,
    }

    if fhs_forecasts is not None:
        model_frames["ewma_fhs"] = fhs_forecasts

    required = {
        "ticker",
        "date",
        "realized_return",
        *RISK_COLUMNS,
        "exception_95",
        "exception_99",
    }

    aligned = None

    for model_name, frame in model_frames.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"{model_name} forecasts must be a pandas DataFrame."
            )

        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{model_name} forecasts missing columns: "
                f"{sorted(missing)}"
            )

        if frame[["ticker", "date"]].isna().any().any():
            raise ValueError(
                f"Missing ticker/date keys in {model_name} forecasts."
            )

        if frame.duplicated(["ticker", "date"]).any():
            raise ValueError(
                f"Duplicate ticker/date keys in {model_name} forecasts."
            )

        selected = frame[
            [
                "ticker",
                "date",
                "realized_return",
                *RISK_COLUMNS,
                "exception_95",
                "exception_99",
            ]
        ].copy()

        selected = _validate_risk_forecasts(selected, model_name)

        rename = {
            column: f"{model_name}_{column}"
            for column in selected.columns
            if column not in ("ticker", "date")
        }
        selected = selected.rename(columns=rename)

        if aligned is None:
            aligned = selected
        else:
            aligned = aligned.merge(
                selected,
                on=["ticker", "date"],
                how="inner",
                validate="one_to_one",
            )

    if aligned is None or aligned.empty:
        raise ValueError("The models have no common forecast dates.")

    # Require all four risk estimates from every model.
    required_risk_columns = [
        f"{model}_{column}"
        for model in model_frames
        for column in RISK_COLUMNS
    ]

    aligned = aligned.dropna(subset=required_risk_columns).copy()

    if aligned.empty:
        raise ValueError(
            "The models have no common dates with complete VaR and ES forecasts."
        )

    # Realized-return missingness must match across every model.
    realized_columns = [
        f"{model}_realized_return"
        for model in model_frames
    ]
    missingness = aligned[realized_columns].isna()

    if not missingness.eq(missingness.iloc[:, 0], axis=0).all().all():
        raise ValueError(
            "Models' realized-return missingness differs on aligned dates."
        )

    for column in realized_columns:
        observed = aligned[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(observed).all():
            raise ValueError("Observed realized returns must be finite.")

    reference_column = realized_columns[0]
    observed_mask = aligned[reference_column].notna()

    for column in realized_columns[1:]:
        if not np.allclose(
            aligned.loc[observed_mask, reference_column],
            aligned.loc[observed_mask, column],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                "Models' realized returns differ on aligned dates."
            )

    return aligned


def build_long_forecasts(aligned):
    """Convert aligned wide forecasts into model/ticker/date rows."""
    available_models = [
        model
        for model in MODEL_NAMES
        if f"{model}_var_95" in aligned.columns
    ]

    forecast_rows = []

    for _, row in aligned.iterrows():
        for model in available_models:
            forecast = {
                "model": model,
                "date": row["date"],
                "ticker": row["ticker"],
                "realized_return": row[f"{model}_realized_return"],
            }

            for suffix in (95, 99):
                forecast[f"var_{suffix}"] = row[
                    f"{model}_var_{suffix}"
                ]
                forecast[f"es_{suffix}"] = row[
                    f"{model}_es_{suffix}"
                ]
                forecast[f"exception_{suffix}"] = row[
                    f"{model}_exception_{suffix}"
                ]

            forecast_rows.append(forecast)

    if not forecast_rows:
        raise ValueError("No aligned model forecasts to convert.")

    return pd.DataFrame(forecast_rows).sort_values(
        ["model", "ticker", "date"]
    ).reset_index(drop=True)


def main():
    returns = load_returns()

    historical = backtest_historical_var(
        returns,
        lookback=LOOKBACK,
        confidence_levels=CONFIDENCE_LEVELS,
    )

    ewma = calculate_ewma_var(
        returns,
        decay_factor=DECAY_FACTOR,
        min_periods=LOOKBACK,
        confidence_levels=CONFIDENCE_LEVELS,
    )

    ewma_fhs = calculate_ewma_fhs_var(
        returns,
        decay_factor=DECAY_FACTOR,
        min_periods=LOOKBACK,
        residual_lookback=FHS_RESIDUAL_LOOKBACK,
        min_residuals=FHS_MIN_RESIDUALS,
        confidence_levels=CONFIDENCE_LEVELS,
    )

    aligned = align_model_forecasts(
        historical,
        ewma,
        ewma_fhs,
    )

    forecasts = build_long_forecasts(aligned)
    diagnostics = calculate_diagnostics(forecasts)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    forecasts.to_csv(
        OUTPUT_DIR / "aligned_forecasts.csv",
        index=False,
    )
    diagnostics.to_csv(
        OUTPUT_DIR / "diagnostics.csv",
        index=False,
    )

    configuration = {
        "data_path": str(DATA_PATH),
        "calendar": CALENDAR_NAME,
        "historical_lookback": LOOKBACK,
        "ewma_decay_factor": DECAY_FACTOR,
        "fhs": {
            "method": "EWMA-filtered historical simulation",
            "residual_lookback": FHS_RESIDUAL_LOOKBACK,
            "min_residuals": FHS_MIN_RESIDUALS,
            "residual_construction": (
                "Each return is standardized using its pre-observation "
                "EWMA volatility forecast."
            ),
            "residual_history": (
                "Rolling window of valid standardized residuals; "
                "previous residuals are retained across missing-return "
                "gaps while EWMA warm-up restarts."
            ),
            "tail_quantile_method": "linear empirical quantile",
        },
        "es_methods": {
            "historical": (
                "Integrated lower-tail empirical quantile mean using "
                "the piecewise-linear empirical quantile function."
            ),
            "ewma": (
                "Gaussian closed-form expected shortfall using the "
                "same pre-observation EWMA volatility forecast."
            ),
            "ewma_fhs": (
                "Integrated lower-tail empirical mean of the same "
                "prior standardized residual history used for FHS VaR."
            ),
            "risk_measure_convention": (
                "Positive loss amounts; ES is constrained to be no "
                "smaller than VaR at the corresponding confidence level."
            ),
        },
        "es_independent_validation": {
            "status": "not_yet_performed",
            "note": (
                "ES forecasts are integrated into the comparison output, "
                "but have not yet undergone independent joint VaR-ES "
                "scoring or calibration validation."
            ),
        },
        "confidence_levels": list(CONFIDENCE_LEVELS),
        "significance_level": ALPHA,
        "return_construction": (
            "Prices aligned to XNYS sessions before pct_change; "
            "no forward-fill; pct_change(fill_method=None)"
        ),
        "ewma_assumptions": [
            "zero expected return",
            "variance initialized with first valid return squared",
            "missing returns reset the EWMA warm-up sequence",
        ],
        "comparison": (
            "Inner join on ticker/date across all included models; "
            "complete VaR and ES forecasts required at both confidence "
            "levels; realized-return missingness and observed values "
            "must match; missing realized returns are retained."
        ),
        "risk_forecast_ordering": [
            "VaR99 >= VaR95",
            "ES95 >= VaR95",
            "ES99 >= VaR99",
            "ES99 >= ES95",
        ],
        "multiple_testing": (
            "Holm correction across model/ticker combinations per "
            "confidence level and diagnostic family. These are VaR "
            "exception diagnostics, not ES validation."
        ),
        "aligned_ticker_date_pairs": int(len(aligned)),
        "aligned_tickers": sorted(aligned["ticker"].unique().tolist()),
        "aligned_start_date": aligned["date"].min().strftime("%Y-%m-%d"),
        "aligned_end_date": aligned["date"].max().strftime("%Y-%m-%d"),
        "forecasts_by_model": (
            forecasts.groupby("model").size().astype(int).to_dict()
        ),
        "unevaluable_forecasts_by_model": (
            forecasts.groupby("model")["realized_return"]
            .apply(lambda values: int(values.isna().sum()))
            .to_dict()
        ),
        "fhs_residual_observations_at_first_forecast": (
            int(
                ewma_fhs.loc[
                    ewma_fhs["var_95"].notna(),
                    "residual_observations",
                ].min()
            )
            if ewma_fhs["var_95"].notna().any()
            else None
        ),
    }

    with (OUTPUT_DIR / "configuration.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(configuration, file, indent=2)

    print("\nMODEL COMPARISON")
    print(f"Aligned ticker/date pairs: {len(aligned):,}")
    print(f"Tickers: {', '.join(configuration['aligned_tickers'])}")
    print(
        f"Evaluation period: {configuration['aligned_start_date']} "
        f"to {configuration['aligned_end_date']}"
    )

    print("\nFORECAST COUNTS")
    print(forecasts.groupby("model").size().to_string())

    summary_columns = [
        "model",
        "ticker",
        "confidence",
        "forecasts",
        "observations",
        "unevaluable_forecasts",
        "exceptions",
        "observed_exception_rate",
        "kupiec_holm_reject",
        "independence_holm_reject",
        "conditional_holm_reject",
    ]

    print("\nVaR BACKTEST SUMMARY")
    print(
        diagnostics[summary_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
