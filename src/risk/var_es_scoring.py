
"""Joint VaR-ES scoring and paired model comparison.

Risk forecasts use the positive-loss convention:
    realized_loss = -realized_return

Lower FZ0 scores indicate better joint VaR-ES forecasts. Raw score levels
are scale-dependent, so model comparisons use paired score differences
for the same ticker and date, with equal ticker weighting.
"""

from itertools import combinations

import numpy as np
import pandas as pd


DEFAULT_CONFIDENCE_LEVELS = (0.95, 0.99)

BASE_COLUMNS = {
    "model",
    "date",
    "ticker",
    "realized_return",
}

RISK_COLUMNS = {
    0.95: ("var_95", "es_95"),
    0.99: ("var_99", "es_99"),
}


def _validate_confidence_levels(confidence_levels):
    levels = tuple(float(level) for level in confidence_levels)

    if not levels:
        raise ValueError("At least one confidence level is required.")

    if len(set(levels)) != len(levels):
        raise ValueError("Confidence levels must be unique.")

    if any(not 0 < level < 1 for level in levels):
        raise ValueError("Confidence levels must be between 0 and 1.")

    unsupported = set(levels) - set(RISK_COLUMNS)
    if unsupported:
        raise ValueError(
            f"Unsupported confidence levels: {sorted(unsupported)}. "
            f"Supported levels are {sorted(RISK_COLUMNS)}."
        )

    return tuple(sorted(levels))


def calculate_fz0_score(
    realized_return,
    var_forecast,
    es_forecast,
    confidence,
):
    """Calculate the FZ0 joint VaR-ES score using positive losses.

    Parameters may be scalars or NumPy-broadcastable arrays.

    A realized return is converted to loss using loss = -return.
    Strict exceedances (loss > VaR) contribute to the tail term.

    Lower scores are better. Scores are not constrained to be positive.
    """
    confidence = float(confidence)

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1.")

    realized_return = np.asarray(realized_return, dtype=float)
    var_forecast = np.asarray(var_forecast, dtype=float)
    es_forecast = np.asarray(es_forecast, dtype=float)

    try:
        realized_return, var_forecast, es_forecast = np.broadcast_arrays(
            realized_return,
            var_forecast,
            es_forecast,
        )
    except ValueError as exc:
        raise ValueError(
            "Returns, VaR, and ES inputs must be broadcast-compatible."
        ) from exc

    if not np.isfinite(realized_return).all():
        raise ValueError("Realized returns must be finite.")

    if not np.isfinite(var_forecast).all():
        raise ValueError("VaR forecasts must be finite.")

    if not np.isfinite(es_forecast).all():
        raise ValueError("ES forecasts must be finite.")

    if (var_forecast < 0).any():
        raise ValueError("VaR forecasts cannot be negative.")

    if (es_forecast <= 0).any():
        raise ValueError("ES forecasts must be strictly positive.")

    if (es_forecast + 1e-12 < var_forecast).any():
        raise ValueError("ES must be greater than or equal to VaR.")

    alpha = 1.0 - confidence
    realized_loss = -realized_return
    exceedance = realized_loss > var_forecast

    score = (
        exceedance.astype(float)
        * (realized_loss - var_forecast)
        / (alpha * es_forecast)
        + var_forecast / es_forecast
        + np.log(es_forecast)
        - 1.0
    )

    if score.ndim == 0:
        return float(score)

    return score


def _validate_forecasts(forecasts, confidence_levels):
    if not isinstance(forecasts, pd.DataFrame):
        raise TypeError("Forecasts must be a pandas DataFrame.")

    levels = _validate_confidence_levels(confidence_levels)

    required = set(BASE_COLUMNS)
    for level in levels:
        required.update(RISK_COLUMNS[level])

    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(
            f"Forecasts are missing required columns: {sorted(missing)}"
        )

    frame = forecasts[list(required)].copy()

    if frame[["model", "ticker", "date"]].isna().any().any():
        raise ValueError("Model, ticker, and date keys cannot be missing.")

    for column in ("model", "ticker"):
        invalid = ~frame[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        if invalid.any():
            raise ValueError(f"{column} values must be non-empty strings.")

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")

    if frame["date"].dt.tz is not None:
        frame["date"] = frame["date"].dt.tz_localize(None)

    frame["date"] = frame["date"].dt.normalize()

    if frame.duplicated(["model", "ticker", "date"]).any():
        raise ValueError("Duplicate model/ticker/date forecast keys found.")

    frame["realized_return"] = pd.to_numeric(
        frame["realized_return"],
        errors="raise",
    )

    observed_returns = frame["realized_return"].dropna().to_numpy(
        dtype=float
    )
    if not np.isfinite(observed_returns).all():
        raise ValueError("Observed realized returns must be finite.")

    for level in levels:
        var_column, es_column = RISK_COLUMNS[level]

        for column in (var_column, es_column):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            values = frame[column].to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"{column} forecasts must be finite and non-missing."
                )

            if (values < 0).any():
                raise ValueError(
                    f"{column} forecasts cannot be negative."
                )

        if (frame[es_column] <= 0).any():
            raise ValueError(
                f"{es_column} forecasts must be strictly positive."
            )

        if (
            frame[es_column] + 1e-12 < frame[var_column]
        ).any():
            raise ValueError(
                f"{es_column} must be greater than or equal to "
                f"{var_column}."
            )

    if 0.95 in levels and 0.99 in levels:
        for lower_column, upper_column in (
            ("var_95", "var_99"),
            ("es_95", "es_99"),
        ):
            if (
                frame[upper_column] + 1e-12
                < frame[lower_column]
            ).any():
                raise ValueError(
                    f"{upper_column} must be greater than or equal to "
                    f"{lower_column}."
                )

    return frame.sort_values(
        ["model", "ticker", "date"]
    ).reset_index(drop=True)


def score_forecasts(
    forecasts,
    confidence_levels=DEFAULT_CONFIDENCE_LEVELS,
):
    """Return one row per model/ticker/date/confidence observation.

    Missing realized returns are retained with evaluable=False and
    score=NaN. Risk forecasts themselves must be complete and valid.
    """
    levels = _validate_confidence_levels(confidence_levels)
    frame = _validate_forecasts(forecasts, levels)

    results = []

    for level in levels:
        var_column, es_column = RISK_COLUMNS[level]

        scored = frame[
            [
                "model",
                "date",
                "ticker",
                "realized_return",
                var_column,
                es_column,
            ]
        ].copy()

        scored = scored.rename(columns={
            var_column: "var_forecast",
            es_column: "es_forecast",
        })

        scored["confidence"] = level
        scored["realized_loss"] = -scored["realized_return"]
        scored["evaluable"] = scored["realized_return"].notna()
        scored["score"] = np.nan

        mask = scored["evaluable"]

        if mask.any():
            scored.loc[mask, "score"] = calculate_fz0_score(
                scored.loc[mask, "realized_return"].to_numpy(),
                scored.loc[mask, "var_forecast"].to_numpy(),
                scored.loc[mask, "es_forecast"].to_numpy(),
                level,
            )

        results.append(scored)

    return pd.concat(results, ignore_index=True).sort_values(
        ["confidence", "model", "ticker", "date"]
    ).reset_index(drop=True)


def summarize_scores(scored):
    """Return (ticker_summary, model_summary).

    Model-level mean scores are calculated by first averaging within
    each ticker and then equally weighting the ticker means. This avoids
    allowing a ticker with more observations to dominate the summary.
    """
    required = {
        "model",
        "ticker",
        "confidence",
        "score",
        "evaluable",
    }
    missing = required - set(scored.columns)

    if missing:
        raise ValueError(
            f"Scored data is missing columns: {sorted(missing)}"
        )

    evaluable = scored.loc[scored["evaluable"]].copy()

    if evaluable.empty:
        raise ValueError("No evaluable scores are available.")

    ticker_summary = (
        evaluable.groupby(
            ["model", "ticker", "confidence"],
            as_index=False,
        )
        .agg(
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            n_evaluable=("score", "count"),
        )
    )

    missing_summary = (
        scored.loc[~scored["evaluable"]]
        .groupby(
            ["model", "ticker", "confidence"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_unevaluable"})
    )

    ticker_summary = ticker_summary.merge(
        missing_summary,
        on=["model", "ticker", "confidence"],
        how="left",
        validate="one_to_one",
    )
    ticker_summary["n_unevaluable"] = (
        ticker_summary["n_unevaluable"].fillna(0).astype(int)
    )

    model_summary = (
        ticker_summary.groupby(
            ["model", "confidence"],
            as_index=False,
        )
        .agg(
            equal_ticker_mean_score=("mean_score", "mean"),
            equal_ticker_median_score=("median_score", "median"),
            tickers=("ticker", "nunique"),
            total_evaluable=("n_evaluable", "sum"),
            total_unevaluable=("n_unevaluable", "sum"),
        )
    )

    return ticker_summary, model_summary


def _moving_block_indices(n_dates, block_length, rng):
    """Sample ordered, non-circular moving blocks of date indices."""
    sampled = []

    while len(sampled) < n_dates:
        max_start = n_dates - block_length
        start = int(rng.integers(0, max_start + 1))
        sampled.extend(range(start, start + block_length))

    return np.asarray(sampled[:n_dates], dtype=int)


def _equal_ticker_mean(matrix):
    values = matrix.to_numpy(dtype=float)
    ticker_counts = np.isfinite(values).sum(axis=0)

    ticker_means = np.divide(
        np.nansum(values, axis=0),
        ticker_counts,
        out=np.full(values.shape[1], np.nan),
        where=ticker_counts > 0,
    )

    valid = np.isfinite(ticker_means)
    if not valid.any():
        return np.nan

    return float(ticker_means[valid].mean())


def compare_paired_scores(
    scored,
    block_length=20,
    n_bootstrap=2000,
    seed=42,
):
    """Compare model pairs using paired FZ0 score differences.

    Difference convention:
        mean_score_difference = score(model_a) - score(model_b)

    Negative values favor model_a.

    The bootstrap resamples common date blocks across all tickers.
    This retains cross-ticker observations from the same sampled dates.
    Within each bootstrap sample, ticker means receive equal weight.

    Confidence intervals are percentile intervals, not hypothesis-test
    p-values. They quantify sampling uncertainty under this bootstrap
    design and should not be interpreted as proof of model calibration.
    """
    required = {
        "model",
        "date",
        "ticker",
        "confidence",
        "score",
        "evaluable",
    }
    missing = required - set(scored.columns)

    if missing:
        raise ValueError(
            f"Scored data is missing columns: {sorted(missing)}"
        )

    if not isinstance(block_length, int) or block_length < 1:
        raise ValueError("block_length must be a positive integer.")

    if not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")

    if scored.duplicated(
        ["model", "ticker", "date", "confidence"]
    ).any():
        raise ValueError("Duplicate scored model/ticker/date/confidence keys.")

    models = sorted(scored["model"].dropna().unique().tolist())
    if len(models) < 2:
        raise ValueError("At least two models are required for comparison.")

    rng = np.random.default_rng(seed)
    rows = []

    for confidence in sorted(scored["confidence"].unique()):
        subset = scored.loc[
            (scored["confidence"] == confidence)
            & scored["evaluable"]
        ]

        for model_a, model_b in combinations(models, 2):
            a = subset.loc[
                subset["model"] == model_a,
                ["date", "ticker", "score"],
            ].rename(columns={"score": "score_a"})

            b = subset.loc[
                subset["model"] == model_b,
                ["date", "ticker", "score"],
            ].rename(columns={"score": "score_b"})

            paired = a.merge(
                b,
                on=["date", "ticker"],
                how="inner",
                validate="one_to_one",
            )

            if paired.empty:
                continue

            paired["score_difference"] = (
                paired["score_a"] - paired["score_b"]
            )

            matrix = paired.pivot(
                index="date",
                columns="ticker",
                values="score_difference",
            ).sort_index()

            # Remove dates with no paired ticker observations.
            matrix = matrix.loc[matrix.notna().any(axis=1)]

            n_dates = len(matrix)
            n_tickers = int(matrix.notna().any(axis=0).sum())
            n_observations = int(matrix.notna().sum().sum())
            point_estimate = _equal_ticker_mean(matrix)

            effective_block_length = min(block_length, n_dates)
            bootstrap_estimates = []

            if n_dates >= 2:
                for _ in range(n_bootstrap):
                    indices = _moving_block_indices(
                        n_dates,
                        effective_block_length,
                        rng,
                    )
                    sample = matrix.iloc[indices]
                    estimate = _equal_ticker_mean(sample)

                    if np.isfinite(estimate):
                        bootstrap_estimates.append(estimate)

            if len(bootstrap_estimates) >= 2:
                ci_lower, ci_upper = np.quantile(
                    bootstrap_estimates,
                    [0.025, 0.975],
                )
                bootstrap_status = "ok"
            else:
                ci_lower = np.nan
                ci_upper = np.nan
                bootstrap_status = "insufficient_dates"

            rows.append({
                "confidence": confidence,
                "model_a": model_a,
                "model_b": model_b,
                "mean_score_difference": point_estimate,
                "ci_lower_95": float(ci_lower),
                "ci_upper_95": float(ci_upper),
                "n_dates": n_dates,
                "n_tickers": n_tickers,
                "n_paired_observations": n_observations,
                "bootstrap_replicates": len(bootstrap_estimates),
                "block_length": effective_block_length,
                "seed": seed,
                "bootstrap_status": bootstrap_status,
            })

    return pd.DataFrame(rows)
