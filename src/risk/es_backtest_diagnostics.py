
"""Independent Expected Shortfall backtest diagnostics.

Implements the Acerbi-Szekely unconditional Test 2 statistic using
the positive-loss convention:

    loss = -realized_return
    exception = loss > VaR

    Z2 = 1 - sum(loss_t * exception_t / ES_t) / (T * alpha)

A negative Z2 indicates that realized normalized tail losses are larger
than expected under the joint VaR-ES calibration condition.

IMPORTANT:
    This module calculates the statistic and descriptive diagnostics.
    It does not calculate p-values or claim statistical rejection.
    Finite-sample inference requires a separately validated calibration
    procedure.

Missing realized returns are retained as unevaluable forecast dates.
"""

import numpy as np
import pandas as pd


DEFAULT_CONFIDENCE_LEVELS = (0.95, 0.99)

RISK_COLUMNS = {
    0.95: ("var_95", "es_95"),
    0.99: ("var_99", "es_99"),
}

DEFAULT_LOW_EXPECTED_EXCEPTION_THRESHOLD = 20.0

BASE_COLUMNS = {
    "model",
    "ticker",
    "date",
    "realized_return",
}


def _validate_confidence_levels(confidence_levels):
    """Validate and normalize supported confidence levels."""
    try:
        levels = tuple(float(level) for level in confidence_levels)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Confidence levels must be an iterable of numeric values."
        ) from exc

    if not levels:
        raise ValueError("At least one confidence level is required.")

    if len(set(levels)) != len(levels):
        raise ValueError("Confidence levels must be unique.")

    if any(not 0 < level < 1 for level in levels):
        raise ValueError(
            "Confidence levels must be strictly between 0 and 1."
        )

    unsupported = set(levels) - set(RISK_COLUMNS)
    if unsupported:
        raise ValueError(
            f"Unsupported confidence levels: {sorted(unsupported)}. "
            f"Supported levels are {sorted(RISK_COLUMNS)}."
        )

    return tuple(sorted(levels))


def _validate_forecasts(forecasts, confidence_levels):
    """Validate forecast keys, realized returns, and VaR/ES estimates."""
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
        valid = frame[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        if not valid.all():
            raise ValueError(
                f"{column} values must be non-empty strings."
            )

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")

    if frame["date"].dt.tz is not None:
        frame["date"] = frame["date"].dt.tz_localize(None)

    frame["date"] = frame["date"].dt.normalize()

    key_columns = ["model", "ticker", "date"]
    if frame.duplicated(key_columns).any():
        raise ValueError(
            "Duplicate model/ticker/date forecast keys were found."
        )

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
            frame[column] = pd.to_numeric(
                frame[column],
                errors="raise",
            )
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


def calculate_acerbi_szekely_test2(
    realized_returns,
    var_forecasts,
    es_forecasts,
    confidence,
):
    """Calculate the Acerbi-Szekely unconditional Test 2 statistic.

    Inputs may be one-dimensional array-like objects of equal length.
    Missing realized returns are not accepted by this low-level function;
    callers should filter to evaluable observations first.

    Returns a dictionary containing the statistic and its components.
    No p-value is calculated.
    """
    confidence = float(confidence)

    if not 0 < confidence < 1:
        raise ValueError("Confidence must be strictly between 0 and 1.")

    try:
        returns = np.asarray(realized_returns, dtype=float)
        var_values = np.asarray(var_forecasts, dtype=float)
        es_values = np.asarray(es_forecasts, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Inputs must contain numeric values.") from exc

    if (
        returns.ndim != 1
        or var_values.ndim != 1
        or es_values.ndim != 1
    ):
        raise ValueError("All inputs must be one-dimensional.")

    if not (
        len(returns) == len(var_values) == len(es_values)
    ):
        raise ValueError("Returns, VaR, and ES must have equal lengths.")

    if len(returns) == 0:
        raise ValueError("At least one evaluable observation is required.")

    if not np.isfinite(returns).all():
        raise ValueError("Realized returns must be finite.")

    if not np.isfinite(var_values).all():
        raise ValueError("VaR forecasts must be finite.")

    if not np.isfinite(es_values).all():
        raise ValueError("ES forecasts must be finite.")

    if (var_values < 0).any():
        raise ValueError("VaR forecasts cannot be negative.")

    if (es_values <= 0).any():
        raise ValueError("ES forecasts must be strictly positive.")

    if (es_values + 1e-12 < var_values).any():
        raise ValueError("ES must be greater than or equal to VaR.")

    alpha = 1.0 - confidence
    losses = -returns
    exceptions = losses > var_values

    normalized_tail_loss = (
        losses[exceptions] / es_values[exceptions]
    )

    tail_loss_sum = float(normalized_tail_loss.sum())
    expected_tail_loss_sum = float(len(returns) * alpha)

    tail_loss_ratio = tail_loss_sum / expected_tail_loss_sum
    statistic = 1.0 - tail_loss_ratio

    exception_count = int(exceptions.sum())

    mean_exception_severity_ratio = (
        float(normalized_tail_loss.mean())
        if exception_count
        else np.nan
    )

    return {
        "acerbi_szekely_z2": float(statistic),
        "normalized_tail_loss_sum": tail_loss_sum,
        "expected_normalized_tail_loss_sum": expected_tail_loss_sum,
        "tail_loss_ratio": float(tail_loss_ratio),
        "exceptions": exception_count,
        "exception_rate": float(exceptions.mean()),
        "expected_exception_rate": float(alpha),
        "mean_exception_severity_es_ratio": (
            mean_exception_severity_ratio
        ),
        "n_observations": int(len(returns)),
    }


def calculate_es_backtest_diagnostics(
    forecasts,
    confidence_levels=DEFAULT_CONFIDENCE_LEVELS,
    low_expected_exception_threshold=(
        DEFAULT_LOW_EXPECTED_EXCEPTION_THRESHOLD
    ),
):
    """Calculate per-model, per-ticker, per-confidence ES diagnostics.

    Missing realized returns are retained in the forecast count and
    separately counted as unevaluable. The Test 2 statistic uses only
    evaluable realized returns.

    A low-tail-sample flag is descriptive. It is not a formal power
    calculation or a statistical decision.
    """
    levels = _validate_confidence_levels(confidence_levels)

    if (
        not isinstance(low_expected_exception_threshold, (int, float))
        or isinstance(low_expected_exception_threshold, bool)
        or not np.isfinite(low_expected_exception_threshold)
        or low_expected_exception_threshold < 0
    ):
        raise ValueError(
            "low_expected_exception_threshold must be finite "
            "and non-negative."
        )

    frame = _validate_forecasts(forecasts, levels)

    rows = []

    for (model, ticker), group in frame.groupby(
        ["model", "ticker"],
        sort=True,
    ):
        group = group.sort_values("date").reset_index(drop=True)

        for confidence in levels:
            var_column, es_column = RISK_COLUMNS[confidence]

            evaluable = group["realized_return"].notna()
            observed = group.loc[evaluable]

            n_forecasts = int(len(group))
            n_evaluable = int(evaluable.sum())
            n_unevaluable = n_forecasts - n_evaluable

            alpha = 1.0 - confidence
            expected_exceptions = n_evaluable * alpha

            row = {
                "model": model,
                "ticker": ticker,
                "confidence": confidence,
                "alpha": alpha,
                "forecasts": n_forecasts,
                "observations": n_evaluable,
                "unevaluable_forecasts": n_unevaluable,
                "expected_exceptions": float(expected_exceptions),
                "low_expected_exception_count": (
                    expected_exceptions
                    < low_expected_exception_threshold
                ),
                "low_tail_sample_threshold": float(
                    low_expected_exception_threshold
                ),
                "inference_status": "not_performed",
                "p_value": np.nan,
                "holm_adjusted_p_value": np.nan,
                "reject_calibration": pd.NA,
            }

            if n_evaluable == 0:
                row.update({
                    "exceptions": 0,
                    "observed_exception_rate": np.nan,
                    "expected_exception_rate": alpha,
                    "normalized_tail_loss_sum": np.nan,
                    "expected_normalized_tail_loss_sum": np.nan,
                    "tail_loss_ratio": np.nan,
                    "acerbi_szekely_z2": np.nan,
                    "mean_exception_severity_es_ratio": np.nan,
                    "diagnostic_status": "no_evaluable_returns",
                })
                rows.append(row)
                continue

            statistic = calculate_acerbi_szekely_test2(
                realized_returns=observed["realized_return"].to_numpy(),
                var_forecasts=observed[var_column].to_numpy(),
                es_forecasts=observed[es_column].to_numpy(),
                confidence=confidence,
            )

            diagnostic_status = (
                "low_expected_exception_count"
                if expected_exceptions < low_expected_exception_threshold
                else "descriptive_only"
            )

            row.update({
                "exceptions": statistic["exceptions"],
                "observed_exception_rate": statistic["exception_rate"],
                "expected_exception_rate": alpha,
                "normalized_tail_loss_sum": (
                    statistic["normalized_tail_loss_sum"]
                ),
                "expected_normalized_tail_loss_sum": (
                    statistic["expected_normalized_tail_loss_sum"]
                ),
                "tail_loss_ratio": statistic["tail_loss_ratio"],
                "acerbi_szekely_z2": (
                    statistic["acerbi_szekely_z2"]
                ),
                "mean_exception_severity_es_ratio": (
                    statistic["mean_exception_severity_es_ratio"]
                ),
                "diagnostic_status": diagnostic_status,
            })

            rows.append(row)

    if not rows:
        raise ValueError("No model/ticker forecasts were available.")

    return pd.DataFrame(rows).sort_values(
        ["confidence", "model", "ticker"]
    ).reset_index(drop=True)
