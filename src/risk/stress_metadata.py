
import hashlib
import json

import numpy as np
import pandas as pd


def _prepare_stress_inputs(
    returns: pd.DataFrame,
    weights: dict[str, float],
    return_column: str,
) -> tuple[dict, pd.DataFrame, pd.Series]:
    """Validate and normalize inputs used by stress-run metadata."""

    if not weights:
        raise ValueError("Weights cannot be empty.")

    weight_series = pd.Series(weights, dtype="float64")

    if not np.isfinite(weight_series.to_numpy()).all():
        raise ValueError("Weights must be finite.")

    if (weight_series < 0).any():
        raise ValueError("Only long-only portfolios are supported.")

    if not np.isclose(weight_series.sum(), 1.0, atol=1e-8):
        raise ValueError("Portfolio weights must sum to 1.")

    required = {"date", "ticker", return_column}
    missing = required - set(returns.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = returns[["date", "ticker", return_column]].copy()
    df = df[df["ticker"].isin(weight_series.index)].copy()

    try:
        df["date"] = pd.to_datetime(
            df["date"], errors="raise", format="mixed"
        )
        df[return_column] = pd.to_numeric(
            df[return_column], errors="raise"
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Dates and returns must be valid and numeric."
        ) from exc

    if df["date"].isna().any():
        raise ValueError("Dates must not be missing.")

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker observations found.")

    values = df[return_column].to_numpy()

    if (
        pd.isna(values).any()
        or not np.isfinite(values).all()
        or (values < -1.0).any()
    ):
        raise ValueError(
            "Returns must be finite, non-missing simple returns "
            "of at least -100%."
        )

    missing_tickers = set(weight_series.index) - set(df["ticker"])
    if missing_tickers:
        raise ValueError(
            f"Missing returns for tickers: {sorted(missing_tickers)}"
        )

    aligned = (
        df.pivot(
            index="date",
            columns="ticker",
            values=return_column,
        )
        .reindex(columns=weight_series.index)
        .sort_index()
        .dropna(how="any")
    )

    if aligned.empty:
        raise ValueError(
            "No complete portfolio return dates available."
        )

    # This canonical payload is both saved as a snapshot and hashed.
    normalized = df.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    normalized["date"] = normalized["date"].map(
        lambda value: value.isoformat()
    )
    normalized[return_column] = normalized[return_column].astype(float)

    input_payload = {
        "return_column": return_column,
        "observations": normalized.to_dict(orient="records"),
        "weights": [
            {
                "ticker": ticker,
                "weight": float(weight_series[ticker]),
            }
            for ticker in weight_series.index
        ],
    }

    return input_payload, aligned, weight_series


def build_stress_input_snapshot(
    returns: pd.DataFrame,
    weights: dict[str, float],
    return_column: str = "simple_return",
) -> dict:
    """
    Build the normalized input snapshot used for fingerprinting.

    Save the returned dictionary as JSON alongside a stress-run record.
    """
    snapshot, _, _ = _prepare_stress_inputs(
        returns=returns,
        weights=weights,
        return_column=return_column,
    )
    return snapshot


def fingerprint_stress_input_snapshot(snapshot: dict) -> str:
    """
    Calculate SHA-256 from a previously saved canonical input snapshot.

    This function hashes the supplied snapshot. Callers should validate
    the snapshot's structure before treating it as a trusted record.
    """
    serialized = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(serialized).hexdigest()


def build_stress_run_metadata(
    returns: pd.DataFrame,
    weights: dict[str, float],
    scenarios: pd.DataFrame,
    return_column: str = "simple_return",
    top_n: int = 10,
) -> dict:
    """
    Build reproducibility metadata for a historical stress run.

    The fingerprint identifies normalized input observations and portfolio
    weights. It does not prove market-data accuracy or source authenticity.
    """
    if (
        not isinstance(top_n, int)
        or isinstance(top_n, bool)
        or top_n < 1
    ):
        raise ValueError("top_n must be a positive integer.")

    snapshot, aligned, weight_series = _prepare_stress_inputs(
        returns=returns,
        weights=weights,
        return_column=return_column,
    )

    fingerprint = fingerprint_stress_input_snapshot(snapshot)

    if "date" not in scenarios.columns:
        raise ValueError("Scenarios must contain a date column.")

    return {
        "metadata_version": 1,
        "return_column": return_column,
        "top_n": top_n,
        "tickers": list(weight_series.index),
        "weights": {
            ticker: float(weight_series[ticker])
            for ticker in weight_series.index
        },
        "complete_observation_count": int(len(aligned)),
        "first_complete_date": aligned.index.min().isoformat(),
        "last_complete_date": aligned.index.max().isoformat(),
        "selected_scenario_dates": [
            pd.Timestamp(date).isoformat()
            for date in scenarios["date"]
        ],
        "input_sha256": fingerprint,
    }
