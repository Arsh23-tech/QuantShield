
import numpy as np
import pandas as pd
import exchange_calendars as xc


def _normalize_date(value, name):
    date = pd.Timestamp(value)

    if pd.isna(date):
        raise ValueError(f"{name} must be a valid date.")

    if date.tz is not None:
        date = date.tz_localize(None)

    return date.normalize()


def get_expected_trading_sessions(
    start_date,
    end_date,
    calendar_name="XNYS",
):
    """Return expected exchange sessions between two inclusive dates."""

    start = _normalize_date(start_date, "start_date")
    end = _normalize_date(end_date, "end_date")

    if start > end:
        raise ValueError("start_date must not be after end_date.")

    calendar = xc.get_calendar(calendar_name)
    sessions = calendar.sessions_in_range(start, end)

    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)

    return sessions.normalize()


def align_returns_to_trading_sessions(
    returns,
    ticker,
    start_date,
    end_date,
    return_column="simple_return",
    calendar_name="XNYS",
):
    """
    Align one ticker's returns to expected exchange sessions.

    source_row_missing=True means no source row existed for that session.
    return_missing=True means the aligned return is unavailable.
    """

    required = {"date", "ticker", return_column}
    missing = required - set(returns.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string.")

    start = _normalize_date(start_date, "start_date")
    end = _normalize_date(end_date, "end_date")

    if start > end:
        raise ValueError("start_date must not be after end_date.")

    df = returns.loc[
        returns["ticker"] == ticker,
        ["date", return_column],
    ].copy()

    df["date"] = pd.to_datetime(df["date"], errors="raise")

    if df["date"].isna().any():
        raise ValueError("Dates must not be missing.")

    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    df["date"] = df["date"].dt.normalize()

    df = df[
        df["date"].between(start, end)
    ].copy()

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates (date/ticker observations) found.")

    df[return_column] = pd.to_numeric(
        df[return_column], errors="raise"
    )

    observed = df[return_column].dropna().to_numpy(dtype=float)

    if not np.isfinite(observed).all():
        raise ValueError("Returns must be finite or NaN.")

    if (observed < -1).any():
        raise ValueError("Returns below -100% are invalid.")

    sessions = get_expected_trading_sessions(
        start,
        end,
        calendar_name=calendar_name,
    )

    df = df.set_index("date").sort_index()

    unexpected = df.index.difference(sessions)

    if len(unexpected):
        raise ValueError(
            "Unexpected trading dates (non-trading dates) found: "
            f"{unexpected.strftime('%Y-%m-%d').tolist()}"
        )

    aligned = df.reindex(sessions)
    aligned.index.name = "date"
    aligned["ticker"] = ticker

    aligned["source_row_missing"] = ~aligned.index.isin(df.index)
    aligned["return_missing"] = aligned[return_column].isna()

    return aligned.reset_index()
