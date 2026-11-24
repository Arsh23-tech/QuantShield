
import pandas as pd
import exchange_calendars as xc


def validate_trading_calendar(
    df: pd.DataFrame,
    calendar_name: str = "XNYS",
) -> dict:
    """
    Compare observed market dates with expected exchange sessions.

    This is a calendar-coverage diagnostic. It does not verify price
    accuracy or establish that every instrument follows this calendar.
    """
    required_columns = ["date", "ticker"]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        return {
            "status": "FAIL",
            "missing_columns": missing_columns,
        }

    issues = {}

    dates = pd.to_datetime(df["date"], errors="coerce")

    invalid_dates = dates.isna()
    if invalid_dates.any():
        issues["invalid_date_rows"] = int(invalid_dates.sum())

    valid_rows = df.loc[~invalid_dates, ["ticker"]].copy()
    valid_rows["date"] = dates.loc[~invalid_dates].dt.normalize()

    invalid_tickers = valid_rows["ticker"].map(
        lambda value: not isinstance(value, str) or not value.strip()
    )

    if invalid_tickers.any():
        issues["invalid_ticker_rows"] = int(invalid_tickers.sum())

    valid_rows = valid_rows.loc[~invalid_tickers]

    if valid_rows.empty:
        return {
            "status": "FAIL",
            "rows_checked": len(df),
            "issues": issues,
            "calendar": calendar_name,
            "missing_sessions": {},
            "unexpected_dates": {},
        }

    calendar = xc.get_calendar(calendar_name)

    observed_dates = pd.DatetimeIndex(
        valid_rows["date"].drop_duplicates()
    ).sort_values()

    expected_sessions = calendar.sessions_in_range(
        observed_dates.min(),
        observed_dates.max(),
    )

    expected_sessions = (
        expected_sessions.tz_localize(None).normalize()
        if expected_sessions.tz is not None
        else expected_sessions.normalize()
    )

    missing_sessions = {}
    unexpected_dates = {}

    for ticker, group in valid_rows.groupby("ticker"):
        ticker_dates = pd.DatetimeIndex(
            group["date"].drop_duplicates()
        ).sort_values()

        missing = expected_sessions.difference(ticker_dates)
        unexpected = ticker_dates.difference(expected_sessions)

        if len(missing):
            missing_sessions[ticker] = [
                date.strftime("%Y-%m-%d") for date in missing
            ]

        if len(unexpected):
            unexpected_dates[ticker] = [
                date.strftime("%Y-%m-%d") for date in unexpected
            ]

    if missing_sessions:
        issues["missing_trading_sessions"] = missing_sessions

    if unexpected_dates:
        issues["unexpected_trading_dates"] = unexpected_dates

    return {
        "status": "PASS" if not issues else "FAIL",
        "rows_checked": len(df),
        "calendar": calendar_name,
        "observed_date_count": len(observed_dates),
        "expected_session_count": len(expected_sessions),
        "missing_sessions": missing_sessions,
        "unexpected_dates": unexpected_dates,
        "issues": issues,
    }
