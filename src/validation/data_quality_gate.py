
from __future__ import annotations

from typing import Any

import pandas as pd

from src.validation.market_validator import validate_market_data
from src.validation.return_validator import validate_returns
from src.validation.trading_calendar_validator import validate_trading_calendar


def validate_market_data_quality(
    prices: pd.DataFrame,
    calendar_name: str = "XNYS",
    extreme_return_threshold: float = 0.20,
) -> dict[str, Any]:
    """
    Run market-data, trading-calendar, and return-quality diagnostics.

    Blocking failures prevent downstream risk calculations. Extreme
    returns are warnings for review and are not automatically rejected.

    This function does not modify, remove, or repair input observations.
    """
    market_report = validate_market_data(prices)
    calendar_report = validate_trading_calendar(
        prices,
        calendar_name=calendar_name,
    )

    return_report: dict[str, Any] = {
        "status": "NOT_RUN",
        "rows_checked": 0,
        "issues": {},
    }

    # Only calculate returns when the basic market-data structure and
    # values pass validation. This avoids calculating returns from
    # known-invalid prices or duplicate observations.
    if market_report.get("status") == "PASS":
        required_columns = {"date", "ticker", "adjusted_close"}

        if required_columns.issubset(prices.columns):
            from src.features.returns import calculate_returns

            calculated_returns = calculate_returns(prices)

            return_report = validate_returns(
                calculated_returns[
                    ["date", "ticker", "simple_return"]
                ],
                extreme_return_threshold=extreme_return_threshold,
            )
        else:
            return_report = {
                "status": "FAIL",
                "rows_checked": len(prices),
                "issues": {
                    "missing_return_input_columns": sorted(
                        required_columns - set(prices.columns)
                    )
                },
            }
    else:
        return_report = {
            "status": "NOT_RUN",
            "rows_checked": 0,
            "issues": {
                "reason": (
                    "Return validation skipped because market-data "
                    "validation failed."
                )
            },
        }

    reports = {
        "market_data": market_report,
        "trading_calendar": calendar_report,
        "returns": return_report,
    }

    blocking_failures = [
        name
        for name, report in reports.items()
        if report.get("status") == "FAIL"
    ]

    skipped_checks = [
        name
        for name, report in reports.items()
        if report.get("status") == "NOT_RUN"
    ]

    # Extreme returns are non-blocking review flags.
    return_issues = return_report.get("issues", {})
    warnings = {}

    if return_issues.get("extreme_return_rows_for_review", 0):
        warnings["extreme_return_rows_for_review"] = return_issues[
            "extreme_return_rows_for_review"
        ]

    if blocking_failures:
        overall_status = "FAIL"
    elif skipped_checks:
        overall_status = "INCOMPLETE"
    else:
        overall_status = "PASS"

    return {
        "status": overall_status,
        "blocking_failures": blocking_failures,
        "skipped_checks": skipped_checks,
        "warnings": warnings,
        "checks": reports,
    }
