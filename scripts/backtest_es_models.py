
"""Run independent Acerbi-Szekely ES backtest diagnostics."""

import json
from pathlib import Path

import pandas as pd

from src.risk.es_backtest_diagnostics import (
    DEFAULT_CONFIDENCE_LEVELS,
    DEFAULT_LOW_EXPECTED_EXCEPTION_THRESHOLD,
    calculate_es_backtest_diagnostics,
)


INPUT_PATH = Path(
    "results/var_model_comparison/aligned_forecasts.csv"
)

OUTPUT_DIR = Path("results/es_backtesting")

LOW_EXPECTED_EXCEPTION_THRESHOLD = (
    DEFAULT_LOW_EXPECTED_EXCEPTION_THRESHOLD
)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Aligned forecast file not found: {INPUT_PATH}. "
            "Run scripts/compare_var_models.py first."
        )

    forecasts = pd.read_csv(
        INPUT_PATH,
        parse_dates=["date"],
    )

    diagnostics = calculate_es_backtest_diagnostics(
        forecasts,
        confidence_levels=DEFAULT_CONFIDENCE_LEVELS,
        low_expected_exception_threshold=(
            LOW_EXPECTED_EXCEPTION_THRESHOLD
        ),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    diagnostics_path = OUTPUT_DIR / "es_diagnostics.csv"
    configuration_path = OUTPUT_DIR / "es_backtest_configuration.json"

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
    )

    configuration = {
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_DIR),
        "method": "Acerbi-Szekely unconditional Test 2",
        "risk_measure_convention": "positive losses",
        "realized_loss": "-realized_return",
        "exception_definition": "realized_loss > VaR",
        "statistic": (
            "Z2 = 1 - sum(loss_t * exception_t / ES_t) "
            "/ (T * alpha)"
        ),
        "null_target": 0.0,
        "interpretation": {
            "negative": (
                "Observed normalized tail losses exceed the "
                "joint VaR-ES calibration target; directionally "
                "consistent with risk underestimation."
            ),
            "positive": (
                "Observed normalized tail losses are below the "
                "joint VaR-ES calibration target; directionally "
                "consistent with conservative risk forecasts."
            ),
            "near_zero": (
                "The sample statistic is near its theoretical "
                "target; this alone does not establish calibration."
            ),
        },
        "confidence_levels": list(DEFAULT_CONFIDENCE_LEVELS),
        "low_expected_exception_threshold": (
            LOW_EXPECTED_EXCEPTION_THRESHOLD
        ),
        "low_tail_sample_flag": (
            "Descriptive warning only; not a formal power calculation."
        ),
        "missing_returns": (
            "Retained in forecast counts and excluded from the "
            "evaluable sample used by the statistic."
        ),
        "inference": {
            "status": "not_performed",
            "p_values": False,
            "holm_adjustment": False,
            "reason": (
                "Finite-sample null calibration has not yet been "
                "implemented and validated. The statistic is reported "
                "descriptively; no statistical rejection is claimed."
            ),
        },
        "forecast_generation": (
            "Not performed. Existing aligned forecasts are consumed "
            "without modifying historical, EWMA, or EWMA-FHS models."
        ),
        "models": sorted(diagnostics["model"].unique().tolist()),
        "tickers": sorted(diagnostics["ticker"].unique().tolist()),
        "evaluation_start": (
            forecasts["date"].min().strftime("%Y-%m-%d")
        ),
        "evaluation_end": (
            forecasts["date"].max().strftime("%Y-%m-%d")
        ),
        "diagnostic_rows": int(len(diagnostics)),
    }

    with configuration_path.open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=2)

    display_columns = [
        "model",
        "ticker",
        "confidence",
        "forecasts",
        "observations",
        "unevaluable_forecasts",
        "exceptions",
        "expected_exceptions",
        "acerbi_szekely_z2",
        "tail_loss_ratio",
        "diagnostic_status",
        "inference_status",
    ]

    print("\nEXPECTED SHORTFALL BACKTEST")
    print(f"Input: {INPUT_PATH}")
    print(f"Evaluation period: {configuration['evaluation_start']} "
          f"to {configuration['evaluation_end']}")
    print(f"Models: {', '.join(configuration['models'])}")
    print(f"Tickers: {', '.join(configuration['tickers'])}")

    print("\nDIAGNOSTICS")
    print(
        diagnostics[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nINFERENCE")
    print(
        "No p-values or calibration rejections were calculated. "
        "The reported statistic is descriptive."
    )

    print(f"\nSaved diagnostics: {diagnostics_path}")
    print(f"Saved configuration: {configuration_path}")


if __name__ == "__main__":
    main()
