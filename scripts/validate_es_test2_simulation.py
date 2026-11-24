
"""Run simulation-based validation of Acerbi-Szekely Test 2."""

import json
from pathlib import Path

import pandas as pd

from src.risk.es_test2_simulation import (
    DEFAULT_CONFIDENCE_LEVELS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_REPLICATIONS,
    DEFAULT_SAMPLE_SIZES,
    ES_ONLY_MISSPECIFICATION_SCALES,
    GARCH_ALPHA,
    GARCH_BETA,
    GARCH_BURN_IN,
    GARCH_UNCONDITIONAL_VARIANCE,
    MISSPECIFICATION_SCALES,
    STUDENT_T_DF,
    run_es_test2_simulation_study,
    validate_test2_implementation,
)


OUTPUT_DIR = Path(
    "results/es_backtesting/simulation_validation"
)

NULL_SCENARIOS = [
    "iid_gaussian",
    "iid_student_t",
    "garch_gaussian",
    "garch_student_t",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("VALIDATING SIMULATION IMPLEMENTATION")

    implementation_check = validate_test2_implementation()

    print(implementation_check)

    print("\nRUNNING SIMULATION STUDY")

    results = run_es_test2_simulation_study(
        sample_sizes=DEFAULT_SAMPLE_SIZES,
        replications=DEFAULT_REPLICATIONS,
        confidence_levels=DEFAULT_CONFIDENCE_LEVELS,
        seed=DEFAULT_RANDOM_SEED,
        student_t_df=STUDENT_T_DF,
        misspecification_scales=MISSPECIFICATION_SCALES,
        es_only_misspecification_scales=(
            ES_ONLY_MISSPECIFICATION_SCALES
        ),
    )

    results_path = OUTPUT_DIR / "simulation_results.csv"

    results.to_csv(
        results_path,
        index=False,
    )

    summary_columns = [
        "scenario",
        "sample_size",
        "confidence",
        "replications",
        "forecast_scale",
        "var_forecast_scale",
        "es_forecast_scale",
        "mean_z2",
        "mcse_mean_z2",
        "median_z2",
        "std_z2",
        "z2_p05",
        "z2_p25",
        "z2_p75",
        "z2_p95",
        "negative_z2_fraction",
        "positive_z2_fraction",
        "mean_exception_rate",
        "target_exception_rate",
        "mean_exception_count",
        "mean_observations",
        "inference_status",
    ]

    summary = results[summary_columns].copy()

    summary_path = OUTPUT_DIR / "simulation_summary.csv"

    summary.to_csv(
        summary_path,
        index=False,
    )

    configuration = {
        "method": "Acerbi-Szekely unconditional Test 2",
        "purpose": (
            "Simulation-based validation of statistic behavior "
            "before implementing finite-sample inference."
        ),
        "statistic": (
            "Z2 = 1 - sum(loss_t * I(loss_t > VaR_t) / ES_t) "
            "/ (T * alpha)"
        ),
        "risk_convention": "positive losses",
        "loss_definition": "loss = -return",
        "exception_definition": "loss > VaR",
        "forecast_constraint": "ES >= VaR > 0",
        "confidence_levels": list(DEFAULT_CONFIDENCE_LEVELS),
        "sample_sizes": list(DEFAULT_SAMPLE_SIZES),
        "replications_per_scenario": DEFAULT_REPLICATIONS,
        "random_seed": DEFAULT_RANDOM_SEED,
        "student_t_degrees_of_freedom": STUDENT_T_DF,
        "garch": {
            "model": "GARCH(1,1)",
            "alpha": GARCH_ALPHA,
            "beta": GARCH_BETA,
            "unconditional_variance": GARCH_UNCONDITIONAL_VARIANCE,
            "omega": (
                GARCH_UNCONDITIONAL_VARIANCE
                * (1.0 - GARCH_ALPHA - GARCH_BETA)
            ),
            "burn_in_observations": GARCH_BURN_IN,
            "conditional_forecasts": (
                "Each observation uses its true pre-observation "
                "conditional volatility."
            ),
        },
        "null_scenarios": NULL_SCENARIOS,
        "joint_alternative_scenarios": {
            name: {
                "var_forecast_scale": float(scale),
                "es_forecast_scale": float(scale),
                "description": (
                    "VaR and ES are multiplied by the same scale; "
                    "the simulated return path is unchanged."
                ),
            }
            for name, scale in MISSPECIFICATION_SCALES.items()
        },
        "es_only_alternative_scenarios": {
            name: {
                "var_forecast_scale": 1.0,
                "es_forecast_scale": float(scale),
                "description": (
                    "VaR remains correctly specified and unchanged. "
                    "Only ES is multiplied by this scale. "
                    "The simulated return path is unchanged."
                ),
            }
            for name, scale in (
                ES_ONLY_MISSPECIFICATION_SCALES.items()
            )
        },
        "es_only_understatement_rationale": (
            "A 10% ES reduction is used instead of 20% so that "
            "ES remains greater than or equal to VaR at both "
            "confidence levels and across the modeled distributions."
        ),
        "paired_alternatives": (
            "All misspecified forecasts reuse the same simulated "
            "paths as their corresponding correctly specified null "
            "scenario."
        ),
        "reported_statistics": [
            "mean and median Z2",
            "standard deviation and Monte Carlo standard error",
            "5th, 25th, 75th, and 95th percentiles of Z2",
            "fraction of negative and positive Z2 values",
            "mean exception rate and theoretical target",
            "mean exception count",
            "separate VaR and ES forecast scaling factors",
        ],
        "inference": {
            "status": "not_performed",
            "p_values": False,
            "bootstrap": False,
            "holm_adjustment": False,
            "calibration_rejections": False,
        },
        "implementation_cross_check": implementation_check,
        "outputs": {
            "detailed_results": str(results_path),
            "summary": str(summary_path),
        },
    }

    configuration_path = (
        OUTPUT_DIR / "simulation_configuration.json"
    )

    with configuration_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(configuration, file, indent=2)

    print("\nNULL SCENARIO RESULTS")

    null_results = results[
        results["scenario"].isin(NULL_SCENARIOS)
    ]

    null_columns = [
        "scenario",
        "sample_size",
        "confidence",
        "mean_z2",
        "mcse_mean_z2",
        "mean_exception_rate",
        "target_exception_rate",
    ]

    print(
        null_results[null_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    joint_scenarios = {
        f"{scenario}_{alternative}"
        for scenario in NULL_SCENARIOS
        for alternative in MISSPECIFICATION_SCALES
    }

    joint_results = results[
        results["scenario"].isin(joint_scenarios)
    ]

    print("\nJOINT VAR AND ES MISSPECIFICATION RESULTS")

    alternative_columns = [
        "scenario",
        "sample_size",
        "confidence",
        "var_forecast_scale",
        "es_forecast_scale",
        "mean_z2",
        "mcse_mean_z2",
        "mean_exception_rate",
        "target_exception_rate",
    ]

    print(
        joint_results[alternative_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    es_only_scenarios = {
        f"{scenario}_{alternative}"
        for scenario in NULL_SCENARIOS
        for alternative in ES_ONLY_MISSPECIFICATION_SCALES
    }

    es_only_results = results[
        results["scenario"].isin(es_only_scenarios)
    ]

    print("\nES-ONLY MISSPECIFICATION RESULTS")

    print(
        es_only_results[alternative_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nINFERENCE")
    print(
        "No p-values, bootstrap inference, or calibration "
        "rejections were calculated."
    )

    print(f"\nDetailed results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Configuration: {configuration_path}")


if __name__ == "__main__":
    main()
