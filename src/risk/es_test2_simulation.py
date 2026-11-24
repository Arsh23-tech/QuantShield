
"""Simulation validation for Acerbi-Szekely unconditional ES Test 2.

All risk forecasts use the positive-loss convention:
    loss = -return
    exception = loss > VaR

Correctly specified scenarios:
    1. IID Gaussian
    2. IID variance-standardized Student-t
    3. Gaussian GARCH(1,1)
    4. Student-t GARCH(1,1)

Joint alternatives scale VaR and ES together.
ES-only alternatives leave VaR unchanged and scale ES separately.

This module evaluates statistic behavior only. It does not calculate
p-values, bootstrap inference, or calibration rejections.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from src.risk.es_backtest_diagnostics import (
    calculate_acerbi_szekely_test2,
)


DEFAULT_CONFIDENCE_LEVELS = (0.95, 0.99)
DEFAULT_SAMPLE_SIZES = (500, 1266, 2500)
DEFAULT_REPLICATIONS = 2000
DEFAULT_RANDOM_SEED = 20261113

STUDENT_T_DF = 7
GARCH_ALPHA = 0.05
GARCH_BETA = 0.90
GARCH_UNCONDITIONAL_VARIANCE = 1.0
GARCH_BURN_IN = 500

MISSPECIFICATION_SCALES = {
    "understated_20pct": 0.80,
    "conservative_20pct": 1.20,
}

ES_ONLY_MISSPECIFICATION_SCALES = {
    "es_understated_10pct": 0.90,
    "es_conservative_20pct": 1.20,
}


@dataclass(frozen=True)
class RiskMultipliers:
    """Unit-scale VaR and ES multipliers for one innovation distribution."""

    var: float
    es: float


def _validate_positive_finite(value, name):
    """Return a finite positive float or raise ValueError."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive.")

    try:
        normalized = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be finite and positive.") from None

    if not np.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be finite and positive.")

    return normalized


def _validate_study_inputs(
    sample_sizes,
    replications,
    confidence_levels,
    seed,
):
    """Validate simulation dimensions and confidence levels."""
    if isinstance(sample_sizes, (str, bytes)):
        raise ValueError("Sample sizes must be a non-empty sequence.")

    try:
        supplied_sizes = tuple(sample_sizes)
    except TypeError:
        raise ValueError(
            "Sample sizes must be a non-empty sequence."
        ) from None

    if not supplied_sizes:
        raise ValueError("At least one sample size is required.")

    normalized_sizes = []

    for value in supplied_sizes:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                "Sample sizes must be integers of at least 2."
            )

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                "Sample sizes must be integers of at least 2."
            ) from None

        if (
            not np.isfinite(numeric_value)
            or not numeric_value.is_integer()
            or numeric_value < 2
        ):
            raise ValueError(
                "Sample sizes must be integers of at least 2."
            )

        normalized_sizes.append(int(numeric_value))

    normalized_sizes = tuple(normalized_sizes)

    if len(set(normalized_sizes)) != len(normalized_sizes):
        raise ValueError("Sample sizes must be unique.")

    if (
        isinstance(replications, (bool, np.bool_))
        or not isinstance(replications, (int, np.integer))
        or replications < 1
    ):
        raise ValueError("replications must be a positive integer.")

    if isinstance(confidence_levels, (str, bytes)):
        raise ValueError(
            "Confidence levels must be a non-empty sequence."
        )

    try:
        supplied_levels = tuple(confidence_levels)
    except TypeError:
        raise ValueError(
            "Confidence levels must be a non-empty sequence."
        ) from None

    if not supplied_levels:
        raise ValueError("At least one confidence level is required.")

    try:
        levels = tuple(sorted(float(value) for value in supplied_levels))
    except (TypeError, ValueError):
        raise ValueError(
            "Confidence levels must be numeric and between 0 and 1."
        ) from None

    if any(not np.isfinite(value) or not 0 < value < 1 for value in levels):
        raise ValueError("Confidence levels must be between 0 and 1.")

    if len(set(levels)) != len(levels):
        raise ValueError("Confidence levels must be unique.")

    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer.")

    return normalized_sizes, levels


def _validate_scale_mapping(scales, label):
    """Validate a mapping of scenario names to positive finite scales."""
    if not isinstance(scales, Mapping) or not scales:
        raise ValueError(
            f"{label} must be a non-empty mapping of names to scales."
        )

    normalized = {}

    for name, scale in scales.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"{label} names must be non-empty strings."
            )

        normalized[name] = _validate_positive_finite(
            scale,
            f"{label} scale for {name}",
        )

    return normalized


def _unit_risk_multipliers(
    distribution,
    confidence,
    student_t_df=STUDENT_T_DF,
):
    """Return exact VaR and ES for a unit-variance innovation."""
    alpha = 1.0 - confidence

    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    if distribution == "gaussian":
        quantile = float(norm.ppf(confidence))
        expected_shortfall = float(norm.pdf(quantile) / alpha)

        return RiskMultipliers(
            var=quantile,
            es=expected_shortfall,
        )

    if distribution == "student_t":
        if (
            isinstance(student_t_df, (bool, np.bool_))
            or not np.isfinite(student_t_df)
            or student_t_df <= 2
        ):
            raise ValueError(
                "Student-t degrees of freedom must exceed 2."
            )

        standard_deviation_scale = np.sqrt(
            (student_t_df - 2.0) / student_t_df
        )

        raw_quantile = float(t.ppf(confidence, df=student_t_df))
        raw_density = float(t.pdf(raw_quantile, df=student_t_df))

        raw_expected_shortfall = (
            (student_t_df + raw_quantile**2)
            / (student_t_df - 1.0)
            * raw_density
            / alpha
        )

        return RiskMultipliers(
            var=standard_deviation_scale * raw_quantile,
            es=standard_deviation_scale * raw_expected_shortfall,
        )

    raise ValueError(
        f"Unsupported innovation distribution: {distribution}"
    )


def _draw_unit_variance_innovations(
    rng,
    size,
    distribution,
    student_t_df=STUDENT_T_DF,
):
    """Draw innovations with zero mean and unit variance."""
    if distribution == "gaussian":
        return rng.standard_normal(size=size)

    if distribution == "student_t":
        if (
            isinstance(student_t_df, (bool, np.bool_))
            or not np.isfinite(student_t_df)
            or student_t_df <= 2
        ):
            raise ValueError(
                "Student-t degrees of freedom must exceed 2."
            )

        draws = rng.standard_t(df=student_t_df, size=size)

        return draws * np.sqrt(
            (student_t_df - 2.0) / student_t_df
        )

    raise ValueError(
        f"Unsupported innovation distribution: {distribution}"
    )


def simulate_iid_returns(
    rng,
    replications,
    sample_size,
    distribution,
    student_t_df=STUDENT_T_DF,
):
    """Generate IID losses and their exact constant conditional risk."""
    innovations = _draw_unit_variance_innovations(
        rng=rng,
        size=(replications, sample_size),
        distribution=distribution,
        student_t_df=student_t_df,
    )

    losses = innovations
    conditional_sigma = np.ones_like(losses)

    return losses, conditional_sigma


def simulate_garch_returns(
    rng,
    replications,
    sample_size,
    distribution,
    student_t_df=STUDENT_T_DF,
    alpha=GARCH_ALPHA,
    beta=GARCH_BETA,
    burn_in=GARCH_BURN_IN,
):
    """Generate GARCH(1,1) losses with a discarded burn-in period.

    The conditional variance at time t uses only information available
    before innovation t is drawn:

        h_t = omega + alpha * loss_(t-1)^2 + beta * h_(t-1)

    The unconditional variance is normalized to one. Initial transient
    observations are discarded using the burn_in parameter.
    """
    if (
        not np.isfinite(alpha)
        or not np.isfinite(beta)
        or alpha < 0
        or beta < 0
        or alpha + beta >= 1
    ):
        raise ValueError(
            "GARCH parameters require alpha >= 0, beta >= 0, "
            "and alpha + beta < 1."
        )

    if (
        isinstance(burn_in, (bool, np.bool_))
        or not isinstance(burn_in, (int, np.integer))
        or burn_in < 0
    ):
        raise ValueError("burn_in must be a non-negative integer.")

    omega = GARCH_UNCONDITIONAL_VARIANCE * (
        1.0 - alpha - beta
    )

    total_size = burn_in + sample_size

    losses = np.empty(
        (replications, sample_size),
        dtype=float,
    )

    conditional_sigma = np.empty_like(losses)

    previous_loss = np.zeros(replications, dtype=float)

    conditional_variance = np.full(
        replications,
        GARCH_UNCONDITIONAL_VARIANCE,
        dtype=float,
    )

    for time_index in range(total_size):
        if time_index > 0:
            conditional_variance = (
                omega
                + alpha * previous_loss**2
                + beta * conditional_variance
            )

        sigma = np.sqrt(conditional_variance)

        innovation = _draw_unit_variance_innovations(
            rng=rng,
            size=replications,
            distribution=distribution,
            student_t_df=student_t_df,
        )

        current_loss = sigma * innovation

        if time_index >= burn_in:
            output_index = time_index - burn_in
            losses[:, output_index] = current_loss
            conditional_sigma[:, output_index] = sigma

        previous_loss = current_loss

    return losses, conditional_sigma


def _calculate_replication_statistics(
    losses,
    conditional_sigma,
    confidence,
    risk_multipliers,
    forecast_scale=1.0,
    var_forecast_scale=None,
    es_forecast_scale=None,
):
    """Calculate Test 2 and exception frequency for every replication.

    forecast_scale is retained for backward compatibility. When the
    separate VaR and ES scales are supplied, they take precedence.
    """
    legacy_scale = _validate_positive_finite(
        forecast_scale,
        "forecast_scale",
    )

    var_scale = (
        legacy_scale
        if var_forecast_scale is None
        else _validate_positive_finite(
            var_forecast_scale,
            "var_forecast_scale",
        )
    )

    es_scale = (
        legacy_scale
        if es_forecast_scale is None
        else _validate_positive_finite(
            es_forecast_scale,
            "es_forecast_scale",
        )
    )

    alpha = 1.0 - confidence

    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("confidence must be between 0 and 1.")

    if (
        not np.isfinite(risk_multipliers.var)
        or not np.isfinite(risk_multipliers.es)
        or risk_multipliers.var <= 0
        or risk_multipliers.es < risk_multipliers.var
    ):
        raise ValueError("Risk multipliers must satisfy ES >= VaR > 0.")

    var_forecast = (
        conditional_sigma
        * risk_multipliers.var
        * var_scale
    )

    es_forecast = (
        conditional_sigma
        * risk_multipliers.es
        * es_scale
    )

    if np.any(es_forecast < var_forecast - 1e-12):
        raise ValueError(
            "Scaled ES forecasts must be greater than or equal "
            "to their corresponding VaR forecasts."
        )

    exceptions = losses > var_forecast

    normalized_tail_loss = np.where(
        exceptions,
        losses / es_forecast,
        0.0,
    )

    n_observations = losses.shape[1]

    z2 = 1.0 - (
        normalized_tail_loss.sum(axis=1)
        / (n_observations * alpha)
    )

    exception_counts = exceptions.sum(axis=1)

    return pd.DataFrame({
        "acerbi_szekely_z2": z2,
        "exception_count": exception_counts,
        "exception_rate": exception_counts / n_observations,
        "observations": n_observations,
    })


def _summarize_replications(
    replication_results,
    scenario,
    sample_size,
    confidence,
    replications,
    var_forecast_scale,
    es_forecast_scale,
):
    """Summarize the Monte Carlo distribution without inference claims."""
    z2 = replication_results["acerbi_szekely_z2"].to_numpy(
        dtype=float
    )

    exception_rates = replication_results["exception_rate"].to_numpy(
        dtype=float
    )

    equal_scales = np.isclose(
        var_forecast_scale,
        es_forecast_scale,
        rtol=0.0,
        atol=1e-12,
    )

    return {
        "scenario": scenario,
        "sample_size": int(sample_size),
        "confidence": float(confidence),
        "alpha": float(1.0 - confidence),
        "replications": int(replications),
        "forecast_scale": (
            float(var_forecast_scale) if equal_scales else np.nan
        ),
        "var_forecast_scale": float(var_forecast_scale),
        "es_forecast_scale": float(es_forecast_scale),
        "mean_z2": float(np.mean(z2)),
        "median_z2": float(np.median(z2)),
        "std_z2": (
            float(np.std(z2, ddof=1))
            if replications > 1 else np.nan
        ),
        "mcse_mean_z2": (
            float(np.std(z2, ddof=1) / np.sqrt(replications))
            if replications > 1 else np.nan
        ),
        "z2_p05": float(np.quantile(z2, 0.05)),
        "z2_p25": float(np.quantile(z2, 0.25)),
        "z2_p75": float(np.quantile(z2, 0.75)),
        "z2_p95": float(np.quantile(z2, 0.95)),
        "negative_z2_fraction": float(np.mean(z2 < 0)),
        "positive_z2_fraction": float(np.mean(z2 > 0)),
        "mean_exception_rate": float(np.mean(exception_rates)),
        "target_exception_rate": float(1.0 - confidence),
        "mean_exception_count": float(
            replication_results["exception_count"].mean()
        ),
        "mean_observations": float(
            replication_results["observations"].mean()
        ),
        "inference_status": "not_performed",
    }


def run_es_test2_simulation_study(
    sample_sizes=DEFAULT_SAMPLE_SIZES,
    replications=DEFAULT_REPLICATIONS,
    confidence_levels=DEFAULT_CONFIDENCE_LEVELS,
    seed=DEFAULT_RANDOM_SEED,
    student_t_df=STUDENT_T_DF,
    misspecification_scales=MISSPECIFICATION_SCALES,
    es_only_misspecification_scales=ES_ONLY_MISSPECIFICATION_SCALES,
):
    """Run null, joint-misspecification, and ES-only scenarios.

    All alternatives reuse the same simulated paths as their corresponding
    correctly specified null scenario. This paired construction isolates
    forecast misspecification from simulation noise.

    Returns one summary row per scenario, sample size, and confidence.
    """
    sample_sizes, confidence_levels = _validate_study_inputs(
        sample_sizes=sample_sizes,
        replications=replications,
        confidence_levels=confidence_levels,
        seed=seed,
    )

    if (
        isinstance(student_t_df, (bool, np.bool_))
        or not np.isfinite(student_t_df)
        or student_t_df <= 2
    ):
        raise ValueError("student_t_df must be finite and exceed 2.")

    joint_scales = _validate_scale_mapping(
        misspecification_scales,
        "misspecification_scales",
    )

    es_only_scales = _validate_scale_mapping(
        es_only_misspecification_scales,
        "es_only_misspecification_scales",
    )

    scenario_definitions = (
        ("iid_gaussian", "iid", "gaussian"),
        ("iid_student_t", "iid", "student_t"),
        ("garch_gaussian", "garch", "gaussian"),
        ("garch_student_t", "garch", "student_t"),
    )

    seed_sequence = np.random.SeedSequence(int(seed))
    child_seeds = iter(
        seed_sequence.spawn(
            len(sample_sizes) * len(scenario_definitions)
        )
    )

    summary_rows = []

    for sample_size in sample_sizes:
        for scenario, process, distribution in scenario_definitions:
            rng = np.random.default_rng(next(child_seeds))

            if process == "iid":
                losses, conditional_sigma = simulate_iid_returns(
                    rng=rng,
                    replications=replications,
                    sample_size=sample_size,
                    distribution=distribution,
                    student_t_df=student_t_df,
                )
            else:
                losses, conditional_sigma = simulate_garch_returns(
                    rng=rng,
                    replications=replications,
                    sample_size=sample_size,
                    distribution=distribution,
                    student_t_df=student_t_df,
                    burn_in=GARCH_BURN_IN,
                )

            for confidence in confidence_levels:
                multipliers = _unit_risk_multipliers(
                    distribution=distribution,
                    confidence=confidence,
                    student_t_df=student_t_df,
                )

                null_results = _calculate_replication_statistics(
                    losses=losses,
                    conditional_sigma=conditional_sigma,
                    confidence=confidence,
                    risk_multipliers=multipliers,
                )

                summary_rows.append(
                    _summarize_replications(
                        replication_results=null_results,
                        scenario=scenario,
                        sample_size=sample_size,
                        confidence=confidence,
                        replications=replications,
                        var_forecast_scale=1.0,
                        es_forecast_scale=1.0,
                    )
                )

                # Joint VaR and ES misspecification.
                for alternative_name, scale in joint_scales.items():
                    alternative_results = (
                        _calculate_replication_statistics(
                            losses=losses,
                            conditional_sigma=conditional_sigma,
                            confidence=confidence,
                            risk_multipliers=multipliers,
                            var_forecast_scale=scale,
                            es_forecast_scale=scale,
                        )
                    )

                    summary_rows.append(
                        _summarize_replications(
                            replication_results=alternative_results,
                            scenario=f"{scenario}_{alternative_name}",
                            sample_size=sample_size,
                            confidence=confidence,
                            replications=replications,
                            var_forecast_scale=scale,
                            es_forecast_scale=scale,
                        )
                    )

                # ES-only misspecification: VaR remains unchanged.
                for alternative_name, es_scale in es_only_scales.items():
                    alternative_results = (
                        _calculate_replication_statistics(
                            losses=losses,
                            conditional_sigma=conditional_sigma,
                            confidence=confidence,
                            risk_multipliers=multipliers,
                            var_forecast_scale=1.0,
                            es_forecast_scale=es_scale,
                        )
                    )

                    summary_rows.append(
                        _summarize_replications(
                            replication_results=alternative_results,
                            scenario=f"{scenario}_{alternative_name}",
                            sample_size=sample_size,
                            confidence=confidence,
                            replications=replications,
                            var_forecast_scale=1.0,
                            es_forecast_scale=es_scale,
                        )
                    )

    return pd.DataFrame(summary_rows).sort_values(
        ["sample_size", "confidence", "scenario"]
    ).reset_index(drop=True)


def validate_test2_implementation(
    sample_size=1266,
    confidence=0.95,
    seed=DEFAULT_RANDOM_SEED,
):
    """Cross-check the vectorized simulation statistic against production."""
    rng = np.random.default_rng(seed)

    losses, conditional_sigma = simulate_iid_returns(
        rng=rng,
        replications=1,
        sample_size=sample_size,
        distribution="gaussian",
    )

    multipliers = _unit_risk_multipliers(
        distribution="gaussian",
        confidence=confidence,
    )

    var_forecasts = conditional_sigma[0] * multipliers.var
    es_forecasts = conditional_sigma[0] * multipliers.es

    direct = calculate_acerbi_szekely_test2(
        realized_returns=-losses[0],
        var_forecasts=var_forecasts,
        es_forecasts=es_forecasts,
        confidence=confidence,
    )

    vectorized = _calculate_replication_statistics(
        losses=losses,
        conditional_sigma=conditional_sigma,
        confidence=confidence,
        risk_multipliers=multipliers,
    ).iloc[0]

    if not np.isclose(
        direct["acerbi_szekely_z2"],
        vectorized["acerbi_szekely_z2"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError(
            "Vectorized simulation statistic does not match "
            "calculate_acerbi_szekely_test2."
        )

    if direct["exceptions"] != int(vectorized["exception_count"]):
        raise AssertionError(
            "Vectorized simulation exception count does not match "
            "the production Test 2 implementation."
        )

    return {
        "statistic_match": True,
        "exception_count_match": True,
        "acerbi_szekely_z2": direct["acerbi_szekely_z2"],
        "exceptions": direct["exceptions"],
        "observations": direct["n_observations"],
    }
