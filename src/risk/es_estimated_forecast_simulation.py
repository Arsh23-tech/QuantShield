
"""Estimated-forecast Monte Carlo validation for VaR and ES models.

This experiment evaluates the actual QuantShield forecasting functions
under known IID and GARCH data-generating processes.

The evaluation window is the intersection of valid historical, EWMA,
and EWMA-FHS forecasts. All forecasts are one-step-ahead and use the
production model implementations.

No p-values, bootstrap inference, or calibration-rejection decisions
are produced. Results are descriptive Monte Carlo diagnostics.

Return convention:
    Production functions receive simple returns as decimal values.
    VaR and ES are reported as positive loss amounts.

DGP convention:
    Innovations have unit variance. Simulated returns are scaled by
    RETURN_SCALE so that IID unconditional daily volatility is 1%.
    GARCH unconditional variance is one in standardized innovation
    space, or RETURN_SCALE**2 in return space.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from src.risk.var_backtesting import backtest_historical_var
from src.risk.ewma_var import calculate_ewma_var
from src.risk.ewma_fhs_var import calculate_ewma_fhs_var
from src.risk.es_backtest_diagnostics import (
    calculate_acerbi_szekely_test2,
)


DEFAULT_CONFIDENCE_LEVELS = (0.95, 0.99)

DGP_NAMES = (
    "iid_gaussian",
    "iid_student_t7",
    "garch_gaussian",
    "garch_student_t7",
)

MODEL_NAMES = (
    "historical",
    "ewma",
    "ewma_fhs",
)

RETURN_SCALE = 0.01
STUDENT_T_DF = 7

GARCH_ALPHA = 0.05
GARCH_BETA = 0.90
GARCH_UNCONDITIONAL_VARIANCE = 1.0
GARCH_BURN_IN = 500

HISTORICAL_LOOKBACK = 252
EWMA_DECAY_FACTOR = 0.94
EWMA_MIN_PERIODS = 252
FHS_RESIDUAL_LOOKBACK = 1000
FHS_MIN_RESIDUALS = 1000

TICKER = "SIM"


def _validate_simulation_inputs(
    replications: int,
    observations: int,
    seed: int,
) -> None:
    """Validate top-level simulation arguments."""

    if (
        not isinstance(replications, int)
        or isinstance(replications, bool)
        or replications < 2
    ):
        raise ValueError("replications must be an integer >= 2.")

    minimum_observations = (
        EWMA_MIN_PERIODS
        + FHS_MIN_RESIDUALS
        + 1
    )

    if (
        not isinstance(observations, int)
        or isinstance(observations, bool)
        or observations < minimum_observations
    ):
        raise ValueError(
            f"observations must be an integer >= "
            f"{minimum_observations} to produce a common "
            "historical, EWMA, and EWMA-FHS evaluation window."
        )

    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer.")


def _standardized_student_t(
    rng: np.random.Generator,
    size: int,
    degrees_of_freedom: int = STUDENT_T_DF,
) -> np.ndarray:
    """Generate Student-t innovations with mean zero and variance one."""
    if degrees_of_freedom <= 2:
        raise ValueError("Student-t degrees of freedom must exceed 2.")

    raw = rng.standard_t(degrees_of_freedom, size=size)
    return raw / np.sqrt(
        degrees_of_freedom / (degrees_of_freedom - 2.0)
    )


def generate_dgp_path(
    dgp_name: str,
    observations: int,
    rng: np.random.Generator,
    return_scale: float = RETURN_SCALE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one complete return path and its known conditional risk.

    Returns
    -------
    returns:
        Columns: date, ticker, simple_return.

    truth:
        Columns: date, true_conditional_volatility,
        true_var_95, true_es_95, true_var_99, true_es_99.

    The conditional risk on date t is calculated from the variance
    known immediately before return t is generated.
    """
    if dgp_name not in DGP_NAMES:
        raise ValueError(
            f"Unsupported DGP {dgp_name!r}. "
            f"Choose from {DGP_NAMES}."
        )

    if (
        not isinstance(observations, int)
        or isinstance(observations, bool)
        or observations <= 0
    ):
        raise ValueError("observations must be a positive integer.")

    if not np.isfinite(return_scale) or return_scale <= 0:
        raise ValueError("return_scale must be finite and positive.")

    is_garch = dgp_name.startswith("garch_")
    is_student_t = dgp_name.endswith("student_t7")

    burn_in = GARCH_BURN_IN if is_garch else 0
    total_observations = observations + burn_in

    if is_student_t:
        innovations = _standardized_student_t(
            rng,
            total_observations,
        )
    else:
        innovations = rng.standard_normal(total_observations)

    if is_garch:
        omega = (
            1.0 - GARCH_ALPHA - GARCH_BETA
        ) * GARCH_UNCONDITIONAL_VARIANCE

        conditional_variance = np.empty(
            total_observations,
            dtype=float,
        )
        standardized_returns = np.empty(
            total_observations,
            dtype=float,
        )

        conditional_variance[0] = (
            GARCH_UNCONDITIONAL_VARIANCE
        )

        for index in range(total_observations):
            standardized_returns[index] = (
                np.sqrt(conditional_variance[index])
                * innovations[index]
            )

            if index + 1 < total_observations:
                conditional_variance[index + 1] = (
                    omega
                    + GARCH_ALPHA
                    * standardized_returns[index] ** 2
                    + GARCH_BETA
                    * conditional_variance[index]
                )

        standardized_returns = standardized_returns[burn_in:]
        conditional_variance = conditional_variance[burn_in:]

    else:
        standardized_returns = innovations
        conditional_variance = np.ones(
            total_observations,
            dtype=float,
        )

    simulated_returns = (
        return_scale * standardized_returns
    )

    true_volatility = (
        return_scale * np.sqrt(conditional_variance)
    )

    dates = pd.bdate_range(
        start="2000-01-03",
        periods=observations,
        name="date",
    )

    returns = pd.DataFrame({
        "date": dates,
        "ticker": TICKER,
        "simple_return": simulated_returns,
    })

    truth = pd.DataFrame({
        "date": dates,
        "true_conditional_volatility": true_volatility,
    })

    for confidence in DEFAULT_CONFIDENCE_LEVELS:
        alpha = 1.0 - confidence

        if is_student_t:
            quantile = t.ppf(
                alpha,
                df=STUDENT_T_DF,
            )

            standardized_scale = np.sqrt(
                (STUDENT_T_DF - 2.0) / STUDENT_T_DF
            )

            conditional_var = (
                -standardized_scale * quantile
                * true_volatility
            )

            # For a standard Student-t variable X:
            # E[X | X <= q] =
            # -((nu + q**2)/(nu - 1)) * pdf(q)/cdf(q)
            lower_tail_mean_magnitude = (
                (STUDENT_T_DF + quantile**2)
                / (STUDENT_T_DF - 1.0)
                * t.pdf(quantile, df=STUDENT_T_DF)
                / alpha
            )

            conditional_es = (
                standardized_scale
                * lower_tail_mean_magnitude
                * true_volatility
            )

        else:
            quantile = norm.ppf(alpha)

            conditional_var = (
                -quantile * true_volatility
            )

            conditional_es = (
                norm.pdf(quantile)
                / alpha
                * true_volatility
            )

        suffix = int(round(confidence * 100))
        truth[f"true_var_{suffix}"] = conditional_var
        truth[f"true_es_{suffix}"] = conditional_es

    if not np.isfinite(simulated_returns).all():
        raise ValueError(
            f"Non-finite simulated returns generated for {dgp_name}."
        )

    if (simulated_returns < -1.0).any():
        raise ValueError(
            "A simulated return breached the production input "
            "floor of -100%. Consider reviewing the DGP scale "
            "or the generated path."
        )

    return returns, truth


def _run_production_forecasters(
    returns: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Run all three production forecasting implementations."""
    historical = backtest_historical_var(
        returns=returns,
        return_column="simple_return",
        lookback=HISTORICAL_LOOKBACK,
        confidence_levels=DEFAULT_CONFIDENCE_LEVELS,
        calendar_name=None,
    )

    ewma = calculate_ewma_var(
        returns=returns,
        return_column="simple_return",
        decay_factor=EWMA_DECAY_FACTOR,
        min_periods=EWMA_MIN_PERIODS,
        confidence_levels=DEFAULT_CONFIDENCE_LEVELS,
    )

    ewma_fhs = calculate_ewma_fhs_var(
        returns=returns,
        return_column="simple_return",
        decay_factor=EWMA_DECAY_FACTOR,
        min_periods=EWMA_MIN_PERIODS,
        residual_lookback=FHS_RESIDUAL_LOOKBACK,
        min_residuals=FHS_MIN_RESIDUALS,
        confidence_levels=DEFAULT_CONFIDENCE_LEVELS,
    )

    return {
        "historical": historical,
        "ewma": ewma,
        "ewma_fhs": ewma_fhs,
    }


def _validate_forecast_frame(
    forecasts: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """Validate the forecast columns needed by this experiment."""
    required = {
        "date",
        "realized_return",
        "var_95",
        "es_95",
        "var_99",
        "es_99",
    }

    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(
            f"{model_name} forecasts are missing columns: "
            f"{sorted(missing)}"
        )

    frame = forecasts.copy()
    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
    ).dt.normalize()

    if frame["date"].duplicated().any():
        raise ValueError(
            f"{model_name} forecasts contain duplicate dates."
        )

    for column in (
        "realized_return",
        "var_95",
        "es_95",
        "var_99",
        "es_99",
    ):
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

    return frame.sort_values("date").reset_index(drop=True)


def _forecast_missing_counts(
    forecasts: pd.DataFrame,
) -> dict[str, int]:
    """Count missing or non-finite risk forecast values."""
    columns = (
        "var_95",
        "es_95",
        "var_99",
        "es_99",
    )

    counts = {}
    for column in columns:
        values = forecasts[column].to_numpy(dtype=float)
        counts[f"missing_or_nonfinite_{column}"] = int(
            (~np.isfinite(values)).sum()
        )

    return counts


def _build_common_evaluation(
    returns: pd.DataFrame,
    truth: pd.DataFrame,
    model_forecasts: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the exact shared evaluation window across all models."""
    normalized = {
        model: _validate_forecast_frame(frame, model)
        for model, frame in model_forecasts.items()
    }

    if set(normalized) != set(MODEL_NAMES):
        raise ValueError(
            f"Expected models {MODEL_NAMES}, received "
            f"{tuple(normalized)}."
        )

    indexed = {
        model: frame.set_index("date", verify_integrity=True)
        for model, frame in normalized.items()
    }

    common_dates = set(truth["date"])

    for frame in indexed.values():
        common_dates &= set(frame.index)

    required_risk_columns = [
        "var_95",
        "es_95",
        "var_99",
        "es_99",
    ]

    for frame in indexed.values():
        valid = np.ones(len(frame), dtype=bool)

        for column in required_risk_columns:
            valid &= np.isfinite(
                frame[column].to_numpy(dtype=float)
            )

        valid &= np.isfinite(
            frame["realized_return"].to_numpy(dtype=float)
        )

        common_dates &= set(frame.index[valid])

    common_dates = sorted(common_dates)

    if not common_dates:
        raise ValueError(
            "The three production models have no common valid "
            "evaluation dates."
        )

    truth_indexed = truth.set_index(
        "date",
        verify_integrity=True,
    )

    returns_indexed = returns.set_index(
        "date",
        verify_integrity=True,
    )

    evaluation = truth_indexed.loc[common_dates].copy()
    evaluation["realized_return"] = (
        returns_indexed.loc[common_dates, "simple_return"]
    )

    for model in MODEL_NAMES:
        frame = indexed[model].loc[common_dates]

        model_realized = frame["realized_return"].to_numpy(
            dtype=float
        )
        input_realized = evaluation["realized_return"].to_numpy(
            dtype=float
        )

        if not np.allclose(
            model_realized,
            input_realized,
            rtol=0.0,
            atol=1e-14,
        ):
            raise ValueError(
                f"{model} realized returns do not match the "
                "original simulated return path."
            )

        for column in required_risk_columns:
            evaluation[f"{model}_{column}"] = (
                frame[column].to_numpy(dtype=float)
            )

        if "ewma_volatility" in frame.columns:
            evaluation[f"{model}_volatility"] = (
                pd.to_numeric(
                    frame["ewma_volatility"],
                    errors="coerce",
                ).to_numpy(dtype=float)
            )

    counts = {
        "common_evaluation_observations": len(common_dates),
        "historical_forecast_rows": len(normalized["historical"]),
        "ewma_forecast_rows": len(normalized["ewma"]),
        "ewma_fhs_forecast_rows": len(normalized["ewma_fhs"]),
    }

    for model, frame in normalized.items():
        counts.update({
            f"{model}_{key}": value
            for key, value in _forecast_missing_counts(frame).items()
        })

    return evaluation.reset_index(), counts


def _calculate_replication_metrics(
    evaluation: pd.DataFrame,
    dgp_name: str,
    replication: int,
    seed: int,
    forecast_counts: dict[str, int],
) -> tuple[list[dict], list[dict]]:
    """Calculate Test 2 and forecast-error metrics for one path."""
    metric_rows = []
    detail_rows = []

    returns = evaluation["realized_return"].to_numpy(dtype=float)

    for model in MODEL_NAMES:
        for confidence in DEFAULT_CONFIDENCE_LEVELS:
            suffix = int(round(confidence * 100))

            var_values = evaluation[
                f"{model}_var_{suffix}"
            ].to_numpy(dtype=float)

            es_values = evaluation[
                f"{model}_es_{suffix}"
            ].to_numpy(dtype=float)

            true_var = evaluation[
                f"true_var_{suffix}"
            ].to_numpy(dtype=float)

            true_es = evaluation[
                f"true_es_{suffix}"
            ].to_numpy(dtype=float)

            statistic = calculate_acerbi_szekely_test2(
                realized_returns=returns,
                var_forecasts=var_values,
                es_forecasts=es_values,
                confidence=confidence,
            )

            var_errors = var_values - true_var
            es_errors = es_values - true_es

            row = {
                "dgp": dgp_name,
                "replication": replication,
                "seed": seed,
                "model": model,
                "confidence": confidence,
                "alpha": 1.0 - confidence,
                "n_observations": len(evaluation),
                "forecast_rows": forecast_counts[
                    f"{model}_forecast_rows"
                ],
                "common_evaluation_observations": len(evaluation),
                "acerbi_szekely_z2": (
                    statistic["acerbi_szekely_z2"]
                ),
                "exceptions": statistic["exceptions"],
                "exception_rate": statistic["exception_rate"],
                "expected_exception_rate": (
                    statistic["expected_exception_rate"]
                ),
                "tail_loss_ratio": statistic["tail_loss_ratio"],
                "mean_exception_severity_es_ratio": (
                    statistic["mean_exception_severity_es_ratio"]
                ),
                "var_bias": float(np.mean(var_errors)),
                "var_mae": float(np.mean(np.abs(var_errors))),
                "var_rmse": float(
                    np.sqrt(np.mean(var_errors**2))
                ),
                "es_bias": float(np.mean(es_errors)),
                "es_mae": float(np.mean(np.abs(es_errors))),
                "es_rmse": float(
                    np.sqrt(np.mean(es_errors**2))
                ),
            }

            volatility_column = f"{model}_volatility"
            if volatility_column in evaluation.columns:
                volatility = evaluation[
                    volatility_column
                ].to_numpy(dtype=float)

                valid_volatility = np.isfinite(volatility)
                if valid_volatility.any():
                    volatility_errors = (
                        volatility[valid_volatility]
                        - evaluation[
                            "true_conditional_volatility"
                        ].to_numpy(dtype=float)[valid_volatility]
                    )

                    row.update({
                        "volatility_bias": float(
                            np.mean(volatility_errors)
                        ),
                        "volatility_mae": float(
                            np.mean(np.abs(volatility_errors))
                        ),
                        "volatility_rmse": float(
                            np.sqrt(
                                np.mean(volatility_errors**2)
                            )
                        ),
                        "volatility_evaluation_observations": int(
                            valid_volatility.sum()
                        ),
                    })
                else:
                    row.update({
                        "volatility_bias": np.nan,
                        "volatility_mae": np.nan,
                        "volatility_rmse": np.nan,
                        "volatility_evaluation_observations": 0,
                    })

            metric_rows.append(row)

        detail = evaluation[
            [
                "date",
                "realized_return",
                "true_conditional_volatility",
                "true_var_95",
                "true_es_95",
                "true_var_99",
                "true_es_99",
            ]
        ].copy()

        detail.insert(0, "dgp", dgp_name)
        detail.insert(1, "replication", replication)
        detail.insert(2, "seed", seed)
        detail.insert(3, "model", model)

        for column in (
            "var_95",
            "es_95",
            "var_99",
            "es_99",
        ):
            detail[column] = evaluation[
                f"{model}_{column}"
            ].to_numpy(dtype=float)

        if f"{model}_volatility" in evaluation.columns:
            detail["estimated_volatility"] = evaluation[
                f"{model}_volatility"
            ].to_numpy(dtype=float)

        detail_rows.extend(detail.to_dict("records"))

    return metric_rows, detail_rows


def _summarize_replications(
    replication_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate replication metrics and calculate Monte Carlo SEs."""
    metric_columns = [
        "acerbi_szekely_z2",
        "exceptions",
        "exception_rate",
        "var_bias",
        "var_mae",
        "var_rmse",
        "es_bias",
        "es_mae",
        "es_rmse",
        "volatility_bias",
        "volatility_mae",
        "volatility_rmse",
    ]

    rows = []

    grouping = ["dgp", "model", "confidence"]

    for keys, group in replication_metrics.groupby(
        grouping,
        sort=True,
        dropna=False,
    ):
        row = dict(zip(grouping, keys))
        row["replications"] = int(len(group))

        for metric in metric_columns:
            if metric not in group.columns:
                continue

            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)

            row[f"mean_{metric}"] = (
                float(np.mean(values))
                if len(values)
                else np.nan
            )

            row[f"mcse_{metric}"] = (
                float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) >= 2
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        grouping
    ).reset_index(drop=True)


def _make_configuration(
    replications: int,
    observations: int,
    seed: int,
) -> dict:
    """Return a reproducibility record for this experiment."""
    return {
        "experiment": "estimated_forecast_validation",
        "replications": replications,
        "post_burn_in_observations_per_replication": observations,
        "seed": seed,
        "ticker": TICKER,
        "return_scale": RETURN_SCALE,
        "iid_unconditional_variance": 1.0,
        "garch": {
            "alpha": GARCH_ALPHA,
            "beta": GARCH_BETA,
            "omega": (
                1.0 - GARCH_ALPHA - GARCH_BETA
            ) * GARCH_UNCONDITIONAL_VARIANCE,
            "unconditional_variance_standardized": (
                GARCH_UNCONDITIONAL_VARIANCE
            ),
            "burn_in": GARCH_BURN_IN,
        },
        "student_t_degrees_of_freedom": STUDENT_T_DF,
        "confidence_levels": list(DEFAULT_CONFIDENCE_LEVELS),
        "models": {
            "historical": {
                "lookback": HISTORICAL_LOOKBACK,
            },
            "ewma": {
                "decay_factor": EWMA_DECAY_FACTOR,
                "min_periods": EWMA_MIN_PERIODS,
            },
            "ewma_fhs": {
                "decay_factor": EWMA_DECAY_FACTOR,
                "min_periods": EWMA_MIN_PERIODS,
                "residual_lookback": FHS_RESIDUAL_LOOKBACK,
                "min_residuals": FHS_MIN_RESIDUALS,
            },
        },
        "dgp_names": list(DGP_NAMES),
        "evaluation_window": (
            "Intersection of valid historical, EWMA, and EWMA-FHS "
            "forecasts with finite realized returns and finite VaR/ES "
            "at both confidence levels."
        ),
        "expected_common_observations_complete_path": (
            observations
            - EWMA_MIN_PERIODS
            - FHS_MIN_RESIDUALS
        ),
        "test2": {
            "formula": (
                "Z2 = 1 - sum(loss_t * I(loss_t > VaR_t) / ES_t) "
                "/ (T * alpha)"
            ),
            "loss_convention": "loss = -realized_return",
            "p_values_calculated": False,
            "calibration_rejections_claimed": False,
        },
        "inference": (
            "Descriptive Monte Carlo summaries only. MCSE is the "
            "sample standard deviation across replication-level "
            "metrics divided by the square root of the number of "
            "replications."
        ),
        "random_stream_strategy": (
            "NumPy SeedSequence with seed, DGP index, and replication "
            "index; streams are independent of loop execution order."
        ),
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def run_estimated_forecast_simulation(
    replications: int = 10,
    observations: int = 2518,
    seed: int = 20261119,
    output_dir: str | Path = (
        "results/es_backtesting/estimated_forecast_validation"
    ),
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run the estimated rolling-forecast Monte Carlo experiment.

    Output files:
        replication_metrics.csv
        summary.csv
        evaluation_forecasts.csv
        configuration.json

    Existing output files are not overwritten unless overwrite=True.
    """
    _validate_simulation_inputs(
        replications=replications,
        observations=observations,
        seed=seed,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_files = [
        output_path / "replication_metrics.csv",
        output_path / "summary.csv",
        output_path / "evaluation_forecasts.csv",
        output_path / "configuration.json",
    ]

    existing = [path for path in output_files if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Simulation output already exists. Use --overwrite "
            "to replace: "
            + ", ".join(str(path) for path in existing)
        )

    all_metric_rows = []
    all_detail_rows = []
    run_records = []

    started = time.perf_counter()

    for dgp_index, dgp_name in enumerate(DGP_NAMES):
        for replication_index in range(replications):
            replication_number = replication_index + 1

            child_seed = np.random.SeedSequence([
                seed,
                dgp_index,
                replication_index,
            ])

            replication_seed = int(
                child_seed.generate_state(1, dtype=np.uint32)[0]
            )

            rng = np.random.default_rng(replication_seed)

            replication_started = time.perf_counter()

            returns, truth = generate_dgp_path(
                dgp_name=dgp_name,
                observations=observations,
                rng=rng,
            )

            forecasts = _run_production_forecasters(returns)

            evaluation, forecast_counts = (
                _build_common_evaluation(
                    returns=returns,
                    truth=truth,
                    model_forecasts=forecasts,
                )
            )

            metric_rows, detail_rows = (
                _calculate_replication_metrics(
                    evaluation=evaluation,
                    dgp_name=dgp_name,
                    replication=replication_number,
                    seed=replication_seed,
                    forecast_counts=forecast_counts,
                )
            )

            all_metric_rows.extend(metric_rows)
            all_detail_rows.extend(detail_rows)

            run_records.append({
                "dgp": dgp_name,
                "replication": replication_number,
                "seed": replication_seed,
                **forecast_counts,
                "elapsed_seconds": (
                    time.perf_counter() - replication_started
                ),
            })

            print(
                f"[{dgp_name}] "
                f"replication {replication_number}/{replications}: "
                f"{len(evaluation)} common observations, "
                f"{run_records[-1]['elapsed_seconds']:.2f}s"
            )

    replication_metrics = pd.DataFrame(all_metric_rows)
    evaluation_forecasts = pd.DataFrame(all_detail_rows)

    summary = _summarize_replications(replication_metrics)

    configuration = _make_configuration(
        replications=replications,
        observations=observations,
        seed=seed,
    )

    configuration["run"] = {
        "total_replication_paths": len(run_records),
        "total_elapsed_seconds": (
            time.perf_counter() - started
        ),
        "replication_records": run_records,
    }

    replication_metrics.to_csv(
        output_path / "replication_metrics.csv",
        index=False,
    )

    summary.to_csv(
        output_path / "summary.csv",
        index=False,
    )

    evaluation_forecasts.to_csv(
        output_path / "evaluation_forecasts.csv",
        index=False,
    )

    with (output_path / "configuration.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=2,
            sort_keys=True,
        )

    print()
    print("ESTIMATED-FORECAST SIMULATION COMPLETE")
    print(f"Output directory: {output_path}")
    print(f"Replication paths: {len(run_records)}")
    print(f"Metric rows: {len(replication_metrics):,}")
    print(f"Evaluation forecast rows: {len(evaluation_forecasts):,}")
    print(
        f"Total elapsed: "
        f"{configuration['run']['total_elapsed_seconds']:.2f}s"
    )

    return {
        "replication_metrics": replication_metrics,
        "summary": summary,
        "evaluation_forecasts": evaluation_forecasts,
    }
