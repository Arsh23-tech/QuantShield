
"""Run independent joint VaR-ES scoring on aligned model forecasts."""

from pathlib import Path
import json

import pandas as pd

from src.risk.var_es_scoring import (
    compare_paired_scores,
    score_forecasts,
    summarize_scores,
)


INPUT_PATH = Path("results/var_model_comparison/aligned_forecasts.csv")
OUTPUT_DIR = Path("results/var_es_scoring")

CONFIDENCE_LEVELS = (0.95, 0.99)

BOOTSTRAP_BLOCK_LENGTH = 20
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 42


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Aligned forecasts not found: {INPUT_PATH}. "
            "Run python -m scripts.compare_var_models first."
        )

    forecasts = pd.read_csv(INPUT_PATH)

    scored = score_forecasts(
        forecasts,
        confidence_levels=CONFIDENCE_LEVELS,
    )

    ticker_summary, model_summary = summarize_scores(scored)

    comparisons = compare_paired_scores(
        scored,
        block_length=BOOTSTRAP_BLOCK_LENGTH,
        n_bootstrap=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scored.to_csv(
        OUTPUT_DIR / "observation_scores.csv",
        index=False,
    )
    ticker_summary.to_csv(
        OUTPUT_DIR / "ticker_score_summary.csv",
        index=False,
    )
    model_summary.to_csv(
        OUTPUT_DIR / "model_score_summary.csv",
        index=False,
    )
    comparisons.to_csv(
        OUTPUT_DIR / "paired_score_comparisons.csv",
        index=False,
    )

    configuration = {
        "input_path": str(INPUT_PATH),
        "score": "FZ0 joint VaR-ES score",
        "risk_convention": "positive loss; realized_loss = -realized_return",
        "interpretation": "lower score is better",
        "confidence_levels": list(CONFIDENCE_LEVELS),
        "aggregation": (
            "Mean score within each ticker, followed by equal weighting "
            "of ticker means. Paired comparisons use the same ticker/date "
            "observations for both models."
        ),
        "bootstrap": {
            "method": "non-circular moving-block bootstrap",
            "sampling_unit": "common dates across all tickers",
            "block_length": BOOTSTRAP_BLOCK_LENGTH,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "95% percentile interval",
        },
        "limitations": [
            "Joint scoring evaluates relative predictive performance; "
            "it does not by itself establish calibration.",
            "Bootstrap intervals depend on the chosen block length and "
            "the historical evaluation sample.",
            "Raw score levels are scale-dependent and should not be used "
            "to rank different tickers against each other.",
            "Paired comparisons are meaningful only on common "
            "ticker/date observations."
        ],
        "observations": int(len(scored)),
        "evaluable_observations": int(scored["evaluable"].sum()),
        "unevaluable_observations": int((~scored["evaluable"]).sum()),
        "models": sorted(scored["model"].unique().tolist()),
        "tickers": sorted(scored["ticker"].unique().tolist()),
    }

    with (OUTPUT_DIR / "configuration.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(configuration, file, indent=2)

    print("\nJOINT VaR-ES SCORING")
    print(f"Input rows: {len(forecasts):,}")
    print(f"Scored rows: {len(scored):,}")
    print(
        f"Evaluable observations: "
        f"{configuration['evaluable_observations']:,}"
    )
    print(
        f"Unevaluable observations: "
        f"{configuration['unevaluable_observations']:,}"
    )

    print("\nEQUAL-TICKER-WEIGHTED MODEL SUMMARY")
    print(
        model_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nPAIRED MODEL COMPARISONS")
    print(
        comparisons.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
