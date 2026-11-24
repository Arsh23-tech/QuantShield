
"""Run estimated rolling-forecast Monte Carlo validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Ensure the repository root is importable when this file is run
# directly as: python scripts/validate_es_estimated_forecasts.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.risk.es_estimated_forecast_simulation import (
    run_estimated_forecast_simulation,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "es_backtesting"
    / "estimated_forecast_validation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate estimated VaR/ES forecasts using "
            "IID and GARCH Monte Carlo data-generating processes."
        )
    )

    parser.add_argument(
        "--replications",
        type=int,
        default=10,
        help="Number of independent paths per DGP (default: 10).",
    )

    parser.add_argument(
        "--observations",
        type=int,
        default=2518,
        help=(
            "Post-burn-in observations per path "
            "(default: 2518)."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261119,
        help="Master random seed (default: 20261119).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for simulation outputs.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output files to be replaced.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = run_estimated_forecast_simulation(
        replications=args.replications,
        observations=args.observations,
        seed=args.seed,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    print()
    print("DESCRIPTIVE MONTE CARLO SUMMARY")
    print(
        results["summary"].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


if __name__ == "__main__":
    main()
