import numpy as np


def empirical_lower_tail_mean(
    observations,
    tail_probability: float,
) -> float:
    """
    Calculate the mean of the lower tail by integrating the
    piecewise-linear empirical quantile function.

    The quantile convention matches np.quantile(..., method="linear").
    Fractional tail intervals are integrated exactly.
    """
    values = np.asarray(observations, dtype=float)

    if values.ndim != 1 or len(values) == 0:
        raise ValueError("observations must be a non-empty 1D array.")

    if not np.isfinite(values).all():
        raise ValueError("observations must contain only finite values.")

    if (
        not np.isfinite(tail_probability)
        or not 0 < tail_probability < 1
    ):
        raise ValueError("tail_probability must be between 0 and 1.")

    values = np.sort(values)

    if len(values) == 1:
        return float(values[0])

    probabilities = np.linspace(0.0, 1.0, len(values))

    # Include every empirical quantile knot inside the integration
    # interval, plus the exact endpoint of the requested tail.
    interior = probabilities[
        (probabilities > 0)
        & (probabilities < tail_probability)
    ]

    integration_points = np.concatenate(
        ([0.0], interior, [tail_probability])
    )

    quantile_values = np.interp(
        integration_points,
        probabilities,
        values,
    )

    area = np.trapezoid(
        quantile_values,
        integration_points,
    )

    return float(area / tail_probability)