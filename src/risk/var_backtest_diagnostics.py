
import math

import numpy as np
import pandas as pd
from scipy.stats import chi2


def _validate_exceptions(exceptions, expected_exception_rate):
    """Validate a chronological series of binary VaR exceptions."""
    if not 0 < expected_exception_rate < 1:
        raise ValueError("expected_exception_rate must be between 0 and 1.")

    values = pd.Series(exceptions)

    if values.empty:
        raise ValueError("At least one exception observation is required.")

    if values.isna().any():
        raise ValueError("Exception observations must not contain missing values.")

    if not values.isin([0, 1, False, True]).all():
        raise ValueError("Exception observations must be binary (0/1 or bool).")

    return values.astype(int).to_numpy()


def _xlogp(x, p):
    """Return x * log(p), defining 0 * log(0) as zero."""
    if x == 0:
        return 0.0
    if p <= 0:
        return -math.inf
    return x * math.log(p)


def _bernoulli_log_likelihood(successes, failures, probability):
    """Log likelihood for binary observations with a common probability."""
    total = successes + failures
    if total == 0:
        return 0.0

    return (
        _xlogp(successes, probability)
        + _xlogp(failures, 1.0 - probability)
    )


def kupiec_coverage_test(exceptions, expected_exception_rate):
    """
    Kupiec unconditional coverage likelihood-ratio test.

    H0: the exception probability equals the expected rate.
    Returns the observed exception rate, LR statistic, p-value,
    and a 5% significance-level decision.
    """
    values = _validate_exceptions(exceptions, expected_exception_rate)
    n = len(values)
    x = int(values.sum())
    observed_rate = x / n

    null_ll = _bernoulli_log_likelihood(
        x, n - x, expected_exception_rate
    )
    alternative_ll = _bernoulli_log_likelihood(
        x, n - x, observed_rate
    )

    lr_stat = max(0.0, 2.0 * (alternative_ll - null_ll))
    p_value = float(chi2.sf(lr_stat, df=1))

    return {
        "observations": n,
        "exceptions": x,
        "expected_exception_rate": expected_exception_rate,
        "observed_exception_rate": observed_rate,
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "reject_at_5pct": p_value < 0.05,
    }


def christoffersen_independence_test(exceptions):
    """
    Christoffersen first-order exception-independence test.

    H0: the probability of an exception does not depend on whether
    the previous observation was an exception.

    Input must be in chronological order.
    """
    values = _validate_exceptions(exceptions, 0.5)

    if len(values) < 2:
        raise ValueError("At least two observations are required.")

    previous = values[:-1]
    current = values[1:]

    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    n0 = n00 + n01
    n1 = n10 + n11
    total_transitions = n0 + n1

    p = (n01 + n11) / total_transitions
    p01 = n01 / n0 if n0 else 0.0
    p11 = n11 / n1 if n1 else 0.0

    null_ll = _bernoulli_log_likelihood(
        n01 + n11, n00 + n10, p
    )
    alternative_ll = (
        _bernoulli_log_likelihood(n01, n00, p01)
        + _bernoulli_log_likelihood(n11, n10, p11)
    )

    lr_stat = max(0.0, 2.0 * (alternative_ll - null_ll))
    p_value = float(chi2.sf(lr_stat, df=1))

    return {
        "observations": len(values),
        "transitions_00": n00,
        "transitions_01": n01,
        "transitions_10": n10,
        "transitions_11": n11,
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "reject_at_5pct": p_value < 0.05,
    }


def conditional_coverage_test(exceptions, expected_exception_rate):
    """
    Christoffersen conditional coverage test.

    Combines unconditional coverage and exception independence.
    The likelihood-ratio statistic is compared with chi-square(df=2).
    """
    values = _validate_exceptions(exceptions, expected_exception_rate)

    if len(values) < 2:
        raise ValueError("At least two observations are required.")

    coverage = kupiec_coverage_test(values, expected_exception_rate)
    independence = christoffersen_independence_test(values)

    lr_stat = coverage["lr_statistic"] + independence["lr_statistic"]
    p_value = float(chi2.sf(lr_stat, df=2))

    return {
        "observations": len(values),
        "exceptions": int(values.sum()),
        "expected_exception_rate": expected_exception_rate,
        "observed_exception_rate": float(values.mean()),
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "reject_at_5pct": p_value < 0.05,
    }
