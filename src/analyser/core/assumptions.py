"""Assumption checks, with the readers' rules of thumb as defaults (plan, section 8).

Each function returns an AssumptionCheck. The checks guide the choice of method;
they do not prove an assumption.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .formatting import fmt_fixed, fmt_value, p_equals
from .results import FAILED, INFO, NOT_RUN, PASSED, WARNING, AssumptionCheck

CLT_N = 40  # Stats 7.2: n ≥ 40 for the large-sample z-test
CHI2_MIN_EXPECTED = 5  # Stats 6.1
RANK_NORMAL_MIN_N = 6  # Stats 7.5: normal approximation of W when n1, n2 > 5
TIES_PLAIN_VARIANCE = 0.20  # Stats 7.5: under 20% ties the plain variance is fine


def normality(x: np.ndarray | None, alpha: float = 0.05, fallback: str | None = None) -> AssumptionCheck:
    """Shapiro–Wilk plus a Q-Q plot. Without raw data the check is listed but not run."""
    method = "Shapiro–Wilk + Q-Q plot"
    rule = "Shapiro–Wilk p-value > α and a straight Q-Q plot"
    if x is None:
        return AssumptionCheck(
            name="normality", method=method, status=NOT_RUN,
            summary=f"normality: {method}", rule=rule,
            detail="Only summary statistics were given, so normality could not be checked here. "
                   "Check it on the raw data (Shapiro–Wilk and the Q-Q plot).",
            fallback=fallback,
        )
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return AssumptionCheck(
            name="normality", method=method, status=NOT_RUN,
            summary="normality: n < 3, Shapiro–Wilk not possible", rule=rule,
            detail="Shapiro–Wilk needs at least 3 observations.", fallback=fallback,
        )
    if np.ptp(x) == 0:
        return AssumptionCheck(
            name="normality", method=method, status=FAILED,
            summary="normality: all values equal", rule=rule,
            detail="All observations are equal, so the data cannot come from a normal distribution "
                   "with σ > 0.", fallback=fallback,
        )
    sample = x
    note = ""
    if n > 5000:
        rng = np.random.default_rng(0)
        sample = rng.choice(x, size=5000, replace=False)
        note = " Shapiro–Wilk was run on a random subsample of 5000 observations."
    w, p = stats.shapiro(sample)
    passed = p > alpha
    return AssumptionCheck(
        name="normality", method=method, status=PASSED if passed else FAILED,
        summary=f"normality: Shapiro–Wilk p {p_equals(p)} + Q-Q plot",
        statistic=float(w), p_value=float(p), rule=rule,
        detail=(
            f"Shapiro–Wilk W = {fmt_fixed(w, 4)}, p {p_equals(p)} "
            + ("> α: no evidence against normality." if passed else "≤ α: the data do not look normal.")
            + " Confirm with the Q-Q plot." + note
        ),
        fallback=None if passed else fallback,
    )


def large_sample(n: int, test_label: str = "t-test", threshold: int = CLT_N) -> AssumptionCheck:
    """n ≥ 40 rule (Stats 7.2)."""
    if n < threshold:
        return AssumptionCheck(
            name="sample size", method=f"n ≥ {threshold} (CLT)", status=INFO,
            summary=f"n = {n} < {threshold}, so {test_label} needs normality",
            rule=f"n ≥ {threshold}",
            detail=f"With n = {n} the central limit theorem cannot be relied on, so the "
                   f"{test_label} is only valid if the population is normal.",
        )
    return AssumptionCheck(
        name="sample size", method=f"n ≥ {threshold} (CLT)", status=PASSED,
        summary=f"n = {n} ≥ {threshold}, so the CLT covers non-normality",
        rule=f"n ≥ {threshold}",
        detail=f"With n = {n} ≥ {threshold} the sample mean is approximately normal even if "
               f"the population is not.",
    )


def mean_test_normality(
    x: np.ndarray | None, n: int, alpha: float, test_label: str, fallback: str
) -> list[AssumptionCheck]:
    """Checks for a test on μ: normality, sample size, and the combined verdict."""
    norm = normality(x, alpha, fallback=fallback)
    size = large_sample(n, test_label)
    if norm.status == FAILED and size.status == PASSED:
        norm.status = WARNING
        norm.detail = (norm.detail or "") + f" Because n ≥ {CLT_N}, the {test_label} remains valid."
        norm.fallback = None
    return [norm, size]


def time_order(column: str | None) -> AssumptionCheck | None:
    if not column:
        return None
    return AssumptionCheck(
        name="independence", method="time order", status=WARNING,
        summary=f"independence: data are in time order ({column}), check autocorrelation",
        rule="random sample",
        detail="The observations are ordered in time, so neighbouring values may be correlated. "
               "The runs test and autocorrelation checks arrive in phase 4.",
    )


def expected_counts(expected: np.ndarray, minimum: float = CHI2_MIN_EXPECTED) -> AssumptionCheck:
    e = np.asarray(expected, dtype=float)
    low = int((e < minimum).sum())
    rule = f"all expected counts ≥ {minimum:g}"
    if low == 0:
        return AssumptionCheck(
            name="expected counts", method="expected count per category", status=PASSED,
            summary=f"expected counts: all ≥ {minimum:g} (min = {fmt_value(e.min())})", rule=rule,
            detail="The χ² approximation is adequate.",
        )
    return AssumptionCheck(
        name="expected counts", method="expected count per category", status=FAILED,
        summary=f"expected counts: {low} of {e.size} below {minimum:g}", rule=rule,
        detail="The χ² approximation is not reliable. Merge categories, or use an exact test.",
        fallback="merge categories or an exact test",
    )


def rank_normal_approximation(n1: int, n2: int, minimum: int = RANK_NORMAL_MIN_N) -> AssumptionCheck:
    ok = n1 >= minimum and n2 >= minimum
    return AssumptionCheck(
        name="normal approximation", method="n₁, n₂ > 5", status=PASSED if ok else WARNING,
        summary=f"normal approximation: n₁ = {n1}, n₂ = {n2}" + (" > 5" if ok else ", not both > 5"),
        rule="n₁ and n₂ > 5 (Stats 7.5)",
        detail=None if ok else "The normal approximation of W is rough for samples of 5 or fewer; "
                               "the exact distribution arrives in phase 3.",
    )


def ties(values: np.ndarray, threshold: float = TIES_PLAIN_VARIANCE) -> AssumptionCheck:
    v = np.asarray(values)
    _, counts = np.unique(v, return_counts=True)
    tied = int(counts[counts > 1].sum())
    share = tied / v.size if v.size else 0.0
    if share < threshold:
        return AssumptionCheck(
            name="ties", method="share of tied observations", status=PASSED,
            summary=f"ties: {share:.0%} of observations tied (< {threshold:.0%})",
            rule=f"under {threshold:.0%} ties: plain variance (Stats 7.5)", statistic=share,
        )
    return AssumptionCheck(
        name="ties", method="share of tied observations", status=INFO,
        summary=f"ties: {share:.0%} tied, tie-corrected variance used",
        rule=f"under {threshold:.0%} ties: plain variance (Stats 7.5)", statistic=share,
    )
