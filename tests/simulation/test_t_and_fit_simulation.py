"""Simulation checks (plan section 11): test size, CI coverage and parameter recovery.

10,000 runs each with a fixed seed. The statistics are computed in one vectorised pass
and the engine's own functions are checked on a subsample, so the suite stays fast.
"""

import numpy as np
import pytest
from scipy import stats

from analyser.fitting.mle import fit
from analyser.inference.one_sample import t_one_sample

RUNS = 10_000
SEED = 20261111


def binomial_band(p: float, n: int, z: float = 4.0) -> tuple[float, float]:
    half = z * np.sqrt(p * (1 - p) / n)
    return p - half, p + half


@pytest.mark.slow
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_t_test_size_and_ci_coverage_under_h0(alternative):
    rng = np.random.default_rng(SEED)
    n, mu, alpha = 12, 50.0, 0.05
    samples = rng.normal(mu, 7.0, size=(RUNS, n))
    rejections, covered = 0, 0
    for row in samples:
        r = t_one_sample(row, mu0=mu, alternative=alternative, alpha=alpha)
        rejections += r.reject
        lo = -np.inf if r.ci.lower is None else r.ci.lower
        hi = np.inf if r.ci.upper is None else r.ci.upper
        covered += lo <= mu <= hi
    lo_b, hi_b = binomial_band(alpha, RUNS)
    assert lo_b <= rejections / RUNS <= hi_b
    lo_c, hi_c = binomial_band(1 - alpha, RUNS)
    assert lo_c <= covered / RUNS <= hi_c


@pytest.mark.slow
def test_t_test_power_matches_noncentral_t():
    rng = np.random.default_rng(SEED + 1)
    n, delta, sigma = 15, 3.0, 6.0
    samples = rng.normal(100 + delta, sigma, size=(RUNS, n))
    t = (samples.mean(axis=1) - 100) / (samples.std(axis=1, ddof=1) / np.sqrt(n))
    crit = stats.t.ppf(0.95, n - 1)
    power_sim = float((t >= crit).mean())
    power_exact = float(stats.nct.sf(crit, n - 1, delta / sigma * np.sqrt(n)))
    lo, hi = binomial_band(power_exact, RUNS)
    assert lo <= power_sim <= hi
    for row in samples[:200]:  # the engine agrees with the vectorised statistic
        assert t_one_sample(row, mu0=100, alternative="greater").reject == (
            (row.mean() - 100) / (row.std(ddof=1) / np.sqrt(n)) >= crit)


@pytest.mark.slow
def test_normal_mle_parameter_recovery_and_wald_coverage():
    rng = np.random.default_rng(SEED + 2)
    n, mu, sigma2 = 50, 12.0, 9.0
    samples = rng.normal(mu, np.sqrt(sigma2), size=(RUNS, n))
    mu_hat = samples.mean(axis=1)
    s2_mle = samples.var(axis=1, ddof=0)
    s2_unbiased = samples.var(axis=1, ddof=1)
    assert abs(mu_hat.mean() - mu) < 4 * np.sqrt(sigma2 / n / RUNS)
    # MLE variance is biased by the factor (n − 1)/n; S² is unbiased
    assert s2_mle.mean() == pytest.approx(sigma2 * (n - 1) / n, rel=0.01)
    assert s2_unbiased.mean() == pytest.approx(sigma2, rel=0.01)
    covered = 0
    for row in samples[:2000]:
        r = fit(row, "normal")
        covered += r.cis["mu"].lower <= mu <= r.cis["mu"].upper
    lo, hi = binomial_band(0.95, 2000)
    assert lo - 0.01 <= covered / 2000 <= hi  # Wald z-interval is slightly narrow at n = 50
