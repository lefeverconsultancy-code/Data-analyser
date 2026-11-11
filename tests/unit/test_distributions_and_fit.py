import math

import numpy as np
import pytest
from scipy import stats

from analyser.distributions import calculator
from analyser.distributions.continuous import Normal
from analyser.fitting.mle import fit
from analyser.inference.categorical import chi2_gof
from analyser.inference.nonparametric import wilcoxon_rank_sum


def test_normal_parameterised_by_variance():
    d = Normal(80, 64)
    assert d.sd() == 8 and d.var() == pytest.approx(64) and d.label == "N(80, 64)"
    assert calculator.probability(d, "<=", 70) == pytest.approx(stats.norm.cdf(70, 80, 8))
    assert calculator.probability(d, ">", 70) == pytest.approx(stats.norm.sf(70, 80, 8))
    assert calculator.between(d, 70, 90) == pytest.approx(stats.norm.cdf(90, 80, 8) - stats.norm.cdf(70, 80, 8))
    assert calculator.quantile(d, 0.95) == pytest.approx(93.1588, abs=1e-4)
    a, b = d.rvs(5, seed=1), d.rvs(5, seed=1)
    assert np.array_equal(a, b)


def test_normal_validation():
    with pytest.raises(ValueError):
        Normal(0, 0)
    with pytest.raises(ValueError):
        Normal(0, -1)
    with pytest.raises(ValueError):
        calculator.quantile(Normal(), 1.0)


def test_normal_mle_matches_closed_form_and_scipy():
    x = np.random.default_rng(11).normal(50, 4, size=200)
    r = fit(x, "normal")
    mu, sigma = stats.norm.fit(x)
    assert r.params["mu"] == pytest.approx(mu)
    assert r.params["sigma2"] == pytest.approx(sigma ** 2)
    assert r.alt_estimates["unbiased"]["sigma2"] == pytest.approx(x.var(ddof=1))
    assert r.loglik == pytest.approx(stats.norm.logpdf(x, mu, sigma).sum())
    assert r.aic == pytest.approx(4 - 2 * r.loglik)
    assert r.bic == pytest.approx(2 * math.log(200) - 2 * r.loglik)
    assert r.standard_errors["mu"] == pytest.approx(sigma / math.sqrt(200))
    assert r.cis["mu"].lower < 50 < r.cis["mu"].upper
    assert any("MLE variance" in n for n in r.notes)


def test_fit_rejects_degenerate_data():
    with pytest.raises(ValueError):
        fit([5.0, 5.0, 5.0], "normal")
    with pytest.raises(KeyError):
        fit([1.0, 2.0], "gumbel")


def test_chi2_against_scipy_and_validation():
    obs, p = [18, 22, 30, 30], [0.25, 0.25, 0.25, 0.25]
    r = chi2_gof(obs, p)
    ref = stats.chisquare(obs, np.array(p) * 100)
    assert r.observed == pytest.approx(ref.statistic) and r.p_value == pytest.approx(ref.pvalue)
    low = chi2_gof([3, 4, 2, 1], [0.25] * 4)
    assert low.checks[0].status == "failed"
    with pytest.raises(ValueError):
        chi2_gof([1, 2], [0.5, 0.6])
    with pytest.raises(ValueError):
        chi2_gof([1, 2], [0.5, 0.5], expected=[1, 2])
    assert chi2_gof(obs, p, ddof=1).details["df"] == 2


def test_wilcoxon_matches_scipy_normal_approximation():
    rng = np.random.default_rng(5)
    a, b = rng.normal(1, 1, 15), rng.normal(0, 1, 12)
    for alt in ("greater", "less", "two-sided"):
        r = wilcoxon_rank_sum(a, b, alt)
        ref = stats.mannwhitneyu(a, b, alternative=alt, method="asymptotic", use_continuity=True)
        assert r.p_value == pytest.approx(ref.pvalue, abs=1e-10)
