import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from analyser.core.results import FAILED, NOT_RUN, PASSED, WARNING
from analyser.inference.one_sample import (
    t_one_sample,
    t_one_sample_summary,
    z_one_sample,
    z_one_sample_summary,
)

RNG = np.random.default_rng(7)
X = RNG.normal(12, 3, size=25)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_t_matches_scipy(alternative):
    r = t_one_sample(X, mu0=11, alternative=alternative)
    ref = stats.ttest_1samp(X, 11, alternative=alternative)
    assert r.observed == pytest.approx(ref.statistic, abs=1e-10)
    assert r.p_value == pytest.approx(ref.pvalue, abs=1e-10)
    ci = ref.confidence_interval(0.95)
    if alternative != "less":
        assert r.ci.lower == pytest.approx(ci.low, abs=1e-8)
    if alternative != "greater":
        assert r.ci.upper == pytest.approx(ci.high, abs=1e-8)


def test_summary_and_data_versions_agree():
    a = t_one_sample(X, mu0=11, alternative="greater")
    b = t_one_sample_summary(len(X), X.mean(), X.std(ddof=1), mu0=11, alternative="greater")
    for field in ("observed", "p_value", "reject", "rejection_region", "h1", "statistic_name"):
        assert getattr(a, field) == getattr(b, field)


def test_uses_sample_variance_n_minus_1():
    r = t_one_sample([1, 2, 3, 4, 5], mu0=2)
    assert r.details["sd"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))


def test_decision_consistent_with_region():
    for mu0 in np.linspace(10, 14, 21):
        for alt in ("two-sided", "greater", "less"):
            r = t_one_sample(X, mu0=mu0, alternative=alt)
            t, cv = r.observed, r.critical_values
            in_region = (("upper" in cv and t >= cv["upper"]) or ("lower" in cv and t <= cv["lower"]))
            assert in_region == r.reject


def test_z_test_matches_formula():
    r = z_one_sample(X, mu0=11, sigma=3, alternative="two-sided")
    z = (X.mean() - 11) / (3 / np.sqrt(len(X)))
    assert r.observed == pytest.approx(z)
    assert r.p_value == pytest.approx(2 * stats.norm.sf(abs(z)))
    assert r.null_distribution == "N(0, 1)"


def test_checks_with_and_without_data():
    with_data = t_one_sample(X, mu0=11)
    assert with_data.checks[0].status in (PASSED, FAILED, WARNING)
    assert with_data.checks[0].p_value is not None
    summary = t_one_sample_summary(16, 118, 9, 115)
    assert summary.checks[0].status == NOT_RUN
    assert summary.checks[1].summary == "n = 16 < 40, so t-test needs normality"


def test_non_normal_large_sample_downgrades_to_warning():
    skewed = np.random.default_rng(3).exponential(2, size=300)
    r = t_one_sample(skewed, mu0=2)
    assert r.checks[0].status == WARNING
    assert r.checks[0].fallback is None


def test_non_normal_small_sample_names_fallback():
    skewed = np.random.default_rng(3).exponential(2, size=20) ** 3
    r = t_one_sample(skewed, mu0=2)
    assert r.checks[0].status == FAILED
    assert "Wilcoxon" in r.checks[0].fallback


def test_time_order_warning_and_missing_values():
    r = t_one_sample([1.0, 2.0, np.nan, 3.0, 4.5], mu0=2, time_ordered="created_at")
    assert any(c.name == "independence" for c in r.checks)
    assert r.inputs["missing_dropped"] == 1
    assert "1 missing value(s) ignored." in r.notes


def test_conclusion_wording():
    r = t_one_sample(X, mu0=0, alternative="greater", variable="pick_time_min", unit="min")
    assert r.reject
    assert r.conclusion.startswith("At a 5% significance level, the data prove that the expected pick time")
    assert "accept" not in r.conclusion.lower()


@pytest.mark.parametrize("bad", [dict(x=[1.0], mu0=0), dict(x=[2.0, 2.0, 2.0], mu0=1)])
def test_invalid_inputs(bad):
    with pytest.raises(ValueError):
        t_one_sample(**bad)
    with pytest.raises(ValueError):
        t_one_sample(X, mu0=1, alternative="sideways")


@settings(max_examples=40, deadline=None)
@given(shift=st.floats(-1e3, 1e3), scale=st.floats(0.1, 100))
def test_location_scale_invariance(shift, scale):
    base = t_one_sample(X, mu0=11)
    moved = t_one_sample(X * scale + shift, mu0=11 * scale + shift)
    assert moved.observed == pytest.approx(base.observed, rel=1e-6, abs=1e-6)
    assert moved.p_value == pytest.approx(base.p_value, rel=1e-6, abs=1e-9)


def test_z_summary_reference_numbers():
    r = z_one_sample_summary(9, 119, 9, 115, "greater")
    assert r.critical_values["xbar_upper"] == pytest.approx(119.9346, abs=1e-4)
