import json
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from analyser.core.results import (
    AssumptionCheck,
    Coefficient,
    ConfidenceInterval,
    EffectSize,
    FitResult,
    ModelResult,
    Result,
    TestResult,
    result_from_json,
)
from analyser.fitting.mle import fit
from analyser.inference.one_sample import t_one_sample, t_one_sample_summary


def _test_result(**overrides):
    base = dict(
        name="One-sample t-test on μ", test_id="t_one_sample", model="m", h0="μ = 1", h1="μ > 1",
        alternative="greater", statistic_name="T", statistic_symbol="T", null_distribution="t(9)",
        observed=1.5, observed_symbol="t", rejection_region="T ≥ 1.833", p_value=0.08,
        p_value_expression="P(T ≥ 1.5 | H0)", alpha=0.05, reject=False,
        conclusion="At a 5% significance level, the data do not prove that μ exceeds 1.",
    )
    base.update(overrides)
    return TestResult(**base)


def test_test_result_json_round_trip_with_nested_objects():
    r = t_one_sample([3.1, 2.9, 3.4, 3.0, 3.3, 2.8, 3.2], mu0=3, alternative="greater")
    text = r.to_json()
    back = Result.from_json(text)
    assert isinstance(back, TestResult)
    assert back == r
    assert isinstance(back.ci, ConfidenceInterval)
    assert isinstance(back.effect_size, EffectSize)
    assert all(isinstance(c, AssumptionCheck) for c in back.checks)
    assert json.loads(text)["kind"] == "test"


def test_fit_result_round_trip_and_fitted_distribution():
    r = fit([4.2, 5.1, 3.9, 4.8, 5.5, 4.4, 4.9, 5.0], "normal")
    back = result_from_json(r.to_json())
    assert isinstance(back, FitResult) and back == r
    assert isinstance(back.cis["mu"], ConfidenceInterval)
    assert back.fitted().params == r.params


def test_model_result_round_trip():
    r = ModelResult(
        name="Simple linear regression", model_id="ols", formula="Y = β₀ + β₁x + ε",
        coefficients=[Coefficient(name="β₀", estimate=1.0, se=0.2, p_value=0.001),
                      Coefficient(name="β₁", estimate=2.5, se=None)],
        anova_table=[{"source": "regression", "df": 1, "SS": 10.0}], r_squared=0.81, n=12,
        residuals=[0.1, -0.2], diagnostics=[AssumptionCheck(name="normality", method="SW",
                                                             status="passed", summary="ok")],
    )
    back = result_from_json(r.to_json())
    assert back == r and isinstance(back.coefficients[0], Coefficient)


def test_non_finite_floats_survive_json():
    r = _test_result(p_value=float("nan"), critical_values={"upper": math.inf, "lower": -math.inf})
    text = r.to_json()
    json.loads(text)  # strict JSON, no NaN tokens
    back = result_from_json(text)
    assert math.isnan(back.p_value)
    assert back.critical_values == {"upper": math.inf, "lower": -math.inf}


def test_never_accept_h0_and_alpha_bounds():
    with pytest.raises(ValueError, match="accept"):
        _test_result(conclusion="We accept H0.")
    with pytest.raises(ValueError):
        _test_result(alpha=1.5)


def test_check_status_is_validated():
    with pytest.raises(ValueError):
        AssumptionCheck(name="x", method="y", status="maybe", summary="z")


def test_loading_wrong_kind_is_refused():
    r = t_one_sample_summary(10, 5, 1, 4)
    with pytest.raises(ValueError):
        FitResult.from_json(r.to_json())


def test_ci_descriptions():
    assert ConfidenceInterval(level=0.95, lower=114.0556, upper=None).describe() == "95% lower bound for μ: 114.06"
    assert ConfidenceInterval(level=0.9, lower=None, upper=3.5).describe() == "90% upper bound for μ: 3.5000"
    assert ConfidenceInterval(level=0.95, lower=1.0, upper=2.0).sides == "two-sided"


@settings(max_examples=60, deadline=None)
@given(
    observed=st.floats(allow_nan=True, allow_infinity=True),
    p=st.floats(min_value=0, max_value=1),
    alpha=st.floats(min_value=0.001, max_value=0.5),
    reject=st.booleans(),
    name=st.text(max_size=30),
    details=st.dictionaries(st.text(max_size=8), st.floats(allow_nan=False) | st.integers() | st.text(max_size=5),
                            max_size=4),
)
def test_round_trip_property(observed, p, alpha, reject, name, details):
    r = _test_result(name=name, observed=observed, p_value=p, alpha=alpha, reject=reject, details=details)
    back = result_from_json(r.to_json())
    # Compare the serialised form: NaN never equals NaN under dataclass equality (Python 3.13+)
    assert back.to_dict() == r.to_dict()
    assert math.isnan(back.observed) if math.isnan(observed) else back.observed == observed
