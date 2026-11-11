"""One-sample tests on μ: t-test (σ unknown) and z-test (σ known).

Each test takes raw data, or summary statistics through the ``*_summary`` variant
(useful for reader exercises that only give n, x̄ and s). Both return a TestResult
with the 8 steps of the Statistics for Engineers reader.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
from scipy import stats

from ..core import assumptions
from ..core.formatting import (
    CLAIM_VERB,
    H1_SIGN,
    conclusion_sentence,
    fmt_num,
    fmt_stat,
    fmt_value,
    humanize,
    normalize_alternative,
    sample_notation,
)
from ..core.registry import register_test
from ..core.results import ConfidenceInterval, EffectSize, TestResult

MAX_STORED_SAMPLE = 10_000


def clean_sample(x: Any) -> tuple[np.ndarray, int]:
    """Numeric array without missing values, and the number of values dropped."""
    arr = np.asarray(x, dtype=float).ravel()
    keep = np.isfinite(arr)
    return arr[keep], int((~keep).sum())


def data_inputs(x: np.ndarray, dropped: int, variable: str | None, unit: str | None) -> dict[str, Any]:
    digest = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]
    inputs: dict[str, Any] = {
        "source": "data",
        "variable": variable,
        "unit": unit,
        "n": int(x.size),
        "missing_dropped": dropped,
        "data_hash": digest,
    }
    if x.size <= MAX_STORED_SAMPLE:
        inputs["sample"] = x.tolist()
    return inputs


def _statistic_formula(symbol: str, mu0: float, scale_symbol: str) -> str:
    if mu0 < 0:
        centre = f"X̄ + {fmt_num(-mu0)}"
    else:
        centre = f"X̄ − {fmt_num(mu0)}"
    return f"{symbol} = ({centre}) / ({scale_symbol}/√n)"


def _mean_test(
    *,
    kind: str,
    n: int,
    mean: float,
    scale: float,
    mu0: float,
    alternative: str,
    alpha: float,
    x: np.ndarray | None,
    about: str | None,
    claim: str | None,
    unit: str | None,
    variable: str | None,
    time_ordered: str | None,
    inputs: dict[str, Any],
) -> TestResult:
    alt = normalize_alternative(alternative)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if n < 2:
        raise ValueError("A test on μ needs at least 2 observations")
    if not (scale > 0 and math.isfinite(scale)):
        raise ValueError(
            "The standard deviation is 0 (all values equal), so the test statistic is undefined"
        )

    se = scale / math.sqrt(n)
    stat = (mean - mu0) / se
    if kind == "t":
        df = n - 1
        null = stats.t(df)
        test_id, name = "t_one_sample", "One-sample t-test on μ"
        symbol, obs_symbol, null_label = "T", "t", f"t({df})"
        scale_symbol, model_sigma = "S", "σ unknown"
        model_dist = "N(μ, σ²)"
        effect = EffectSize(name="d", value=(mean - mu0) / scale, formula="(x̄ − μ₀)/s", label="Cohen's d")
        fallback = "Wilcoxon signed rank or sign test"
        label = "t-test"
    else:
        df = None
        null = stats.norm()
        test_id, name = "z_one_sample", "One-sample z-test on μ (σ known)"
        symbol, obs_symbol, null_label = "Z", "z", "N(0, 1)"
        scale_symbol, model_sigma = "σ", "σ known"
        model_dist = f"N(μ, {fmt_num(scale ** 2)})"
        effect = EffectSize(name="d", value=(mean - mu0) / scale, formula="(x̄ − μ₀)/σ", label="Cohen's d")
        fallback = "large-sample z-test (n ≥ 40) or a nonparametric test"
        label = "z-test"

    margin = "t·s/√n" if kind == "t" else "z·σ/√n"

    # Step 6a: critical values and the equivalent region for x̄
    crit: dict[str, float] = {}
    if alt == "greater":
        c = float(null.ppf(1 - alpha))
        xc = mu0 + c * se
        crit = {"upper": c, "xbar_upper": xc}
        region = f"{symbol} ≥ {fmt_stat(c)}  (equivalently x̄ ≥ {fmt_value(xc)})"
        p = float(null.sf(stat))
        p_expr = f"P({symbol} ≥ {fmt_stat(stat)} | H0)"
        ci = ConfidenceInterval(level=1 - alpha, lower=mean - c * se, upper=None, parameter="μ",
                                method=f"x̄ − {margin}")
    elif alt == "less":
        c = float(null.ppf(1 - alpha))
        xc = mu0 - c * se
        crit = {"lower": -c, "xbar_lower": xc}
        region = f"{symbol} ≤ {fmt_stat(-c)}  (equivalently x̄ ≤ {fmt_value(xc)})"
        p = float(null.cdf(stat))
        p_expr = f"P({symbol} ≤ {fmt_stat(stat)} | H0)"
        ci = ConfidenceInterval(level=1 - alpha, lower=None, upper=mean + c * se, parameter="μ",
                                method=f"x̄ + {margin}")
    else:
        c = float(null.ppf(1 - alpha / 2))
        lo, hi = mu0 - c * se, mu0 + c * se
        crit = {"lower": -c, "upper": c, "xbar_lower": lo, "xbar_upper": hi}
        region = (
            f"{symbol} ≤ {fmt_stat(-c)} or {symbol} ≥ {fmt_stat(c)}  "
            f"(equivalently x̄ ≤ {fmt_value(lo)} or x̄ ≥ {fmt_value(hi)})"
        )
        p = float(min(1.0, 2 * null.sf(abs(stat))))
        p_expr = f"2·P({symbol} ≥ {fmt_stat(abs(stat))} | H0)"
        ci = ConfidenceInterval(level=1 - alpha, lower=mean - c * se, upper=mean + c * se, parameter="μ",
                                method=f"x̄ ± {margin}")
    reject = p <= alpha

    # Step 8
    if claim is None:
        subject = about or (f"the expected {humanize(variable)}" if variable else "μ")
        unit_txt = f" {unit}" if unit else ""
        claim = f"{subject} {CLAIM_VERB[alt]} {fmt_num(mu0)}{unit_txt}"

    checks = assumptions.mean_test_normality(x, n, alpha, label, fallback)
    order_check = assumptions.time_order(time_ordered)
    if order_check is not None:
        checks.append(order_check)

    details: dict[str, Any] = {"n": n, "mean": mean, "se": se}
    if kind == "t":
        details.update({"sd": scale, "df": df})
    else:
        details.update({"sigma": scale})

    return TestResult(
        name=name,
        test_id=test_id,
        model=f"{sample_notation('X', n)} random sample from {model_dist}, {model_sigma}",
        h0=f"μ = {fmt_num(mu0)}",
        h1=f"μ {H1_SIGN[alt]} {fmt_num(mu0)}",
        alternative=alt,
        statistic_name=_statistic_formula(symbol, mu0, scale_symbol),
        statistic_symbol=symbol,
        null_distribution=null_label,
        observed=float(stat),
        observed_symbol=obs_symbol,
        rejection_region=region,
        critical_values=crit,
        p_value=p,
        p_value_expression=p_expr,
        alpha=alpha,
        reject=reject,
        conclusion=conclusion_sentence(alpha, reject, claim),
        ci=ci,
        effect_size=effect,
        checks=checks,
        details=details,
        inputs=inputs,
        settings={"alpha": alpha},
    )


# ---------------------------------------------------------------------------
# t-test
# ---------------------------------------------------------------------------


@register_test(
    "t_one_sample",
    name="One-sample t-test on μ",
    samples=1,
    data="continuous",
    parameter="mu",
    assumptions=("random_sample", "normal_or_n_ge_40"),
    params=("mu0", "alternative", "alpha"),
    sources=("Stats 5.1", "L&M 7.4"),
    summary_function="analyser.inference.one_sample:t_one_sample_summary",
    fallback="Wilcoxon signed rank or sign test",
)
def t_one_sample(
    x,
    mu0: float,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    *,
    about: str | None = None,
    claim: str | None = None,
    unit: str | None = None,
    variable: str | None = None,
    time_ordered: str | None = None,
) -> TestResult:
    """One-sample t-test of H0: μ = μ₀ with σ unknown.

    ``about`` names μ in plain English for step 8 (e.g. "the expected pick time");
    ``claim`` replaces the whole H1 claim. ``time_ordered`` names a time column if the
    data are in time order, which adds an independence warning.
    """
    arr, dropped = clean_sample(x)
    inputs = data_inputs(arr, dropped, variable, unit)
    inputs.update({"mu0": mu0, "alternative": alternative})
    result = _mean_test(
        kind="t", n=arr.size, mean=float(arr.mean()) if arr.size else math.nan,
        scale=float(arr.std(ddof=1)) if arr.size > 1 else math.nan,
        mu0=mu0, alternative=alternative, alpha=alpha, x=arr, about=about, claim=claim,
        unit=unit, variable=variable, time_ordered=time_ordered, inputs=inputs,
    )
    if dropped:
        result.notes.append(f"{dropped} missing value(s) ignored.")
    return result


def t_one_sample_summary(
    n: int,
    mean: float,
    sd: float,
    mu0: float,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    *,
    about: str | None = None,
    claim: str | None = None,
    unit: str | None = None,
    variable: str | None = None,
) -> TestResult:
    """The t-test from n, x̄ and s (sample standard deviation, divisor n − 1)."""
    inputs = {"source": "summary", "n": int(n), "mean": mean, "sd": sd, "mu0": mu0,
              "alternative": alternative, "variable": variable, "unit": unit}
    return _mean_test(
        kind="t", n=int(n), mean=float(mean), scale=float(sd), mu0=mu0, alternative=alternative,
        alpha=alpha, x=None, about=about, claim=claim, unit=unit, variable=variable,
        time_ordered=None, inputs=inputs,
    )


t_one_sample.from_summary = t_one_sample_summary


# ---------------------------------------------------------------------------
# z-test, σ known
# ---------------------------------------------------------------------------


@register_test(
    "z_one_sample",
    name="One-sample z-test on μ (σ known)",
    samples=1,
    data="continuous",
    parameter="mu",
    assumptions=("random_sample", "normal_or_n_ge_40", "sigma_known"),
    params=("mu0", "sigma", "alternative", "alpha"),
    sources=("Stats 4.1–4.2",),
    summary_function="analyser.inference.one_sample:z_one_sample_summary",
)
def z_one_sample(
    x,
    mu0: float,
    sigma: float,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    *,
    about: str | None = None,
    claim: str | None = None,
    unit: str | None = None,
    variable: str | None = None,
    time_ordered: str | None = None,
) -> TestResult:
    """z-test of H0: μ = μ₀ when the population σ is known."""
    arr, dropped = clean_sample(x)
    inputs = data_inputs(arr, dropped, variable, unit)
    inputs.update({"mu0": mu0, "sigma": sigma, "alternative": alternative})
    result = _mean_test(
        kind="z", n=arr.size, mean=float(arr.mean()) if arr.size else math.nan, scale=float(sigma),
        mu0=mu0, alternative=alternative, alpha=alpha, x=arr, about=about, claim=claim,
        unit=unit, variable=variable, time_ordered=time_ordered, inputs=inputs,
    )
    if dropped:
        result.notes.append(f"{dropped} missing value(s) ignored.")
    return result


def z_one_sample_summary(
    n: int,
    mean: float,
    sigma: float,
    mu0: float,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    *,
    about: str | None = None,
    claim: str | None = None,
    unit: str | None = None,
    variable: str | None = None,
) -> TestResult:
    """The z-test from n, x̄ and the known σ."""
    inputs = {"source": "summary", "n": int(n), "mean": mean, "sigma": sigma, "mu0": mu0,
              "alternative": alternative, "variable": variable, "unit": unit}
    return _mean_test(
        kind="z", n=int(n), mean=float(mean), scale=float(sigma), mu0=mu0, alternative=alternative,
        alpha=alpha, x=None, about=about, claim=claim, unit=unit, variable=variable,
        time_ordered=None, inputs=inputs,
    )


z_one_sample.from_summary = z_one_sample_summary
