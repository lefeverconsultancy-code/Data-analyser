"""Categorical tests. Phase 0: χ² goodness of fit against a specified distribution
(Stats 6.1). Estimated-parameter fits, cross tables and Fisher exact follow in phase 4."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy import stats

from ..core import assumptions
from ..core.formatting import conclusion_sentence, fmt_num, fmt_stat, subscript
from ..core.registry import register_test
from ..core.results import EffectSize, TestResult


@register_test(
    "chi2_gof",
    name="χ² goodness-of-fit test",
    samples=1,
    data="categorical",
    parameter="p",
    alternatives=("two-sided",),
    assumptions=("random_sample", "expected_counts_ge_5"),
    params=("observed", "probs", "expected", "ddof", "alpha"),
    sources=("Stats 6.1", "L&M 10.3"),
)
def chi2_gof(
    observed: Sequence[float],
    probs: Sequence[float] | None = None,
    *,
    expected: Sequence[float] | None = None,
    categories: Sequence[str] | None = None,
    ddof: int = 0,
    alpha: float = 0.05,
    claim: str | None = None,
) -> TestResult:
    """χ² test of H0: the category probabilities are p₀₁,…,p₀ₖ.

    Give either ``probs`` (the H0 probabilities; expected counts are n·p₀ᵢ) or
    ``expected`` counts directly. ``ddof`` is the number of parameters estimated
    from the data (df = k − 1 − ddof).
    """
    obs = np.asarray(observed, dtype=float)
    if obs.ndim != 1 or obs.size < 2:
        raise ValueError("observed must be a list of at least 2 counts")
    if (obs < 0).any():
        raise ValueError("observed counts cannot be negative")
    n = float(obs.sum())
    k = obs.size
    if (probs is None) == (expected is None):
        raise ValueError("give either probs or expected, not both")
    if probs is not None:
        p0 = np.asarray(probs, dtype=float)
        if p0.size != k or (p0 < 0).any() or not math.isclose(p0.sum(), 1.0, abs_tol=1e-6):
            raise ValueError("probs must have one value per category and sum to 1")
        exp = n * p0
    else:
        exp = np.asarray(expected, dtype=float)
        if exp.size != k or (exp <= 0).any():
            raise ValueError("expected must have one positive count per category")
        if not math.isclose(exp.sum(), n, rel_tol=0.01):
            raise ValueError(f"expected counts sum to {exp.sum():g}, observed counts to {n:g}")
        p0 = exp / exp.sum()
    df = k - 1 - int(ddof)
    if df < 1:
        raise ValueError("degrees of freedom k − 1 − ddof must be at least 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")

    chi2 = float(((obs - exp) ** 2 / exp).sum())
    c = float(stats.chi2.ppf(1 - alpha, df))
    p = float(stats.chi2.sf(chi2, df))
    reject = p <= alpha
    cats = list(categories) if categories is not None else [str(i + 1) for i in range(k)]

    h0_parts = ", ".join(f"p{subscript(i + 1)} = {fmt_num(v, 4)}" for i, v in enumerate(p0))
    h0 = h0_parts if k <= 6 else f"pᵢ = p₀ᵢ for all {k} categories"
    claim = claim or "the category probabilities differ from the H0 distribution"
    table = [
        {"category": cat, "observed": float(o), "expected": float(e),
         "contribution": float((o - e) ** 2 / e)}
        for cat, o, e in zip(cats, obs, exp)
    ]
    inputs: dict[str, Any] = {
        "source": "counts", "observed": obs.tolist(), "categories": cats,
        "probs": p0.tolist() if probs is not None else None,
        "expected": exp.tolist(), "ddof": ddof,
    }
    return TestResult(
        name="χ² goodness-of-fit test",
        test_id="chi2_gof",
        model=f"N₁,…,N{subscript(k)} multinomial with n = {fmt_num(n)} and unknown probabilities "
              f"p₁,…,p{subscript(k)}",
        h0=h0,
        h1="at least one pᵢ differs",
        alternative="two-sided",
        statistic_name="χ² = Σ (Nᵢ − E₀Nᵢ)² / E₀Nᵢ",
        statistic_symbol="χ²",
        null_distribution=f"χ²({df})",
        null_is_approximate=True,
        observed=chi2,
        observed_symbol="χ²",
        rejection_region=f"χ² ≥ {fmt_stat(c)}",
        critical_values={"upper": c},
        p_value=p,
        p_value_expression=f"P(χ² ≥ {fmt_stat(chi2)} | H0)",
        alpha=alpha,
        reject=reject,
        conclusion=conclusion_sentence(alpha, reject, claim),
        effect_size=EffectSize(name="w", value=math.sqrt(chi2 / n), formula="√(χ²/n)",
                               label="Cohen's w"),
        checks=[assumptions.expected_counts(exp)],
        details={"n": n, "k": k, "df": df, "table": table,
                 "min_expected": float(exp.min()), "max_contribution": max(t["contribution"] for t in table)},
        inputs=inputs,
        settings={"alpha": alpha},
    )
