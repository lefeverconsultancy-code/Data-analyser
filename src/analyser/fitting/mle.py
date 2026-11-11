"""Maximum likelihood fitting. Phase 0: closed-form MLE (the normal distribution).

Numeric optimisation, profile-likelihood and bootstrap CIs, χ²/KS/AD goodness of fit
and AIC ranking of candidates follow in phase 2.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from ..core import assumptions
from ..core.registry import get_distribution, register_model
from ..core.results import ConfidenceInterval, FitResult, GoodnessOfFit
from ..inference.one_sample import clean_sample, data_inputs


@register_model(
    "fit_mle",
    name="Maximum likelihood fit",
    data="continuous",
    result="fit",
    params=("distribution", "level"),
    distributions=("normal",),
    sources=("L&M 5.2", "PrEE 6.5"),
)
def fit(
    x,
    distribution: str = "normal",
    *,
    level: float = 0.95,
    variable: str | None = None,
    unit: str | None = None,
) -> FitResult:
    """Fit a registered distribution by maximum likelihood.

    Standard errors come from the Fisher information; CIs are Wald intervals
    (estimate ± z·SE) at the given level.
    """
    if not 0 < level < 1:
        raise ValueError("level must lie strictly between 0 and 1")
    cls = get_distribution(distribution)
    arr, dropped = clean_sample(x)
    est = cls.mle(arr)
    params: dict[str, float] = est["params"]
    se: dict[str, float] = est["se"]
    dist = cls(**params)
    n = arr.size
    k = len(params)
    loglik = float(np.sum(dist.logpdf(arr)))
    z = float(stats.norm.ppf(0.5 + level / 2))
    cis = {
        name: ConfidenceInterval(level=level, lower=v - z * se[name], upper=v + z * se[name],
                                 parameter=cls.param_symbols.get(name, name),
                                 method="Wald: estimate ± z·SE (Fisher information)")
        for name, v in params.items()
    }
    gof: list[GoodnessOfFit] = []
    checks = []
    if distribution == "normal":
        check = assumptions.normality(arr, alpha=1 - level)
        if check.statistic is not None:
            gof.append(GoodnessOfFit(test="Shapiro–Wilk", statistic=check.statistic,
                                     p_value=check.p_value, note="normality"))
        checks.append(check)

    inputs: dict[str, Any] = data_inputs(arr, dropped, variable, unit)
    result = FitResult(
        name=f"{cls.__name__} fit (MLE)",
        distribution=distribution,
        distribution_label=f"{cls.notation}({', '.join(cls.param_symbols.get(p, p) for p in cls.param_names)})",
        method="mle",
        params=params,
        param_symbols=dict(cls.param_symbols),
        param_labels=est.get("labels", {}),
        standard_errors=se,
        cis=cis,
        n=n,
        k=k,
        loglik=loglik,
        aic=2 * k - 2 * loglik,
        bic=k * math.log(n) - 2 * loglik,
        gof=gof,
        alt_estimates=est.get("alt", {}),
        checks=checks,
        inputs=inputs,
        settings={"level": level},
    )
    if distribution == "normal":
        result.notes.append(
            "σ² here is the MLE variance (divisor n). The unbiased S² (divisor n − 1), "
            "the default elsewhere in the tool, is listed under 'Also'."
        )
    if dropped:
        result.notes.append(f"{dropped} missing value(s) ignored.")
    return result
