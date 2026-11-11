"""Nonparametric tests. Phase 0: Wilcoxon rank-sum test with the normal approximation and
continuity correction (Stats 7.5). Exact p-values, the Hodges–Lehmann CI for the shift and
the sign / signed-rank tests follow in phase 3."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from ..core import assumptions
from ..core.formatting import conclusion_sentence, fmt_num, fmt_stat, fmt_value, normalize_alternative, sample_notation
from ..core.registry import register_test
from ..core.results import EffectSize, TestResult
from .one_sample import clean_sample


@register_test(
    "wilcoxon_rank_sum",
    name="Wilcoxon rank-sum test",
    samples=2,
    data="continuous",
    paired=False,
    parameter="shift",
    assumptions=("independent_samples", "same_shape"),
    params=("alternative", "alpha", "continuity", "tie_correction"),
    sources=("Stats 7.5", "L&M 14.3"),
)
def wilcoxon_rank_sum(
    x,
    y,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    *,
    continuity: bool = True,
    tie_correction: str = "auto",
    x_label: str = "X",
    y_label: str = "Y",
    claim: str | None = None,
) -> TestResult:
    """Wilcoxon rank-sum test of H0: f_X = f_Y against a shift alternative.

    W is the rank sum of the x-sample. ``tie_correction``: 'auto' uses the tie-corrected
    variance when 20% or more of the observations are tied (Stats 7.5), 'always' or 'never'.
    """
    alt = normalize_alternative(alternative)
    a, da = clean_sample(x)
    b, db = clean_sample(y)
    n1, n2 = a.size, b.size
    if n1 < 1 or n2 < 1:
        raise ValueError("both samples need at least one observation")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if tie_correction not in ("auto", "always", "never"):
        raise ValueError("tie_correction must be 'auto', 'always' or 'never'")
    N = n1 + n2
    combined = np.concatenate([a, b])
    ranks = stats.rankdata(combined)
    w = float(ranks[:n1].sum())
    mean_w = n1 * (N + 1) / 2
    var_plain = n1 * n2 * (N + 1) / 12
    _, t = np.unique(combined, return_counts=True)
    var_ties = n1 * n2 * (N**3 - float((t**3).sum())) / (12 * N * (N - 1)) if N > 1 else var_plain
    ties_check = assumptions.ties(combined)
    use_ties = tie_correction == "always" or (tie_correction == "auto" and ties_check.status != "passed")
    var_w = var_ties if use_ties else var_plain
    sd_w = math.sqrt(var_w)
    cc = 0.5 if continuity else 0.0
    z_norm = stats.norm()

    if alt == "greater":
        z = (w - cc - mean_w) / sd_w
        p = float(z_norm.sf(z))
        c = math.ceil(mean_w + cc + z_norm.ppf(1 - alpha) * sd_w - 1e-9)
        region = f"W ≥ {c}"
        crit = {"upper": float(c)}
        p_expr = f"P(W ≥ {fmt_stat(w)} | H0) ≈ P(Z ≥ ({fmt_num(w - cc)} − {fmt_num(mean_w)})/√{fmt_value(var_w)})"
    elif alt == "less":
        z = (w + cc - mean_w) / sd_w
        p = float(z_norm.cdf(z))
        c = math.floor(mean_w - cc - z_norm.ppf(1 - alpha) * sd_w + 1e-9)
        region = f"W ≤ {c}"
        crit = {"lower": float(c)}
        p_expr = f"P(W ≤ {fmt_stat(w)} | H0) ≈ P(Z ≤ ({fmt_num(w + cc)} − {fmt_num(mean_w)})/√{fmt_value(var_w)})"
    else:
        z = (abs(w - mean_w) - cc) / sd_w
        z = max(z, 0.0)
        p = float(min(1.0, 2 * z_norm.sf(z)))
        zc = z_norm.ppf(1 - alpha / 2)
        lo = math.floor(mean_w - cc - zc * sd_w + 1e-9)
        hi = math.ceil(mean_w + cc + zc * sd_w - 1e-9)
        region = f"W ≤ {lo} or W ≥ {hi}"
        crit = {"lower": float(lo), "upper": float(hi)}
        p_expr = f"2·P(Z ≥ (|{fmt_stat(w)} − {fmt_num(mean_w)}| − {fmt_num(cc)})/√{fmt_value(var_w)})"
    reject = p <= alpha

    verbs = {"greater": "tend to be larger than", "less": "tend to be smaller than",
             "two-sided": "differ in location from"}
    claim = claim or f"the {x_label} values {verbs[alt]} the {y_label} values"
    h1 = {"greater": "f_X(x) = f_Y(x − a) with a > 0", "less": "f_X(x) = f_Y(x − a) with a < 0",
          "two-sided": "f_X(x) = f_Y(x − a) with a ≠ 0"}[alt]
    checks = [assumptions.rank_normal_approximation(n1, n2), ties_check]
    effect = EffectSize(name="r", value=float(z / math.sqrt(N)), formula="z/√N", label="rank-biserial r")
    notes = []
    if da or db:
        notes.append(f"{da + db} missing value(s) ignored.")
    inputs: dict[str, Any] = {"source": "data", "x": a.tolist(), "y": b.tolist(),
                              "x_label": x_label, "y_label": y_label}
    return TestResult(
        name="Wilcoxon rank-sum test",
        test_id="wilcoxon_rank_sum",
        model=f"independent random samples {sample_notation('X', n1)} ({x_label}) and "
              f"{sample_notation('Y', n2)} ({y_label}); f_X and f_Y unknown",
        h0="f_X(x) = f_Y(x)",
        h1=h1,
        alternative=alt,
        statistic_name="W = Σ R(Xᵢ), the rank sum of the X-sample",
        statistic_symbol="W",
        null_distribution=f"N({fmt_num(mean_w)}, {fmt_value(var_w)})",
        null_is_approximate=True,
        observed=w,
        observed_symbol="w",
        rejection_region=region,
        critical_values=crit,
        p_value=p,
        p_value_expression=p_expr,
        alpha=alpha,
        reject=reject,
        conclusion=conclusion_sentence(alpha, reject, claim),
        effect_size=effect,
        checks=checks,
        details={"n1": n1, "n2": n2, "mean_w": mean_w, "var_w": var_w, "var_plain": var_plain,
                 "var_ties": var_ties, "z": float(z), "continuity": continuity,
                 "tie_corrected": use_ties},
        inputs=inputs,
        settings={"alpha": alpha},
        notes=notes,
    )
