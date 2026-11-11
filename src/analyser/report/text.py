"""Plain-text renderer. A TestResult prints in the 8-step format of the plan (section 9):

    One-sample t-test on μ                                        α = 0.05
    1. Model       X₁,…,X₁₆ random sample from N(μ, σ²), σ unknown
    2. Hypotheses  H0: μ = 115   vs   H1: μ > 115
    ...
"""

from __future__ import annotations

from ..core.formatting import fmt_fixed, fmt_num, fmt_p, fmt_stat, fmt_value, p_equals, wrap_hanging
from ..core.results import FitResult, ModelResult, Result, TestResult

WIDTH = 70  # header width; the conclusion wraps at this width
LABEL = 15  # "1. Model       " — step labels are padded to 15 characters
FOOTER = 14  # "Effect size   ", "Checks        "


def _step(label: str, text: str) -> str:
    return f"{label:<{LABEL}}{text}"


def render_test(r: TestResult) -> str:
    alpha_txt = f"α = {fmt_num(r.alpha)}"
    gap = max(2, WIDTH - len(r.name) - len(alpha_txt))
    null = f"{r.statistic_symbol} ~ {r.null_distribution}"
    if r.null_is_approximate:
        null += ", approximately"
    region_side = "in" if r.reject else "outside"
    p_side = "p ≤ α" if r.reject else "p > α"
    decision = f"{r.decision}: {r.observed_symbol} is {region_side} the rejection region, {p_side}"

    lines = [
        f"{r.name}{' ' * gap}{alpha_txt}",
        _step("1. Model", r.model),
        _step("2. Hypotheses", f"H0: {r.h0}   vs   H1: {r.h1}"),
        _step("3. Statistic", r.statistic_name),
        _step("4. Under H0", null),
        _step("5. Observed", f"{r.observed_symbol} = {fmt_stat(r.observed)}"),
        f"6. a. Reject H0 if {r.rejection_region}",
        f"   b. p-value = {r.p_value_expression} {p_equals(r.p_value)}",
        _step("7. Decision", decision),
        *wrap_hanging(r.conclusion, f"{'8. Conclusion':<{LABEL}}", WIDTH),
    ]
    if r.ci is not None:
        lines.append(f"CI   {r.ci.describe()}")
    if r.effect_size is not None:
        lines.append(f"{'Effect size':<{FOOTER}}{r.effect_size.describe()}")
    if r.checks:
        lines.append(f"{'Checks':<{FOOTER}}" + " | ".join(c.summary for c in r.checks))
    for note in r.notes:
        lines.append(f"{'Note':<{FOOTER}}{note}")
    return "\n".join(lines)


def render_fit(r: FitResult) -> str:
    title = f"{r.name}"
    lines = [title, f"{'Distribution':<{FOOTER}}{r.distribution_label}, n = {r.n}"]
    width = max(len(r.param_labels.get(k, k)) for k in r.params) if r.params else 10
    lines.append("Estimates")
    for key, value in r.params.items():
        label = r.param_labels.get(key, r.param_symbols.get(key, key))
        se = r.standard_errors.get(key)
        ci = r.cis.get(key)
        parts = [f"  {label:<{width}}  {fmt_value(value)}"]
        if se is not None:
            parts.append(f"SE {fmt_value(se)}")
        if ci is not None:
            parts.append(f"{fmt_fixed(ci.level * 100, 0)}% CI ({fmt_value(ci.lower)}, {fmt_value(ci.upper)})")
        lines.append("   ".join(parts))
    if r.loglik is not None:
        lines.append(
            f"{'Likelihood':<{FOOTER}}ln L̂ = {fmt_value(r.loglik)}   AIC = {fmt_value(r.aic)}   "
            f"BIC = {fmt_value(r.bic)}   (k = {r.k})"
        )
    for name, est in r.alt_estimates.items():
        txt = ", ".join(f"{r.param_symbols.get(k, k)} = {fmt_value(v)}" for k, v in est.items())
        lines.append(f"{'Also':<{FOOTER}}{name}: {txt}")
    for g in r.gof:
        p = f", p {p_equals(g.p_value)}" if g.p_value is not None else ""
        lines.append(f"{'Fit check':<{FOOTER}}{g.test}: statistic = {fmt_fixed(g.statistic, 4)}{p}"
                     + (f"  ({g.note})" if g.note else ""))
    if r.checks:
        lines.append(f"{'Checks':<{FOOTER}}" + " | ".join(c.summary for c in r.checks))
    for note in r.notes:
        lines.append(f"{'Note':<{FOOTER}}{note}")
    return "\n".join(lines)


def render_model(r: ModelResult) -> str:
    lines = [r.name, f"{'Model':<{FOOTER}}{r.formula}", f"{'n':<{FOOTER}}{r.n}"]
    if r.coefficients:
        lines.append(f"{'term':<16}{'estimate':>12}{'SE':>12}{'statistic':>12}{'p-value':>10}")
        for c in r.coefficients:
            lines.append(
                f"{c.name:<16}{fmt_value(c.estimate):>12}"
                f"{(fmt_value(c.se) if c.se is not None else ''):>12}"
                f"{(fmt_stat(c.statistic) if c.statistic is not None else ''):>12}"
                f"{(fmt_p(c.p_value) if c.p_value is not None else ''):>10}"
            )
    if r.r_squared is not None:
        adj = f"   adjusted R² = {fmt_fixed(r.adj_r_squared, 3)}" if r.adj_r_squared is not None else ""
        lines.append(f"{'Fit':<{FOOTER}}R² = {fmt_fixed(r.r_squared, 3)}{adj}")
    if r.diagnostics:
        lines.append(f"{'Checks':<{FOOTER}}" + " | ".join(c.summary for c in r.diagnostics))
    if r.conclusion:
        lines.extend(wrap_hanging(r.conclusion, f"{'Conclusion':<{FOOTER}}", WIDTH))
    return "\n".join(lines)


def render_text(result: Result) -> str:
    """Render any result object as plain text."""
    if isinstance(result, TestResult):
        return render_test(result)
    if isinstance(result, FitResult):
        return render_fit(result)
    if isinstance(result, ModelResult):
        return render_model(result)
    raise TypeError(f"No text renderer for {type(result).__name__}")
