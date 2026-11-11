"""Render any result object in the app: TestResult, FitResult or ModelResult."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from analyser.core.formatting import fmt_fixed, fmt_num, fmt_p, fmt_stat, fmt_value, p_equals
from analyser.core.registry import REGISTRY
from analyser.core.results import AssumptionCheck, FitResult, ModelResult, Result, TestResult
from analyser.describe import plots
from analyser.report.text import render_text

STATUS = {
    "passed": ("✅", "passed"),
    "failed": ("❌", "failed"),
    "warning": ("⚠️", "warning"),
    "not_run": ("⏸️", "not run"),
    "info": ("ℹ️", "note"),
}

ASSUMPTION_TEXT = {
    "random_sample": "the observations are a random sample (independent)",
    "normal_or_n_ge_40": "the population is normal, or n ≥ 40",
    "sigma_known": "the population σ is known",
    "expected_counts_ge_5": "all expected counts are at least 5",
    "independent_samples": "the two samples are independent random samples",
    "same_shape": "both distributions have the same shape (shift alternative)",
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def render_checks(checks: list[AssumptionCheck]) -> None:
    if not checks:
        return
    st.markdown("**Assumption checks**")
    for c in checks:
        icon, label = STATUS.get(c.status, ("•", c.status))
        line = f"{icon} **{label}**: {c.summary}"
        if c.fallback:
            line += f"  \n&nbsp;&nbsp;&nbsp;&nbsp;Alternative: {c.fallback}"
        st.markdown(line)
        if c.detail or c.rule:
            st.caption(" ".join(x for x in (c.detail, f"Rule: {c.rule}." if c.rule else None) if x))


def _sample(r: Result):
    return r.inputs.get("sample")


def render_test(r: TestResult, key: str) -> None:
    st.subheader(r.name)
    m = st.columns(4)
    m[0].metric(f"observed {r.observed_symbol}", fmt_stat(r.observed))
    m[1].metric("p-value", fmt_p(r.p_value))
    m[2].metric("α", fmt_num(r.alpha))
    n = r.details.get("n", r.details.get("n1"))
    if n is not None:
        m[3].metric("n" if "n" in r.details else "n₁, n₂",
                    fmt_num(n) if "n" in r.details else f"{r.details['n1']}, {r.details['n2']}")
    (st.success if r.reject else st.info)(f"**{r.decision}.** {r.conclusion}")

    st.markdown("**The 8 steps**")
    d = r.details
    entry = REGISTRY.entries("test")
    meta = next((e.metadata for e in entry if e.id == r.test_id), {})
    with st.expander(f"1. Model: {r.model}"):
        if meta.get("assumptions"):
            st.markdown("Assumptions of this test:\n" + "\n".join(
                f"- {ASSUMPTION_TEXT.get(a, a)}" for a in meta["assumptions"]))
    with st.expander(f"2. Hypotheses: H0: {r.h0}  vs  H1: {r.h1}"):
        st.markdown(f"Alternative: **{r.alternative}**. H0 is rejected only with enough evidence; "
                    "otherwise we fail to reject it (we never 'accept' H0).")
    with st.expander(f"3. Statistic: {r.statistic_name}"):
        rows = {k: v for k, v in d.items() if isinstance(v, (int, float))}
        if rows:
            st.dataframe(pd.DataFrame({"quantity": list(rows), "value": [fmt_value(float(v)) for v in rows.values()]}),
                         hide_index=True, width="content")
    with st.expander(f"4. Under H0: {r.statistic_symbol} ~ {r.null_distribution}"
                     + (" (approximately)" if r.null_is_approximate else "")):
        st.markdown("The distribution of the statistic if H0 is true; it supplies the critical "
                    "values and the p-value (replacing the printed tables).")
    with st.expander(f"5. Observed: {r.observed_symbol} = {fmt_stat(r.observed)}"):
        st.markdown(f"Full precision: `{r.observed!r}`")
    with st.expander(f"6. Reject H0 if {r.rejection_region};  p-value = {fmt_p(r.p_value)}"):
        st.markdown(f"a. Rejection region: {r.rejection_region}  \n"
                    f"b. p-value = {r.p_value_expression} = {r.p_value:.6g}")
        if r.critical_values:
            st.caption("Critical values: " + ", ".join(f"{k} = {fmt_value(v)}" for k, v in r.critical_values.items()))
    with st.expander(f"7. Decision: {r.decision}"):
        st.markdown(f"{'p ≤ α' if r.reject else 'p > α'}: {fmt_p(r.p_value)} {'≤' if r.reject else '>'} "
                    f"{fmt_num(r.alpha)}, and {r.observed_symbol} is "
                    f"{'in' if r.reject else 'outside'} the rejection region.")
    with st.expander("8. Conclusion", expanded=True):
        st.markdown(r.conclusion)
    extra = []
    if r.ci is not None:
        extra.append(f"**CI:** {r.ci.describe()}" + (f" ({r.ci.method})" if r.ci.method else ""))
    if r.effect_size is not None:
        extra.append(f"**Effect size:** {r.effect_size.describe()}"
                     + (f" ({r.effect_size.label})" if r.effect_size.label else ""))
    if extra:
        st.markdown("  \n".join(extra))
    render_checks(r.checks)
    for note in r.notes:
        st.caption(note)

    figs = [plots.null_distribution_plot(r)]
    if _sample(r):
        figs.append(plots.qq_plot(_sample(r), unit=r.inputs.get("unit")))
    cols = st.columns(len([f for f in figs if f is not None]) or 1)
    for col, fig in zip(cols, [f for f in figs if f is not None]):
        col.plotly_chart(fig, key=f"{key}_{id(fig)}", width="stretch")
    with st.expander("Text report (8-step format)"):
        st.code(render_text(r), language=None)


def render_fit(r: FitResult, key: str) -> None:
    st.subheader(r.name)
    m = st.columns(4)
    m[0].metric("n", r.n)
    m[1].metric("ln L̂", fmt_value(r.loglik))
    m[2].metric("AIC", fmt_value(r.aic))
    m[3].metric("BIC", fmt_value(r.bic))
    rows = []
    for k, v in r.params.items():
        ci = r.cis.get(k)
        rows.append({
            "parameter": r.param_labels.get(k, r.param_symbols.get(k, k)),
            "estimate": fmt_value(v),
            "SE": fmt_value(r.standard_errors[k]) if k in r.standard_errors else "",
            f"{fmt_fixed(ci.level * 100, 0)}% CI" if ci else "CI": f"({fmt_value(ci.lower)}, {fmt_value(ci.upper)})" if ci else "",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for name, est in r.alt_estimates.items():
        st.caption(f"{name.capitalize()}: " + ", ".join(
            f"{r.param_symbols.get(k, k)} = {fmt_value(v)}" for k, v in est.items()))
    for g in r.gof:
        st.markdown(f"**{g.test}:** statistic = {fmt_fixed(g.statistic, 4)}"
                    + (f", p {p_equals(g.p_value)}" if g.p_value is not None else ""))
    render_checks(r.checks)
    for note in r.notes:
        st.caption(note)
    if _sample(r):
        hist, qq = plots.fit_plots(r, _sample(r))
        c1, c2 = st.columns(2)
        c1.plotly_chart(hist, key=f"{key}_hist", width="stretch")
        c2.plotly_chart(qq, key=f"{key}_qq", width="stretch")
    with st.expander("Text report"):
        st.code(render_text(r), language=None)


def render_model(r: ModelResult, key: str) -> None:
    st.subheader(r.name)
    st.markdown(f"Model: {r.formula}")
    if r.coefficients:
        st.dataframe(pd.DataFrame([{
            "term": c.name, "estimate": fmt_value(c.estimate),
            "SE": fmt_value(c.se) if c.se is not None else "",
            "CI": f"({fmt_value(c.ci_lower)}, {fmt_value(c.ci_upper)})" if c.ci_lower is not None else "",
            "statistic": fmt_stat(c.statistic) if c.statistic is not None else "",
            "p-value": fmt_p(c.p_value) if c.p_value is not None else "",
        } for c in r.coefficients]), hide_index=True, width="stretch")
    if r.anova_table:
        st.dataframe(pd.DataFrame(r.anova_table), hide_index=True)
    if r.r_squared is not None:
        st.markdown(f"R² = {fmt_fixed(r.r_squared, 3)}")
    render_checks(r.diagnostics)
    if r.conclusion:
        st.markdown(r.conclusion)
    with st.expander("Text report"):
        st.code(render_text(r), language=None)


def render_result(r: Result, key: str = "result") -> None:
    """Render any result object, with a JSON download."""
    if isinstance(r, TestResult):
        render_test(r, key)
    elif isinstance(r, FitResult):
        render_fit(r, key)
    elif isinstance(r, ModelResult):
        render_model(r, key)
    else:  # pragma: no cover - future result kinds
        st.json(r.to_dict())
    st.download_button("Download result (JSON)", r.to_json(), file_name=f"{_slug(r.name)}-{r.id}.json",
                       mime="application/json", key=f"{key}_download_{r.id}")
