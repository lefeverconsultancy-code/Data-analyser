"""Analyse page (phase 0 proof of concept): one-sample t-test, normal MLE fit, normal calculator."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from analyser.app import state
from analyser.app.render import render_result
from analyser.core.formatting import fmt_value, humanize
from analyser.describe import plots
from analyser.distributions import calculator
from analyser.distributions.continuous import Normal

S = state.session()


def page() -> None:
    st.title("Analyse")

    if not S.datasets:
        st.info("No data yet. Import a dataset first, or load the sample order data.")
        c = st.columns(2)
        c[0].page_link("pages/import_data.py", label="Import data", icon=":material/upload_file:")
        if c[1].button("Load the sample order data", type="primary", key="an_sample"):
            try:
                state.load_sample_orders()
                st.rerun()
            except FileNotFoundError as exc:
                st.error(str(exc))
        return

    names = list(S.datasets)
    active = state.active_dataset()
    ds_name = st.selectbox("Dataset", names, index=names.index(active.name) if active else 0, key="an_ds")
    state.set_active(ds_name)
    ds = S.datasets[ds_name]
    numeric = [c for c in ds.numeric_columns() if ds.columns[c].role != "id"]
    if not numeric:
        st.warning("This dataset has no numeric columns to analyse.")
        return
    preferred = next((c for c in numeric if c.startswith("duration")), numeric[0])
    var_name = st.selectbox("Variable", numeric, index=numeric.index(preferred), key=f"an_var_{ds_name}")
    v = ds.var(var_name)
    x = v.numeric()
    if x.size < 2:
        st.warning("This variable has fewer than 2 values.")
        return
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    unit = f" {v.unit}" if v.unit else ""
    st.caption(f"n = {x.size}" + (f" ({v.missing} missing ignored)" if v.missing else "")
               + f" · x̄ = {fmt_value(mean)}{unit} · s = {fmt_value(sd)}{unit} · S² = {fmt_value(sd ** 2)} (divisor n − 1)")


    def show_last(kind: str) -> None:
        rid = st.session_state.get(f"last_{kind}")
        if rid:
            try:
                render_result(S.get(rid), key=f"an_{kind}")
            except KeyError:
                pass


    tab_t, tab_fit, tab_norm = st.tabs(["One-sample t-test", "Normal fit (MLE)", "Normal distribution"])

    with tab_t:
        c = st.columns(3)
        mu0 = c[0].number_input("μ₀, the value under H0", value=float(round(mean)), key="an_mu0", format="%g")
        alt_label = c[1].selectbox("H1", ["μ ≠ μ₀ (two-sided)", "μ > μ₀", "μ < μ₀"], key="an_alt")
        alpha = c[2].number_input("α", min_value=0.001, max_value=0.5, value=S.settings.alpha, step=0.01,
                                  format="%g", key="an_alpha")
        about = st.text_input("μ in words, for the conclusion", value=f"the expected {humanize(var_name)}",
                              key=f"an_about_{var_name}")
        if st.button("Run t-test", type="primary", key="an_run_t"):
            alternative = {"μ ≠ μ₀ (two-sided)": "two-sided", "μ > μ₀": "greater", "μ < μ₀": "less"}[alt_label]
            try:
                r = S.run_test("t_one_sample", x, mu0=mu0, alternative=alternative, alpha=alpha, about=about,
                               unit=v.unit, variable=v.name, time_ordered=v.time_ordered_by)
                r.inputs["dataset"] = ds.name
                st.session_state["last_test"] = r.id
            except ValueError as exc:
                st.error(str(exc))
        show_last("test")

    with tab_fit:
        level = st.number_input("Confidence level for the parameter CIs", min_value=0.5, max_value=0.999,
                                value=S.settings.confidence, step=0.01, format="%g", key="an_level")
        if st.button("Fit normal distribution (MLE)", type="primary", key="an_run_fit"):
            try:
                r = S.run_model("fit_mle", x, "normal", level=level, variable=v.name, unit=v.unit)
                r.inputs["dataset"] = ds.name
                st.session_state["last_fit"] = r.id
            except ValueError as exc:
                st.error(str(exc))
        show_last("fit")

    with tab_norm:
        st.caption("Probabilities and percentiles of N(μ, σ²). Defaults are the sample's x̄ and S².")
        c = st.columns(4)
        mu = c[0].number_input("μ", value=float(round(mean, 4)), format="%g", key="an_n_mu")
        sigma2 = c[1].number_input("σ²", min_value=1e-12, value=float(round(sd ** 2, 4)), format="%g", key="an_n_s2")
        xv = c[2].number_input("x", value=float(round(mean - sd, 2)), format="%g", key="an_n_x")
        pv = c[3].number_input("percentile p", min_value=0.0001, max_value=0.9999, value=0.95, format="%g",
                               key="an_n_p")
        dist = Normal(mu, sigma2)
        below = calculator.probability(dist, "<=", xv)
        q = calculator.quantile(dist, pv)
        m = st.columns(3)
        m[0].metric(f"P(X ≤ {xv:g})", f"{below:.4f}")
        m[1].metric(f"P(X > {xv:g})", f"{1 - below:.4f}")
        m[2].metric(f"{pv * 100:g}th percentile", fmt_value(q))
        sd_ = np.sqrt(sigma2)
        grid = np.linspace(mu - 4 * sd_, mu + 4 * sd_, 400)
        fig = go.Figure()
        shade = grid <= xv
        fig.add_trace(go.Scatter(x=grid[shade], y=dist.pdf(grid[shade]), fill="tozeroy", mode="none",
                                 fillcolor="rgba(42, 120, 214, 0.25)", name=f"P(X ≤ {xv:g})", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=grid, y=dist.pdf(grid), mode="lines", name=dist.label,
                                 line=dict(color=plots.DATA, width=2),
                                 hovertemplate="x %{x:.3f}<br>density %{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[q, q], y=[0, float(dist.pdf(q))], mode="lines", name=f"{pv * 100:g}th percentile",
                                 line=dict(color=plots.MODEL, width=2, dash="dash"), hoverinfo="name"))
        fig.update_layout(title=f"X ~ {dist.label}", xaxis_title="x", yaxis_title="density", **plots.LAYOUT)
        st.plotly_chart(fig, key="an_norm_plot", width="stretch")


page()
