"""Plotly figures for the app (phase 0: Q-Q plot, histogram with fitted density,
null distribution with the rejection region). Static matplotlib versions for saved
reports follow in phase 6.

Colours follow a validated categorical order: slot 1 (blue) for the data, slot 2
(orange) for the fitted model or observed statistic; reference lines are neutral grey.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from ..core.results import FitResult, TestResult

DATA = "#2a78d6"  # categorical slot 1
MODEL = "#eb6834"  # categorical slot 2
NEUTRAL = "#8a8985"
REGION = "rgba(235, 104, 52, 0.18)"  # slot 2, tinted, for the rejection region

LAYOUT = dict(
    margin=dict(l=10, r=10, t=48, b=10),
    height=380,
    legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
    hovermode="closest",
)


def plotting_positions(n: int) -> np.ndarray:
    """(i − 0.5)/n, the same positions as the Excel workbook."""
    return (np.arange(1, n + 1) - 0.5) / n


def qq_plot(sample, dist=None, *, title: str = "Normal Q-Q plot", unit: str | None = None) -> go.Figure:
    """Sample quantiles against theoretical quantiles.

    With ``dist`` (a fitted Distribution) the x-axis is on the data scale and the
    reference line is y = x; without it, standard-normal quantiles and the line
    x̄ + s·z.
    """
    x = np.sort(np.asarray(sample, dtype=float))
    x = x[np.isfinite(x)]
    n = x.size
    p = plotting_positions(n)
    if dist is not None:
        theo = np.asarray(dist.quantile(p), dtype=float)
        line_x = np.array([theo.min(), theo.max()])
        line_y = line_x
        x_title = f"theoretical quantile, {dist.label}"
    else:
        theo = stats.norm.ppf(p)
        mean, sd = float(x.mean()), float(x.std(ddof=1)) if n > 1 else 0.0
        line_x = np.array([theo.min(), theo.max()])
        line_y = mean + sd * line_x
        x_title = "standard normal quantile z"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y, mode="lines", name="reference line",
        line=dict(color=NEUTRAL, width=2, dash="dash"), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=theo, y=x, mode="markers", name="observations",
        marker=dict(color=DATA, size=8, opacity=0.75, line=dict(width=0.5, color="white")),
        hovertemplate="theoretical %{x:.3f}<br>observed %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title=x_title,
                      yaxis_title=f"sample quantile{f' ({unit})' if unit else ''}", **LAYOUT)
    return fig


def histogram_with_fit(sample, dist=None, *, title: str = "Histogram with fitted density",
                       unit: str | None = None, bins: int | None = None) -> go.Figure:
    """Density-scaled histogram, with the fitted pdf overlaid when ``dist`` is given."""
    x = np.asarray(sample, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    k = bins or max(5, min(40, math.ceil(math.sqrt(n))))  # √n bins
    counts, edges = np.histogram(x, bins=k)
    widths = np.diff(edges)
    density = counts / (n * widths)
    centres = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centres, y=density, width=widths, name="data",
        marker=dict(color=DATA, line=dict(width=2, color="rgba(0,0,0,0)")),
        customdata=np.column_stack([edges[:-1], edges[1:], counts]),
        hovertemplate="%{customdata[0]:.2f} to %{customdata[1]:.2f}<br>%{customdata[2]} observations<extra></extra>",
    ))
    if dist is not None:
        grid = np.linspace(edges[0] - widths[0], edges[-1] + widths[-1], 300)
        fig.add_trace(go.Scatter(
            x=grid, y=dist.pdf(grid), mode="lines", name=f"fitted {dist.label}",
            line=dict(color=MODEL, width=2),
            hovertemplate="x %{x:.3f}<br>density %{y:.4f}<extra></extra>",
        ))
    fig.update_layout(title=title, xaxis_title=unit or "value", yaxis_title="density",
                      bargap=0.04, **LAYOUT)
    return fig


def _null_curve(r: TestResult):
    """(frozen scipy distribution, label) for the test's statistic under H0, if known."""
    if r.test_id == "t_one_sample":
        return stats.t(r.details["df"]), "T"
    if r.test_id == "z_one_sample":
        return stats.norm(), "Z"
    if r.test_id == "chi2_gof":
        return stats.chi2(r.details["df"]), "χ²"
    if r.test_id == "wilcoxon_rank_sum":
        return stats.norm(r.details["mean_w"], math.sqrt(r.details["var_w"])), "W"
    return None, None


def null_distribution_plot(r: TestResult) -> go.Figure | None:
    """The statistic's distribution under H0 with the rejection region shaded and the observed value."""
    dist, symbol = _null_curve(r)
    if dist is None:
        return None
    lo, hi = dist.ppf(0.0005), dist.ppf(0.9995)
    lo = min(lo, r.observed) - 0.05 * (hi - lo)
    hi = max(hi, r.observed) + 0.05 * (hi - lo)
    grid = np.linspace(lo, hi, 600)
    pdf = dist.pdf(grid)
    fig = go.Figure()
    upper = r.critical_values.get("upper")
    lower = r.critical_values.get("lower")
    first = True
    for bound, side in ((upper, "upper"), (lower, "lower")):
        if bound is None:
            continue
        mask = grid >= bound if side == "upper" else grid <= bound
        fig.add_trace(go.Scatter(
            x=grid[mask], y=pdf[mask], fill="tozeroy", mode="none", fillcolor=REGION,
            name="rejection region", showlegend=first, legendgroup="region", hoverinfo="skip",
        ))
        first = False
    fig.add_trace(go.Scatter(
        x=grid, y=pdf, mode="lines", name=f"{symbol} under H0: {r.null_distribution}",
        line=dict(color=NEUTRAL, width=2),
        hovertemplate=f"{symbol} = %{{x:.3f}}<br>density %{{y:.4f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[r.observed, r.observed], y=[0, float(dist.pdf(r.observed)) * 1.0 + pdf.max() * 0.08],
        mode="lines", name=f"observed {r.observed_symbol} = {r.observed:.3f}",
        line=dict(color=MODEL, width=2), hoverinfo="name",
    ))
    fig.update_layout(title="Test statistic under H0", xaxis_title=symbol, yaxis_title="density", **LAYOUT)
    return fig


def fit_plots(r: FitResult, sample) -> tuple[go.Figure, go.Figure]:
    dist = r.fitted()
    unit = r.inputs.get("unit")
    label = r.inputs.get("variable") or "value"
    axis = f"{label} ({unit})" if unit else label
    return (histogram_with_fit(sample, dist, unit=axis),
            qq_plot(sample, dist, title=f"Q-Q plot against {dist.label}", unit=unit))
