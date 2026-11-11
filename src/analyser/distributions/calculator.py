"""Probability calculator (phase 0 core; shaded plots and approximations follow in phase 1).

    >>> from analyser.distributions.continuous import Normal
    >>> round(probability(Normal(80, 64), "<=", 70), 4)
    0.1056
"""

from __future__ import annotations

from .base import Distribution

_OPS = ("<=", "<", ">=", ">", "==")


def probability(dist: Distribution, op: str, x: float) -> float:
    """P(X op x) for op in '<=', '<', '>=', '>', '=='."""
    if op not in _OPS:
        raise ValueError(f"op must be one of {_OPS}")
    discrete = dist.family == "discrete"
    if op == "<=":
        return float(dist.cdf(x))
    if op == "<":
        return float(dist.cdf(x - 1)) if discrete and float(x).is_integer() else float(dist.cdf(x))
    if op == ">":
        return float(dist.sf(x))
    if op == ">=":
        return float(dist.sf(x - 1)) if discrete and float(x).is_integer() else float(dist.sf(x))
    return float(dist.pmf(x)) if discrete else 0.0


def between(dist: Distribution, a: float, b: float) -> float:
    """P(a < X ≤ b)."""
    if b < a:
        raise ValueError("b must be at least a")
    return dist.interval_probability(a, b)


def quantile(dist: Distribution, p: float) -> float:
    """The p-quantile, e.g. p = 0.95 for the 95th percentile."""
    if not 0 < p < 1:
        raise ValueError("p must lie strictly between 0 and 1")
    return float(dist.quantile(p))
