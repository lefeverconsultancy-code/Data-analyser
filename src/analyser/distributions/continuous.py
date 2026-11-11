"""Continuous distributions fitted to data. Phase 0: the normal distribution."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from ..core.registry import register_distribution
from .base import Distribution


@register_distribution(
    "normal",
    name="Normal",
    family="continuous",
    params=("mu", "sigma2"),
    support="all reals",
    notation="N(μ, σ²)",
    use="Measurements, process control",
    estimator="x̄; S² (unbiased) or (n−1)S²/n (MLE)",
    sources=("PrEE 6.5", "L&M 4.3"),
)
class Normal(Distribution):
    """N(μ, σ²), parameterised by the variance as in the readers."""

    family = "continuous"
    param_names = ("mu", "sigma2")
    param_symbols = {"mu": "μ", "sigma2": "σ²"}
    notation = "N"

    def __init__(self, mu: float = 0.0, sigma2: float = 1.0) -> None:
        super().__init__(mu=mu, sigma2=sigma2)

    def validate(self) -> None:
        if not math.isfinite(self.params["mu"]):
            raise ValueError("μ must be finite")
        if not (self.params["sigma2"] > 0 and math.isfinite(self.params["sigma2"])):
            raise ValueError("σ² must be positive and finite")

    def _scipy(self):
        return stats.norm(loc=self.params["mu"], scale=math.sqrt(self.params["sigma2"]))

    @classmethod
    def mle(cls, x: np.ndarray) -> dict[str, Any]:
        """μ̂ = x̄ and σ̂² = (1/n)Σ(xᵢ − x̄)², with standard errors from the Fisher information."""
        x = np.asarray(x, dtype=float)
        n = x.size
        if n < 2:
            raise ValueError("A normal fit needs at least 2 observations")
        mu = float(x.mean())
        sigma2 = float(((x - mu) ** 2).mean())
        if sigma2 == 0:
            raise ValueError("All observations are equal; σ² would be 0")
        return {
            "params": {"mu": mu, "sigma2": sigma2},
            "se": {"mu": math.sqrt(sigma2 / n), "sigma2": sigma2 * math.sqrt(2.0 / n)},
            "labels": {"mu": "μ (MLE = x̄)", "sigma2": "σ² (MLE, divisor n)"},
            "alt": {"unbiased": {"mu": mu, "sigma2": float(x.var(ddof=1))}},
        }
