"""Distribution base class: family, parameters, support, pdf/pmf, cdf, quantile,
mean, variance and a seeded random sampler, on top of scipy.stats."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..core.formatting import fmt_num


class Distribution:
    """A fully specified distribution. Subclasses set the class attributes and ``_scipy``."""

    registry_id: ClassVar[str] = ""
    family: ClassVar[str] = "continuous"  # or "discrete"
    param_names: ClassVar[tuple[str, ...]] = ()
    param_symbols: ClassVar[dict[str, str]] = {}
    notation: ClassVar[str] = ""  # e.g. "N" -> "N(80, 64)"

    def __init__(self, **params: float) -> None:
        missing = [p for p in self.param_names if p not in params]
        extra = [p for p in params if p not in self.param_names]
        if missing or extra:
            raise TypeError(
                f"{type(self).__name__} takes parameters {self.param_names}; "
                f"missing {missing}, unexpected {extra}"
            )
        self.params: dict[str, float] = {k: float(params[k]) for k in self.param_names}
        self.validate()
        self._frozen = self._scipy()

    # -- subclass hooks ---------------------------------------------------------
    def validate(self) -> None:
        """Raise ValueError for invalid parameter values."""

    def _scipy(self):  # pragma: no cover - abstract
        raise NotImplementedError

    @classmethod
    def mle(cls, x: np.ndarray) -> dict[str, Any]:  # pragma: no cover - abstract
        """Closed-form MLE: {'params': ..., 'se': ..., 'labels': ...}."""
        raise NotImplementedError(f"{cls.__name__} has no closed-form MLE yet")

    # -- common interface --------------------------------------------------------
    @property
    def label(self) -> str:
        args = ", ".join(fmt_num(v, 6) for v in self.params.values())
        return f"{self.notation}({args})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{k}={v:g}' for k, v in self.params.items())})"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.params == other.params  # type: ignore[attr-defined]

    def pdf(self, x):
        if self.family == "discrete":
            return self._frozen.pmf(x)
        return self._frozen.pdf(x)

    pmf = pdf

    def logpdf(self, x):
        if self.family == "discrete":
            return self._frozen.logpmf(x)
        return self._frozen.logpdf(x)

    def cdf(self, x):
        return self._frozen.cdf(x)

    def sf(self, x):
        return self._frozen.sf(x)

    def quantile(self, q):
        return self._frozen.ppf(q)

    ppf = quantile

    def mean(self) -> float:
        return float(self._frozen.mean())

    def var(self) -> float:
        return float(self._frozen.var())

    def sd(self) -> float:
        return float(self._frozen.std())

    def support(self) -> tuple[float, float]:
        lo, hi = self._frozen.support()
        return float(lo), float(hi)

    def interval_probability(self, a: float, b: float) -> float:
        """P(a < X ≤ b)."""
        return float(self.cdf(b) - self.cdf(a))

    def rvs(self, size: int | tuple[int, ...] = 1, seed: int | np.random.Generator | None = None):
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        return self._frozen.rvs(size=size, random_state=rng)

    def to_dict(self) -> dict[str, Any]:
        return {"distribution": self.registry_id, "params": dict(self.params)}
