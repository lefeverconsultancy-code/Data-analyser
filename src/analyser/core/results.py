"""Result objects: every fit, test and model returns one of these.

One result shape means one renderer per output (text, app, JSON). Every result
round-trips through JSON: ``Result.from_json(r.to_json()) == r``.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
import types
import typing
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

SCHEMA_VERSION = 1

# Status values for AssumptionCheck
PASSED = "passed"
FAILED = "failed"
WARNING = "warning"
NOT_RUN = "not_run"
INFO = "info"
CHECK_STATUSES = (PASSED, FAILED, WARNING, NOT_RUN, INFO)


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Small value objects
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class AssumptionCheck:
    """One assumption check, e.g. normality by Shapiro–Wilk.

    ``summary`` is the short form printed on the report's Checks line;
    ``detail`` explains the outcome and ``fallback`` names the method to use instead.
    """

    name: str
    method: str
    status: str
    summary: str
    statistic: float | None = None
    p_value: float | None = None
    rule: str | None = None
    detail: str | None = None
    fallback: str | None = None

    def __post_init__(self) -> None:
        if self.status not in CHECK_STATUSES:
            raise ValueError(f"status must be one of {CHECK_STATUSES}, got {self.status!r}")

    @property
    def ok(self) -> bool:
        return self.status in (PASSED, INFO)


@dataclass(kw_only=True)
class ConfidenceInterval:
    """A confidence interval or one-sided bound. ``None`` means unbounded on that side."""

    level: float
    lower: float | None
    upper: float | None
    parameter: str = "μ"
    method: str = ""

    @property
    def sides(self) -> str:
        if self.lower is not None and self.upper is not None:
            return "two-sided"
        return "lower bound" if self.lower is not None else "upper bound"

    def describe(self) -> str:
        """'95% lower bound for μ: 114.06' or '95% CI for μ: (110.2, 125.8)'."""
        from .formatting import fmt_pct, fmt_value

        level = fmt_pct(self.level)
        if self.sides == "two-sided":
            return f"{level} CI for {self.parameter}: ({fmt_value(self.lower)}, {fmt_value(self.upper)})"
        bound = self.lower if self.lower is not None else self.upper
        return f"{level} {self.sides} for {self.parameter}: {fmt_value(bound)}"


@dataclass(kw_only=True)
class EffectSize:
    name: str
    value: float
    formula: str = ""
    label: str = ""

    def describe(self) -> str:
        from .formatting import fmt_fixed

        lhs = f"{self.name} = {self.formula}" if self.formula else self.name
        return f"{lhs} = {fmt_fixed(self.value, 2)}"


@dataclass(kw_only=True)
class GoodnessOfFit:
    test: str
    statistic: float
    p_value: float | None
    df: float | None = None
    note: str = ""


@dataclass(kw_only=True)
class Coefficient:
    name: str
    estimate: float
    se: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    statistic: float | None = None
    p_value: float | None = None


# ---------------------------------------------------------------------------
# Result base class and JSON round-trip
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class Result:
    """Fields shared by every result: identity, inputs, settings snapshot, notes."""

    kind: ClassVar[str] = "result"

    name: str
    id: str = field(default_factory=_new_id)
    created_at: str = field(default_factory=_now)
    inputs: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "schema_version": SCHEMA_VERSION}
        for f in dataclasses.fields(self):
            out[f.name] = _encode(getattr(self, f.name))
        return out

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Result":
        kind = data.get("kind")
        target = RESULT_TYPES.get(kind, cls) if cls is Result else cls
        if target.kind != kind:
            raise ValueError(f"Cannot load a {kind!r} result as {target.__name__}")
        return _decode_dataclass(target, data)

    @classmethod
    def from_json(cls, text: str) -> "Result":
        return cls.from_dict(json.loads(text))

    def summary(self) -> str:
        return self.name


@dataclass(kw_only=True)
class TestResult(Result):
    """A hypothesis test, with the 8 steps of the Statistics for Engineers reader as fields."""

    __test__ = False  # not a pytest test class
    kind: ClassVar[str] = "test"

    test_id: str
    model: str  # step 1
    h0: str  # step 2
    h1: str
    alternative: str
    statistic_name: str  # step 3, e.g. "T = (X̄ − 115) / (S/√n)"
    statistic_symbol: str  # "T"
    null_distribution: str  # step 4, e.g. "t(15)"
    observed: float  # step 5
    observed_symbol: str  # "t"
    rejection_region: str  # step 6a, e.g. "T ≥ 1.753  (equivalently x̄ ≥ 118.94)"
    p_value: float  # step 6b
    p_value_expression: str  # "P(T ≥ 1.333 | H0)"
    alpha: float
    reject: bool  # step 7
    conclusion: str  # step 8, plain English
    null_is_approximate: bool = False
    critical_values: dict[str, float] = field(default_factory=dict)
    ci: ConfidenceInterval | None = None
    effect_size: EffectSize | None = None
    checks: list[AssumptionCheck] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if "accept" in self.conclusion.lower():
            raise ValueError("A conclusion never 'accepts' H0; use 'fail to reject'.")

    @property
    def decision(self) -> str:
        return "Reject H0" if self.reject else "Fail to reject H0"

    def summary(self) -> str:
        from .formatting import fmt_stat, p_equals

        return (
            f"{self.name}: {self.observed_symbol} = {fmt_stat(self.observed)}, "
            f"p {p_equals(self.p_value)} → {self.decision[0].lower()}{self.decision[1:]}"
        )


@dataclass(kw_only=True)
class FitResult(Result):
    """A fitted distribution: estimates with SEs and CIs, likelihood, AIC/BIC, goodness of fit."""

    kind: ClassVar[str] = "fit"

    distribution: str  # registry id, e.g. "normal"
    distribution_label: str  # "N(μ, σ²)"
    method: str  # "mle" | "mom"
    params: dict[str, float]
    param_symbols: dict[str, str] = field(default_factory=dict)
    param_labels: dict[str, str] = field(default_factory=dict)
    standard_errors: dict[str, float] = field(default_factory=dict)
    cis: dict[str, ConfidenceInterval] = field(default_factory=dict)
    n: int = 0
    k: int = 0
    loglik: float | None = None
    aic: float | None = None
    bic: float | None = None
    gof: list[GoodnessOfFit] = field(default_factory=list)
    alt_estimates: dict[str, dict[str, float]] = field(default_factory=dict)
    checks: list[AssumptionCheck] = field(default_factory=list)

    def fitted(self):
        """The fitted Distribution object."""
        from .registry import get_distribution

        return get_distribution(self.distribution)(**self.params)

    def summary(self) -> str:
        from .formatting import fmt_value

        est = ", ".join(
            f"{self.param_symbols.get(k, k)} = {fmt_value(v)}" for k, v in self.params.items()
        )
        return f"{self.name}: {est}; AIC = {fmt_value(self.aic)}"


@dataclass(kw_only=True)
class ModelResult(Result):
    """A statistical model (regression, ANOVA): coefficients, ANOVA table, fit measures, diagnostics."""

    kind: ClassVar[str] = "model"

    model_id: str
    formula: str
    response: str = ""
    predictors: list[str] = field(default_factory=list)
    coefficients: list[Coefficient] = field(default_factory=list)
    anova_table: list[dict[str, Any]] = field(default_factory=list)
    r_squared: float | None = None
    adj_r_squared: float | None = None
    n: int = 0
    residuals: list[float] = field(default_factory=list)
    fitted_values: list[float] = field(default_factory=list)
    diagnostics: list[AssumptionCheck] = field(default_factory=list)
    conclusion: str = ""

    def summary(self) -> str:
        from .formatting import fmt_fixed

        r2 = f"; R² = {fmt_fixed(self.r_squared, 3)}" if self.r_squared is not None else ""
        return f"{self.name}: {self.formula}{r2}"


RESULT_TYPES: dict[str, type[Result]] = {
    TestResult.kind: TestResult,
    FitResult.kind: FitResult,
    ModelResult.kind: ModelResult,
}


def result_from_dict(data: dict[str, Any]) -> Result:
    """Load any result object from its dict form, using its 'kind'."""
    return Result.from_dict(data)


def result_from_json(text: str) -> Result:
    return Result.from_json(text)


# ---------------------------------------------------------------------------
# Encoding helpers: numpy types, non-finite floats, nested dataclasses
# ---------------------------------------------------------------------------

_FLOAT_TAG = "__float__"


def _encode(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _encode(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if math.isnan(x):
            return {_FLOAT_TAG: "nan"}
        if math.isinf(x):
            return {_FLOAT_TAG: "inf" if x > 0 else "-inf"}
        return x
    if isinstance(value, np.ndarray):
        return [_encode(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    return value


def _decode_plain(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {_FLOAT_TAG}:
            return float(value[_FLOAT_TAG])
        return {k: _decode_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_plain(v) for v in value]
    return value


def _decode(value: Any, hint: Any) -> Any:
    if value is None:
        return None
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        for a in non_none:
            if dataclasses.is_dataclass(a) and isinstance(value, dict):
                return _decode_dataclass(a, value)
        return _decode(value, non_none[0]) if len(non_none) == 1 else _decode_plain(value)
    if dataclasses.is_dataclass(hint) and isinstance(value, dict):
        return _decode_dataclass(hint, value)
    if origin is list and args:
        return [_decode(v, args[0]) for v in value]
    if origin is dict and len(args) == 2:
        return {k: _decode(v, args[1]) for k, v in value.items()}
    if hint is float:
        decoded = _decode_plain(value)
        return float(decoded) if decoded is not None else None
    return _decode_plain(value)


def _decode_dataclass(cls: type, data: dict[str, Any]) -> Any:
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            kwargs[f.name] = _decode(data[f.name], hints.get(f.name, Any))
    return cls(**kwargs)
