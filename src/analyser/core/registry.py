"""Registry of distributions, tests and models.

Every engine function or class registers itself with metadata (data type, samples,
paired, assumptions, parameters). The guided selector, the app's menus and the docs
read this registry, so adding a test means adding one decorated function.

    @register_test(id="t_one_sample", name="One-sample t-test on μ", samples=1,
                   data="continuous", assumptions=["random_sample", "normal_or_n_ge_40"])
    def t_one_sample(x, mu0, alternative="two-sided", alpha=0.05) -> TestResult: ...
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

KINDS = ("distribution", "test", "model")

# Engine modules that register things on import. discover() imports them so that
# list_registered() is complete no matter which modules the caller imported first.
ENGINE_MODULES = (
    "analyser.distributions.continuous",
    "analyser.inference.one_sample",
    "analyser.inference.categorical",
    "analyser.inference.nonparametric",
    "analyser.fitting.mle",
)

DATA_TYPES = ("continuous", "discrete", "counts", "binary", "categorical", "any")


@dataclass(frozen=True)
class RegistryEntry:
    kind: str
    id: str
    name: str
    obj: Any = field(compare=False, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def qualname(self) -> str:
        return f"{getattr(self.obj, '__module__', '?')}.{getattr(self.obj, '__qualname__', '?')}"

    def as_row(self) -> dict[str, Any]:
        row = {"kind": self.kind, "id": self.id, "name": self.name, "object": self.qualname}
        row.update(self.metadata)
        return row


class Registry:
    def __init__(self) -> None:
        self._entries: dict[str, dict[str, RegistryEntry]] = {k: {} for k in KINDS}

    def register(self, kind: str, id: str, name: str, obj: Any, **metadata: Any) -> RegistryEntry:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not id or not id.replace("_", "").isalnum():
            raise ValueError(f"Registry id must be a non-empty identifier, got {id!r}")
        entry = RegistryEntry(kind=kind, id=id, name=name, obj=obj, metadata=dict(metadata))
        existing = self._entries[kind].get(id)
        if existing is not None and existing.qualname != entry.qualname:
            raise ValueError(
                f"{kind} id {id!r} is already registered by {existing.qualname}"
            )
        self._entries[kind][id] = entry  # same object re-imported (e.g. app reload): replace
        return entry

    def get(self, kind: str, id: str) -> RegistryEntry:
        try:
            return self._entries[kind][id]
        except KeyError:
            discover()
            if id in self._entries[kind]:
                return self._entries[kind][id]
            known = ", ".join(sorted(self._entries[kind])) or "none"
            raise KeyError(f"No {kind} registered as {id!r}. Registered: {known}") from None

    def entries(self, kind: str | None = None) -> list[RegistryEntry]:
        kinds: Iterable[str] = KINDS if kind is None else (kind,)
        out: list[RegistryEntry] = []
        for k in kinds:
            if k not in KINDS:
                raise ValueError(f"kind must be one of {KINDS}")
            out.extend(sorted(self._entries[k].values(), key=lambda e: e.id))
        return out

    def clear(self) -> None:  # used by tests only
        for k in KINDS:
            self._entries[k].clear()


REGISTRY = Registry()


def discover() -> None:
    """Import all engine modules so every decorated object is registered."""
    for module in ENGINE_MODULES:
        importlib.import_module(module)


def _as_tuple(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def register_distribution(
    id: str,
    *,
    name: str,
    family: str = "continuous",
    params: Iterable[str],
    support: str,
    notation: str = "",
    use: str = "",
    sources: Iterable[str] = (),
    **metadata: Any,
) -> Callable:
    """Register a Distribution class. family: 'continuous', 'discrete' or 'sampling'."""
    if family not in ("continuous", "discrete", "sampling"):
        raise ValueError("family must be 'continuous', 'discrete' or 'sampling'")

    def decorator(cls):
        REGISTRY.register(
            "distribution", id, name, cls,
            family=family, params=_as_tuple(params), support=support,
            notation=notation, use=use, sources=_as_tuple(sources), **metadata,
        )
        cls.registry_id = id
        return cls

    return decorator


def register_test(
    id: str,
    *,
    name: str,
    samples: int | str,
    data: str,
    paired: bool = False,
    parameter: str | None = None,
    alternatives: Iterable[str] = ("two-sided", "less", "greater"),
    assumptions: Iterable[str] = (),
    params: Iterable[str] = (),
    sources: Iterable[str] = (),
    **metadata: Any,
) -> Callable:
    """Register a test function that returns a TestResult."""
    if data not in DATA_TYPES:
        raise ValueError(f"data must be one of {DATA_TYPES}")
    if not (samples in (1, 2) or samples == "k"):
        raise ValueError("samples must be 1, 2 or 'k'")

    def decorator(func):
        REGISTRY.register(
            "test", id, name, func,
            samples=samples, data=data, paired=paired, parameter=parameter,
            alternatives=_as_tuple(alternatives), assumptions=_as_tuple(assumptions),
            params=_as_tuple(params), sources=_as_tuple(sources), **metadata,
        )
        func.registry_id = id
        return func

    return decorator


def register_model(
    id: str,
    *,
    name: str,
    data: str = "continuous",
    response: str = "continuous",
    predictors: str = "any",
    assumptions: Iterable[str] = (),
    params: Iterable[str] = (),
    result: str = "model",
    sources: Iterable[str] = (),
    **metadata: Any,
) -> Callable:
    """Register a model (statistical model returning ModelResult, or a fit returning FitResult)."""
    if result not in ("model", "fit"):
        raise ValueError("result must be 'model' or 'fit'")

    def decorator(func):
        REGISTRY.register(
            "model", id, name, func,
            data=data, response=response, predictors=predictors,
            assumptions=_as_tuple(assumptions), params=_as_tuple(params),
            result=result, sources=_as_tuple(sources), **metadata,
        )
        func.registry_id = id
        return func

    return decorator


def list_registered(kind: str | None = None, *, as_frame: bool = False):
    """Everything registered, optionally of one kind ('distribution', 'test', 'model').

    Returns a list of dicts, or a pandas DataFrame with ``as_frame=True``.
    """
    discover()
    rows = [e.as_row() for e in REGISTRY.entries(kind)]
    if as_frame:
        import pandas as pd

        return pd.DataFrame(rows)
    return rows


def get_test(id: str) -> Callable:
    return REGISTRY.get("test", id).obj


def get_distribution(id: str):
    return REGISTRY.get("distribution", id).obj


def get_model(id: str) -> Callable:
    return REGISTRY.get("model", id).obj
