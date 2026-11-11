"""Session: settings, the ordered history of results, and the datasets in use.

    s = Session(alpha=0.05)
    s.add_dataset(ds)
    r = s.run_test("t_one_sample", ds.var("pick_time_min").numeric(), mu0=12)
    s.save("analysis.json");  s2 = Session.load("analysis.json")
"""

from __future__ import annotations

import datetime as _dt
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import TIME_UNITS, Dataset
from .registry import REGISTRY
from .results import Result, result_from_dict

SESSION_SCHEMA = 1

# Quartile methods: name → numpy method. Excel QUARTILE.INC is numpy's "linear" (the default).
QUARTILE_METHODS = {"QUARTILE.INC": "linear", "QUARTILE.EXC": "weibull"}


@dataclass
class Settings:
    alpha: float = 0.05
    quartile_method: str = "QUARTILE.INC"
    seed: int = 20261111
    time_unit: str = "min"
    currency: str = "EUR"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("α must lie strictly between 0 and 1")
        if self.quartile_method not in QUARTILE_METHODS:
            raise ValueError(f"quartile method must be one of {tuple(QUARTILE_METHODS)}")
        if self.time_unit not in TIME_UNITS:
            raise ValueError(f"time unit must be one of {tuple(TIME_UNITS)}")
        if not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a non-negative whole number")
        if not self.currency or len(self.currency) > 5:
            raise ValueError("currency must be a short code such as EUR")

    @property
    def numpy_quantile_method(self) -> str:
        return QUARTILE_METHODS[self.quartile_method]

    @property
    def confidence(self) -> float:
        return 1 - self.alpha


class Session:
    def __init__(self, settings: Settings | None = None, **overrides: Any) -> None:
        self.settings = settings or Settings(**overrides)
        if settings is not None and overrides:
            for k, v in overrides.items():
                setattr(self.settings, k, v)
            self.settings.validate()
        self.results: list[Result] = []
        self.datasets: dict[str, Dataset] = {}
        self.created_at = _dt.datetime.now().astimezone().isoformat(timespec="seconds")

    # -- settings ---------------------------------------------------------------
    def update_settings(self, **changes: Any) -> None:
        candidate = Settings(**{**asdict(self.settings), **changes})
        self.settings = candidate

    def rng(self, offset: int = 0) -> np.random.Generator:
        """A generator seeded from the session seed (plus an offset per use)."""
        return np.random.default_rng(self.settings.seed + offset)

    def quantiles(self, x, q) -> np.ndarray:
        """Quantiles with the session's quartile method (QUARTILE.INC by default)."""
        return np.quantile(np.asarray(x, dtype=float), q, method=self.settings.numpy_quantile_method)

    # -- datasets ---------------------------------------------------------------
    def add_dataset(self, ds: Dataset, *, replace: bool = True) -> Dataset:
        if ds.name in self.datasets and not replace:
            raise ValueError(f"a dataset named {ds.name!r} already exists")
        self.datasets[ds.name] = ds
        return ds

    # -- results ----------------------------------------------------------------
    def add(self, result: Result) -> Result:
        """Append a result, stamping the session settings onto it."""
        result.settings = {**asdict(self.settings), **result.settings}
        self.results.append(result)
        return result

    def run_test(self, test_id: str, *args: Any, **kwargs: Any) -> Result:
        """Run a registered test with the session α (unless given) and record the result."""
        func = REGISTRY.get("test", test_id).obj
        if "alpha" in inspect.signature(func).parameters:
            kwargs.setdefault("alpha", self.settings.alpha)
        return self.add(func(*args, **kwargs))

    def run_model(self, model_id: str, *args: Any, **kwargs: Any) -> Result:
        """Run a registered model or fit (confidence level 1 − α unless given) and record it."""
        func = REGISTRY.get("model", model_id).obj
        if "level" in inspect.signature(func).parameters:
            kwargs.setdefault("level", self.settings.confidence)
        return self.add(func(*args, **kwargs))

    def get(self, result_id: str) -> Result:
        for r in self.results:
            if r.id == result_id:
                return r
        raise KeyError(result_id)

    def remove(self, result_id: str) -> None:
        self.results = [r for r in self.results if r.id != result_id]

    def history(self) -> pd.DataFrame:
        rows = [{"#": i + 1, "id": r.id, "kind": r.kind, "name": r.name, "created": r.created_at,
                 "summary": r.summary()} for i, r in enumerate(self.results)]
        return pd.DataFrame(rows, columns=["#", "id", "kind", "name", "created", "summary"])

    # -- save and reload --------------------------------------------------------
    def to_dict(self, include_data: bool = True) -> dict[str, Any]:
        return {
            "schema_version": SESSION_SCHEMA,
            "created_at": self.created_at,
            "saved_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "settings": asdict(self.settings),
            "results": [r.to_dict() for r in self.results],
            "datasets": {k: d.to_dict() for k, d in self.datasets.items()} if include_data else {},
        }

    def to_json(self, include_data: bool = True) -> str:
        return json.dumps(self.to_dict(include_data), ensure_ascii=False, allow_nan=False)

    def save(self, path: str | Path, include_data: bool = True) -> Path:
        path = Path(path)
        path.write_text(self.to_json(include_data), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Session":
        if d.get("schema_version") != SESSION_SCHEMA:
            raise ValueError(f"unsupported session file version {d.get('schema_version')!r}")
        s = cls(Settings(**d["settings"]))
        s.created_at = d.get("created_at", s.created_at)
        s.results = [result_from_dict(r) for r in d.get("results", [])]
        s.datasets = {k: Dataset.from_dict(v) for k, v in d.get("datasets", {}).items()}
        return s

    @classmethod
    def from_json(cls, text: str) -> "Session":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: str | Path) -> "Session":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
