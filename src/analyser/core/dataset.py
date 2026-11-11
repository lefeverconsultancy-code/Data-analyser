"""Dataset: the data table plus column metadata (kind, role, unit), source and cleaning log."""

from __future__ import annotations

import copy as _copy
import datetime as _dt
import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .variable import ANALYSIS_ROLES, Variable

# Column roles assigned at import (brief: flexible data input)
ROLES = ("id", "timestamp", "start", "end", "duration", "count", "amount", "category", "resource")
ROLE_LABELS = {
    "id": "ID",
    "timestamp": "timestamp",
    "start": "start time",
    "end": "end time",
    "duration": "duration",
    "count": "count",
    "amount": "amount (money)",
    "category": "category",
    "resource": "resource",
    None: "(not used)",
}
AMOUNT_KINDS = ("revenue", "cost", "profit", "price", "other")
KINDS = ("numeric", "datetime", "time", "text", "empty")
TIME_UNITS = {"s": 1.0, "min": 60.0, "h": 3600.0}
TIME_UNIT_LABELS = {"s": "seconds", "min": "minutes", "h": "hours"}


def timedelta_to_unit(values: pd.Series, unit: str) -> pd.Series:
    """Convert a timedelta Series to floats in the given time unit."""
    if unit not in TIME_UNITS:
        raise ValueError(f"time unit must be one of {tuple(TIME_UNITS)}")
    return values.dt.total_seconds() / TIME_UNITS[unit]


def convert_time_unit(values: pd.Series, from_unit: str, to_unit: str) -> pd.Series:
    return values * TIME_UNITS[from_unit] / TIME_UNITS[to_unit]


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class CleaningStep:
    step: int
    action: str
    detail: str
    column: str | None = None
    rows_affected: int | None = None
    at: str = field(default_factory=_now)


class CleaningLog:
    """Ordered record of every read, parse, mapping, derivation and cleaning step."""

    def __init__(self, steps: list[CleaningStep] | None = None) -> None:
        self.steps: list[CleaningStep] = list(steps or [])

    def add(self, action: str, detail: str, column: str | None = None,
            rows_affected: int | None = None) -> CleaningStep:
        step = CleaningStep(step=len(self.steps) + 1, action=action, detail=detail,
                            column=column, rows_affected=rows_affected)
        self.steps.append(step)
        return step

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[CleaningStep]:
        return iter(self.steps)

    def actions(self) -> list[str]:
        return [s.action for s in self.steps]

    def to_list(self) -> list[dict[str, Any]]:
        return [asdict(s) for s in self.steps]

    @classmethod
    def from_list(cls, items: list[dict[str, Any]]) -> "CleaningLog":
        return cls([CleaningStep(**i) for i in items])

    def to_frame(self) -> pd.DataFrame:
        cols = ["step", "action", "column", "detail", "rows_affected", "at"]
        return pd.DataFrame(self.to_list(), columns=cols)


@dataclass
class ColumnInfo:
    name: str
    kind: str
    role: str | None = None
    role_confirmed: bool = False
    unit: str | None = None
    amount_kind: str | None = None
    original_name: str | None = None
    derived: bool = False
    formula: str | None = None
    role_reason: str | None = None
    parse_info: dict[str, Any] = field(default_factory=dict)


class Dataset:
    """A clean, typed table with column metadata, ready for analysis.

    Columns hold numbers (float), date-times, times of day (timedelta) or text. Durations
    are stored as numbers in the dataset's time unit; amounts in its currency.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        name: str = "dataset",
        columns: dict[str, ColumnInfo] | None = None,
        raw: pd.DataFrame | None = None,
        source: dict[str, Any] | None = None,
        log: CleaningLog | None = None,
        time_unit: str = "min",
        currency: str = "EUR",
        mapping: Any = None,
        derivations: list[str] | None = None,
    ) -> None:
        if time_unit not in TIME_UNITS:
            raise ValueError(f"time_unit must be one of {tuple(TIME_UNITS)}")
        self.data = data.reset_index(drop=True)
        self.name = name
        self.columns: dict[str, ColumnInfo] = columns or {
            c: ColumnInfo(name=c, kind=_infer_kind(self.data[c])) for c in self.data.columns
        }
        for c in self.data.columns:
            self.columns.setdefault(c, ColumnInfo(name=c, kind=_infer_kind(self.data[c])))
        self.raw = raw
        self.source = dict(source or {})
        self.log = log or CleaningLog()
        self.time_unit = time_unit
        self.currency = currency
        self.mapping = mapping
        self.derivations: list[str] = list(derivations or [])

    # -- basics -----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"Dataset({self.name!r}, {len(self)} rows × {self.data.shape[1]} columns)"

    @property
    def column_names(self) -> list[str]:
        return list(self.data.columns)

    def copy(self, name: str | None = None) -> "Dataset":
        new = _copy.deepcopy(self)
        if name:
            new.name = name
        return new

    def data_hash(self) -> str:
        h = pd.util.hash_pandas_object(self.data, index=False).values
        return hashlib.sha256(h.tobytes()).hexdigest()[:16]

    # -- roles --------------------------------------------------------------------
    def role_of(self, column: str) -> str | None:
        return self.columns[column].role

    def columns_with_role(self, role: str, *, amount_kind: str | None = None) -> list[str]:
        out = [c for c, info in self.columns.items() if info.role == role and c in self.data]
        if amount_kind is not None:
            out = [c for c in out if self.columns[c].amount_kind == amount_kind]
        return out

    def first_with_role(self, role: str, *, amount_kind: str | None = None) -> str | None:
        cols = self.columns_with_role(role, amount_kind=amount_kind)
        return cols[0] if cols else None

    def numeric_columns(self) -> list[str]:
        return [c for c in self.data.columns if self.columns[c].kind == "numeric"]

    def time_column(self) -> str | None:
        """The arrival/creation timestamp, else the start time: the column that orders the rows."""
        return self.first_with_role("timestamp") or self.first_with_role("start")

    def is_time_ordered(self) -> str | None:
        """Name of the time column if the rows are sorted by it, else None."""
        col = self.time_column()
        if col is None:
            return None
        s = self.data[col].dropna()
        if len(s) > 2 and s.is_monotonic_increasing:
            return col
        return None

    # -- columns --------------------------------------------------------------------
    def add_column(
        self,
        name: str,
        values: pd.Series | np.ndarray | list,
        *,
        role: str | None = None,
        unit: str | None = None,
        formula: str | None = None,
        amount_kind: str | None = None,
        replace: bool = False,
    ) -> None:
        if name in self.data.columns and not replace:
            raise ValueError(f"column {name!r} already exists")
        series = pd.Series(values, index=self.data.index) if not isinstance(values, pd.Series) else values
        self.data[name] = series.reindex(self.data.index)
        self.columns[name] = ColumnInfo(
            name=name, kind=_infer_kind(self.data[name]), role=role, role_confirmed=True,
            unit=unit, amount_kind=amount_kind, derived=formula is not None, formula=formula,
        )
        missing = int(self.data[name].isna().sum())
        detail = f"{formula}" if formula else "column added"
        if missing:
            detail += f" ({missing} missing)"
        self.log.add("derive" if formula else "add_column", detail, column=name,
                     rows_affected=len(self.data) - missing)

    def drop_rows(self, mask: pd.Series, reason: str) -> int:
        mask = mask.reindex(self.data.index).fillna(False).astype(bool)
        count = int(mask.sum())
        if count:
            self.data = self.data.loc[~mask].reset_index(drop=True)
            if self.raw is not None and len(self.raw) == len(mask):
                self.raw = self.raw.loc[~mask.values].reset_index(drop=True)
        self.log.add("drop_rows", reason, rows_affected=count)
        return count

    def var(self, name: str, analysis_role: str | None = "response") -> Variable:
        """One column as a Variable, with its role, unit and time-order flag."""
        if name not in self.data.columns:
            raise KeyError(f"No column {name!r} in {self.name}; columns: {self.column_names}")
        if analysis_role is not None and analysis_role not in ANALYSIS_ROLES:
            raise ValueError(f"analysis_role must be one of {ANALYSIS_ROLES}")
        info = self.columns[name]
        return Variable(
            name=name, values=self.data[name], kind=info.kind, role=info.role,
            analysis_role=analysis_role, unit=info.unit, dataset=self.name,
            time_ordered_by=self.is_time_ordered(),
        )

    def describe_columns(self) -> pd.DataFrame:
        rows = []
        for c in self.data.columns:
            info = self.columns[c]
            s = self.data[c]
            rows.append({
                "column": c, "kind": info.kind, "role": ROLE_LABELS.get(info.role, info.role),
                "unit": info.unit or "", "missing": int(s.isna().sum()),
                "example": example_value(s), "derived": info.derived,
            })
        return pd.DataFrame(rows)

    # -- serialisation (used by Session save/reload) ----------------------------------
    def to_dict(self) -> dict[str, Any]:
        cols = {}
        for c in self.data.columns:
            kind = self.columns[c].kind
            s = self.data[c]
            if kind == "numeric":
                values = [None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)
                          for v in s.astype(float)]
            elif kind == "datetime":
                values = [None if pd.isna(v) else pd.Timestamp(v).isoformat() for v in s]
            elif kind == "time":
                values = [None if pd.isna(v) else pd.Timedelta(v).total_seconds() for v in s]
            else:
                values = [None if pd.isna(v) else str(v) for v in s]
            cols[c] = values
        return {
            "name": self.name,
            "time_unit": self.time_unit,
            "currency": self.currency,
            "order": list(self.data.columns),
            "data": cols,
            "columns": {c: asdict(i) for c, i in self.columns.items() if c in self.data.columns},
            "source": self.source,
            "log": self.log.to_list(),
            "mapping": self.mapping.to_dict() if hasattr(self.mapping, "to_dict") else self.mapping,
            "derivations": self.derivations,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Dataset":
        columns = {c: ColumnInfo(**i) for c, i in d["columns"].items()}
        frame = {}
        for c in d["order"]:
            kind = columns[c].kind
            values = d["data"][c]
            if kind == "numeric":
                frame[c] = pd.Series([np.nan if v is None else v for v in values], dtype=float)
            elif kind == "datetime":
                frame[c] = pd.to_datetime(pd.Series(values, dtype=object), format="ISO8601")
            elif kind == "time":
                frame[c] = pd.to_timedelta(pd.Series([np.nan if v is None else v for v in values],
                                                     dtype=float), unit="s")
            else:
                frame[c] = pd.Series([np.nan if v is None else v for v in values], dtype=object)
        mapping = d.get("mapping")
        if isinstance(mapping, dict):
            from ..io.mapping import ColumnMapping  # local import: io sits above core

            mapping = ColumnMapping.from_dict(mapping)
        return cls(
            pd.DataFrame(frame, columns=d["order"]), name=d["name"], columns=columns,
            source=d.get("source"), log=CleaningLog.from_list(d.get("log", [])),
            time_unit=d.get("time_unit", "min"), currency=d.get("currency", "EUR"),
            mapping=mapping, derivations=d.get("derivations"),
        )


def example_value(s: pd.Series) -> str:
    """The first non-missing value, shown compactly: 2 not 2.0, 9.13333 not 9.133333333333333."""
    present = s.dropna()
    if present.empty:
        return ""
    v = present.iloc[0]
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}"
    return str(v)


def _infer_kind(s: pd.Series) -> str:
    if s.isna().all():
        return "empty"
    if pd.api.types.is_bool_dtype(s):
        return "text"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_timedelta64_dtype(s):
        return "time"
    return "text"
