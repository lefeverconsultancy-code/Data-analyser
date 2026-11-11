"""Column roles: suggest them from names and content, then apply the confirmed mapping.

Roles: id, timestamp, start, end, duration, count, amount, category, resource.
Names are matched in English and Dutch (aantal, omzet, kosten, duur, medewerker, ...).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core.dataset import AMOUNT_KINDS, ROLES, TIME_UNITS, Dataset, convert_time_unit, timedelta_to_unit
from .clean import is_missing, parse_datetimes, parse_numbers

KEYWORDS: dict[str, tuple[str, ...]] = {
    "id": ("id", "nr", "no", "number", "nummer", "code", "key", "ref", "reference", "referentie"),
    "start": ("start", "begin", "started", "gestart", "aanvang"),
    "end": ("end", "eind", "einde", "finish", "finished", "stop", "klaar", "complete", "completed",
            "gereed", "done"),
    "timestamp": ("created", "time", "date", "datum", "tijd", "timestamp", "arrival", "aankomst",
                  "received", "ontvangen", "ordered", "besteld", "moment", "datetime", "at"),
    "duration": ("duration", "duur", "pick_time", "picktime", "process_time", "lead_time", "leadtime",
                 "cycle", "doorlooptijd", "seconds", "sec", "minutes", "min", "hours", "uren",
                 "hrs", "wait", "waiting", "wachttijd", "service"),
    "count": ("count", "qty", "quantity", "aantal", "orders", "lines", "regels", "orderregels", "picks",
              "units", "stuks", "items", "colli", "pallets", "n"),
    "amount": ("revenue", "omzet", "cost", "costs", "kosten", "price", "prijs", "amount", "bedrag",
               "profit", "winst", "margin", "marge", "eur", "euro", "sales", "turnover", "value",
               "waarde", "total", "totaal", "kostprijs"),
    "resource": ("picker", "employee", "medewerker", "operator", "dock", "machine", "resource",
                 "truck", "worker", "user", "gebruiker", "station", "werkplek", "driver", "chauffeur"),
    "category": ("channel", "kanaal", "type", "category", "categorie", "product", "region", "regio",
                 "status", "warehouse", "magazijn", "process", "proces", "segment", "klant",
                 "customer", "supplier", "leverancier", "zone", "group", "groep", "week", "month",
                 "maand", "day", "dag", "shift", "ploeg", "carrier", "vervoerder"),
}
AMOUNT_KEYWORDS = {
    "revenue": ("revenue", "omzet", "sales", "turnover", "opbrengst"),
    "cost": ("cost", "costs", "kosten", "kostprijs", "expense", "uitgaven"),
    "profit": ("profit", "winst", "margin", "marge"),
    "price": ("price", "prijs", "tarief", "rate"),
}
UNIT_KEYWORDS = {"s": ("sec", "seconds", "seconden", "_s"), "min": ("min", "minutes", "minuten"),
                 "h": ("hours", "hrs", "uren", "uur", "_h")}


def _words(name: str) -> list[str]:
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return [w for w in re.split(r"[^a-zA-Z0-9]+", name.lower()) if w]


def _has(name: str, keys: tuple[str, ...]) -> bool:
    words = _words(name)
    joined = "_".join(words)
    return any(k in words or (len(k) > 3 and k in joined) for k in keys)


@dataclass
class RoleSuggestion:
    role: str | None
    confidence: float
    reason: str
    amount_kind: str | None = None
    unit: str | None = None


def _unit_from_name(name: str) -> str | None:
    words = _words(name)
    joined = "_" + "_".join(words)
    for unit, keys in UNIT_KEYWORDS.items():
        if any(k.strip("_") in words or (k.startswith("_") and joined.endswith(k)) for k in keys):
            return unit
    return None


def _amount_kind(name: str) -> str:
    for kind, keys in AMOUNT_KEYWORDS.items():
        if _has(name, keys):
            return kind
    return "other"


def suggest_role(ds: Dataset, column: str) -> RoleSuggestion:
    info = ds.columns[column]
    s = ds.data[column]
    kind = info.kind
    n = int(s.notna().sum())
    unique = int(s.nunique(dropna=True))
    name = column

    if kind == "empty":
        return RoleSuggestion(None, 1.0, "column is empty")
    if kind in ("datetime", "time"):
        if _has(name, KEYWORDS["start"]):
            return RoleSuggestion("start", 0.9, "date-times; name says start")
        if _has(name, KEYWORDS["end"]):
            return RoleSuggestion("end", 0.9, "date-times; name says end")
        if kind == "time" and _has(name, KEYWORDS["duration"]):
            return RoleSuggestion("duration", 0.8, "h:mm values; name says duration")
        return RoleSuggestion("timestamp", 0.7 if _has(name, KEYWORDS["timestamp"]) else 0.5,
                              "date-time values")
    if kind == "numeric":
        p = info.parse_info
        if _has(name, KEYWORDS["id"]) and unique == n and _words(name)[-1] in KEYWORDS["id"]:
            return RoleSuggestion("id", 0.8, "unique numbers; name says ID")
        if p.get("currencies") or _has(name, KEYWORDS["amount"]):
            return RoleSuggestion("amount", 0.85, "money: currency sign or name", amount_kind=_amount_kind(name))
        if p.get("units") or _has(name, KEYWORDS["duration"]):
            unit = (p.get("units") or [None])[0] or _unit_from_name(name)
            return RoleSuggestion("duration", 0.8, "time values: unit or name", unit=unit or ds.time_unit)
        if _has(name, KEYWORDS["category"]) and unique <= 60:
            return RoleSuggestion("category", 0.6, "numbered groups (e.g. week); name says category")
        nonneg = bool((s.dropna() >= 0).all())
        if p.get("integer") and nonneg:
            if _has(name, KEYWORDS["count"]):
                return RoleSuggestion("count", 0.85, "whole numbers ≥ 0; name says count")
            return RoleSuggestion("count", 0.4, "whole numbers ≥ 0")
        return RoleSuggestion(None, 0.3, "numbers without a clear role")
    # text
    if _has(name, KEYWORDS["id"]) and unique >= 0.9 * n:
        return RoleSuggestion("id", 0.85, "mostly unique text; name says ID")
    if _has(name, KEYWORDS["resource"]):
        return RoleSuggestion("resource", 0.8, "name says person, dock or machine")
    if unique >= 0.9 * n and n > 20:
        return RoleSuggestion("id", 0.6, "mostly unique text")
    if unique <= max(30, 0.2 * n) or _has(name, KEYWORDS["category"]):
        return RoleSuggestion("category", 0.7, f"text with {unique} distinct values")
    return RoleSuggestion(None, 0.2, "free text")


def suggest_roles(ds: Dataset) -> dict[str, RoleSuggestion]:
    out = {c: suggest_role(ds, c) for c in ds.data.columns}
    # Only one arrival timestamp: when start and end exist, another date-time stays 'timestamp'.
    return out


@dataclass
class ColumnMapping:
    """The confirmed role per column, plus duration source units and amount kinds."""

    roles: dict[str, str | None] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)  # source unit of numeric durations
    amount_kinds: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ColumnMapping":
        return cls(roles=dict(d.get("roles", {})), units=dict(d.get("units", {})),
                   amount_kinds=dict(d.get("amount_kinds", {})))

    @classmethod
    def from_suggestions(cls, ds: Dataset) -> "ColumnMapping":
        m = cls()
        for c, info in ds.columns.items():
            if c not in ds.data.columns:
                continue
            m.roles[c] = info.role
            if info.role == "duration" and info.kind == "numeric":
                m.units[c] = info.unit or ds.time_unit
            if info.role == "amount":
                m.amount_kinds[c] = info.amount_kind or "other"
        return m


def _raw(ds: Dataset, col: str) -> pd.Series | None:
    if ds.raw is not None and col in ds.raw.columns and len(ds.raw) == len(ds.data):
        return ds.raw[col]
    return None


def apply_mapping(ds: Dataset, mapping: ColumnMapping, *, time_unit: str | None = None,
                  currency: str | None = None, inplace: bool = False) -> Dataset:
    """Confirm roles and convert columns to match them. Every change is logged.

    - id, category, resource → text (from the original cells, so '000123' keeps its zeros)
    - timestamp, start, end → date-time (or time of day)
    - duration → number in the dataset's time unit (converted from its source unit)
    - count, amount → number
    """
    out = ds if inplace else ds.copy()
    if time_unit:
        if time_unit not in TIME_UNITS:
            raise ValueError(f"time unit must be one of {tuple(TIME_UNITS)}")
        out.time_unit = time_unit
    if currency:
        out.currency = currency
    dayfirst = (out.source.get("options") or {}).get("dayfirst")
    for col, role in mapping.roles.items():
        if col not in out.data.columns:
            raise KeyError(f"mapping names column {col!r}, which is not in the data")
        if role is not None and role not in ROLES:
            raise ValueError(f"unknown role {role!r} for {col}; roles: {ROLES}")
        info = out.columns[col]
        s = out.data[col]
        raw = _raw(out, col)
        before_kind = info.kind
        if role in ("id", "category", "resource"):
            src = raw if raw is not None else s
            text = [np.nan if is_missing(v) else (
                str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v).strip())
                for v in src.tolist()]
            out.data[col] = pd.Series(text, index=out.data.index, dtype=object)
            info.kind = "text"
            info.unit = None
        elif role in ("timestamp", "start", "end"):
            if info.kind not in ("datetime", "time"):
                parsed = parse_datetimes(raw if raw is not None else s.astype(str), dayfirst)
                if parsed.info["parsed"] < 0.5 * max(1, parsed.info["non_missing"]):
                    raise ValueError(f"{col} does not contain date-times, so it cannot be a {role}")
                out.data[col] = parsed.values
                info.kind = "datetime"
                info.parse_info = parsed.info
        elif role == "duration":
            if info.kind == "time":
                out.data[col] = timedelta_to_unit(s, out.time_unit)
                info.kind = "numeric"
                out.log.add("convert", f"h:mm durations converted to {out.time_unit}", column=col)
            elif info.kind == "numeric":
                src_unit = mapping.units.get(col) or info.unit or out.time_unit
                if src_unit not in TIME_UNITS:
                    raise ValueError(f"unknown time unit {src_unit!r} for {col}")
                if src_unit != out.time_unit:
                    out.data[col] = convert_time_unit(s, src_unit, out.time_unit)
                    out.log.add("convert", f"converted from {src_unit} to {out.time_unit}", column=col)
            else:
                parsed = parse_numbers(raw if raw is not None else s.astype(str))
                if info.kind != "text" or parsed.info["parsed"] < 0.5 * max(1, parsed.info["non_missing"]):
                    raise ValueError(f"{col} is {info.kind}; a duration needs numbers or h:mm values")
                src_unit = mapping.units.get(col) or (parsed.info["units"] or [None])[0] or out.time_unit
                out.data[col] = convert_time_unit(parsed.values, src_unit, out.time_unit)
                info.kind = "numeric"
                info.parse_info = parsed.info
            info.unit = out.time_unit
        elif role in ("count", "amount"):
            if info.kind != "numeric":
                parsed = parse_numbers(raw if raw is not None else s.astype(str))
                if parsed.info["parsed"] < 0.5 * max(1, parsed.info["non_missing"]):
                    raise ValueError(f"{col} does not contain numbers, so it cannot be a {role}")
                out.data[col] = parsed.values
                info.kind = "numeric"
                info.parse_info = parsed.info
            if role == "amount":
                info.unit = out.currency
                info.amount_kind = mapping.amount_kinds.get(col) or info.amount_kind or "other"
                if info.amount_kind not in AMOUNT_KINDS:
                    raise ValueError(f"amount kind must be one of {AMOUNT_KINDS}")
            else:
                info.unit = None
        info.role = role
        info.role_confirmed = True
        change = f" ({before_kind} → {info.kind})" if before_kind != info.kind else ""
        extra = ""
        if role == "amount":
            extra = f", {info.amount_kind}, {out.currency}"
        elif role == "duration":
            extra = f", {out.time_unit}"
        out.log.add("set_role", f"role: {role or 'not used'}{extra}{change}", column=col)
    out.mapping = mapping
    return out
