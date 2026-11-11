"""Parsing raw text into typed columns (Dutch and English formats), data-quality checks,
and wide-to-long reshaping. Every change to a Dataset is written to its cleaning log."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core.dataset import ColumnInfo, Dataset
from ..core.results import FAILED, INFO, PASSED, WARNING, AssumptionCheck

MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "#n/a", "n.v.t.", "nvt", "?", "nat"}
NUMERIC_SHARE = 0.95  # a column is numeric if ≥ 95% of non-missing values parse as numbers

CURRENCY_TOKENS = {"€": "EUR", "eur": "EUR", "$": "USD", "usd": "USD", "£": "GBP", "gbp": "GBP"}
UNIT_TOKENS = {
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s", "seconde": "s", "seconden": "s",
    "min": "min", "mins": "min", "minute": "min", "minutes": "min", "minuut": "min", "minuten": "min",
    "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h", "uur": "h", "u": "h",
}
_CURRENCY_RE = re.compile(r"^(?:€|\$|£|eur|usd|gbp)\s*|\s*(?:€|\$|£|eur|usd|gbp)$", re.I)
_UNIT_RE = re.compile(r"^(.*?\d)\s*([a-z]+)\.?$", re.I)
_CORE_RE = re.compile(r"^\d[\d.,]*$|^[.,]\d+$")
DATE_RE = re.compile(
    r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?)?"
    r"(?:\s*(?:Z|[+-]\d{2}:?\d{2}))?$"
)
TIME_RE = re.compile(r"^\d{1,3}:\d{2}(?::\d{2}(?:[.,]\d+)?)?$")


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip().lower() in MISSING_TOKENS


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


@dataclass
class _Token:
    core: str
    negative: bool
    currency: str | None
    unit: str | None
    percent: bool


def _tokenise_number(text: str) -> _Token | None:
    t = text.strip().replace(" ", " ").replace(" ", " ")
    negative = False
    if t.startswith("(") and t.endswith(")"):
        negative, t = True, t[1:-1].strip()
    currency = None
    m = _CURRENCY_RE.search(t)
    if m:
        currency = CURRENCY_TOKENS[m.group(0).strip().lower()]
        t = _CURRENCY_RE.sub("", t).strip()
    if t[:1] in "-−+":
        negative = negative or t[0] in "-−"
        t = t[1:].strip()
        if not currency:  # "-€ 12,50"
            m = _CURRENCY_RE.search(t)
            if m:
                currency = CURRENCY_TOKENS[m.group(0).strip().lower()]
                t = _CURRENCY_RE.sub("", t).strip()
    percent = t.endswith("%")
    if percent:
        t = t[:-1].strip()
    unit = None
    m = _UNIT_RE.match(t)
    if m and m.group(2).lower() in UNIT_TOKENS:
        unit = UNIT_TOKENS[m.group(2).lower()]
        t = m.group(1)
    t = t.replace(" ", "").replace("'", "")
    if not _CORE_RE.match(t):
        return None
    return _Token(core=t, negative=negative, currency=currency, unit=unit, percent=percent)


def _decimal_vote(core: str) -> str | None:
    """'.' or ',' if this number shows which separator is the decimal one, else None."""
    has_dot, has_comma = "." in core, "," in core
    if has_dot and has_comma:
        return "," if core.rfind(",") > core.rfind(".") else "."
    for sep, other in ((",", "."), (".", ",")):
        if (sep == "," and has_comma) or (sep == "." and has_dot):
            if core.count(sep) > 1:
                return other  # repeated separator = thousands grouping
            after = core.split(sep)[1]
            if len(after) != 3 or core.startswith(sep):
                return sep  # "12,5" or "0.25": decimal
            return None  # "1,234" or "1.234": ambiguous
    return None


def detect_decimal(cores: list[str], hint: str | None = None) -> tuple[str, bool]:
    """Pick the decimal separator for a column. Returns (separator, was_ambiguous)."""
    votes = Counter(v for v in (_decimal_vote(c) for c in cores) if v)
    if votes:
        return votes.most_common(1)[0][0], False
    ambiguous = any(("," in c or "." in c) for c in cores)
    return (hint or "."), ambiguous


def _to_float(core: str, decimal: str) -> float:
    if decimal == ",":
        core = core.replace(".", "").replace(",", ".")
    else:
        core = core.replace(",", "")
    return float(core)


@dataclass
class ParsedColumn:
    kind: str
    values: pd.Series
    info: dict[str, Any] = field(default_factory=dict)


def parse_numbers(raw: pd.Series, decimal: str | None = None, decimal_hint: str | None = None) -> ParsedColumn:
    """Parse text such as '1.234,56', '€ 12,50', '(3,2)', '12 min' or '45%' to floats.

    ``decimal`` forces the decimal separator; otherwise it is detected per column, with
    ``decimal_hint`` (e.g. ',' for semicolon-separated files) for ambiguous columns.
    """
    tokens: list[_Token | None] = []
    missing = []
    for v in raw.tolist():
        if is_missing(v):
            tokens.append(None)
            missing.append(True)
        else:
            tokens.append(_tokenise_number(str(v)))
            missing.append(False)
    cores = [t.core for t in tokens if t is not None]
    ambiguous = False
    if decimal is None:
        decimal, ambiguous = detect_decimal(cores, decimal_hint)
    out = np.full(len(tokens), np.nan)
    failures: list[str] = []
    currencies, units = set(), set()
    percent = False
    for i, (tok, miss) in enumerate(zip(tokens, missing)):
        if miss:
            continue
        if tok is None:
            failures.append(str(raw.iloc[i]))
            continue
        try:
            x = _to_float(tok.core, decimal)
        except ValueError:
            failures.append(str(raw.iloc[i]))
            continue
        out[i] = -x if tok.negative else x
        if tok.currency:
            currencies.add(tok.currency)
        if tok.unit:
            units.add(tok.unit)
        percent = percent or tok.percent
    non_missing = len(tokens) - sum(missing)
    values = pd.Series(out, index=raw.index, dtype=float)
    finite = values.dropna()
    info = {
        "decimal": decimal,
        "decimal_ambiguous": ambiguous,
        "separators_seen": any(("," in c or "." in c) for c in cores),
        "non_missing": non_missing,
        "parsed": non_missing - len(failures),
        "failures": len(failures),
        "failure_examples": failures[:5],
        "currencies": sorted(currencies),
        "units": sorted(units),
        "percent": percent,
        "integer": bool(len(finite) and np.all(np.isclose(finite, np.round(finite)))),
    }
    return ParsedColumn(kind="numeric", values=values, info=info)


# ---------------------------------------------------------------------------
# Dates and times
# ---------------------------------------------------------------------------


def detect_dayfirst(values: list[str], hint: bool | None = None) -> tuple[bool, bool]:
    """Decide day-first vs month-first for dates like 03-11-2026. Returns (dayfirst, ambiguous)."""
    first_big = second_big = seen = False
    for v in values:
        parts = re.split(r"[-/.]", v.strip().split(" ")[0].split("T")[0])
        if len(parts) < 3 or len(parts[0]) == 4:
            continue  # ISO yyyy-mm-dd: no ambiguity
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        seen = True
        first_big |= a > 12
        second_big |= b > 12
    default = True if hint is None else hint
    if not seen:
        return default, False
    if first_big and not second_big:
        return True, False
    if second_big and not first_big:
        return False, False
    return default, True


def _date_formats(sample: str, dayfirst: bool) -> list[str]:
    date_part = re.split(r"[ T]", sample.strip())[0]
    sep = next((ch for ch in "-/." if ch in date_part), "-")
    parts = date_part.split(sep)
    if len(parts[0]) == 4:
        dates = [f"%Y{sep}%m{sep}%d"]
    else:
        year = "%Y" if len(parts[-1]) == 4 else "%y"
        dates = [f"%d{sep}%m{sep}{year}"] if dayfirst else [f"%m{sep}%d{sep}{year}"]
    times = ["", " %H:%M", " %H:%M:%S", "T%H:%M", "T%H:%M:%S", " %H:%M:%S.%f", "T%H:%M:%S.%f"]
    return [d + t for d in dates for t in times]


def parse_datetimes(raw: pd.Series, dayfirst: bool | None = None) -> ParsedColumn:
    """Parse date-time text. Day-first (Dutch, 03-11-2026 = 3 November) unless the data show otherwise."""
    s = raw.map(lambda v: None if is_missing(v) else str(v).strip()).astype(object)
    present = [v for v in s.tolist() if isinstance(v, str)]
    detected, ambiguous = detect_dayfirst(present, dayfirst)
    use_dayfirst = detected if dayfirst is None else dayfirst
    best: pd.Series | None = None
    best_ok, best_fmt = -1, None
    if present:
        for fmt in _date_formats(present[0], use_dayfirst):
            parsed = pd.to_datetime(s, format=fmt, errors="coerce")
            ok = int(parsed.notna().sum())
            if ok > best_ok:
                best, best_ok, best_fmt = parsed, ok, fmt
            if ok == len(present):
                break
    if best is None or best_ok < len(present):
        mixed = pd.to_datetime(s, format="mixed", dayfirst=use_dayfirst, errors="coerce")
        if best is None or int(mixed.notna().sum()) > best_ok:
            best, best_ok, best_fmt = mixed, int(mixed.notna().sum()), "mixed"
    if getattr(best.dt, "tz", None) is not None:
        best = best.dt.tz_convert(None)
    failures = [v for v, p in zip(s, best) if isinstance(v, str) and pd.isna(p)]
    info = {
        "dayfirst": use_dayfirst,
        "dayfirst_ambiguous": bool(ambiguous and dayfirst is None),
        "format": best_fmt,
        "non_missing": len(present),
        "parsed": best_ok,
        "failures": len(failures),
        "failure_examples": failures[:5],
    }
    return ParsedColumn(kind="datetime", values=best, info=info)


def parse_times(raw: pd.Series) -> ParsedColumn:
    """Parse times of day or durations written as h:mm or h:mm:ss into timedeltas."""
    def conv(v: Any) -> Any:
        if is_missing(v):
            return None
        t = str(v).strip().replace(",", ".")
        return t + ":00" if t.count(":") == 1 else t

    s = raw.map(conv)
    values = pd.to_timedelta(s, errors="coerce")
    present = int(sum(isinstance(c, str) for c in s))
    failures = [str(r) for r, v, c in zip(raw, values, s) if isinstance(c, str) and pd.isna(v)]
    return ParsedColumn(kind="time", values=values, info={
        "non_missing": present, "parsed": present - len(failures), "failures": len(failures),
        "failure_examples": failures[:5], "format": "h:mm[:ss]",
    })


def infer_column(raw: pd.Series, *, decimal: str | None = None, decimal_hint: str | None = None,
                 dayfirst: bool | None = None) -> ParsedColumn:
    """Detect the kind of a raw text column and parse it."""
    present = [str(v).strip() for v in raw.tolist() if not is_missing(v)]
    if not present:
        return ParsedColumn(kind="empty", values=pd.Series(np.nan, index=raw.index, dtype=float),
                            info={"non_missing": 0})
    n = len(present)
    if sum(bool(DATE_RE.match(v)) for v in present) >= NUMERIC_SHARE * n:
        parsed = parse_datetimes(raw, dayfirst)
        if parsed.info["parsed"] >= NUMERIC_SHARE * n:
            return parsed
    if sum(bool(TIME_RE.match(v)) for v in present) >= NUMERIC_SHARE * n:
        parsed = parse_times(raw)
        if parsed.info["parsed"] >= NUMERIC_SHARE * n:
            return parsed
    numbers = parse_numbers(raw, decimal=decimal, decimal_hint=decimal_hint)
    if numbers.info["parsed"] >= NUMERIC_SHARE * n:
        return numbers
    text = pd.Series([np.nan if is_missing(v) else str(v).strip() for v in raw.tolist()],
                     index=raw.index, dtype=object)
    return ParsedColumn(kind="text", values=text, info={
        "non_missing": n, "unique": len(set(present)),
        "numeric_share": numbers.info["parsed"] / n,
    })


def parse_frame(raw: pd.DataFrame, *, decimal: str | None = None, decimal_hint: str | None = None,
                dayfirst: bool | None = None) -> tuple[pd.DataFrame, dict[str, ColumnInfo], list[str]]:
    """Parse every column; returns the typed frame, column infos and log lines."""
    data, infos, lines = {}, {}, []
    for col in raw.columns:
        parsed = infer_column(raw[col], decimal=decimal, decimal_hint=decimal_hint, dayfirst=dayfirst)
        data[col] = parsed.values
        infos[col] = ColumnInfo(name=col, kind=parsed.kind, original_name=col, parse_info=parsed.info)
        lines.append(_describe_parse(col, parsed))
    return pd.DataFrame(data, index=raw.index), infos, lines


def _describe_parse(col: str, parsed: ParsedColumn) -> str:
    i = parsed.info
    if parsed.kind == "numeric":
        if not i.get("separators_seen"):
            msg = "whole numbers" if i.get("integer") else "numbers"
        else:
            msg = f"numbers, decimal '{i['decimal']}'"
        if i.get("decimal_ambiguous"):
            msg += " (ambiguous; assumed)"
        if i.get("currencies"):
            msg += f", currency {'/'.join(i['currencies'])}"
        if i.get("units"):
            msg += f", unit {'/'.join(i['units'])}"
    elif parsed.kind == "datetime":
        msg = f"date-times, {'day-first' if i['dayfirst'] else 'month-first'} ({i['format']})"
        if i.get("dayfirst_ambiguous"):
            msg += ", order ambiguous: assumed day-first"
    elif parsed.kind == "time":
        msg = "times h:mm[:ss]"
    elif parsed.kind == "empty":
        msg = "empty"
    else:
        msg = f"text, {i.get('unique', '?')} distinct values"
    if i.get("failures"):
        msg += f"; {i['failures']} value(s) could not be parsed (e.g. {i['failure_examples'][:2]})"
    return msg


# ---------------------------------------------------------------------------
# Data-quality checks
# ---------------------------------------------------------------------------


def quality_checks(ds: Dataset) -> list[AssumptionCheck]:
    """Flag missing values, duplicates, negative durations, unparsed text, mixed units and time order."""
    checks: list[AssumptionCheck] = []
    df = ds.data

    miss = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    if miss:
        listing = ", ".join(f"{c}: {n}" for c, n in list(miss.items())[:6])
        checks.append(AssumptionCheck(
            name="missing values", method="count per column", status=WARNING,
            summary=f"missing values in {len(miss)} column(s) ({listing})",
            detail="Analyses drop missing values per variable and report how many were dropped.",
        ))
    else:
        checks.append(AssumptionCheck(name="missing values", method="count per column",
                                      status=PASSED, summary="no missing values"))

    dup_rows = int(df.duplicated().sum())
    checks.append(AssumptionCheck(
        name="duplicate rows", method="identical rows", status=WARNING if dup_rows else PASSED,
        summary=f"{dup_rows} duplicate row(s)" if dup_rows else "no duplicate rows",
    ))
    for col in ds.columns_with_role("id"):
        d = int(df[col].dropna().duplicated().sum())
        if d:
            checks.append(AssumptionCheck(
                name="duplicate IDs", method=f"repeated values in {col}", status=WARNING,
                summary=f"{d} repeated ID(s) in {col}",
                detail="Repeated IDs can be fine (several lines per order), but check that rows are not double-counted.",
            ))

    for col in ds.columns_with_role("duration"):
        if ds.columns[col].kind == "numeric":
            neg = int((df[col] < 0).sum())
            if neg:
                checks.append(AssumptionCheck(
                    name="negative durations", method=f"{col} < 0", status=FAILED,
                    summary=f"{neg} negative value(s) in {col}",
                    detail="End before start, swapped columns or a midnight crossing. Fix or drop these rows.",
                ))

    for col, info in ds.columns.items():
        if col not in df.columns:
            continue
        p = info.parse_info
        if p.get("failures"):
            checks.append(AssumptionCheck(
                name="unparsed values", method=f"text in {info.kind} column", status=WARNING,
                summary=f"{p['failures']} value(s) in {col} could not be read as {info.kind}",
                detail=f"Examples: {p.get('failure_examples')}. They are treated as missing.",
            ))
        if len(p.get("units", [])) > 1 or len(p.get("currencies", [])) > 1:
            mixed = p.get("units") if len(p.get("units", [])) > 1 else p.get("currencies")
            checks.append(AssumptionCheck(
                name="mixed units", method=f"unit suffixes in {col}", status=FAILED,
                summary=f"mixed units in {col}: {', '.join(mixed)}",
                detail="Values were written in different units; convert them to one unit before analysis.",
            ))
        if info.role == "count" and info.kind == "numeric":
            s = df[col].dropna()
            if (s < 0).any() or not np.allclose(s, np.round(s)):
                checks.append(AssumptionCheck(
                    name="count values", method=f"{col} whole and ≥ 0", status=WARNING,
                    summary=f"{col} has negative or non-whole values for a count",
                ))
        if p.get("dayfirst_ambiguous"):
            checks.append(AssumptionCheck(
                name="date order", method=f"day/month order in {col}", status=INFO,
                summary=f"{col}: day/month order ambiguous, read as day-first",
                detail="No day above 12 was found. Override the date order on import if this is wrong.",
            ))

    tcol = ds.time_column()
    if tcol is not None:
        order = df[tcol].sort_values(kind="stable").index
        flagged = []
        for col in ds.numeric_columns():
            if ds.columns[col].role in ("id",):
                continue
            x = df.loc[order, col].dropna().to_numpy(dtype=float)
            if len(x) >= 10 and np.std(x) > 0:
                r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
                if abs(r1) > 2 / np.sqrt(len(x)):
                    flagged.append(f"{col} (r₁ = {r1:.2f})")
        checks.append(AssumptionCheck(
            name="time order", method=f"lag-1 autocorrelation in order of {tcol}",
            status=WARNING if flagged else INFO,
            summary=(f"time-ordered data: autocorrelation in {', '.join(flagged)}" if flagged
                     else f"time-ordered data ({tcol}): no lag-1 autocorrelation flagged"),
            rule="|r₁| ≤ 2/√n",
            detail="Observations close in time may be correlated, which breaks the 'random sample' "
                   "assumption of most tests. The runs test arrives in phase 4.",
        ))
    return checks


# ---------------------------------------------------------------------------
# Reshaping and simple cleaning steps
# ---------------------------------------------------------------------------


def wide_to_long(ds: Dataset, value_columns: list[str], *, group_name: str = "group",
                 value_name: str = "value", keep: list[str] | None = None) -> Dataset:
    """Reshape a wide table (one column per group) to one row per observation."""
    keep = keep or []
    long = ds.data[keep + value_columns].melt(id_vars=keep, value_vars=value_columns,
                                              var_name=group_name, value_name=value_name)
    long = long.dropna(subset=[value_name]).reset_index(drop=True)
    new = Dataset(long, name=f"{ds.name} (long)", log=ds.log, time_unit=ds.time_unit,
                  currency=ds.currency, source=ds.source)
    for c in keep:
        new.columns[c] = ds.columns[c]
    new.columns[group_name].role = "category"
    new.columns[group_name].role_confirmed = True
    unit = {ds.columns[c].unit for c in value_columns}
    new.columns[value_name].unit = unit.pop() if len(unit) == 1 else None
    new.log.add("reshape", f"wide to long: {len(value_columns)} columns → '{group_name}' and "
                           f"'{value_name}'", rows_affected=len(long))
    return new


def drop_missing(ds: Dataset, columns: list[str]) -> int:
    mask = ds.data[columns].isna().any(axis=1)
    return ds.drop_rows(mask, f"dropped rows with missing values in {', '.join(columns)}")


def drop_duplicates(ds: Dataset) -> int:
    return ds.drop_rows(ds.data.duplicated(), "dropped duplicate rows")
