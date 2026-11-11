"""Derived measures built from column roles. Every derivation is logged with its formula.

Row-level (added as columns):   inter-arrival times, durations (end − start),
                                waiting times (created → start), profit, margin %
Table-level (new Datasets):     counts and totals per hour/day/week, hour-of-day and
                                day-of-week profiles, picks per resource-hour
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from ..core.dataset import Dataset, timedelta_to_unit

# (label, period rule, date_range frequency); weeks run Monday to Sunday
FREQS = {"h": ("hour", "h", "h"), "D": ("day", "D", "D"), "W": ("week", "W-SUN", "W-MON")}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _need(ds: Dataset, column: str | None, role: str, **kw) -> str:
    col = column or ds.first_with_role(role, **kw)
    if col is None:
        label = f"{role} ({kw['amount_kind']})" if kw.get("amount_kind") else role
        raise ValueError(f"No column with role '{label}'; map one on import first")
    if col not in ds.data.columns:
        raise KeyError(col)
    return col


def _as_timedelta(ds: Dataset, col: str) -> pd.Series:
    s = ds.data[col]
    if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
        return s
    raise ValueError(f"{col} must hold date-times or times of day")


# ---------------------------------------------------------------------------
# Row-level derivations
# ---------------------------------------------------------------------------


def add_interarrival(ds: Dataset, timestamp: str | None = None, *, by: str | None = None,
                     name: str | None = None) -> str:
    """Time since the previous arrival, in the dataset's time unit (sorted by time; per group if ``by``)."""
    col = _need(ds, timestamp, "timestamp")
    name = name or f"interarrival_{ds.time_unit}"
    frame = ds.data[[col] + ([by] if by else [])].copy()
    frame = frame.sort_values(col, kind="stable")
    if by:
        gaps = frame.groupby(by, sort=False)[col].diff()
    else:
        gaps = frame[col].diff()
    values = timedelta_to_unit(gaps, ds.time_unit).reindex(ds.data.index)
    formula = f"{col}ᵢ − {col}ᵢ₋₁ (sorted by time{', per ' + by if by else ''}), in {ds.time_unit}"
    ds.add_column(name, values, role="duration", unit=ds.time_unit, formula=formula, replace=True)
    ds.derivations.append(f"interarrival:{col}:{by or ''}")
    return name


def add_duration(ds: Dataset, start: str | None = None, end: str | None = None, *,
                 name: str | None = None, crosses_midnight: bool = False) -> str:
    """end − start in the dataset's time unit. Times of day can wrap past midnight if allowed."""
    s_col = _need(ds, start, "start")
    e_col = _need(ds, end, "end")
    delta = _as_timedelta(ds, e_col) - _as_timedelta(ds, s_col)
    if crosses_midnight and pd.api.types.is_timedelta64_dtype(delta):
        delta = delta.where(delta >= pd.Timedelta(0), delta + pd.Timedelta(days=1))
    name = name or f"duration_{ds.time_unit}"
    formula = f"{e_col} − {s_col}, in {ds.time_unit}"
    ds.add_column(name, timedelta_to_unit(delta, ds.time_unit), role="duration", unit=ds.time_unit,
                  formula=formula, replace=True)
    ds.derivations.append(f"duration:{s_col}:{e_col}")
    return name


def add_waiting(ds: Dataset, created: str | None = None, start: str | None = None, *,
                name: str | None = None) -> str:
    """Waiting time from creation (timestamp) to start, in the dataset's time unit."""
    c_col = _need(ds, created, "timestamp")
    s_col = _need(ds, start, "start")
    delta = _as_timedelta(ds, s_col) - _as_timedelta(ds, c_col)
    name = name or f"waiting_{ds.time_unit}"
    ds.add_column(name, timedelta_to_unit(delta, ds.time_unit), role="duration", unit=ds.time_unit,
                  formula=f"{s_col} − {c_col}, in {ds.time_unit}", replace=True)
    ds.derivations.append(f"waiting:{c_col}:{s_col}")
    return name


def add_profit(ds: Dataset, revenue: str | None = None, cost: str | None = None) -> tuple[str, str]:
    """profit = revenue − cost and margin % = profit / revenue × 100 (missing when revenue is 0)."""
    r_col = _need(ds, revenue, "amount", amount_kind="revenue")
    c_col = _need(ds, cost, "amount", amount_kind="cost")
    profit = ds.data[r_col] - ds.data[c_col]
    ds.add_column("profit", profit, role="amount", unit=ds.currency, amount_kind="profit",
                  formula=f"{r_col} − {c_col}", replace=True)
    rev = ds.data[r_col].where(ds.data[r_col] != 0)
    ds.add_column("margin_pct", profit / rev * 100, unit="%",
                  formula=f"profit / {r_col} × 100", replace=True)
    ds.derivations.append(f"profit:{r_col}:{c_col}")
    return "profit", "margin_pct"


def available_derivations(ds: Dataset) -> dict[str, str]:
    """Row-level derivations the mapped roles allow: key → description."""
    out = {}
    ts, st, en = ds.first_with_role("timestamp"), ds.first_with_role("start"), ds.first_with_role("end")
    if ts:
        out["interarrival"] = f"inter-arrival times from {ts}"
    if st and en:
        out["duration"] = f"durations {en} − {st}"
    if ts and st:
        out["waiting"] = f"waiting times {st} − {ts}"
    if ds.first_with_role("amount", amount_kind="revenue") and ds.first_with_role("amount", amount_kind="cost"):
        out["profit"] = "profit and margin % from revenue and cost"
    return out


DERIVERS: dict[str, Callable[[Dataset], object]] = {
    "interarrival": add_interarrival,
    "duration": add_duration,
    "waiting": add_waiting,
    "profit": add_profit,
}


def derive(ds: Dataset, keys: list[str] | None = None) -> list[str]:
    """Run the requested (default: all available) row-level derivations; returns the keys run."""
    keys = list(available_derivations(ds)) if keys is None else keys
    for key in keys:
        if key not in DERIVERS:
            raise ValueError(f"unknown derivation {key!r}; choose from {list(DERIVERS)}")
        DERIVERS[key](ds)
    return keys


# ---------------------------------------------------------------------------
# Table-level derivations
# ---------------------------------------------------------------------------


def period_table(ds: Dataset, freq: str = "D", *, timestamp: str | None = None,
                 value: str | None = None, agg: str = "count", by: str | None = None,
                 fill_zeros: bool = True, weekdays_only: bool = False,
                 hours: tuple[int, int] | None = None) -> Dataset:
    """Counts (or totals of ``value``) per hour, day or week, as a new Dataset.

    Periods without events are included as zeros (``fill_zeros``), which matters when the
    counts are later fitted, e.g. to a Poisson distribution. ``weekdays_only`` drops
    Saturdays and Sundays; ``hours=(8, 18)`` keeps hourly periods from 08:00 up to 18:00.
    """
    if freq not in FREQS:
        raise ValueError(f"freq must be one of {tuple(FREQS)}")
    if agg not in ("count", "sum", "mean"):
        raise ValueError("agg must be 'count', 'sum' or 'mean'")
    if agg != "count" and value is None:
        raise ValueError(f"agg='{agg}' needs a value column")
    col = _need(ds, timestamp, "timestamp")
    label, rule, step = FREQS[freq]
    frame = ds.data[[col] + ([value] if value else []) + ([by] if by else [])].dropna(subset=[col])
    period = frame[col].dt.to_period(rule).dt.start_time.rename("period")
    keys = [period] + ([frame[by]] if by else [])
    grouped = frame.groupby(keys)
    if agg == "count":
        out = grouped.size().rename("count")
        out_name, unit = "count", None
    else:
        out = getattr(grouped[value], agg)().rename(f"{value}_{agg}")
        out_name, unit = f"{value}_{agg}", ds.columns[value].unit
    out = out.reset_index()
    if fill_zeros:
        full = pd.date_range(period.min(), period.max(), freq=step)
        if by:
            idx = pd.MultiIndex.from_product([full, sorted(frame[by].dropna().unique())],
                                             names=["period", by])
            out = out.set_index(["period", by]).reindex(idx).reset_index()
        else:
            out = out.set_index("period").reindex(full).rename_axis("period").reset_index()
        if agg in ("count", "sum"):
            out[out_name] = out[out_name].fillna(0)
    if weekdays_only and freq in ("h", "D"):
        out = out[out["period"].dt.day_of_week < 5].reset_index(drop=True)
    if hours is not None and freq == "h":
        lo, hi = hours
        if not 0 <= lo < hi <= 24:
            raise ValueError("hours must be (start, end) with 0 ≤ start < end ≤ 24")
        out = out[(out["period"].dt.hour >= lo) & (out["period"].dt.hour < hi)].reset_index(drop=True)
    new = Dataset(out, name=f"{ds.name}: {out_name} per {label}", time_unit=ds.time_unit,
                  currency=ds.currency, source={"derived_from": ds.name})
    new.columns["period"].role = "timestamp"
    new.columns["period"].role_confirmed = True
    new.columns[out_name].role = "count" if agg == "count" else "amount" if unit == ds.currency else None
    new.columns[out_name].unit = unit
    new.columns[out_name].formula = (f"number of rows per {label}" if agg == "count"
                                     else f"{agg} of {value} per {label}")
    new.columns[out_name].derived = True
    if by:
        new.columns[by].role = "category"
    note = f"{agg} per {label} from {col}" + (f" by {by}" if by else "")
    note += ", empty periods as 0" if fill_zeros and agg != "mean" else ""
    note += ", weekdays only" if weekdays_only else ""
    note += f", hours {hours[0]:02d}:00–{hours[1]:02d}:00" if hours is not None and freq == "h" else ""
    ds.log.add("period_table", note, rows_affected=len(out))
    new.log.add("derive", f"built from '{ds.name}': {note}", rows_affected=len(out))
    return new


def profile_table(ds: Dataset, by: str = "hour", *, timestamp: str | None = None) -> Dataset:
    """Average number of events per hour of day or per day of week."""
    if by not in ("hour", "weekday"):
        raise ValueError("by must be 'hour' or 'weekday'")
    col = _need(ds, timestamp, "timestamp")
    t = ds.data[col].dropna()
    days = t.dt.normalize()
    if by == "hour":
        counts = t.groupby([days, t.dt.hour]).size()
        per = counts.groupby(level=1).agg(["sum", "mean"])
        n_days = days.nunique()
        out = pd.DataFrame({"hour": range(24)}).merge(
            per.rename(columns={"sum": "total"}).reset_index(names="hour"), on="hour", how="left")
        out["total"] = out["total"].fillna(0)
        out["mean_per_day"] = out["total"] / n_days
        out = out[["hour", "total", "mean_per_day"]]
        key = "hour"
    else:
        per_day = t.groupby(days).size()
        dow = per_day.index.day_of_week
        out = pd.DataFrame({"weekday": WEEKDAYS})
        out["days"] = [int((dow == i).sum()) for i in range(7)]
        out["total"] = [float(per_day[dow == i].sum()) for i in range(7)]
        out["mean_per_day"] = out["total"] / out["days"].where(out["days"] > 0)
        key = "weekday"
    new = Dataset(out, name=f"{ds.name}: {by} profile", time_unit=ds.time_unit, currency=ds.currency,
                  source={"derived_from": ds.name})
    new.columns[key].role = "category"
    new.columns["total"].role = "count"
    ds.log.add("profile", f"{by} profile from {col}")
    new.log.add("derive", f"{by}-of-{'day' if by == 'hour' else 'week'} profile from '{ds.name}' ({col})")
    return new


def picks_per_resource_hour(ds: Dataset, *, resource: str | None = None, start: str | None = None,
                            end: str | None = None, count: str | None = None) -> Dataset:
    """Picks per resource-hour, per resource and day.

    Hours worked = last end − first start of that resource on that day; picks = the sum of
    ``count`` (e.g. order lines) or the number of rows.
    """
    r_col = _need(ds, resource, "resource")
    s_col = _need(ds, start, "start")
    e_col = _need(ds, end, "end")
    c_col = count or ds.first_with_role("count")
    frame = ds.data[[r_col, s_col, e_col] + ([c_col] if c_col else [])].dropna(subset=[r_col, s_col, e_col])
    frame = frame.assign(day=frame[s_col].dt.normalize())
    g = frame.groupby([r_col, "day"])
    hours = (g[e_col].max() - g[s_col].min()).dt.total_seconds() / 3600
    picks = g[c_col].sum() if c_col else g.size()
    out = pd.DataFrame({"picks": picks, "hours": hours}).reset_index()
    out["picks_per_hour"] = out["picks"] / out["hours"].where(out["hours"] > 0)
    new = Dataset(out, name=f"{ds.name}: picks per {r_col}-hour", time_unit=ds.time_unit,
                  currency=ds.currency, source={"derived_from": ds.name})
    new.columns[r_col].role = "resource"
    new.columns["day"].role = "timestamp"
    new.columns["picks"].role = "count"
    new.columns["hours"].unit = "h"
    new.columns["picks_per_hour"].formula = (f"{'sum of ' + c_col if c_col else 'rows'} / "
                                             f"(last {e_col} − first {s_col}) per {r_col} per day")
    new.columns["picks_per_hour"].derived = True
    ds.log.add("resource_table", f"picks per {r_col}-hour per day")
    new.log.add("derive", f"built from '{ds.name}': {new.columns['picks_per_hour'].formula}")
    return new
