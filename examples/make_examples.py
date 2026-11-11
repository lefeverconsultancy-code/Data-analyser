"""Generate the sample order datasets in examples/ (fixed seed, so the files are reproducible).

    python examples/make_examples.py

orders_nl.csv  Dutch export: semicolon separator, decimal comma, day-first dates
orders.xlsx    the same orders as an Excel sheet "Orders" with two title rows above the
               header (header on row 4)

Four weeks of weekday orders at one warehouse: arrivals follow a Poisson process whose
rate varies by hour of day; two pickers pick orders first come, first served; pick time
grows with the number of order lines. Revenue and cost give profit and margin. The last
orders of the export are not yet picked, so pick_start/pick_end are empty there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261111
HERE = Path(__file__).parent
START = pd.Timestamp("2026-10-05")  # a Monday
DAYS = 20  # four working weeks
# Orders per hour from 08:00 to 18:00
HOURLY_RATE = [3, 5, 7, 7, 6, 4, 6, 6, 5, 3]
PICKERS = ["Anna", "Bram"]
CHANNELS = {"web": 0.6, "phone": 0.15, "EDI": 0.25}


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    order_no = 1000
    day_list = [START + pd.Timedelta(days=d) for d in range(28) if (START + pd.Timedelta(days=d)).day_of_week < 5][:DAYS]
    for day in day_list:
        # Non-homogeneous Poisson arrivals: piecewise-constant rate per hour
        for h, rate in enumerate(HOURLY_RATE):
            k = rng.poisson(rate)
            minutes = np.sort(rng.uniform(0, 60, size=k))
            for m in minutes:
                order_no += 1
                created = day + pd.Timedelta(hours=8 + h, minutes=float(m))
                lines = 1 + int(rng.poisson(4))
                channel = rng.choice(list(CHANNELS), p=list(CHANNELS.values()))
                rows.append({"order_id": f"ORD-{order_no:06d}", "created_at": created.floor("s"),
                             "channel": channel, "lines": lines})
    orders = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)

    # Two pickers, first come first served; each picker is free from 08:00 each day
    free = {p: pd.Timestamp.min for p in PICKERS}
    starts, ends, who = [], [], []
    for _, o in orders.iterrows():
        day_start = o.created_at.normalize() + pd.Timedelta(hours=8)
        picker = min(PICKERS, key=lambda p: max(free[p], day_start))
        begin = max(o.created_at, free[picker], day_start) + pd.Timedelta(seconds=float(rng.uniform(20, 90)))
        pick_minutes = max(1.0, 4.0 + 1.6 * o.lines + rng.normal(0, 1.5))
        end = begin + pd.Timedelta(minutes=pick_minutes)
        free[picker] = end
        starts.append(begin.floor("s"))
        ends.append(end.floor("s"))
        who.append(picker)
    orders["pick_start"] = starts
    orders["pick_end"] = ends
    orders["picker"] = who

    price = rng.normal(38.0, 8.0, size=len(orders)).clip(12, None)
    orders["revenue"] = (orders["lines"] * price).round(2)
    pick_cost = (orders["pick_end"] - orders["pick_start"]).dt.total_seconds() / 60 * 0.75
    shipping = orders["channel"].map({"web": 6.95, "phone": 8.50, "EDI": 4.20})
    orders["cost"] = (0.62 * orders["revenue"] + pick_cost + shipping).round(2)

    # The export was taken while the last orders were still waiting to be picked
    cutoff = orders["created_at"].iloc[-1] - pd.Timedelta(minutes=8)
    open_orders = orders["pick_start"] > cutoff
    orders.loc[open_orders, ["pick_start", "pick_end", "picker"]] = None
    return orders


def dutch_number(x: float) -> str:
    """1234.5 -> '1.234,50'."""
    if pd.isna(x):
        return ""
    s = f"{x:,.2f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def dutch_datetime(t) -> str:
    return "" if pd.isna(t) else pd.Timestamp(t).strftime("%d-%m-%Y %H:%M:%S")


def write_csv(orders: pd.DataFrame, path: Path) -> None:
    out = pd.DataFrame({
        "order_id": orders["order_id"],
        "created_at": orders["created_at"].map(dutch_datetime),
        "channel": orders["channel"],
        "lines": orders["lines"].astype(int).astype(str),
        "pick_start": orders["pick_start"].map(dutch_datetime),
        "pick_end": orders["pick_end"].map(dutch_datetime),
        "picker": orders["picker"].fillna(""),
        "revenue": orders["revenue"].map(dutch_number),
        "cost": orders["cost"].map(dutch_number),
    })
    out.to_csv(path, sep=";", index=False, encoding="utf-8")


def write_excel(orders: pd.DataFrame, path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        orders.to_excel(xw, sheet_name="Orders", index=False, startrow=3)
        ws = xw.sheets["Orders"]
        ws["A1"] = "Order export, warehouse Zaandam"
        ws["A2"] = "Exported 30-10-2026 17:45"
        for col in ("B", "E", "F"):
            for cell in ws[col][4:]:
                cell.number_format = "dd-mm-yyyy hh:mm:ss"


def main() -> None:
    orders = generate()
    write_csv(orders, HERE / "orders_nl.csv")
    write_excel(orders, HERE / "orders.xlsx")
    print(f"{len(orders)} orders written to orders_nl.csv and orders.xlsx")


if __name__ == "__main__":
    main()
