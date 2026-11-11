"""Loading, role mapping, derived measures, checks and profiles on the sample order data."""

import io

import numpy as np
import pandas as pd
import pytest

from analyser.core.dataset import Dataset
from analyser.io.clean import quality_checks, wide_to_long
from analyser.io.derive import (
    add_duration,
    available_derivations,
    derive,
    period_table,
    picks_per_resource_hour,
    profile_table,
)
from analyser.io.load import list_sheets, load, sniff_separator
from analyser.io.mapping import ColumnMapping, apply_mapping, suggest_roles
from analyser.io.profiles import apply_profile, list_profiles, load_profile, profile_from_dataset, save_profile

EXPECTED_ROLES = {"order_id": "id", "created_at": "timestamp", "channel": "category", "lines": "count",
                  "pick_start": "start", "pick_end": "end", "picker": "resource", "revenue": "amount",
                  "cost": "amount"}


def test_load_dutch_csv(orders_csv):
    ds = load(orders_csv)
    assert len(ds) == 1059 and ds.source["separator"] == ";"
    assert ds.columns["revenue"].kind == "numeric" and ds.columns["revenue"].parse_info["decimal"] == ","
    assert ds.columns["created_at"].kind == "datetime" and ds.columns["created_at"].parse_info["dayfirst"]
    assert ds.data["created_at"].iloc[0] == pd.Timestamp("2026-10-05 08:08:40")
    assert {c: i.role for c, i in ds.columns.items()} == EXPECTED_ROLES
    assert ds.columns["revenue"].amount_kind == "revenue" and ds.columns["cost"].amount_kind == "cost"
    assert ds.log.actions()[0] == "read" and ds.log.actions().count("parse") == 9


def test_excel_and_csv_give_the_same_data(orders_csv, orders_xlsx):
    assert list_sheets(orders_xlsx) == ["Orders"]
    a, b = load(orders_csv), load(orders_xlsx, header_row=3)
    assert b.log.steps[0].action == "skip_rows"
    for col in ("created_at", "pick_end"):
        pd.testing.assert_series_equal(a.data[col], b.data[col], check_names=False)
    np.testing.assert_allclose(a.data["revenue"], b.data["revenue"])


def test_pasted_table_from_excel():
    text = "Proces\tDuur (min)\tDatum\nInpak\t12,5\t03-11-2026\nInpak\t11,0\t04-11-2026\nPick\t7,25\t25-11-2026\n"
    ds = load(text)
    assert ds.source["file_type"] == "paste" and ds.source["separator"] == "\t"
    assert ds.data["Duur (min)"].tolist() == [12.5, 11.0, 7.25]
    assert ds.columns["Duur (min)"].role == "duration" and ds.columns["Duur (min)"].unit == "min"
    assert ds.data["Datum"].iloc[2] == pd.Timestamp("2026-11-25")


def test_cp1252_csv_with_title_rows_and_duplicate_headers():
    raw = "Export magazijn\n\nid;waarde;waarde\n1;€ 12,50;3\n2;€ 7,25;4\n".encode("cp1252")
    ds = load(io.BytesIO(raw), header_row=2)
    assert ds.source["encoding"] == "cp1252"
    assert list(ds.data.columns) == ["id", "waarde", "waarde_2"]
    assert ds.data["waarde"].tolist() == [12.5, 7.25]
    assert ds.columns["waarde"].parse_info["currencies"] == ["EUR"]
    assert "rename" in ds.log.actions()


def test_long_pasted_table():
    text = "minutes\n" + "\n".join(f"{i},5" for i in range(400))
    ds = load(text)
    assert len(ds) == 400 and ds.data["minutes"].iloc[3] == 3.5


def test_sniff_separator():
    assert sniff_separator("a;b;c\n1,5;2;3\n") == ";"
    assert sniff_separator("a,b,c\n1.5,2,3\n") == ","


def test_apply_mapping_converts_and_logs(orders_csv):
    ds = load(orders_csv)
    m = ColumnMapping.from_suggestions(ds)
    m.roles["lines"] = "category"
    out = apply_mapping(ds, m, time_unit="h")
    assert out.columns["lines"].kind == "text" and out.data["lines"].iloc[0] == "2"
    assert out.time_unit == "h" and out.columns["revenue"].unit == "EUR"
    assert out.log.actions().count("set_role") == 9
    assert ds.columns["lines"].role == "count"  # original untouched


def test_numeric_ids_keep_leading_zeros_and_duration_units():
    ds = load("article\tpick_seconds\n000123\t90\n000124\t30\n")
    assert ds.columns["pick_seconds"].unit == "s"
    m = ColumnMapping(roles={"article": "id", "pick_seconds": "duration"}, units={"pick_seconds": "s"})
    out = apply_mapping(ds, m)
    assert out.data["article"].tolist() == ["000123", "000124"]
    assert out.data["pick_seconds"].tolist() == [1.5, 0.5] and out.columns["pick_seconds"].unit == "min"


def test_invalid_mapping_is_refused(orders_csv):
    ds = load(orders_csv)
    with pytest.raises(ValueError):
        apply_mapping(ds, ColumnMapping(roles={"channel": "timestamp"}))
    with pytest.raises(ValueError):
        apply_mapping(ds, ColumnMapping(roles={"channel": "weight"}))
    with pytest.raises(KeyError):
        apply_mapping(ds, ColumnMapping(roles={"nope": "id"}))


def test_row_derivations(mapped_orders):
    ds = mapped_orders
    assert set(available_derivations(ds)) == {"interarrival", "duration", "waiting", "profit"}
    derive(ds)
    d = ds.data
    first = d.iloc[0]
    assert first["duration_min"] == pytest.approx((first["pick_end"] - first["pick_start"]).total_seconds() / 60)
    assert first["waiting_min"] == pytest.approx((first["pick_start"] - first["created_at"]).total_seconds() / 60)
    assert np.isnan(first["interarrival_min"])
    assert d["interarrival_min"].iloc[1] == pytest.approx((d["created_at"].iloc[1] - d["created_at"].iloc[0]).total_seconds() / 60)
    assert first["profit"] == pytest.approx(first["revenue"] - first["cost"])
    assert first["margin_pct"] == pytest.approx(first["profit"] / first["revenue"] * 100)
    assert ds.columns["duration_min"].role == "duration" and ds.columns["duration_min"].formula
    assert ds.log.actions().count("derive") == 5


def test_duration_across_midnight_for_times_of_day():
    ds = load("start\tend\n23:50\t00:10\n08:00\t08:30\n")
    ds = apply_mapping(ds, ColumnMapping(roles={"start": "start", "end": "end"}))
    add_duration(ds, crosses_midnight=True)
    assert ds.data["duration_min"].tolist() == [20.0, 30.0]


def test_period_and_profile_tables(mapped_orders):
    ds = mapped_orders
    daily = period_table(ds, "D")
    assert daily.data["count"].sum() == len(ds)
    assert (daily.data["count"] == 0).sum() == 6  # three weekends inside the range, filled with zeros
    weekdays = period_table(ds, "D", weekdays_only=True)
    assert len(weekdays) == 20 and (weekdays.data["count"] > 0).all()
    hourly = period_table(ds, "h", weekdays_only=True, hours=(8, 18))
    assert len(hourly) == 200 and hourly.data["count"].sum() == len(ds)
    revenue = period_table(ds, "W", value="revenue", agg="sum")
    assert revenue.data["revenue_sum"].sum() == pytest.approx(ds.data["revenue"].sum())
    by_channel = period_table(ds, "W", by="channel")
    assert by_channel.data["count"].sum() == len(ds)
    week = profile_table(ds, "weekday")
    assert week.data["days"].tolist()[:5] == [4, 4, 4, 4, 4]
    hours = profile_table(ds, "hour")
    assert hours.data["total"].sum() == len(ds)
    picks = picks_per_resource_hour(ds)
    assert set(picks.data["picker"]) == {"Anna", "Bram"} and (picks.data["picks_per_hour"] > 0).all()


def test_quality_checks_flag_problems():
    text = ("id;start;end;minutes\n1;01-11-2026 08:00;01-11-2026 08:10;10\n"
            "1;01-11-2026 09:00;01-11-2026 08:50;-10\n2;02-11-2026 10:00;;abc\n2;02-11-2026 10:00;;abc\n")
    ds = load(text, file_type="paste")
    ds = apply_mapping(ds, ColumnMapping(roles={"id": "id", "start": "start", "end": "end", "minutes": "duration"}))
    names = {c.name: c for c in quality_checks(ds)}
    assert names["missing values"].status == "warning"
    assert names["duplicate rows"].status == "warning"
    assert names["duplicate IDs"].status == "warning"
    assert names["negative durations"].status == "failed"
    assert "time order" in names


def test_mixed_units_flagged():
    ds = load("t\n12 min\n30 s\n2 min\n")
    names = {c.name: c for c in quality_checks(ds)}
    assert names["mixed units"].status == "failed"


def test_wide_to_long():
    ds = Dataset(pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, np.nan]}), name="wide")
    long = wide_to_long(ds, ["A", "B"], group_name="supplier", value_name="days")
    assert len(long) == 3 and long.columns["supplier"].role == "category"
    assert long.log.actions()[-1] == "reshape"


def test_profile_save_and_apply(orders_csv, isolated_home):
    ds = load(orders_csv)
    m = ColumnMapping.from_suggestions(ds)
    ds = apply_mapping(ds, m)
    derive(ds, ["duration", "profit"])
    profile = profile_from_dataset(ds, "Zaandam orders")
    path = save_profile(profile)
    assert path.parent == isolated_home / "profiles" and path.name == "zaandam-orders.json"
    assert [p.name for p in list_profiles()] == ["Zaandam orders"]
    again = apply_profile(orders_csv, load_profile("Zaandam orders"))
    assert {"duration_min", "profit", "margin_pct"} <= set(again.data.columns)
    assert "profile" in again.log.actions()
    pd.testing.assert_series_equal(again.data["duration_min"], ds.data["duration_min"])


def test_profile_refuses_missing_columns(orders_csv):
    ds = apply_mapping(load(orders_csv), ColumnMapping.from_suggestions(load(orders_csv)))
    profile = profile_from_dataset(ds, "orders")
    with pytest.raises(ValueError, match="expects column"):
        apply_profile("a;b\n1;2\n", profile)


def test_suggestions_in_dutch():
    ds = load("ordernummer;aantal;omzet;kosten;medewerker;datum\nA1;3;12,50;8,10;Piet;03-11-2026\n"
              "A2;5;20,00;11,00;Kees;04-11-2026\n", file_type="paste")
    roles = {c: s.role for c, s in suggest_roles(ds).items()}
    assert roles == {"ordernummer": "id", "aantal": "count", "omzet": "amount", "kosten": "amount",
                     "medewerker": "resource", "datum": "timestamp"}
