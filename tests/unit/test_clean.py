import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from analyser.io.clean import detect_dayfirst, infer_column, parse_datetimes, parse_numbers, parse_times


def nums(values, **kw):
    return parse_numbers(pd.Series(values, dtype=object), **kw)


@pytest.mark.parametrize("values,expected,decimal", [
    (["12,5", "3,25", "1.234,56"], [12.5, 3.25, 1234.56], ","),       # Dutch
    (["12.5", "3.25", "1,234.56"], [12.5, 3.25, 1234.56], "."),       # English
    (["€ 12,50", "€1.000,00", "-€ 3,10"], [12.5, 1000.0, -3.1], ","),  # currency
    (["(3,2)", "−4,5", "+2,0"], [-3.2, -4.5, 2.0], ","),               # negatives
    (["1 234,5", "2 000,25"], [1234.5, 2000.25], ","),                 # space thousands
])
def test_number_formats(values, expected, decimal):
    p = nums(values)
    assert p.info["decimal"] == decimal
    assert p.values.tolist() == pytest.approx(expected)


def test_ambiguous_thousands_use_hint():
    assert nums(["1.234", "5.678"], decimal_hint=",").values.tolist() == [1234.0, 5678.0]
    assert nums(["1.234", "5.678"], decimal_hint=".").values.tolist() == [1.234, 5.678]
    assert nums(["1.234"], decimal_hint=",").info["decimal_ambiguous"]


def test_missing_tokens_failures_units():
    p = nums(["12", "", "n.v.t.", "abc", "14", "NA"])
    assert np.isnan(p.values[1]) and np.isnan(p.values[2]) and np.isnan(p.values[5])
    assert p.info["failures"] == 1 and p.info["failure_examples"] == ["abc"]
    u = nums(["12 min", "30 s", "1,5 uur"])
    assert u.info["units"] == ["h", "min", "s"]
    assert u.values.tolist() == [12.0, 30.0, 1.5]


@settings(max_examples=100, deadline=None)
@given(st.lists(st.floats(-1e7, 1e7, allow_nan=False).map(lambda v: round(v, 2)), min_size=1, max_size=20))
def test_dutch_round_trip(values):
    def dutch(x):
        s = f"{x:,.2f}"
        return s.replace(",", "§").replace(".", ",").replace("§", ".")
    parsed = parse_numbers(pd.Series([dutch(v) for v in values]), decimal=",").values
    assert parsed.tolist() == pytest.approx(values, abs=1e-9)


def test_dayfirst_detection():
    assert detect_dayfirst(["03-11-2026", "25-11-2026"]) == (True, False)
    assert detect_dayfirst(["11/25/2026", "03/11/2026"]) == (False, False)
    assert detect_dayfirst(["03-11-2026", "04-11-2026"]) == (True, True)  # ambiguous → day-first
    assert detect_dayfirst(["2026-11-03"]) == (True, False)


def test_datetime_parsing_dutch_and_iso():
    p = parse_datetimes(pd.Series(["03-11-2026 08:15", "25-11-2026 14:02", ""]))
    assert p.values.iloc[0] == pd.Timestamp("2026-11-03 08:15")
    assert pd.isna(p.values.iloc[2]) and p.info["failures"] == 0
    iso = parse_datetimes(pd.Series(["2026-11-03T08:15:00", "2026-11-04T09:00:00"]))
    assert iso.values.iloc[1] == pd.Timestamp("2026-11-04 09:00")
    us = parse_datetimes(pd.Series(["03/11/2026", "04/11/2026"]), dayfirst=False)
    assert us.values.iloc[0] == pd.Timestamp("2026-03-11")


def test_times_and_kind_inference():
    t = parse_times(pd.Series(["0:12:30", "1:05", ""]))
    assert t.values.iloc[0] == pd.Timedelta(minutes=12, seconds=30)
    assert t.values.iloc[1] == pd.Timedelta(hours=1, minutes=5)
    assert infer_column(pd.Series(["08:15", "09:30"])).kind == "time"
    assert infer_column(pd.Series(["03-11-2026", "04-11-2026"])).kind == "datetime"
    assert infer_column(pd.Series(["12,5", "13"])).kind == "numeric"
    assert infer_column(pd.Series(["web", "phone"])).kind == "text"
    assert infer_column(pd.Series(["", ""])).kind == "empty"
