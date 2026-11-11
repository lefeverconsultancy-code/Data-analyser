import numpy as np
import pandas as pd
import pytest

from analyser.core.dataset import Dataset
from analyser.core.session import Session, Settings
from analyser.io.derive import derive


def test_settings_defaults_and_validation():
    s = Settings()
    assert (s.alpha, s.quartile_method, s.time_unit, s.currency) == (0.05, "QUARTILE.INC", "min", "EUR")
    assert s.numpy_quantile_method == "linear"
    for bad in (dict(alpha=0), dict(quartile_method="R7"), dict(time_unit="day"), dict(seed=-1)):
        with pytest.raises(ValueError):
            Settings(**bad)


def test_quartile_inc_matches_excel():
    s = Session()
    # Excel QUARTILE.INC({1,2,3,4,5,6,7,8}, 1) = 2.75 and (…, 3) = 6.25
    assert s.quantiles([1, 2, 3, 4, 5, 6, 7, 8], [0.25, 0.75]).tolist() == [2.75, 6.25]
    s.update_settings(quartile_method="QUARTILE.EXC")
    # Excel QUARTILE.EXC gives 2.25 and 6.75
    assert s.quantiles([1, 2, 3, 4, 5, 6, 7, 8], [0.25, 0.75]).tolist() == [2.25, 6.75]


def test_run_uses_session_alpha_and_records_history():
    s = Session(alpha=0.10, seed=42)
    x = np.random.default_rng(1).normal(10, 2, 30)
    r = s.run_test("t_one_sample", x, mu0=10)
    assert r.alpha == 0.10 and r.settings["seed"] == 42
    f = s.run_model("fit_mle", x, "normal")
    assert f.cis["mu"].level == pytest.approx(0.90)
    h = s.history()
    assert list(h["kind"]) == ["test", "fit"] and len(s.results) == 2
    assert s.get(r.id) is r
    s.remove(r.id)
    assert [x.kind for x in s.results] == ["fit"]


def test_rng_is_reproducible():
    s = Session(seed=7)
    assert s.rng().normal() == Session(seed=7).rng().normal()


def test_session_save_and_reload(mapped_orders, tmp_path):
    ds = mapped_orders
    derive(ds)
    s = Session(alpha=0.01, time_unit="min")
    s.add_dataset(ds)
    s.run_test("t_one_sample", ds.var("duration_min").numeric(), mu0=12, about="the expected pick time")
    s.run_model("fit_mle", ds.var("duration_min").numeric(), "normal")
    path = s.save(tmp_path / "session.json")
    back = Session.load(path)
    assert back.settings == s.settings
    assert [r.to_dict() for r in back.results] == [r.to_dict() for r in s.results]
    b = back.datasets["orders"]
    pd.testing.assert_frame_equal(b.data, ds.data, check_dtype=False)
    assert b.columns["revenue"].role == "amount" and b.columns["duration_min"].formula
    assert b.log.to_list() == ds.log.to_list()
    assert b.mapping.roles == ds.mapping.roles
    assert b.is_time_ordered() == "created_at"


def test_dataset_var_and_time_order(mapped_orders):
    v = mapped_orders.var("revenue")
    assert v.unit == "EUR" and v.role == "amount" and v.analysis_role == "response"
    assert v.time_ordered_by == "created_at" and v.n == 1059
    with pytest.raises(KeyError):
        mapped_orders.var("nope")
    with pytest.raises(TypeError):
        mapped_orders.var("channel").numeric()


def test_dataset_drop_rows_logs():
    ds = Dataset(pd.DataFrame({"a": [1.0, np.nan, 3.0]}), name="t")
    assert ds.drop_rows(ds.data["a"].isna(), "drop missing a") == 1
    assert len(ds) == 2 and ds.log.steps[-1].rows_affected == 1
