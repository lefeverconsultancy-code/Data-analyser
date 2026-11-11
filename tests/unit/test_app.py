"""The app renders every page and runs the proof of concept end to end (headless AppTest)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from analyser import __main__ as entry  # noqa: E402
from analyser.core.results import result_from_json  # noqa: E402
from analyser.inference.one_sample import t_one_sample_summary  # noqa: E402

APP = str(Path(entry.__file__).parent / "app" / "streamlit_app.py")


def no_errors(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.fixture
def app():
    return AppTest.from_file(APP, default_timeout=90).run()


def test_entry_point_runs_the_app_locally():
    args = entry.streamlit_args(["--server.port", "8600"])
    assert args[:3] == ["streamlit", "run", APP]
    assert "--server.address=localhost" in args and "--browser.gatherUsageStats=false" in args
    assert args[-2:] == ["--server.port", "8600"]


def test_every_page_renders_empty(app):
    no_errors(app)
    for page, title in [("pages/analyse.py", "Analyse"), ("pages/results.py", "Results"),
                        ("pages/session.py", "Session"), ("pages/import_data.py", "Import data")]:
        app.switch_page(page).run()
        no_errors(app)
        assert app.title[0].value == title


def test_import_map_derive_then_t_test_and_fit(app):
    app.radio(key="imp_kind").set_value("Sample order data").run()
    assert app.selectbox(key="imp_example").value == "orders_nl.csv"
    app.button(key="imp_read").click().run()
    no_errors(app)
    app.button(key="imp_confirm").click().run()
    no_errors(app)
    app.multiselect[0].set_value(["count per day", "day-of-week profile"]).run()
    app.button(key="imp_add").click().run()
    no_errors(app)
    session = app.session_state["session"]
    assert set(session.datasets) == {"orders_nl", "orders_nl: count per day", "orders_nl: weekday profile"}
    assert any("no duplicate rows" in m.value for m in app.markdown)

    app.switch_page("pages/analyse.py").run()
    no_errors(app)
    assert app.selectbox(key="an_var_orders_nl").value == "duration_min"
    app.number_input(key="an_mu0").set_value(12.0).run()
    app.button(key="an_run_t").click().run()
    no_errors(app)
    app.button(key="an_run_fit").click().run()
    no_errors(app)
    kinds = [r.kind for r in session.results]
    assert kinds == ["test", "fit"]
    texts = [c.value for c in app.code]
    assert any(t.startswith("One-sample t-test on μ") and "8. Conclusion" in t for t in texts)
    assert any(t.startswith("Normal fit (MLE)") for t in texts)

    app.switch_page("pages/results.py").run()
    no_errors(app)
    assert app.selectbox(key="res_select").value == session.results[-1].id
    app.selectbox(key="res_select").set_value(session.results[0].id).run()
    no_errors(app)
    assert any("The 8 steps" in m.value for m in app.markdown)

    app.switch_page("pages/session.py").run()
    no_errors(app)


def test_results_page_renders_a_result_loaded_from_json(app):
    r = t_one_sample_summary(16, 118, 9, 115, "greater", about="the expected IQ of technical students")
    restored = result_from_json(r.to_json())
    app.session_state["session"].add(restored)
    app.switch_page("pages/results.py").run()
    no_errors(app)
    assert any("Fail to reject H0" in i.value for i in app.info)


def test_analyse_page_offers_sample_data(app):
    app.switch_page("pages/analyse.py").run()
    app.button(key="an_sample").click().run()
    no_errors(app)
    assert "sample orders" in app.session_state["session"].datasets
