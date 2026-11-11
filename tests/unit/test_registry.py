import pytest

from analyser.core.registry import (
    REGISTRY,
    Registry,
    get_distribution,
    get_test,
    list_registered,
    register_distribution,
    register_model,
    register_test,
)


def test_phase0_methods_are_registered():
    rows = list_registered()
    ids = {(r["kind"], r["id"]) for r in rows}
    assert ("distribution", "normal") in ids
    assert ("test", "t_one_sample") in ids
    assert ("model", "fit_mle") in ids
    t = next(r for r in rows if r["id"] == "t_one_sample")
    assert t["samples"] == 1 and t["data"] == "continuous" and "normal_or_n_ge_40" in t["assumptions"]


def test_listing_by_kind_and_as_frame():
    assert all(r["kind"] == "test" for r in list_registered("test"))
    frame = list_registered(as_frame=True)
    assert {"kind", "id", "name"} <= set(frame.columns)
    with pytest.raises(ValueError):
        list_registered("widget")


def test_lookup_helpers():
    assert get_test("t_one_sample").registry_id == "t_one_sample"
    assert get_distribution("normal")(0, 1).cdf(0) == pytest.approx(0.5)
    with pytest.raises(KeyError, match="Registered"):
        get_test("no_such_test")


def test_decorators_attach_metadata_and_reject_duplicates():
    reg_before = {e.id for e in REGISTRY.entries("test")}

    @register_test("demo_test_tmp", name="Demo", samples=2, data="continuous", paired=True,
                   assumptions=["normal"])
    def demo(x, y):
        return None

    try:
        entry = REGISTRY.get("test", "demo_test_tmp")
        assert entry.metadata["paired"] is True and entry.metadata["assumptions"] == ("normal",)
        with pytest.raises(ValueError, match="already registered"):
            @register_test("demo_test_tmp", name="Other", samples=1, data="continuous")
            def other(x):
                return None
    finally:
        REGISTRY._entries["test"].pop("demo_test_tmp", None)
    assert {e.id for e in REGISTRY.entries("test")} == reg_before


def test_decorator_validation():
    with pytest.raises(ValueError):
        register_test("bad", name="Bad", samples=3, data="continuous")
    with pytest.raises(ValueError):
        register_test("bad", name="Bad", samples=1, data="colours")
    with pytest.raises(ValueError):
        register_distribution("bad", name="Bad", family="mixed", params=["a"], support="x")
    with pytest.raises(ValueError):
        register_model("bad", name="Bad", result="table")


def test_separate_registry_instance():
    r = Registry()
    r.register("model", "m1", "Model one", object())
    assert [e.id for e in r.entries("model")] == ["m1"]
    with pytest.raises(ValueError):
        r.register("model", "not valid!", "x", object())
