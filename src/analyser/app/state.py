"""App state: one Session per browser tab, the active dataset, and the sample files."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from analyser.core.dataset import Dataset
from analyser.core.session import Session


def session() -> Session:
    if "session" not in st.session_state:
        st.session_state["session"] = Session()
    return st.session_state["session"]


def replace_session(new: Session) -> None:
    st.session_state["session"] = new
    st.session_state["active_dataset"] = next(iter(new.datasets), None)
    st.session_state.pop("last_result", None)


def active_dataset() -> Dataset | None:
    s = session()
    name = st.session_state.get("active_dataset")
    if name in s.datasets:
        return s.datasets[name]
    return next(iter(s.datasets.values()), None)


def set_active(name: str) -> None:
    st.session_state["active_dataset"] = name


def examples_dir() -> Path | None:
    """The repository's examples/ folder (present when running from a clone)."""
    here = Path(__file__).resolve()
    for parent in here.parents[:5]:
        candidate = parent / "examples"
        if (candidate / "orders_nl.csv").exists():
            return candidate
    return None


def example_files() -> dict[str, Path]:
    d = examples_dir()
    if d is None:
        return {}
    files = sorted((p for p in d.iterdir() if p.suffix in (".csv", ".xlsx")), key=lambda p: (p.suffix != ".csv", p.name))
    return {p.name: p for p in files}


def load_sample_orders() -> Dataset:
    """Import the sample order CSV with the suggested mapping and all row-level derivations."""
    from analyser.io.derive import derive
    from analyser.io.load import load
    from analyser.io.mapping import ColumnMapping, apply_mapping

    files = example_files()
    if "orders_nl.csv" not in files:
        raise FileNotFoundError("examples/orders_nl.csv not found; run python examples/make_examples.py")
    s = session()
    ds = load(files["orders_nl.csv"], name="sample orders", time_unit=s.settings.time_unit,
              currency=s.settings.currency)
    ds = apply_mapping(ds, ColumnMapping.from_suggestions(ds), inplace=True)
    derive(ds)
    s.add_dataset(ds)
    set_active(ds.name)
    return ds
