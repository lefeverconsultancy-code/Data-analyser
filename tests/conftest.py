from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Profiles and saved sessions go to a temporary folder, never the user's home."""
    monkeypatch.setenv("ANALYSER_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def orders_csv() -> Path:
    return EXAMPLES / "orders_nl.csv"


@pytest.fixture
def orders_xlsx() -> Path:
    return EXAMPLES / "orders.xlsx"


@pytest.fixture
def mapped_orders(orders_csv):
    from analyser.io.load import load
    from analyser.io.mapping import ColumnMapping, apply_mapping

    ds = load(orders_csv, name="orders")
    return apply_mapping(ds, ColumnMapping.from_suggestions(ds))
