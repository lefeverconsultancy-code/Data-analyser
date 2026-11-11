"""Reference-case harness: runs every case in tests/reference/*.yaml.

Add a worked example by appending a case to cases.yaml (or a new YAML file in this
folder); no Python needed.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from analyser.report.text import render_text

HERE = Path(__file__).parent


def load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in sorted(HERE.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for case in data.get("cases", []):
            case["_file"] = path.name
            cases.append(case)
    ids = [c["id"] for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"duplicate case ids: {duplicates}"
    return cases


CASES = load_cases()


def resolve(dotted: str):
    module, _, attr = dotted.partition(":")
    obj = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def get_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(obj, dict):
            obj = obj[part]
        elif isinstance(obj, (list, tuple)) and part.lstrip("-").isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def check(result: Any, exp: dict[str, Any]) -> list[str]:
    """Return a list of failure messages for one expectation."""
    if "get" in exp:
        label = exp["get"]
        actual = get_path(result, exp["get"])
    else:
        label = f"{exp['call']}({', '.join(map(str, exp.get('args', [])))})"
        actual = getattr(result, exp["call"])(*exp.get("args", []))
    failures = []
    if "equals" in exp and actual != exp["equals"]:
        failures.append(f"{label}: expected {exp['equals']!r}, got {actual!r}")
    if "value" in exp and abs(float(actual) - float(exp["value"])) > float(exp["tol"]):
        failures.append(f"{label}: published {exp['value']} ± {exp['tol']}, got {float(actual):.10g}")
    if "exact" in exp and abs(float(actual) - float(exp["exact"])) > float(exp["exact_tol"]):
        failures.append(f"{label}: exact {exp['exact']} ± {exp['exact_tol']}, got {float(actual):.12g}")
    return failures


def test_cases_are_well_formed():
    assert len(CASES) >= 5
    for case in CASES:
        assert {"id", "title", "source", "call", "expect"} <= set(case), case.get("id")
        for exp in case["expect"]:
            assert ("get" in exp) != ("call" in exp), (case["id"], exp)
            assert {"value", "exact", "equals"} & set(exp), (case["id"], exp)
            if "value" in exp:
                assert "tol" in exp, (case["id"], exp)
            if "exact" in exp:
                assert "exact_tol" in exp, (case["id"], exp)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_reference_case(case: dict[str, Any]) -> None:
    result = resolve(case["call"])(*case.get("args", []), **case.get("kwargs", {}))
    failures = [msg for exp in case["expect"] for msg in check(result, exp)]
    assert not failures, f"{case['id']} ({case['source']}):\n  " + "\n  ".join(failures)
    if "render" in case:
        expected = (HERE / case["render"]).read_text(encoding="utf-8").rstrip("\n")
        assert render_text(result) == expected
