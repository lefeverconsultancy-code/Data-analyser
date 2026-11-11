import pytest

from analyser.core.formatting import (
    conclusion_sentence,
    fmt_num,
    fmt_p,
    fmt_pct,
    fmt_stat,
    fmt_value,
    humanize,
    normalize_alternative,
    sample_notation,
)


def test_numbers():
    assert fmt_stat(1.33333) == "1.333"
    assert fmt_stat(-1.7530) == "−1.753"
    assert fmt_stat(78.0) == "78"
    assert fmt_p(0.10116) == "0.1012"
    assert fmt_p(0.00001) == "< 0.0001"
    assert fmt_value(118.9443) == "118.94"
    assert fmt_value(0.012345) == "0.012345"
    assert fmt_num(115.0) == "115" and fmt_num(15.1) == "15.1" and fmt_num(0.05) == "0.05"
    assert fmt_pct(0.05) == "5%" and fmt_pct(0.025) == "2.5%" and fmt_pct(0.1) == "10%"


def test_notation_and_language():
    assert sample_notation("X", 16) == "X₁,…,X₁₆"
    assert conclusion_sentence(0.05, False, "μ exceeds 3") == \
        "At a 5% significance level, the data do not prove that μ exceeds 3."
    assert humanize("duration_min") == "duration"
    assert humanize("pickTime") == "pick time"


@pytest.mark.parametrize("raw,expected", [(">", "greater"), ("right", "greater"), ("<", "less"),
                                          ("two_sided", "two-sided"), ("≠", "two-sided")])
def test_alternatives(raw, expected):
    assert normalize_alternative(raw) == expected
