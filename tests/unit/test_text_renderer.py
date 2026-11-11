from pathlib import Path

from analyser.fitting.mle import fit
from analyser.inference.categorical import chi2_gof
from analyser.inference.one_sample import t_one_sample, t_one_sample_summary
from analyser.report.text import render_text

EXAMPLE = Path(__file__).resolve().parents[1] / "reference" / "section9_example.txt"


def test_section9_example_prints_exactly():
    r = t_one_sample_summary(16, 118, 9, 115, "greater", 0.05, about="the expected IQ of technical students")
    assert render_text(r) == EXAMPLE.read_text(encoding="utf-8").rstrip("\n")


def test_two_sided_and_left_sided_wording():
    two = render_text(t_one_sample_summary(16, 118, 9, 115, "two-sided"))
    assert "H1: μ ≠ 115" in two
    assert "Reject H0 if T ≤ −2.131 or T ≥ 2.131" in two
    assert "2·P(T ≥ 1.333 | H0)" in two
    assert "95% CI for μ: (" in two
    left = render_text(t_one_sample_summary(16, 112, 9, 115, "less"))
    assert "T ≤ −1.753" in left and "upper bound for μ" in left and "is less than 115" in left


def test_reject_wording_and_small_p():
    r = t_one_sample_summary(100, 130, 9, 115, "greater")
    text = render_text(r)
    assert "Reject H0: t is in the rejection region, p ≤ α" in text
    assert "= P(T ≥ 16.667 | H0) < 0.0001" in text
    assert "the data prove that" in text


def test_every_step_present_in_order():
    text = render_text(t_one_sample([1.2, 1.5, 1.1, 1.9, 1.4], mu0=1))
    positions = [text.index(f"{i}. ") for i in range(1, 9)]
    assert positions == sorted(positions)


def test_other_result_kinds_render():
    assert "χ² ~ χ²(3), approximately" in render_text(chi2_gof([290, 219, 225, 266], expected=[260, 250, 250, 240]))
    fit_text = render_text(fit([4.2, 5.1, 3.9, 4.8, 5.5, 4.4], "normal"))
    assert "σ² (MLE, divisor n)" in fit_text and "AIC" in fit_text
