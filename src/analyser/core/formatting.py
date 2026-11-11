"""Number and text formatting shared by the engine and the renderers.

Conventions (plan, section 9):
- test statistics and critical values: 3 decimals (t = 1.333)
- p-values: 4 decimals; below 0.0001 shown as "< 0.0001"
- values on the data scale (x̄ bounds, CI limits): at least 2 decimals, more for small numbers
- negative numbers use the true minus sign (−), as in the readers
"""

from __future__ import annotations

import math
import re
import textwrap

MINUS = "−"
SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

ALTERNATIVES = {
    "two-sided": "two-sided",
    "two_sided": "two-sided",
    "two sided": "two-sided",
    "!=": "two-sided",
    "≠": "two-sided",
    "ne": "two-sided",
    "greater": "greater",
    ">": "greater",
    "right": "greater",
    "right-sided": "greater",
    "gt": "greater",
    "less": "less",
    "<": "less",
    "left": "less",
    "left-sided": "less",
    "lt": "less",
}

H1_SIGN = {"two-sided": "≠", "greater": ">", "less": "<"}
CLAIM_VERB = {"two-sided": "differs from", "greater": "exceeds", "less": "is less than"}


def normalize_alternative(alternative: str) -> str:
    """Map the many ways of writing an alternative to 'two-sided', 'greater' or 'less'."""
    key = str(alternative).strip().lower()
    if key not in ALTERNATIVES:
        raise ValueError(
            f"Unknown alternative {alternative!r}; use 'two-sided', 'greater' or 'less'."
        )
    return ALTERNATIVES[key]


def _minus(text: str) -> str:
    return text.replace("-", MINUS) if text.startswith("-") else text


def fmt_fixed(x: float, decimals: int) -> str:
    """Fixed decimals with a true minus sign; avoids '−0.000'."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if math.isinf(x):
        return "∞" if x > 0 else MINUS + "∞"
    s = f"{x:.{decimals}f}"
    if float(s) == 0:
        s = f"{0:.{decimals}f}"
    return _minus(s)


def is_integer_valued(x: float, tol: float = 1e-9) -> bool:
    return x is not None and math.isfinite(x) and abs(x - round(x)) < tol


def fmt_stat(x: float) -> str:
    """Test statistic or critical value: 3 decimals, or a whole number when integer-valued."""
    if x is not None and is_integer_valued(x):
        return _minus(str(int(round(x))))
    return fmt_fixed(x, 3)


def fmt_p(p: float) -> str:
    """p-value with 4 decimals; below 0.0001 as '< 0.0001'."""
    if p is None or math.isnan(p):
        return "n/a"
    if p < 0.0001:
        return "< 0.0001"
    return f"{p:.4f}"


def p_equals(p: float) -> str:
    """'= 0.1012' or '< 0.0001', for use after an expression."""
    s = fmt_p(p)
    return s if s.startswith("<") else f"= {s}"


def value_decimals(x: float) -> int:
    """Decimals for a value on the data scale: at least 2, enough for ~5 significant digits."""
    if x is None or not math.isfinite(x) or x == 0:
        return 2
    return max(2, min(8, 4 - math.floor(math.log10(abs(x)))))


def fmt_value(x: float, decimals: int | None = None) -> str:
    """A value on the data scale (x̄ bound, CI limit, estimate)."""
    if x is None:
        return "n/a"
    return fmt_fixed(x, value_decimals(x) if decimals is None else decimals)


def fmt_num(x: float, sig: int = 10) -> str:
    """A user-given number (μ₀, α, parameters) without trailing zeros: 115, 15.1, 0.05."""
    if x is None:
        return "n/a"
    if is_integer_valued(x) and abs(x) < 1e15:
        return _minus(str(int(round(x))))
    return _minus(f"{x:.{sig}g}")


def fmt_pct(fraction: float) -> str:
    """0.05 -> '5%', 0.025 -> '2.5%'."""
    return f"{fraction * 100:.10g}%"


def subscript(n: int) -> str:
    return str(n).translate(SUBSCRIPT_DIGITS)


def sample_notation(symbol: str, n: int) -> str:
    """'X₁,…,X₁₆'."""
    return f"{symbol}₁,…,{symbol}{subscript(n)}"


_UNIT_SUFFIXES = {"s", "sec", "min", "h", "hrs", "eur", "usd", "pct", "perc"}


def humanize(name: str) -> str:
    """'duration_min' -> 'duration', 'pickTime' -> 'pick time': a column name as words."""
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ").replace("-", " ").split()
    if len(words) > 1 and words[-1].lower() in _UNIT_SUFFIXES:
        words = words[:-1]
    return " ".join(w.lower() if not w.isupper() else w for w in words) or name


def conclusion_sentence(alpha: float, reject: bool, claim: str) -> str:
    """Step 8 in plain English. Never says 'accept H0'."""
    verb = "prove" if reject else "do not prove"
    return f"At a {fmt_pct(alpha)} significance level, the data {verb} that {claim}."


def wrap_hanging(text: str, first_prefix: str, width: int) -> list[str]:
    """Wrap text so continuation lines align under the first character after the prefix."""
    indent = " " * len(first_prefix)
    return textwrap.wrap(
        text,
        width=width,
        initial_indent=first_prefix,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [first_prefix.rstrip()]
