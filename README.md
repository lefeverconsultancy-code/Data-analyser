# Data-analyser

Tool for analysing data using statistics, probability, and simulations.

A local Python application: a tested library with a Streamlit app on top, running only on
your own computer (no hosting, no accounts). It imports logistics and financial data
(orders received and picked, revenue and profit, process times), describes it, fits
distributions and runs hypothesis tests. Every test is reported in the 8-step procedure of
the *Statistics for Engineers* reader.

**Status: phase 0 (backbone) complete.** See [Phase 0](#phase-0-what-works) below.

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/lefeverconsultancy-code/data-analyser.git
cd data-analyser
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run the app

```bash
python -m analyser
```

The app opens in your browser at http://localhost:8501. It listens on localhost only and
sends no usage statistics. Extra Streamlit options pass through, e.g.
`python -m analyser --server.port 8600`.

Quick tour: **Import data** → *Sample order data* → *Read data* → *Confirm mapping* →
*Add to session*, then **Analyse** → *Run t-test* and *Fit normal distribution (MLE)*.
**Results** lists every result and can open saved result files; **Session** holds the
settings (α, quartile method, seed, time unit, currency), save/reload and the list of
registered methods.

## Use it from Python

```python
from analyser.io.load import load
from analyser.io.mapping import ColumnMapping, apply_mapping
from analyser.io.derive import derive
from analyser.core.session import Session
from analyser.report.text import render_text

ds = load("examples/orders_nl.csv")                 # Dutch CSV: ; separator, decimal comma, day-first dates
ds = apply_mapping(ds, ColumnMapping.from_suggestions(ds))
derive(ds)                                          # inter-arrival, durations, waiting times, profit, margin %

s = Session(alpha=0.05)
r = s.run_test("t_one_sample", ds.var("duration_min").numeric(), mu0=12,
               alternative="greater", about="the expected pick time", unit="min")
print(render_text(r))
fit = s.run_model("fit_mle", ds.var("duration_min").numeric(), "normal")
s.save("analysis.json")
```

## Tests

```bash
python -m pytest                 # everything (about 30 s)
python -m pytest -m "not slow"   # skip the 10,000-run simulation checks
```

GitHub Actions runs the full suite on every push (Ubuntu and Windows, Python 3.11 and 3.13,
plus a pandas 2.2 job).

- `tests/unit/`: one file per area, including property-based tests (hypothesis) and a
  headless test of every app page.
- `tests/reference/`: the reference-case harness. `cases.yaml` holds worked examples with
  their published answers; `test_reference_cases.py` runs them. Add a case by appending to
  the YAML; no Python needed.
- `tests/simulation/`: test size, CI coverage, power and parameter recovery (10,000 runs,
  fixed seed).

### Reference-case notes

| Case | Published | Exact | Why they differ |
| --- | --- | --- | --- |
| z-test, IQ, σ known (Stats 4.1) | p = 9.18% | p = 9.12% | reader rounds z to 1.33 for the table |
| χ² family composition (Stats 6.1.6) | χ² = 12.62 | 12.53 with E₀N = 1000·p | reader rounds expected counts to 260, 250, 250, 240 |
| Wilcoxon cities vs towns (Stats 7.5.4) | p = 15.87% | p = 15.92% | reader rounds z = 0.998 to 1.00 |
| Wilcoxon tie variance (Stats 7.5) | var(W) = 90.71 | 90.53 | **reader error**: its own formula gives 90.53; ties can only lower the variance below 90.67 |

The harness checks both the published value (with a tolerance that covers the rounding)
and the exact value.

## Project layout

```
src/analyser/
├── __main__.py      python -m analyser starts the local app
├── core/            dataset, variable, results, registry, assumptions, session, formatting
├── io/              load (CSV, Excel, pasted), clean (parsing, checks), mapping, derive, profiles
├── describe/        plots (phase 0: Q-Q, histogram with fit, null distribution)
├── distributions/   base, continuous (normal), calculator
├── fitting/         mle (normal)
├── inference/       one_sample (t, z), categorical (χ² GOF), nonparametric (Wilcoxon rank sum)
├── estimation/ models/ simulate/ simstudy/ guide/   later phases
├── report/          text (8-step renderer)
└── app/             streamlit_app.py, render.py, state.py, pages/
tests/  unit/  reference/  simulation/
examples/  orders_nl.csv, orders.xlsx, make_examples.py
```

Five layers, each calling only the one below: interfaces (app, Python API) → workflow
(Session) → analysis engine (inference, fitting, distributions) → core (Dataset, result
objects, registry, checks) → backends (numpy, pandas, scipy).

### Adding a test

Write one decorated function that returns a `TestResult`; the registry, the Session and the
app's method list pick it up.

```python
from analyser.core.registry import register_test

@register_test("my_test", name="My test", samples=1, data="continuous",
               assumptions=["random_sample"], sources=["L&M 7.4"])
def my_test(x, mu0, alternative="two-sided", alpha=0.05) -> TestResult: ...
```

Add the module to `ENGINE_MODULES` in `core/registry.py` so `list_registered()` sees it.

## Phase 0: what works

- **Import**: CSV, Excel (choose sheet and header row) and pasted tables; Dutch and English
  numbers (decimal comma or point, thousands separators, € signs, accounting negatives),
  day-first dates detected automatically, `h:mm` times. Columns get a suggested role (ID,
  timestamp, start, end, duration, count, amount, category, resource) from their name in
  English or Dutch and their content; you confirm it. Durations convert to one time unit.
- **Derived measures**: inter-arrival times, durations, waiting times, profit and margin %,
  counts and totals per hour, day or week (empty periods as 0; optional weekdays-only and
  operating hours), hour-of-day and day-of-week profiles, picks per resource-hour.
- **Checks**: missing values, duplicate rows and IDs, negative durations, unparsed text,
  mixed units, ambiguous date order, time-ordered data with lag-1 autocorrelation.
  Every step is written to the dataset's cleaning log.
- **Import profiles**: save a read + mapping + derivations and reapply it in one click
  (stored in `~/.data-analyser/profiles`, or `$ANALYSER_HOME/profiles`).
- **Result objects** (`TestResult` with the 8 steps, `FitResult`, `ModelResult`,
  `AssumptionCheck`) with an exact JSON round-trip.
- **Registry** with `@register_distribution`, `@register_test`, `@register_model` and
  `list_registered()`.
- **Session**: settings (α, quartile method, seed, time unit, currency), result history,
  save and reload including datasets.
- **8-step text renderer**: the plan's section 9 example prints character for character.
- **Proof of concept**: the normal distribution and calculator, one-sample t-test and normal
  MLE fit, end to end in the app on the sample order data. The z-test, χ² goodness of fit and
  Wilcoxon rank-sum test are in early versions so the harness covers all reference cases.

## Conventions

- Variance: S² (divisor n − 1) and s by default; the MLE variance appears only inside
  distribution fits, labelled as such.
- Quartiles: Excel QUARTILE.INC (numpy `method="linear"`).
- Decisions read "reject H0" or "fail to reject H0", never "accept H0".
- p-values to 4 decimals, "< 0.0001" below that. Output in English.
- Excel is an import format only.
