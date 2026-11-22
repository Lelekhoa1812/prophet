"""Smoke test: every helper example in the spec catalogue and every pl/ps/sc python snippet in skills/*.md must run.

Motivation vs Logic
Motivation: the planner copies skill snippets and catalogue signatures. A snippet that names a helper wrongly, or with an
argument that does not exist, costs a failed cell and a retry. This test runs them all, so a wrong name fails here first.
Logic: every snippet runs in one sandbox run, each in its own namespace after a shared prelude (a small synthetic sales
table with one entity, and the variables the docs assume, such as ``s`` and ``x``). Each snippet's result is written to
out/results.json, so one failure does not hide the others. Snippets that are not Python, or that need data this fixture
cannot give, are skipped with the reason written next to them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prophet_agent.sandbox.catalogue import catalogue_entries
from prophet_agent.sandbox.runner import run_python

SKILLS = Path(__file__).resolve().parents[3] / "src" / "prophet_agent" / "skills"

# Variables the docs assume. Each snippet runs after this, so a snippet may redefine any of them.
PRELUDE = """
import numpy as np, pandas as pd
import prophet_lib as pl
import prophet_stats as ps
import prophet_scan as sc
df = pl.load("in/sales.csv")
s, _rep = pl.prepare_series(df, value="units", time="date", freq="D")
x = s.to_numpy(dtype=float)
y = s
history = s
"""

# Skill snippets that need more than the prelude: the extra definitions run before the snippet.
EXTRA: dict[str, str] = {
    "choose-model-by-backtest.md:19": "treatment = lambda h: h.rolling(3, min_periods=1).median()",
    "bayesian-shrinkage-hierarchies.md:11":"rates = np.array([0.10, 0.22, 0.05, 0.31])\nexposure = np.array([120, 80, 40, 200])",
    "changepoints-and-regimes.md:11": "history = s.to_numpy(dtype=float)",
    "deliverables-and-handover.md:16": "fc = pd.DataFrame({'forecast': s.values}, index=s.index)",
    "forecast-combination-and-conformal.md:10": "f_model = x[-30:] * 1.02\nf_naive = np.repeat(x[-1], 30)",
    "forecast-combination-and-conformal.md:17": ("past_residuals = np.diff(x)[:120]\ncombined = np.repeat(float(x[-1]), 30)"),
    "intervals-and-trust-statement.md:20": ("y_true = x[-40:]\nq10 = y_true - 8.0\nq90 = y_true + 8.0"),
    "knn-and-analogue-imputation.md:12": ("df = pd.DataFrame({'demand': np.where(np.arange(len(x)) % 17 == 3, np.nan, x)})"),
    "timesfm-control-and-flat-forecasts.md:16": "history = s.to_numpy(dtype=float)",
}

# Snippets that cannot run in this fixture, with the reason.
SKIP: dict[str, str] = {
    "web-research-playbook.md:19": "a JSON argument block for the web_research tool, not Python",
    "smoothing-without-leakage.md:16": "defines a function only; nothing is called, so there is nothing to run",
}

# One example per catalogued helper, so every signature in the spec is also called once.
CATALOGUE_EXAMPLES: dict[str, str] = {
    "pl.load": "t = pl.load('in/sales.csv')\nassert len(t) > 0",
    "pl.save_series": "pl.save_series(s, 'out/series.json', name='units')",
    "pl.prepare_series": "out_s, rep = pl.prepare_series(df, value='units', time='date', freq='D')\nassert 'periods' in rep",
    "pl.panel": ("series, table = pl.panel(df.assign(sku='A'), 'sku', 'date', 'units', freq='D')\n"
                 "assert list(series) == ['A'] and len(table) == 1"),
    "pl.save_panel": ("series, table = pl.panel(df.assign(sku='A'), 'sku', 'date', 'units', freq='D')\n"
                      "pl.save_panel(series, 'out/panel.json')"),
    "pl.load_series": ("pl.save_series(s, 'out/series.json', name='units')\nback = pl.load_series('out/series.json')\n"
                       "assert back.name == 'units'"),
    "sc.scan": "r = sc.scan(df, 'date', 'units', entity_cols=['sku'], freq='D')\nassert 'findings' in r",
    "sc.write_scan": "r = sc.scan(df, 'date', 'units', entity_cols=['sku'], freq='D')\nsc.write_scan(r, 'out/scan.json')",
    "ps.changepoints": "shifts, rep = ps.changepoints(x, min_size=20, max_changes=5)\nassert 'n_changes' in rep",
    "ps.hampel": "flags, rep = ps.hampel(x, window=7, n_sigmas=3.0, causal=True)\nassert 'count' in rep",
    "ps.stl_outliers": "flags, rep = ps.stl_outliers(x, period=7, k=3.0)\nassert 'count' in rep",
    "ps.change_test": ("verdict, rep = ps.change_test(df['units'], recent=7, dates=df['date'])\n"
                       "assert verdict in ('rise', 'decline', 'within usual variation', 'not enough history')"),
    "pl.detect_anomalies": "res = pl.detect_anomalies(x, season=7)\nassert 'candidates' in res and 'values' in res",
    "pl.dominant_period": "period, strength = pl.dominant_period(x)\nassert strength >= 0",
    "pl.find_season": "res = pl.find_season(x)\nassert 'strongest_lag' in res",
    "pl.read_table": "t, info = pl.read_table('in/sales.csv')\nassert len(t) > 0",
    "pl.to_number": "n, info = pl.to_number(pd.Series(['1,200', '3.5', 'n/a']))\nassert n.iloc[0] == 1200 and isinstance(info, dict)",
    "pl.to_datetime": "t, info = pl.to_datetime(pd.Series(['01/02/2024']), dayfirst=True)\nassert t.iloc[0].month == 2",
    "pl.tidy_text": "t = pl.tidy_text(pd.Series(['  Acme  Ltd ', 'Acme Ltd']))\nassert t.nunique() == 1",
    "pl.build_series": "b = pl.build_series(df, 'units', 'date', freq='D')\nassert len(b) > 0",
    "pl.chart": ("pl.chart('line', data=s, title='Daily units held steady through the fixture period', "
                 "subtitle='synthetic fixture, daily', unit='units', path='out/c.png')"),
    "pl.result": "assert pl.result('total', 12.5, 'GBP') == 12.5",
    "pl.list_holidays": "h = pl.list_holidays('AU', start='2024-01-01', end='2024-12-31')\nassert isinstance(h, list)",
    "ps.Ledger": "led = ps.Ledger()\nled.add('impute', 'two gaps', 'linear')\nassert len(led.entries) == 1\nled.save('out/ledger.json')",
}

_SNIPPET = re.compile(r"```python\n(.*?)```", re.S)
_HELPER_CALL = re.compile(r"\b(pl|ps|sc)\.[A-Za-z_]")


def _skill_snippets() -> list[tuple[str, str]]:
    out = []
    for path in sorted(SKILLS.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for m in _SNIPPET.finditer(text):
            code = m.group(1)
            if not _HELPER_CALL.search(code):
                continue
            line = text[:m.start()].count("\n") + 1
            out.append((f"{path.name}:{line}", code))
    return out


def _driver(cases: list[tuple[str, str, str]]) -> str:
    """Program that runs each case in its own namespace and writes out/results.json."""
    payload = json.dumps(cases)
    return f'''
import json, io, contextlib, traceback
cases = json.loads({payload!r})
results = {{}}
for label, extra, code in cases:
    ns = {{}}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile({PRELUDE!r} + extra + "\\n" + code, label, "exec"), ns)
        results[label] = "ok"
    except BaseException as exc:
        last = traceback.format_exc().strip().splitlines()[-1]
        results[label] = "{{}}: {{}}".format(type(exc).__name__, last)[:300]
json.dump(results, open("out/results.json", "w"), indent=1)
'''


@pytest.fixture(scope="module")
def snippet_results(tmp_path_factory) -> dict[str, str]:
    csv = tmp_path_factory.mktemp("fixture") / "sales.csv"
    rng = np.random.default_rng(7)
    days = pd.date_range("2024-01-01", periods=400, freq="D")
    units = 50 + 10 * np.sin(2 * np.pi * np.arange(400) / 7) + rng.normal(0, 2, 400)
    units[[40, 41, 200]] = np.nan                                  # a few gaps, as real files have
    pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "units": units, "sku": "A"}).to_csv(csv, index=False)

    cases = []
    for label, code in _skill_snippets():
        if label in SKIP:
            continue
        cases.append((label, EXTRA.get(label, ""), code))
    for name, code in CATALOGUE_EXAMPLES.items():
        cases.append((f"catalogue:{name}", "", code))
    names = {f"{alias}.{fn}" for alias, _, fn in catalogue_entries()}
    missing_examples = sorted(names - set(CATALOGUE_EXAMPLES) - {"pl.Ledger"})
    assert not missing_examples, f"catalogue helpers without a smoke example: {missing_examples}"

    res = run_python(_driver(cases), timeout_s=300, mem_mb=1536, inputs={"sales.csv": csv})
    assert res.ok, res.stderr[-1500:]
    return json.loads((Path(res.workdir) / "out" / "results.json").read_text(encoding="utf-8"))


def test_every_skill_snippet_runs(snippet_results):
    failed = {k: v for k, v in snippet_results.items() if not k.startswith("catalogue:") and v != "ok"}
    assert not failed, "skill snippets that fail:\n" + "\n".join(f"  {k}: {v}" for k, v in sorted(failed.items()))


def test_every_catalogue_example_runs(snippet_results):
    failed = {k: v for k, v in snippet_results.items() if k.startswith("catalogue:") and v != "ok"}
    assert not failed, "catalogue examples that fail:\n" + "\n".join(f"  {k}: {v}" for k, v in sorted(failed.items()))


def test_the_snippet_set_is_not_empty():
    snippets = _skill_snippets()
    assert len(snippets) >= 15, f"expected the skills to hold python snippets; found {len(snippets)}"
