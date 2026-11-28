"""The signature catalogue in the sandbox spec: every name exists and its parameters match the code.

Motivation vs Logic
Motivation: a spec that names helpers which do not exist, or lists the wrong parameters, sends the planner into the same
guessing that caused half the failed cells.
Logic: the catalogue is generated with inspect from the real functions. These tests re-read the generated text, check each
name resolves to a callable, check the parameter names against the signature, and check the spec carries the catalogue
and stays under the length budget.
"""
from __future__ import annotations

import inspect
import json
import re

from prophet_agent.sandbox import catalogue
from prophet_agent.sandbox import prophet_lib, prophet_scan, prophet_stats
from prophet_agent.tools.sandbox_tool import SANDBOX_PYTHON_SPEC

MODULES = {"pl": prophet_lib, "ps": prophet_stats, "sc": prophet_scan}
LINE = re.compile(r"^(\w+)\.(\w+)\((.*?)\) -> ", re.M)
PARAM_NAME = re.compile(r"(?:^|, )\*{0,2}([A-Za-z_]\w*)(?=$|=|, )")


def _catalogue_lines() -> list[str]:
    return catalogue.build_catalogue().splitlines()


def test_every_catalogued_name_exists_and_is_callable():
    for alias, module, name in catalogue.catalogue_entries():
        assert getattr(module, name, None) is not None, f"{alias}.{name} does not exist"
        assert callable(getattr(module, name)), f"{alias}.{name} is not callable"


def test_every_line_names_a_real_helper_with_the_real_parameters():
    seen = 0
    for line in _catalogue_lines():
        if line.startswith("ps.Ledger()"):           # the Ledger line lists methods, checked by the Ledger test below
            assert hasattr(prophet_stats.Ledger, "entries") and isinstance(prophet_stats.Ledger.entries, property)
            continue
        m = LINE.match(line)
        assert m, f"unparsed catalogue line: {line}"
        alias, name, params = m.groups()
        module = MODULES[alias]
        fn = getattr(module, name)
        expected = [p.name for p in inspect.signature(fn).parameters.values() if p.name != "self"]
        listed = PARAM_NAME.findall(params)
        assert listed == expected, f"{alias}.{name}: catalogue lists {listed}, code has {expected}"
        seen += 1
    assert seen == len(catalogue.catalogue_entries())


def test_the_planner_names_appear_with_their_real_keywords():
    text = catalogue.build_catalogue()
    assert "ps.hampel(x, window=7, n_sigmas=3.0, causal=True) -> (ndarray,dict)" in text
    assert "sc.scan(df, time_col, value_col, entity_cols=None, freq=None, max_entities=200) -> dict" in text
    assert "pl.save_series(series, path='out/series.json', *, freq=None, covariates=None, name=None, provenance=None, dates=None) -> str" in text
    assert "ps.Ledger()" in text and "entries" in text


def test_the_spec_carries_the_generated_catalogue_and_stays_under_budget():
    description = SANDBOX_PYTHON_SPEC["description"]
    assert catalogue.build_catalogue() in description
    assert len(description) < 3500, len(description)
    assert "not polars" in description                 # the polars confusion is named in the spec
    assert len(json.dumps(SANDBOX_PYTHON_SPEC)) < 4000


def test_the_catalogue_covers_the_most_used_helpers_and_the_panel_ones():
    names = {f"{alias}.{name}" for alias, _, name in catalogue.catalogue_entries()}
    for needed in ("pl.load", "pl.save_series", "pl.prepare_series", "sc.scan", "ps.changepoints", "ps.hampel",
                   "pl.detect_anomalies", "pl.find_season", "pl.panel", "pl.save_panel"):
        assert needed in names, needed
