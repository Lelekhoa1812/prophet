"""Provenance guard for the SciRes defect (workflow run SciRes-20261124-100411, steps 36 to 39).

Motivation vs Logic
Motivation: with web research on and no upload, the planner found no reachable NOAA file and wrote numpy code that generated a
"representative" daily CO2 series (seeded random noise around a trend). The code was titled as NOAA GML data, so the
numbers reached the answer labelled as the source. Nothing in the sandbox, the series files or the prompt stopped it.
Logic: (c) sandbox code that builds numbers (a random generator or a typed number list) while its comments or prints name
an external source, and reads no input file, is refused before it runs. (b) a series file may carry a provenance block;
a block that claims a retrieved source must name a source id and URL with a research fetch record, a synthetic series may
never carry a source name, and a series with no block is labelled "source not recorded". (a) the planner prompt says it
must never generate, simulate or represent data for a named source, and must say when a source was not retrieved.
"""
from __future__ import annotations

import json
import time
import uuid

import pytest

from prophet_agent.sandbox import prophet_lib as pl
from prophet_agent.tools import series_io
from prophet_agent.tools.sandbox_tool import sandbox_python_tool

URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_daily_mlo.txt"

# The planner's cell from steps 36 and 38 of the SciRes run, shortened. Same shape: a seeded random generator and a title
# that names NOAA GML.
SCIRES_CODE = """import pandas as pd
import numpy as np

# Create representative CO2 daily data from NOAA GML Mauna Loa
# Source: https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_daily_mlo.txt
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
ppm = 415 + 2.5 * np.arange(len(dates)) / 365.25 + np.random.normal(0, 0.3, len(dates))
df = pd.DataFrame({'date': dates, 'ppm': ppm})
df.to_csv('out/co2_daily_mlo.csv', index=False)
print('Source: NOAA GML Mauna Loa Observatory daily CO2 averages')
"""

# A typed list of numbers under a NOAA label: the other way to invent a series inside the sandbox.
TYPED_LIST_CODE = ("# NOAA GML daily CO2 in ppm\nppm = [" + ", ".join(f"{415 + i * 0.1:.2f}" for i in range(24)) + "]\n"
                   "print(len(ppm))\n")


def _reason(code: str):
    from prophet_agent.tools.sandbox_tool import invented_source_reason
    return invented_source_reason(code)


def _key(label: str) -> str:
    import hashlib
    return hashlib.sha256(f"provenance-guard:{label}:{uuid.uuid4().hex}".encode()).hexdigest()


# ------------------------------------------------------------------ (c) sandbox guard

def test_sandbox_refuses_random_numbers_written_under_a_named_source():
    key = _key("scires")
    tool = sandbox_python_tool()
    with pytest.raises(ValueError) as err:
        tool.fn({"code": SCIRES_CODE}, key)
    message = str(err.value)
    assert "NOAA" in message and "not retrieved" in message
    from prophet_agent.sandbox.runner import SANDBOX_ROOT
    assert not (SANDBOX_ROOT / key[:32]).exists(), "a refused cell must not start a run"


def test_sandbox_refuses_a_typed_number_list_written_under_a_named_source():
    reason = _reason(TYPED_LIST_CODE)
    assert reason is not None and "NOAA" in reason
    with pytest.raises(ValueError):
        sandbox_python_tool().fn({"code": TYPED_LIST_CODE}, _key("typed"))


def test_sandbox_still_runs_random_numbers_that_name_no_source():
    code = "import numpy as np\nnp.random.seed(7)\nx = np.random.normal(0, 1, 50)\nprint(round(float(x.mean()), 3))\n"
    assert _reason(code) is None
    res = sandbox_python_tool().fn({"code": code}, _key("plain-random"))
    assert res["ok"] is True


def test_sandbox_still_runs_a_bootstrap_of_an_uploaded_file_that_names_a_source():
    code = ("import pandas as pd\nimport numpy as np\n# NOAA GML file, bootstrap resample of the upload\n"
            "df = pd.read_csv('in/co2.csv')\nidx = np.random.default_rng(1).integers(0, len(df), len(df))\n"
            "print(len(df.iloc[idx]))\n")
    assert _reason(code) is None


def test_sandbox_guard_does_not_look_at_plain_analysis_code():
    code = "import pandas as pd\nprint(pd.Series([1, 2, 3]).mean())\n"
    assert _reason(code) is None


# ------------------------------------------------------------------ (b) provenance on series files

def _fetch_record(tmp_path, monkeypatch, url=URL, ok=True):
    """A research fetch record: the cache entry the page fetcher writes for a page it read (research/fetch.py)."""
    from prophet_agent.research.search import cache_key
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    path = tmp_path / "cache" / "pages" / f"{cache_key('page', url)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"url": url, "final_url": url, "ok": ok, "saved_at": time.time()}), encoding="utf-8")


def _saved(tmp_path, monkeypatch, provenance, values=(410.1, 410.4, 410.2)):
    """Write a series with save_series under a fresh run folder, as the planner's sandbox cell would."""
    root = tmp_path / "sandbox"
    monkeypatch.setattr(series_io, "SANDBOX_ROOT", root)
    run_id = uuid.uuid4().hex
    kwargs = {} if provenance is None else {"provenance": provenance}
    pl.save_series(list(values), str(root / run_id / "out" / "series.json"), **kwargs)
    return {"source": {"run_id": run_id, "file": "out/series.json"}}


def test_a_value_copied_from_a_fetched_page_keeps_its_provenance(tmp_path, monkeypatch):
    _fetch_record(tmp_path, monkeypatch)
    prov = {"origin": "retrieved", "source_name": "NOAA GML Mauna Loa daily CO2",
            "sources": [{"id": "S2", "url": URL}]}
    args = _saved(tmp_path, monkeypatch, prov)
    assert [v for _, arr in series_io.resolve_series(args) for v in arr.tolist()] == [410.1, 410.4, 410.2]
    kept = series_io.series_provenance(args)
    assert kept == prov
    label = series_io.provenance_label(kept)
    assert "NOAA GML" in label and "S2" in label and URL in label


def test_a_retrieved_claim_without_a_fetch_record_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "empty-cache"))
    prov = {"origin": "retrieved", "source_name": "NOAA GML", "sources": [{"id": "S2", "url": URL}]}
    args = _saved(tmp_path, monkeypatch, prov)
    with pytest.raises(ValueError, match="fetch record"):
        series_io.resolve_series(args)


def test_a_retrieved_claim_needs_a_source_id_and_a_url(tmp_path, monkeypatch):
    _fetch_record(tmp_path, monkeypatch)
    args = _saved(tmp_path, monkeypatch, {"origin": "retrieved", "source_name": "NOAA GML", "sources": [{"url": URL}]})
    with pytest.raises(ValueError, match="id"):
        series_io.resolve_series(args)


def test_a_synthetic_series_can_never_carry_a_named_source(tmp_path, monkeypatch):
    args = _saved(tmp_path, monkeypatch, {"origin": "synthetic", "source_name": "NOAA GML Mauna Loa"})
    with pytest.raises(ValueError, match="synthetic"):
        series_io.resolve_series(args)


def test_a_synthetic_series_is_labelled_synthetic_and_names_no_source(tmp_path, monkeypatch):
    args = _saved(tmp_path, monkeypatch, {"origin": "synthetic"})
    assert series_io.resolve_series(args)
    label = series_io.provenance_label(series_io.series_provenance(args))
    assert label.startswith("synthetic") and "NOAA" not in label


def test_a_series_with_no_recorded_source_is_labelled_and_still_accepted(tmp_path, monkeypatch):
    args = _saved(tmp_path, monkeypatch, None)
    assert series_io.resolve_series(args)
    assert series_io.series_provenance(args) is None
    assert series_io.provenance_label(None) == "source not recorded"


def test_a_user_supplied_series_is_labelled_as_the_person_supplied_it(tmp_path, monkeypatch):
    args = _saved(tmp_path, monkeypatch, {"origin": "user", "source_name": "the uploaded CSV"})
    assert series_io.resolve_series(args)
    assert series_io.provenance_label(series_io.series_provenance(args)).startswith("user-supplied")


def test_values_passed_in_a_call_are_checked_the_same_way():
    assert series_io.series_provenance({"values": [1.0, 2.0]}) is None
    with pytest.raises(ValueError, match="origin"):
        series_io.check_provenance({"source_name": "NOAA GML"}, "series")


# ------------------------------------------------------------------ (a) planner prompt

def test_the_planner_prompt_forbids_representing_data_for_a_named_source():
    from prophet_agent import prompts
    planner = prompts.current("planner")
    text = planner.text.lower()
    assert planner.version == "planner.v37"
    assert "never generate, simulate" in text and "represent" in text
    assert "not retrieved" in text
    assert "provenance" in text and "origin" in text
