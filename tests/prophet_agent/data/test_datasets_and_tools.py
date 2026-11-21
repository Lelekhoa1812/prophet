"""Uploaded files, the analysis helpers that read them, and the tools built on them (no real model, no network)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.datasets import DatasetStore, clean_name
from prophet_agent.forecast import timesfm as timesfm_mod
from prophet_agent.forecast.rolling import rolling_backtest
from prophet_agent.forecast.types import Forecast
from prophet_agent.sandbox.runner import run_python
from prophet_agent.tools import default_tools, tool_catalogue
from prophet_agent.tools.series_io import resolve_series


class StubEngine:
    checkpoint, global_context = "stub", 64

    def forecast_batch(self, series_list, horizon):
        n = len(series_list)
        point = np.stack([np.full(horizon, float(np.asarray(x)[-1])) for x in series_list])
        q = np.stack([np.tile(p[:, None], (1, 9)) + np.linspace(-1, 1, 9) for p in point])
        return Forecast(point, q)


@pytest.fixture()
def stub(monkeypatch):
    monkeypatch.setattr(timesfm_mod.TimesFMEngine, "get", classmethod(lambda cls: StubEngine()))


def test_store_cleans_names_hashes_and_caps(tmp_path):
    store = DatasetStore(tmp_path)
    meta = store.add("../../etc/pass wd.csv", b"a,b\n1,2\n")
    assert meta["name"] == "pass wd.csv" and len(meta["sha256"]) == 64
    assert store.path(meta["id"]).read_bytes() == b"a,b\n1,2\n"
    with pytest.raises(ValueError):
        store.add("x.csv", b"")
    with pytest.raises(KeyError):
        store.get("../..")
    assert clean_name("") == "data"
    assert store.mount([meta["id"], meta["id"]]).keys() == {"pass wd.csv", "pass wd_2.csv"}


def test_sandbox_mounts_inputs_read_only_and_exposes_the_helper(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("x\n1\n")
    res = run_python("import prophet_lib\nprint(open('in/d.csv').read().strip())\n"
                     "try:\n    open('in/d.csv','w')\nexcept PermissionError as e:\n    print('blocked')\n",
                     inputs={"d.csv": f})
    assert res.ok, res.stderr
    assert "blocked" in res.stdout


PROFILE = """
import json, prophet_lib as pl
print(json.dumps(pl.profile('in/f.csv')))
"""


def _profile(tmp_path, text, name="f.csv", encoding="utf-8"):
    f = tmp_path / name
    f.write_bytes(text.encode(encoding))
    res = run_python(PROFILE, inputs={name: f}, timeout_s=60)
    assert res.ok, res.stderr
    return json.loads(res.stdout)


def test_profile_reads_semicolons_decimal_commas_and_flags_sentinels(tmp_path):
    rows = "\n".join(f"{d:02d}/03/2024;{1 + i / 10:.1f}".replace(".", ",") + ";" + ("-999" if i % 7 == 0 else str(i))
                     for i, d in enumerate(range(1, 29)))
    p = _profile(tmp_path, "Date;Value;Code\n" + rows + "\n")
    cols = {c["name"]: c for c in p["columns"]}
    assert p["read"]["delimiter"] == "';'"
    assert cols["Value"]["kind"] == "number" and cols["Value"]["decimal_comma_values"] > 0
    assert cols["Code"]["sentinel_candidate"]["value"] == -999
    assert cols["Date"]["kind"] == "time"


def test_profile_flags_ambiguous_day_month_and_headerless_and_bad_rows(tmp_path):
    p = _profile(tmp_path, "when,v\n" + "\n".join(f"0{(i % 9) + 1}/0{((i + 3) % 9) + 1}/2023,{i}" for i in range(30)) + "\n")
    assert any("day-first and month-first" in i for i in p["issues"])
    q = _profile(tmp_path, "1,10\n2,12\n3,11\n4,15\n5,14\n6,16\n")
    assert q["read"]["header"].startswith("none found") and q["rows"] == 6
    r = _profile(tmp_path, "a,b\n1,2\n3\n4,5\n6,7,8\n")
    assert r["read"]["problems"] and r["rows"] == 2


def test_profile_handles_other_encodings(tmp_path):
    p = _profile(tmp_path, "Datum;Menge\n01.01.2024;5\n02.01.2024;6\n03.01.2024;7\n", encoding="utf-16")
    assert p["rows"] == 3


def test_series_io_by_values_and_by_reference_with_guards(tmp_path, stub):
    res = run_python("import prophet_lib as pl\npl.save_series({'a':[1,2,3,4],'b':[5,6,7,8]})\n"
                     "import json; print(json.load(open('out/series.json'))['series'].keys())")
    assert res.ok, res.stderr
    named = resolve_series({"source": {"run_id": res.run_id, "file": "out/series.json"}})
    assert [n for n, _ in named] == ["a", "b"]
    assert [n for n, _ in resolve_series({"source": {"run_id": res.run_id, "name": "b"}})] == ["b"]
    for bad in ({"values": [1, float("nan")]}, {"values": []}, {}, {"values": [1], "source": {}},
                {"source": {"run_id": "../" * 8, "file": "out/series.json"}},
                {"source": {"run_id": res.run_id, "file": "../manifest.json"}},
                {"source": {"run_id": res.run_id, "name": "zzz"}}):
        with pytest.raises(ValueError):
            resolve_series(bad)


def test_save_series_refuses_unresolved_gaps():
    res = run_python("import prophet_lib as pl, numpy as np\npl.save_series([1.0, float('nan'), 3.0])\n")
    assert not res.ok and "missing values" in res.stderr


def test_forecast_tool_takes_a_series_by_reference_and_negative_history_is_not_clipped(stub):
    from prophet_agent.tools.timesfm_tool import _run
    ref = run_python("import prophet_lib as pl\npl.save_series([-3.0, -2.0, -4.0, -1.0, -2.5])\n")
    out = _run({"source": {"run_id": ref.run_id}, "horizon": 3}, "k")
    assert out["point"] == [-2.5, -2.5, -2.5] and out["config"]["history_points"] == 5
    multi = run_python("import prophet_lib as pl\npl.save_series({'a':[1.0,2.0],'b':[3.0,4.0]})\n")
    assert sorted(_run({"source": {"run_id": multi.run_id}, "horizon": 2}, "k")["series"]) == ["a", "b"]


def test_rolling_backtest_reports_all_models_and_refuses_short_series(stub):
    rng = np.random.default_rng(1)
    x = 50 + 10 * np.sin(np.arange(120) * 2 * np.pi / 12) + rng.normal(0, 1, 120)
    out = rolling_backtest(x, 12, n_origins=4, season=12, engine=StubEngine())
    assert out["origins"] == 4 and out["season_used"] == 12
    assert set(out["models"]) == {"timesfm", "naive", "seasonal_naive", "drift", "mean_last_season", "tsb_expected"}
    assert out["models"]["seasonal_naive"]["mase"] < out["models"]["naive"]["mase"]
    assert out["best_baseline"] == "seasonal_naive" and 0 <= out["timesfm_range_coverage_80"] <= 1
    with pytest.raises(ValueError):
        rolling_backtest(x[:20], 12, engine=StubEngine())


def test_tool_set_offers_data_tools_only_with_files():
    assert "inspect_data" not in default_tools()
    assert "inspect_data" in default_tools({"a.csv": __import__("pathlib").Path("a.csv")})
    assert {"timesfm_forecast", "backtest_series", "forecast_report", "sandbox_python", "inspect_data"} <= set(tool_catalogue())
    assert all(t.title and t.activity and t.doc for t in tool_catalogue().values())


class Scripted:
    def __init__(self, plans, evals):
        self.plans, self.evals = list(plans), list(evals)

    def decide(self, messages, schema, name):
        return (self.plans if name == "plan" else self.evals).pop(0)


def test_mission_with_an_uploaded_file_runs_inspect_data_through_the_api(tmp_path):
    brain = Scripted([{"action": "call_tool", "tool": "inspect_data", "args": {}, "rationale": "look first"},
                      {"action": "finish", "rationale": "done"}],
                     [{"verdict": "met", "reason": "ok", "answer": "read it"}])
    service = MissionService(tmp_path / "t.sqlite", lambda: brain, tool_catalogue(), tools_factory=default_tools)
    client = TestClient(create_app(service), base_url="http://127.0.0.1:8000")
    up = client.post("/api/datasets?name=sales.csv", content=b"week,units\n" + b"\n".join(b"2024-01-%02d,%d" % (d, d) for d in range(1, 29)))
    assert up.status_code == 201
    did = up.json()["id"]
    assert client.get("/api/datasets").json()[0]["id"] == did
    assert client.post("/api/datasets?name=e.csv", content=b"").status_code == 422
    assert client.post("/api/missions", json={"goal": "forecast my sales", "dataset_ids": ["0" * 32]}).status_code == 422
    # NIM keys are absent in tests, so build the mission through the service as the route would.
    mid = service.create("forecast my sales", dataset_ids=[did])
    service.start(mid)
    service.wait(mid, timeout=60)
    snap = client.get(f"/api/missions/{mid}").json()
    assert snap["status"] == "completed", snap["events"][-3:]
    assert snap["datasets"][0]["name"] == "sales.csv"
    step = next(s for s in snap["steps"] if s["name"].startswith("act"))
    assert step["output"]["result"]["rows"] == 28 and step["output"]["result"]["columns"], "the planner sees a compact profile"
