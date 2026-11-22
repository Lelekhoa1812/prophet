"""Panel pack: one forecast_report call delivers a whole panel (up to 50 series), with a grid, a workbook, a report and a PDF.

No real TimesFM: the engine is a fake that forecasts the last value plus 1.0, so a naive centre rule is visibly different
from TimesFM's centre. The packs themselves are built in the real sandbox, as the other artefact tests do.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prophet_agent.artifacts import MAX_PANEL, build_panel_pack
from prophet_agent.forecast import timesfm as timesfm_mod
from prophet_agent.forecast.types import Forecast
from prophet_agent.sandbox.runner import verify_manifest
from prophet_agent.tools import report_tool as rt
from prophet_agent.tools import series_io

RUN = "b" * 32
N, H = 120, 4


class PanelEngine:
    checkpoint = "stub"
    global_context = 64

    def forecast_batch(self, series_list, horizon):
        point = np.stack([np.full(horizon, float(np.asarray(s)[-1]) + 1.0) for s in series_list])
        return Forecast(point, np.repeat(point[:, :, None], 9, axis=2) + np.linspace(-2, 2, 9))


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch):
    monkeypatch.setattr(timesfm_mod.TimesFMEngine, "get", classmethod(lambda cls: PanelEngine()))


def _values(offset: float, n: int = N) -> list[float]:
    return [round(50 + offset + 0.1 * i + 3 * np.sin(2 * np.pi * i / 7), 4) for i in range(n)]


def _daily(start: str, n: int) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, periods=n)]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A sandbox root under tmp_path holding a three-station file (one station dated), so no real data is read."""
    monkeypatch.setattr(series_io, "SANDBOX_ROOT", tmp_path)
    out = tmp_path / RUN / "out"
    out.mkdir(parents=True)
    stations = {"Station A": _values(0), "Station B": _values(10), "Station C": _values(20)}
    (out / "series.json").write_text(json.dumps({"series": stations, "dates": {"Station A": _daily("2025-01-01", N)}}))
    many = {f"s{i}": _values(i) for i in range(51)}
    (out / "many.json").write_text(json.dumps({"series": many}))
    return stations


def _args(**kw):
    base = {"source": {"run_id": RUN, "file": "out/series.json"}, "horizon": H, "title": "Air quality, 14 days"}
    base.update(kw)
    return base


def _series(name: str, n: int = 90, h: int = 7, dated: bool = False) -> dict:
    hist = [10 + (i % 7) * 0.5 + 0.01 * i for i in range(n)]
    pt = [hist[-1] + 0.1 * k for k in range(1, h + 1)]
    item = {"name": name, "history": hist, "point": pt, "q10": [p - 1 for p in pt], "q90": [p + 1 for p in pt]}
    if dated:
        every = _daily("2025-01-01", n + h)
        item.update(dates=every[:n], future_dates=every[n:])
    return item


# --------------------------------------------------------------------------------------------- the builder --

def test_panel_of_three_yields_every_kind_of_file_and_the_manifest():
    res = build_panel_pack(series=[_series("A", dated=True), _series("B"), _series("C")], title="Three stations")
    assert res.ok, res.stderr
    names = {Path(o.path).name for o in res.outputs}
    assert {"forecast_grid.png", "forecast_grid.svg", "forecast.xlsx", "report.docx", "report.pdf"} <= names
    assert "forecast_grid_2.png" not in names
    assert verify_manifest(Path(res.workdir) / "manifest.json") == []


def test_workbook_summary_has_a_row_per_series_and_the_document_a_table():
    from docx import Document
    import openpyxl
    series = [_series("A", dated=True), _series("B"), _series("C")]
    res = build_panel_pack(series=series, title="Three stations", note="Synthetic.")
    assert res.ok, res.stderr
    out = Path(res.workdir) / "out"
    wb = openpyxl.load_workbook(out / "forecast.xlsx", read_only=True)
    summary = list(wb["Summary"].iter_rows(values_only=True))
    assert summary[0][0] == "Series" and len(summary) - 1 == 3
    assert [r[0] for r in summary[1:]] == ["A", "B", "C"]
    b = summary[2]
    assert b[1] == pytest.approx(series[1]["history"][-1]) and b[2] == pytest.approx(series[1]["point"][0])
    assert b[3] == pytest.approx(series[1]["point"][-1]) and b[4] < b[3] < b[5]
    forecast_rows = list(wb["Forecast"].iter_rows(values_only=True))
    wb.close()
    assert len(forecast_rows) - 1 == 3 * 7
    assert forecast_rows[1][1] == series[0]["future_dates"][0] and forecast_rows[8][1] is None  # B has no dates
    doc = Document(out / "report.docx")
    assert len(doc.inline_shapes) == 1 and len(doc.tables[0].rows) == 4
    assert any("Synthetic." in p.text for p in doc.paragraphs)


def test_more_than_twelve_series_go_on_to_a_second_grid_figure():
    res = build_panel_pack(series=[_series(f"s{i:02d}", n=60, h=3) for i in range(13)], title="Thirteen")
    assert res.ok, res.stderr
    names = {Path(o.path).name for o in res.outputs}
    assert {"forecast_grid.png", "forecast_grid_2.png", "forecast_grid_2.svg"} <= names
    assert verify_manifest(Path(res.workdir) / "manifest.json") == []


def test_a_panel_outside_one_to_fifty_series_is_refused_before_any_build():
    with pytest.raises(ValueError, match="1 to 50"):
        build_panel_pack(series=[_series(f"s{i}") for i in range(MAX_PANEL + 1)], title="too many")
    assert MAX_PANEL == 50
    with pytest.raises(ValueError, match="1 to 50"):
        build_panel_pack(series=[], title="none")


def test_mismatched_lengths_and_duplicate_names_are_refused():
    bad = _series("A")
    bad["q10"] = bad["q10"][:-1]
    with pytest.raises(ValueError, match="same length"):
        build_panel_pack(series=[bad], title="t")
    with pytest.raises(ValueError, match="more than once"):
        build_panel_pack(series=[_series("A"), _series("A")], title="t")


# ------------------------------------------------------------------------------------- the report tool --

def test_panel_call_forecasts_every_series_and_builds_one_pack(sandbox):
    out = rt._run(_args(), "k")
    assert out["n_series"] == 3 and set(out["forecast"]["series"]) == set(sandbox)
    assert len(out["files"]) == 5 and out["manifest"].endswith("manifest.json")
    assert verify_manifest(out["manifest"]) == []


def test_centre_dict_applies_naive_only_to_the_named_series(sandbox):
    out = rt._run(_args(centre={"Station B": "naive"}), "k")
    fcs = out["forecast"]["series"]
    assert fcs["Station B"]["centre"]["rule"] == "naive"
    assert "centre" not in fcs["Station A"] and "centre" not in fcs["Station C"]
    assert all(v == pytest.approx(sandbox["Station B"][-1], abs=1e-3) for v in fcs["Station B"]["point"])
    assert fcs["Station A"]["point"][0] == pytest.approx(sandbox["Station A"][-1] + 1.0, abs=1e-3)


def test_a_single_centre_rule_applies_to_every_series(sandbox):
    out = rt._run(_args(centre="naive"), "k")
    assert all(fc["centre"]["rule"] == "naive" for fc in out["forecast"]["series"].values())


def test_calibrate_with_a_panel_is_refused_not_ignored(sandbox):
    with pytest.raises(ValueError, match="calibrate applies to one series"):
        rt._run(_args(calibrate=True), "k")


def test_control_options_with_a_panel_are_refused(sandbox):
    with pytest.raises(ValueError, match="apply to one series"):
        rt._run(_args(ignore_ranges=[[3, 5]]), "k")


def test_bad_centre_names_and_rules_are_refused(sandbox):
    with pytest.raises(ValueError, match="not in the panel"):
        rt._run(_args(centre={"Station Z": "naive"}), "k")
    with pytest.raises(ValueError, match="must be one of"):
        rt._run(_args(centre={"Station A": "median"}), "k")


def test_a_file_with_more_than_fifty_series_is_refused(sandbox):
    with pytest.raises(ValueError, match="up to 50"):
        rt._run({"source": {"run_id": RUN, "file": "out/many.json"}, "horizon": H, "title": "t"}, "k")


def test_one_series_keeps_the_flat_result_with_no_panel_keys():
    out = rt._run({"values": _values(0), "horizon": H, "title": "One"}, "k")
    assert "n_series" not in out and set(out) == {"forecast", "run_id", "manifest", "files"}
    assert "point" in out["forecast"] and "series" not in out["forecast"]


def test_observe_gives_a_line_per_series_with_range_and_files(sandbox):
    out = rt._run(_args(centre={"Station B": "naive"}), "k")
    view = rt.observe(out)
    assert view["n_series"] == 3 and set(view["series"]) == set(sandbox)
    line = view["series"]["Station B"]
    assert line["centre"] == "naive" and line["health"] and "point_last_step" in line and "q10_last_step" in line
    assert set(line["range_vs_centre_pct"]) == {"first_step", "last_step"}
    assert view["series"]["Station A"]["centre"] == "timesfm"
    assert view["run_id"] == out["run_id"] and view["manifest"] == out["manifest"] and len(view["files"]) == 5


def test_observe_shows_at_most_thirty_series_and_counts_the_rest():
    fc = {"point": [1.0, 2.0], "quantiles": {"0.1": [0.5, 1.5], "0.9": [1.5, 2.5]}, "health": {"verdict": "ok"}}
    result = {"forecast": {"series": {f"s{i}": fc for i in range(35)}}, "n_series": 35, "run_id": "r", "manifest": "m", "files": []}
    view = rt.observe(result)
    assert len(view["series"]) == 30 and view["not_shown"] == 5 and view["n_series"] == 35
