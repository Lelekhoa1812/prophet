"""pl.chart: the standard a chart must meet (finding title, subtitle, zero-based bars, direct labels, caps) and its report."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "prophet_agent" / "sandbox"))
import prophet_lib as pl  # noqa: E402

IDX = pd.date_range("2024-01-01", periods=60, freq="W")
S = pd.Series(100 + np.arange(60) * 1.5, index=IDX)


def test_line_chart_writes_png_svg_and_the_plotted_numbers(tmp_path):
    r = pl.chart("line", pd.DataFrame({"A": S, "B": S * 0.5}), title="A stays well above B all year", subtitle="Units per week",
                 unit="units", path=str(tmp_path / "l.png"))
    assert r["problems"] == [] and r["checks"]["labels_overlap"] and r["checks"]["colour_blind_safe_palette"]
    names = {Path(f).suffix for f in r["files"]}
    assert names == {".png", ".svg", ".csv"}
    csv = pd.read_csv(tmp_path / "l.csv", index_col=0)
    assert list(csv.columns) == ["A", "B"] and len(csv) == 60            # the CSV holds exactly what was drawn
    assert (tmp_path / "l.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_title_must_state_a_finding_and_a_subtitle_is_required(tmp_path):
    with pytest.raises(ValueError, match="finding"):
        pl.chart("line", S, title="Sales chart", subtitle="x", path=str(tmp_path / "a.png"))
    with pytest.raises(ValueError, match="subtitle"):
        pl.chart("line", S, title="Sales rose steadily this year", path=str(tmp_path / "a.png"))


def test_too_many_lines_or_bars_are_refused_with_the_fix(tmp_path):
    many = pd.DataFrame({f"s{i}": S for i in range(7)})
    with pytest.raises(ValueError, match="Other"):
        pl.chart("line", many, title="Seven lines are far too many", subtitle="x", path=str(tmp_path / "a.png"))
    bars = pd.Series(range(1, 20), index=[f"c{i}" for i in range(19)])
    with pytest.raises(ValueError, match="Other"):
        pl.chart("bar", bars, title="Nineteen bars are too many here", subtitle="x", path=str(tmp_path / "b.png"))


def test_bars_are_sorted_start_at_zero_and_carry_values(tmp_path):
    r = pl.chart("bar", pd.Series({"North": 512, "South": 400, "Online": 800}), title="Online is the largest channel at 800 units",
                 subtitle="Units by channel, Q3", unit="units", path=str(tmp_path / "b.png"))
    assert r["checks"]["zero_baseline"] and r["problems"] == []
    assert list(pd.read_csv(tmp_path / "b.csv", index_col=0).index) == ["Online", "North", "South"]


def test_negative_bars_still_include_zero(tmp_path):
    r = pl.chart("bar", pd.Series({"a": -30, "b": 12}), title="Region a lost thirty units overall", subtitle="Change in units",
                 unit="units", path=str(tmp_path / "n.png"))
    assert r["checks"]["zero_baseline"]


def test_forecast_chart_marks_the_origin_nests_bands_and_checks_them(tmp_path):
    pt = S.iloc[-1] + np.arange(1, 13) * 1.5
    r = pl.chart("forecast", title="Demand keeps rising for another quarter", subtitle="Units per week", unit="units", history=S,
                 point=list(pt), bands=[(50, list(pt - 9), list(pt + 9)), (80, list(pt - 20), list(pt + 20))],
                 path=str(tmp_path / "f.png"))
    assert r["problems"] == [] and r["checks"]["bands"] == [80.0, 50.0] and r["checks"]["forecast_origin_marked"]
    csv = pd.read_csv(tmp_path / "f.csv", index_col=0)
    assert {"history", "forecast", "lower_80", "upper_80", "lower_50", "upper_50"} <= set(csv.columns)
    with pytest.raises(ValueError, match="lower value above"):
        pl.chart("forecast", title="Demand keeps rising for another quarter", subtitle="x", history=S, point=list(pt),
                 bands=[(80, list(pt + 5), list(pt))], path=str(tmp_path / "g.png"))


def test_missing_points_are_reported_not_zeroed(tmp_path):
    s = S.copy()
    s.iloc[10:13] = np.nan
    r = pl.chart("line", s, title="Three weeks of data are missing", subtitle="Units per week", unit="units", path=str(tmp_path / "m.png"))
    assert any("missing" in p for p in r["problems"])


def test_histogram_scatter_and_small_multiples_render(tmp_path):
    rng = np.random.default_rng(2)
    assert pl.chart("hist", rng.normal(50, 10, 200), title="Order sizes cluster near fifty units", subtitle="Units per order",
                    unit="units", path=str(tmp_path / "h.png"))["problems"] == []
    df = pd.DataFrame({"x": rng.normal(size=80)})
    df["y"] = df["x"] * 2 + rng.normal(size=80)
    r = pl.chart("scatter", df, x="x", y="y", trend=True, title="Price and demand move together here", subtitle="Each dot is one week",
                 path=str(tmp_path / "s.png"))
    assert r["checks"]["correlation"] > 0.8
    r = pl.chart("small_multiples", pd.DataFrame({"A": S, "B": S * 0.5, "C": S * 2}), title="All three products rise together",
                 subtitle="Units per week", path=str(tmp_path / "sm.png"))
    assert r["checks"]["shared_y_axis"]


def test_query_reads_a_large_file_in_place_and_is_fenced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    n = 1_200_000
    rng = np.random.default_rng(0)
    big = pd.DataFrame({"region": rng.choice(["N", "S", "E", "W"], n), "units": rng.integers(1, 10, n),
                        "day": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")})
    f = tmp_path / "big.csv"
    big.to_csv(f, index=False)
    rep = pl.size_report(f)
    assert rep["rows"] == n and "pl.query" in rep["strategy"] and {c["name"] for c in rep["columns"]} == {"region", "units", "day"}
    got = pl.query("SELECT region, sum(units) AS units, count(*) AS rows FROM t GROUP BY 1 ORDER BY 1", t=f)
    exp = big.groupby("region").agg(units=("units", "sum"), rows=("units", "size")).reset_index()
    assert got["units"].tolist() == exp["units"].tolist() and got["rows"].sum() == n
    assert pl.query("SELECT * FROM t", max_rows=100, t=f).attrs["truncated"] is True
    with pytest.raises(Exception, match="(?i)permission|not allowed|disabled"):
        pl.query("SELECT * FROM read_csv_auto('/etc/hosts')", t=f)
    with pytest.raises(ValueError, match="one SELECT"):
        pl.query("COPY t TO 'x.csv'", t=f)
    with pytest.raises(ValueError, match="one SELECT"):
        pl.query("SELECT 1; SELECT 2", t=f)


def test_small_files_are_told_to_load_whole(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pd.DataFrame({"a": range(10)}).to_csv(tmp_path / "s.csv", index=False)
    assert "pl.load" in pl.size_report(tmp_path / "s.csv")["strategy"]


def test_a_table_with_named_columns_draws_like_a_frame_with_an_index(tmp_path):
    df = pd.DataFrame({"Week": IDX[:40], "Online": range(40), "Store": range(40, 80)})
    r = pl.chart("line", df, x="Week", y=["Online", "Store"], title="Store sells more than online every week", subtitle="Units per week",
                 unit="units", path=str(tmp_path / "t.png"))
    assert r["problems"] == [] and list(pd.read_csv(tmp_path / "t.csv", index_col=0).columns) == ["Online", "Store"]
    b = pl.chart("bar", pd.DataFrame({"c": ["a", "b"], "v": [3, 5]}), x="c", y="v", title="Category b is larger than category a",
                 subtitle="Units", unit="units", path=str(tmp_path / "b2.png"))
    assert b["checks"]["zero_baseline"]
    with pytest.raises(ValueError, match="not a column"):
        pl.chart("line", df, x="Nope", y="Online", title="A title with several words", subtitle="s", path=str(tmp_path / "e.png"))


def test_group_draws_one_line_per_region_not_one_tangled_line(tmp_path):
    rows = pd.DataFrame({"month": ["2024-01", "2024-01", "2024-02", "2024-02"], "region": ["North", "South", "North", "South"],
                         "amount": [10.0, 4.0, 12.0, 5.0]})
    r = pl.chart("line", rows, x="month", y="amount", group="region", title="North earned more than South in both months",
                 subtitle="Revenue by month, 2024", unit="USD", path=str(tmp_path / "g.png"))
    assert r["problems"] == []
    csv = pd.read_csv(tmp_path / "g.csv", index_col=0)
    assert list(csv.columns) == ["North", "South"] and len(csv) == 2


def test_repeated_x_without_group_is_refused_with_the_fix(tmp_path):
    rows = pd.DataFrame({"month": ["2024-01", "2024-01", "2024-02", "2024-02"], "region": ["North", "South", "North", "South"],
                         "amount": [10.0, 4.0, 12.0, 5.0]})
    with pytest.raises(ValueError, match="group="):
        pl.chart("line", rows, x="month", y="amount", title="North earned more than South in both months",
                 subtitle="Revenue by month, 2024", path=str(tmp_path / "g.png"))
