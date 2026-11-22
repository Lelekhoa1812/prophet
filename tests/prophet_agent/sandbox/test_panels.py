"""Panels: entities that share dates are not summed silently (prophet_lib: prepare_series, build_series, panel, save_panel).

Motivation vs Logic
Motivation: three stations sharing five dates gave a first PM2.5 value of 42.0 (the sum of the three). The Beijing mission
had 12 stations, so a default sum was a wrong number that looked right.
Logic: when dates repeat across a text column that names a few entities, the helpers raise a ValueError that names the
column and the fix. entity= and key= select one entity; agg= given explicitly is a choice the caller made; pl.panel
prepares every entity with its own report row, and pl.save_panel writes them for the forecast tools.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prophet_agent.sandbox import prophet_lib as pl
from prophet_agent.sandbox.runner import run_python


def _three_stations() -> pd.DataFrame:
    """Three stations on the same 30 days. Station C misses two days, so it has gaps to fill."""
    days = pd.date_range("2024-03-01", periods=30, freq="D")
    rows = []
    for station, base in (("Beijing", 80.0), ("Shanghai", 50.0), ("Chengdu", 65.0)):
        for i, d in enumerate(days):
            if station == "Chengdu" and i in (10, 11):
                continue
            rows.append({"date": d.strftime("%Y-%m-%d"), "station": station, "pm25": base + (i % 7)})
    return pd.DataFrame(rows)


def test_prepare_series_refuses_stations_that_share_dates():
    with pytest.raises(ValueError, match=r"names several entities.*entity='station'.*pl\.panel"):
        pl.prepare_series(_three_stations(), value="pm25", time="date")


def test_build_series_refuses_stations_that_share_dates():
    with pytest.raises(ValueError, match="entity='station'"):
        pl.build_series(_three_stations(), "pm25", "date", freq="D")


def test_an_explicit_agg_is_an_accepted_choice_and_sums_as_asked():
    df = _three_stations()
    s, rep = pl.prepare_series(df, value="pm25", time="date", agg="sum")
    assert rep["duplicate_dates"] == 58 and s.iloc[0] == pytest.approx(80.0 + 50.0 + 65.0)   # 88 rows on 30 dates


def test_entity_and_key_select_one_station_with_a_unique_daily_index():
    df = _three_stations()
    s, rep = pl.prepare_series(df, value="pm25", time="date", freq="D", entity="station", key="Shanghai")
    assert len(s) == 30 and s.index.is_unique and rep["duplicate_dates"] == 0
    assert s.iloc[0] == pytest.approx(50.0)
    b = pl.build_series(df, "pm25", "date", freq="D", entity="station", key="Beijing")
    assert b.index.is_unique and len(b) == 30


def test_entity_without_key_and_an_unknown_key_are_refused():
    df = _three_stations()
    with pytest.raises(ValueError, match=r"needs key=.*pl\.panel"):
        pl.prepare_series(df, value="pm25", time="date", entity="station")
    with pytest.raises(ValueError, match="no rows with station == 'Osaka'"):
        pl.prepare_series(df, value="pm25", time="date", entity="station", key="Osaka")


def test_a_single_entity_table_is_not_mistaken_for_a_panel():
    df = _three_stations()
    one = df[df["station"] == "Beijing"]
    s, rep = pl.prepare_series(one, value="pm25", time="date", freq="D")
    assert len(s) == 30 and rep["duplicate_dates"] == 0


def test_panel_returns_one_series_per_entity_and_a_report_row_each():
    series, table = pl.panel(_three_stations(), "station", "date", "pm25", freq="D")
    assert sorted(series) == ["Beijing", "Chengdu", "Shanghai"]
    assert all(s.index.is_unique and len(s) == 30 for s in series.values())
    assert list(table["entity"]) == ["Beijing", "Chengdu", "Shanghai"]
    row = table.set_index("entity").loc["Chengdu"]
    assert row["gaps_filled"] == 2 and row["periods"] == 30
    assert row["missing_share"] == pytest.approx(2 / 30, abs=1e-3)
    assert {"rows", "first", "last", "zero_share", "missing_share"} <= set(table.columns)


def test_panel_report_shares_are_within_range():
    _, table = pl.panel(_three_stations(), "station", "date", "pm25", freq="D")
    assert ((table["zero_share"] >= 0) & (table["zero_share"] <= 1)).all()
    assert ((table["missing_share"] >= 0) & (table["missing_share"] <= 1)).all()


def test_save_panel_writes_every_entity_in_the_multi_series_format(tmp_path):
    series, _ = pl.panel(_three_stations(), "station", "date", "pm25", freq="D")
    path = str(tmp_path / "out" / "series.json")
    assert pl.save_panel(series, path) == path
    back = pl.load_series(path)
    assert sorted(back) == ["Beijing", "Chengdu", "Shanghai"]
    assert np.allclose(back["Shanghai"].values, series["Shanghai"].values)
    assert pl.load_series(path, name="Chengdu").index[0] == series["Chengdu"].index[0]


def test_save_panel_accepts_the_pair_and_refuses_other_input(tmp_path):
    pair = pl.panel(_three_stations(), "station", "date", "pm25", freq="D")
    assert pl.save_panel(pair, str(tmp_path / "p.json")).endswith("p.json")
    with pytest.raises(ValueError, match="dict of entity series"):
        pl.save_panel([1, 2, 3], str(tmp_path / "q.json"))


def test_three_stations_in_a_sandbox_cell_give_three_series(tmp_path):
    csv = tmp_path / "stations.csv"
    _three_stations().to_csv(csv, index=False)
    code = (
        "import prophet_lib as pl\n"
        "df = pl.load('in/stations.csv')\n"
        "series, table = pl.panel(df, 'station', 'date', 'pm25', freq='D')\n"
        "pl.save_panel(series, 'out/series.json')\n"
        "print(len(series), list(table['entity']))\n"
    )
    r = run_python(code, inputs={"stations.csv": csv})
    assert r.ok, r.stderr
    assert r.stdout.strip() == "3 ['Beijing', 'Chengdu', 'Shanghai']"
