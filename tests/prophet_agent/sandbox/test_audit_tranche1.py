"""Regression tests for the silent-corruption defects in the 2026-11-22 audit (docs/prophet/AUDIT_2026-11-22.md, items 2 to 6, 12)."""
import numpy as np
import pandas as pd


def test_an_open_trailing_gap_is_not_filled_with_the_last_value():
    from prophet_agent.sandbox import prophet_lib as pl
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=6, freq="D").strftime("%Y-%m-%d"),
                       "units": [1.0, 2.0, 3.0, np.nan, np.nan, np.nan]})
    s, rep = pl.prepare_series(df, value="units", time="date", freq="D", trim_tail=False)
    assert np.isnan(s.iloc[-1]) and np.isnan(s.iloc[-3:]).all()          # audit: [1,2,3,3,3,3] before
    assert any("left open" in n for n in rep["notes"])
    # an interior gap is still filled
    df2 = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5, freq="D").strftime("%Y-%m-%d"),
                        "units": [1.0, np.nan, 3.0, 4.0, 5.0]})
    s2, _ = pl.prepare_series(df2, value="units", time="date", freq="D")
    assert s2.iloc[1] == 2.0


def test_clipping_keeps_the_end_where_the_traceback_prints():
    from prophet_agent.sandbox.runner import MAX_STREAM_CHARS, _clip
    text = "x" * 31_000 + "\nTraceback (most recent call last)\nKeyError: 'units'"
    clipped = _clip(text)
    assert clipped.endswith("KeyError: 'units'") and len(clipped) <= MAX_STREAM_CHARS + 100
    assert "characters cut from the middle" in clipped
    short = "fine"
    assert _clip(short) == short


def test_a_seasonal_year_forecast_longer_than_the_season_repeats_the_last_season():
    from prophet_agent.forecast import baselines as b
    x = np.cumsum(np.random.default_rng(1).normal(size=200)) + 100
    p, band = b.seasonal_year(x, 60, 52)                  # audit: IndexError before
    assert len(p) == 60 and np.isfinite(p).all()
    assert np.allclose(p[52:60], p[0:8])                  # step 53 is step 1 one season later


def test_two_carried_files_with_one_name_are_both_mounted_and_neither_overwrites(tmp_path, monkeypatch):
    # Was a refusal (no silent overwrite). Round 5 (live Beijing rep6, five failed calls): the second file is mounted under
    # its run's prefix instead, so both are readable and neither replaces the other.
    from prophet_agent.tools import sandbox_tool as st
    run_a, run_b = "a" * 32, "b" * 32
    for run in (run_a, run_b):
        (tmp_path / run / "out").mkdir(parents=True)
        (tmp_path / run / "out" / "series.json").write_text("{}")
    monkeypatch.setattr(st, "SANDBOX_ROOT", tmp_path)
    both = st.carried_inputs([{"run_id": run_a, "file": "out/series.json"}, {"run_id": run_b, "file": "out/series.json"}])
    assert both == {"carried/series.json": tmp_path / run_a / "out" / "series.json",
                    "carried/bbbbbbbb_series.json": tmp_path / run_b / "out" / "series.json"}
    assert list(st.carried_inputs([{"run_id": run_a, "file": "out/series.json"}])) == ["carried/series.json"]


def test_the_scan_reads_day_first_dates_as_day_first_and_flags_the_ambiguous():
    from prophet_agent.sandbox import prophet_scan as sc
    dayfirst, finding = sc._day_order(pd.Series(["01/02/2024", "15/02/2024", "28/02/2024"]), "date")
    assert dayfirst is True and finding is None                       # 15 and 28 settle it: day first
    month_first, finding = sc._day_order(pd.Series(["01/02/2024", "02/03/2024", "05/04/2024"]), "date")
    assert finding is not None and finding["id"] == "time.ambiguous_day_month" and finding["severity"] == "major"
    mixed, finding = sc._day_order(pd.Series(["15/02/2024", "02/13/2024"]), "date")
    assert finding["id"] == "time.mixed_day_month" and finding["severity"] == "blocker"


def test_epoch_seconds_are_read_as_seconds_not_nanoseconds():
    from prophet_agent.sandbox import prophet_scan as sc
    df = pd.DataFrame({"t": [1704067200 + 86400 * i for i in range(40)], "v": np.arange(40, dtype=float)})
    res = sc.scan(df, "t", "v", freq="D")
    assert res.summary["time_range"] == ["2024-01-01 00:00:00", "2024-02-09 00:00:00"]


def test_list_holidays_with_no_dates_returns_the_current_year_instead_of_raising():
    # Audit item 23: pd.Timestamp(None) is NaT, so the old default raised on .year.
    from prophet_agent.sandbox import prophet_lib as pl
    days = pl.list_holidays("AU")
    assert len(days) >= 5
    assert {d["date"][:4] for d in days} == {str(pd.Timestamp.today().year)}


def test_the_catalogue_lists_the_cleaning_helpers_the_planner_needs():
    # Audit item 23: to_number, to_datetime and tidy_text were in the code but not in the planner's catalogue.
    from prophet_agent.sandbox import catalogue
    names = {f"{alias}.{name}" for alias, _, name in catalogue.catalogue_entries()}
    assert {"pl.to_number", "pl.to_datetime", "pl.tidy_text"} <= names
    assert "pl.tidy_text(" in catalogue.build_catalogue()


def _two_numbered_stores():
    days = pd.date_range("2024-01-01", periods=5, freq="D").strftime("%Y-%m-%d").tolist()
    rows = [{"date": d, "store_id": sid, "sales_units": 10.0 + i + sid} for d in days for i, sid in enumerate((1, 2))]
    return pd.DataFrame(rows)


def test_numeric_store_ids_sharing_dates_are_refused_not_summed():
    # Audit item 13: the guard skipped numeric columns, so store_id 1 and 2 were added together without a word.
    import pytest
    from prophet_agent.sandbox import prophet_lib as pl
    with pytest.raises(ValueError, match="store_id"):
        pl.prepare_series(_two_numbered_stores(), value="sales_units", time="date", freq="D")
    s, _ = pl.prepare_series(_two_numbered_stores(), value="sales_units", time="date", freq="D", entity="store_id", key=1)
    assert len(s) == 5


def test_a_numeric_measure_that_varies_within_a_date_is_not_mistaken_for_entities():
    from prophet_agent.sandbox import prophet_lib as pl
    days = pd.date_range("2024-01-01", periods=5, freq="D").strftime("%Y-%m-%d").tolist()
    rows = [{"date": d, "sales_units": 3.0, "quantity": float(q)} for d in days for q in (1.0, 2.0)]
    s, _ = pl.prepare_series(pd.DataFrame(rows), value="sales_units", time="date", freq="D")   # no refusal
    assert len(s) == 5
