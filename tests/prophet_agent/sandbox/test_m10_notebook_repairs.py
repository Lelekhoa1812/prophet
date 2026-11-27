"""VN Unemployment notebook (2026-11-27): 23 sandbox cells failed. The repairs: pandas 3 offset aliases, no weekday-closure
check on a grid coarser than a day, no interpolation of a series that is mostly empty, and errors that name the next step."""
import pandas as pd
import pytest

from prophet_agent.sandbox import prophet_lib as pl
from prophet_agent.tools import sandbox_tool as st


@pytest.mark.parametrize("old, new", [("A", "YE"), ("Y", "YE"), ("A-DEC", "YE-DEC"), ("Y-DEC", "YE-DEC"), ("Q", "QE"),
                                      ("M", "ME"), ("2M", "2ME"), ("H", "h"), ("T", "min"), ("S", "s"), ("MS", "MS"),
                                      ("D", "D"), ("W-MON", "W-MON")])
def test_legacy_offset_aliases_map_to_the_ones_pandas_accepts(old, new):
    assert pl.freq_alias(old) == new
    pd.tseries.frequencies.to_offset(pl.freq_alias(old))


def _annual(years, values):
    return pd.DataFrame({"y": [str(y) for y in years], "v": [str(v) for v in values]})


def test_an_annual_series_prepares_with_a_legacy_alias_and_no_closed_day_error():
    s, _ = pl.prepare_series(_annual(range(2000, 2024), [1 + i / 10 for i in range(24)]), "v", "y", freq="A")
    assert len(s) == 24


def test_five_annual_points_over_twenty_four_years_are_not_interpolated_into_a_series():
    df = _annual([1997, 1998, 2009, 2019, 2020], [2.9, 2.3, 2.6, 2.0, 2.1])
    with pytest.raises(ValueError, match="only 5 of 24 periods"):
        pl.prepare_series(df, "v", "y", freq="YE")
    with pytest.raises(ValueError, match="search_open_data"):
        pl.build_series(df, "v", "y", freq="A", fill="interpolate")


def test_a_series_with_a_few_gaps_still_fills_them():
    years = [y for y in range(2000, 2024) if y not in (2005, 2006, 2015)]
    s, rep = pl.prepare_series(_annual(years, [1.0 + i for i in range(len(years))]), "v", "y", freq="YE")
    assert len(s) == 24 and s.notna().all()


def _result(stderr="", stdout="", ok=False):
    return {"ok": ok, "run_id": "nope", "stdout": stdout, "stderr": stderr, "outputs": [], "timed_out": False}


def test_sandbox_errors_name_the_next_correct_step():
    net = st.observe(_result("PermissionError: network access is blocked in the sandbox"))
    assert "download_web_data" in net["hint"]
    req = st.observe(_result("ModuleNotFoundError: No module named 'requests'"))
    assert "download_open_data" in req["hint"]
    freq = st.observe(_result("ValueError: Invalid frequency: A. Did you mean Y?"))
    assert "'YE'" in freq["hint"]
    missing = st.observe(_result("FileNotFoundError: [Errno 2] No such file or directory: 'in/carried/web_research.json'"))
    assert "carry=" in missing["hint"] and "download_web_data" in missing["hint"]


def test_a_cell_that_caught_the_network_refusal_is_not_read_as_a_success():
    out = st.observe(_result(stdout="Error: <urlopen error network access is blocked in the sandbox>", ok=True))
    assert out["blocked"] is True and "download_web_data" in out["hint"]
