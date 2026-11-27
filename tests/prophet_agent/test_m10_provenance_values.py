"""VN Unemployment (2026-11-27): a series refused for its provenance came back as inline values and was forecast."""
import numpy as np
import pytest

from prophet_agent.tools import series_io


@pytest.fixture(autouse=True)
def clean_registry():
    series_io.REFUSED_SERIES.clear()
    yield
    series_io.REFUSED_SERIES.clear()


INTERP = [2.868, 2.29, 2.2397272727272726, 2.1894545454545455, 2.139181818181818, 2.088909090909091, 2.0386363636363636,
          1.9883636363636363, 1.938090909090909, 1.8878181818181818, 1.837]


def test_values_copied_from_a_refused_series_are_refused_in_whole_or_in_part():
    series_io._remember_refused({"series": INTERP}, "synthetic, cannot be labelled World Bank")
    with pytest.raises(ValueError, match="already refused"):
        series_io.resolve_series({"values": INTERP})
    with pytest.raises(ValueError, match="already refused"):
        series_io.resolve_series({"values": INTERP[2:]})


def test_other_values_and_short_checks_still_pass():
    series_io._remember_refused({"series": INTERP}, "synthetic")
    assert series_io.resolve_series({"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})[0][0] == "series"
    assert series_io.resolve_series({"values": INTERP[:4]})[0][0] == "series", "a short run is too common to match"
    out = series_io.resolve_series({"values": list(np.linspace(5, 6, 30))})
    assert len(out[0][1]) == 30


def test_recency_is_ignored_for_dataset_and_history_queries_only():
    from prophet_agent.tools import research_tools as rt
    assert rt._recency({"recency": "year"}, ["Vietnam unemployment rate quarterly data download"]) == (None, None)
    assert rt._recency({"recency": "year"}, ["Vietnam central bank rate decision"]) == rt.RECENCY["year"]
    assert rt._recency({"recency": "year"}, ["Vietnam monthly statistics", "Vietnam election news"]) == rt.RECENCY["year"]
