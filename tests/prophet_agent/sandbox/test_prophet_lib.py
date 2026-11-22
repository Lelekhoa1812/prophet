

def test_carried_paths_resolve_by_file_name_and_load_series_round_trips(tmp_path, monkeypatch):
    import pandas as pd
    import pytest
    from prophet_agent.sandbox import prophet_lib as pl
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in" / "carried").mkdir(parents=True)
    s = pd.Series([1.0, 2.0, 3.5], index=pd.date_range("2024-01-01", periods=3), name="x")
    pl.save_series(s, "in/carried/series.json")
    back = pl.load_series("in/carried/out/series.json")           # the path a planner guesses: resolved by file name
    assert list(back.values) == [1.0, 2.0, 3.5] and str(back.index[0].date()) == "2024-01-01"
    (tmp_path / "in" / "a.csv").write_text("d,v\n2024-01-01,1\n2024-01-02,2\n")
    assert len(pl.load("in/carried/a.csv", run_id="abc")) == 2   # run_id accepted and ignored; found by name in in/
    with pytest.raises(FileNotFoundError, match="Files you can open"):
        pl.load("in/missing.csv")
