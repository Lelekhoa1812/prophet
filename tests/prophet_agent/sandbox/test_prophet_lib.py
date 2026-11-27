

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


def test_a_flags_array_explains_a_text_key_read():
    import numpy as np
    import pytest
    from prophet_agent.sandbox import prophet_stats as ps
    flags, report = ps.hampel(np.r_[np.ones(30), 50.0, np.ones(30)])
    assert flags.sum() >= 1 and flags[30]                       # still an ordinary boolean array
    with pytest.raises(TypeError, match=r"hampel\(\) returned a boolean array of flags.*report keys"):
        flags["is_outlier"]


def test_report_aliases_analysts_write_are_answered():
    import numpy as np
    import pandas as pd
    import prophet_stats as ps
    import prophet_lib  # noqa: F401  installs the pandas additions
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 1, 120), rng.normal(5, 1, 120)]
    x[30] = 40
    changes, _ = ps.changepoints(x)
    c = changes[0]
    assert c["mean_before"] == c["before_mean"] and c["jump"] == c["shift"] and c["at"] == c["index"]
    flags, rep = ps.hampel(x)
    assert rep["outliers"].sum() == rep["count"] == rep["n_outliers"] and rep.get("nope") is None
    d = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=5)) - pd.Timestamp("2024-01-03")
    assert int(d.abs().argmin()) == 2


def test_flag_positions_and_finding_field_names_are_answered():
    import numpy as np
    import prophet_stats as ps
    from prophet_scan import Finding
    x = np.r_[np.zeros(30) + np.linspace(0, 1, 30), [50.0], np.linspace(1, 2, 30)]
    flags, rep = ps.hampel(x)
    assert list(rep["outlier_indices"]) == list(np.flatnonzero(flags)) and 30 in rep["indices"]
    f = Finding(id="F1", message="level shift in 2024", decision_needed="cut or keep", severity="major")
    assert f["title"] == "level shift in 2024" and f["decision"] == "cut or keep" and f.get("nope") is None


def test_pl_reaches_the_statistics_helpers_and_change_points_give_a_ratio():
    import numpy as np
    import prophet_lib as pl
    import pytest
    flags, rep = pl.hampel(np.r_[np.ones(20), [9.0], np.ones(20)])
    assert rep["count"] >= 1
    changes, _ = pl.changepoints(np.r_[np.full(60, 10.0), np.full(60, 20.0)] + np.random.default_rng(0).normal(0, 0.1, 120))
    assert abs(changes[0]["ratio"] - 2.0) < 0.05
    with pytest.raises(AttributeError, match="neither has prophet_stats"):
        pl.no_such_helper


def test_calling_an_agent_tool_from_the_sandbox_says_how_to_call_it():
    import prophet_lib as pl
    import pytest
    with pytest.raises(AttributeError, match="call backtest_series as its own step"):
        pl.backtest_series


def test_a_flag_array_is_true_when_any_flag_is_set():
    import numpy as np
    import prophet_stats as ps
    flags, rep = ps.hampel(np.r_[np.ones(20), [9.0], np.ones(20)])
    assert bool(flags) and bool(rep["outliers"]) and int(flags.sum()) == rep["count"]
    calm, rep2 = ps.hampel(np.ones(30) + np.arange(30) * 0.01)
    assert not calm and not rep2["outliers"]


def test_result_prints_the_figure_an_answer_will_quote(capsys):
    import numpy as np
    from prophet_agent.sandbox import prophet_lib as pl
    assert pl.result("total_14d", np.float64(575419.3712), "GBP") == 575419.3712
    pl.result({"share": 0.18333333, "n": np.int64(135)})
    out = capsys.readouterr().out.splitlines()
    assert out == ["RESULT total_14d: 575419.3712 GBP", "RESULT share: 0.1833", "RESULT n: 135"]
    from prophet_agent.runtime.grounding import unverified
    history = [{"kind": "tool", "tool": "sandbox_python", "args": {}, "result": {"stdout": "\n".join(out)}}]
    assert unverified("The fortnight total is £575,419 (18% share, 135 days, about £575.4k).", history, "") == []


def test_the_prompts_tell_the_planner_to_print_and_the_composer_to_quote_as_printed():
    from prophet_agent import prompts
    planner, composer = prompts.current("planner"), prompts.current("composer")
    assert planner.version == "planner.v36" and composer.version == "composer.v8"
    assert "pl.result" in planner.text and "per-day edges" in planner.text and "total_range_80" in planner.text
    assert "written as the evidence prints it" in composer.text and "per-day" in composer.text


def test_a_daily_grid_over_a_trading_series_names_the_fix(tmp_path):
    """Panel round 12 (SPX): build_series(freq='D') over trading days added the weekends, and save_series's error did not
    say how to keep only observed days; the planner repeated the save seven times."""
    import pandas as pd
    import pytest
    from prophet_agent.sandbox import prophet_lib as pl
    days = pd.bdate_range("2026-01-05", periods=30)
    df = pd.DataFrame({"d": days.strftime("%Y-%m-%d"), "v": [str(100 + i) for i in range(30)]})
    s = pl.build_series(df, value="v", time="d", freq="D")
    with pytest.raises(ValueError, match="without freq"):
        pl.save_series(s, str(tmp_path / "s.json"))
    pl.save_series(pl.build_series(df, value="v", time="d"), str(tmp_path / "s.json"))
