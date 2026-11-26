"""Panel round 3 EPL (2026-11-26): a simulated season is the forecast; one schedule source is used."""
import pytest

from prophet_agent.forecast import sports
from prophet_agent.runtime import contracts as c


def test_a_simulated_season_closes_the_forecast_items_of_the_checklist():
    obs = [{"kind": "tool", "tool": "inspect_data", "args": {}, "result": {"columns": ["Date: time", "Home: text"]}},
           {"kind": "tool", "tool": "simulate_season", "args": {"results": "epl.csv"},
            "result": {"teams": [], "backtest": {"comparisons": []}}}]
    items = c.progress(obs)
    names = " ".join(it["item"] for it in items)
    assert "simulate the rest of the season" in names
    assert c.SERIES_ITEM not in names and "timesfm_forecast" not in names
    assert all(it["status"] == "done" for it in items)


def _m(day, home, away, hg=None, ag=None):
    return sports.Match(date=f"2026-09-{day:02d}", home=home, away=away, home_goals=hg, away_goals=ag)


def test_unplayed_rows_in_the_results_are_the_schedule_and_a_second_fixtures_file_is_not_used():
    teams = ["A", "B", "C", "D"]
    pairs = [(h, a) for h in teams for a in teams if h != a]
    played = [sports.Match(date=f"2026-08-{1 + i % 28:02d}", home=h, away=a, home_goals=(i + 1) % 3, away_goals=i % 2)
              for i, (h, a) in enumerate(pairs * 2)]
    unplayed = [_m(20, "C", "D"), _m(21, "D", "C"), _m(22, "B", "D"), _m(23, "D", "B")]
    other_source = [_m(20, "Club C", "Club D")]
    out = sports.simulate_season(played + unplayed, other_source, n_sims=50, backtest=False)
    assert out["settings"]["n_remaining"] == 4
    assert any("fixtures file was not used" in n for n in out.get("notes", []))


def test_fixtures_with_other_team_names_are_refused():
    teams = ["A", "B", "C", "D"]
    played = [_m(1 + i, h, a, 1, 0) for i, (h, a) in enumerate([(h, a) for h in teams for a in teams if h != a])]
    with pytest.raises(ValueError, match="names do not match"):
        sports.simulate_season(played, [_m(25, "Club A", "B")], n_sims=50, backtest=False)
