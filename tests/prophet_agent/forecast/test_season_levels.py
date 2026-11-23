"""The level measure in choose_season: a cycle that is small against day-to-day noise is missed by the one-step changes and
found in the detrended levels. Random walks, the null the changes test guards against, must still get no season."""
import numpy as np

from prophet_agent.forecast.rolling import backtest_season
from prophet_agent.forecast.season import choose_season


def _annual_daily(n=1612, amplitude=10.0, noise=3.0, seed=3):
    """Daily temperature-like series: a yearly sine of the given amplitude under white noise."""
    t = np.arange(n)
    return 50 + amplitude * np.sin(2 * np.pi * t / 365) + np.random.default_rng(seed).normal(0, noise, n)


def test_an_annual_cycle_under_daily_noise_is_found_when_no_hint_is_given():
    out = choose_season(_annual_daily())
    assert out["season"] == 365
    assert out["method"] == "levels"
    assert out["level_strengths"][365] > out["level_threshold"]
    # The one-step changes do not carry it at this noise level, which is why the level measure exists.
    assert abs(out["strengths"][365]) < out["threshold"]


def test_the_annual_cycle_is_found_with_a_daily_hint_and_the_reason_says_levels():
    out = choose_season(_annual_daily(), freq_hint="D")
    assert out["season"] == 365
    assert "levels repeat every 365 periods" in out["reason"]


def test_a_random_walk_daily_series_still_gets_no_season_on_both_paths():
    rng = np.random.default_rng(21)
    hits = sum(choose_season(np.cumsum(rng.normal(size=1612)), freq_hint="D")["season"] != 1 for _ in range(20))
    assert hits == 0


def test_a_short_random_walk_false_season_rate_stays_within_the_alpha_budget():
    rng = np.random.default_rng(11)
    hits = sum(choose_season(np.cumsum(rng.normal(size=60)))["season"] != 1 for _ in range(200))
    assert hits / 200 <= 0.05, hits / 200


def test_a_business_day_random_walk_gets_no_season_from_the_level_measure():
    rng = np.random.default_rng(8)
    hits = sum(choose_season(np.cumsum(rng.normal(size=600)), freq_hint="B")["season"] != 1 for _ in range(30))
    assert hits == 0


def test_the_backtest_default_season_finds_the_annual_cycle_on_a_daily_series():
    # What the Berlin check depends on: with no season given, the backtest measures on the history before its test windows
    # and must choose 365 rather than the weak season 1 it chose before the level measure.
    x = _annual_daily()
    season, reason = backtest_season(x, horizon=14, n_origins=12, given=None)
    assert season == 365
    assert "measured on the first" in reason
