"""choose_season reports the shortest lag that repeats, not a multiple of it (found while wiring the health check to it).

Motivation vs Logic
Motivation: a smooth cycle of 24 steps repeats at 48, 96, 168 and so on with almost the same autocorrelation. choose_season took
the largest, so a clean 24-step sine with no hint was reported as 48 (or 168 with a trend). find_season already applies the
rule "the shortest lag within SEASON_TOLERANCE of the best peak"; choose_season did not. The health check now reads this season,
so the alias would reach the health verdict too.
Logic: the same tie-break, applied to the lags that already clear the bar. Whether a season is found is unchanged (the strongest
lag must still clear the bar); only which of the equal lags is reported changes. Synthetic series only.
"""
import numpy as np

from prophet_agent.forecast.season import choose_season


def test_a_clean_24_step_cycle_is_reported_as_24_not_a_multiple():
    t = np.arange(900)
    x = 100 + 10 * np.sin(2 * np.pi * t / 24) + np.random.default_rng(0).normal(0, 0.5, 900)
    out = choose_season(x)
    assert out["season"] == 24, out["reason"]


def test_a_clean_24_step_cycle_with_a_daily_hint_is_reported_as_24_when_it_is_a_candidate_cycle():
    # Hourly candidates are 24 and 168; the shorter repeat of the same cycle must be the one reported.
    t = np.arange(900)
    x = 100 + 10 * np.sin(2 * np.pi * t / 24) + np.random.default_rng(1).normal(0, 0.5, 900)
    out = choose_season(x, freq_hint="H")
    assert out["season"] == 24, out["reason"]


def test_a_weekly_cycle_is_still_seven_and_not_a_multiple_of_it():
    t = np.arange(700)
    x = 50 + 4 * np.sin(2 * np.pi * t / 7) + np.random.default_rng(4).normal(0, 1, 700)
    assert choose_season(x, freq_hint="D")["season"] == 7
