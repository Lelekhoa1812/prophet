"""Tests for the principal-analyst scan (prophet_scan). Synthetic frames with fixed seeds, so every run is the same."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from prophet_agent.sandbox import prophet_scan as sc

START = "2022-01-01"


def _pattern(n: int, level: float, seed: int, season: float = 0.3, period: int = 7, noise: float = 0.05) -> np.ndarray:
    """A positive series with a cycle of ``period`` steps and multiplicative noise. Clean by construction."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return level * (1 + season * np.sin(2 * np.pi * t / period)) * (1 + noise * rng.standard_normal(n))


def _long(series: dict, n: int, freq: str = "D", start: str = START, entity: str = "sku") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq)
    parts = [pd.DataFrame({entity: name, "date": idx, "units": vals}) for name, vals in series.items()]
    return pd.concat(parts, ignore_index=True)


def _clean_frame(n: int = 400) -> pd.DataFrame:
    return _long({"A": _pattern(n, 40, 1), "B": _pattern(n, 60, 2), "C": _pattern(n, 80, 3)}, n)


def _scan(df: pd.DataFrame, entity_cols=("sku",), **kw) -> dict:
    return sc.scan(df, "date", "units", list(entity_cols) if entity_cols is not None else None, **kw)


def _find(result: dict, fid: str, scope: str | None = None) -> list[dict]:
    return [f for f in result["findings"] if f["id"] == fid and (scope is None or f["scope"] == scope)]


def _blocking(result: dict) -> list[dict]:
    return [f for f in result["findings"] if f["severity"] in ("blocker", "major")]


# ---------------------------------------------------------------------------------------------------------------------
# A clean table, determinism and the output contract
# ---------------------------------------------------------------------------------------------------------------------

def test_clean_frame_yields_no_blocker_or_major_finding():
    r = _scan(_clean_frame())
    assert not _blocking(r), [(f["id"], f["scope"]) for f in _blocking(r)]
    assert r["summary"]["blocked"] is False


def test_every_finding_has_the_contract_fields():
    df = _clean_frame()
    df.loc[df.sku == "A", "units"] = df.loc[df.sku == "A", "units"].where(np.arange(400) % 9 != 0, 0.0)
    r = _scan(df)
    assert r["findings"], "expected at least one finding on a table with zeros"
    for f in r["findings"]:
        assert set(f) == {"id", "severity", "scope", "evidence", "decision_needed", "next_step"}
        assert f["severity"] in ("blocker", "major", "minor", "info")
        assert isinstance(f["scope"], str)
        assert f["decision_needed"] is None or isinstance(f["decision_needed"], str)


def test_output_is_strict_json_without_nan():
    df = _clean_frame()
    df.loc[5, "units"] = np.nan
    r = _scan(df)
    json.dumps(r, allow_nan=False)  # raises if any NaN or infinity leaked into the result


def test_scan_is_deterministic():
    df = _clean_frame()
    df.loc[df.sku == "B", "units"] *= np.where(np.arange(400) % 50 == 0, 4.0, 1.0)
    a = json.dumps(_scan(df), sort_keys=True)
    b = json.dumps(_scan(df), sort_keys=True)
    assert a == b


def test_scan_never_mutates_the_input():
    df = _clean_frame()
    df.loc[3, "units"] = np.nan
    df = pd.concat([df, df.head(4)], ignore_index=True)  # duplicates, so every branch has work to do
    before = df.copy(deep=True)
    _scan(df)
    pd.testing.assert_frame_equal(df, before)


def test_write_scan_writes_strict_json(tmp_path):
    r = _scan(_clean_frame())
    target = tmp_path / "out" / "scan.json"
    written = sc.write_scan(r, str(target))
    assert written == str(target) and target.exists()
    loaded = json.loads(target.read_text(encoding="utf-8"), parse_constant=lambda c: pytest.fail(c))
    assert loaded["summary"]["severity_counts"] == r["summary"]["severity_counts"]


def test_modal_frequency_is_reported_for_an_inferred_grain():
    r = _scan(_clean_frame())
    found = _find(r, "time.modal_frequency", "all")
    assert found and found[0]["severity"] == "info"
    assert found[0]["evidence"]["grain"] == "D"
    assert found[0]["evidence"]["source"]


# ---------------------------------------------------------------------------------------------------------------------
# Keys, duplicates and the time grid
# ---------------------------------------------------------------------------------------------------------------------

def test_identical_duplicate_timestamps_are_major_for_the_right_entity():
    df = _clean_frame()
    dup = df[df.sku == "B"].head(5)
    r = _scan(pd.concat([df, dup], ignore_index=True))
    found = _find(r, "time.duplicate_timestamps", "B")
    assert found and found[0]["severity"] == "major"
    assert found[0]["evidence"]["extra_rows"] == 5
    assert found[0]["evidence"]["conflicting_stamps"] == 0
    assert not _find(r, "time.duplicate_timestamps", "A")


def test_conflicting_duplicate_timestamps_are_a_blocker():
    df = _clean_frame()
    dup = df[df.sku == "B"].head(3).copy()
    dup["units"] = dup["units"] * 2.0
    r = _scan(pd.concat([df, dup], ignore_index=True))
    found = _find(r, "time.duplicate_timestamps", "B")
    assert found and found[0]["severity"] == "blocker"
    assert found[0]["evidence"]["conflicting_stamps"] == 3
    assert r["summary"]["blocked"] is True


def test_stamps_off_the_grid_are_reported_for_the_entity():
    df = _clean_frame()
    idx = df.index[df.sku == "C"][::10]  # every tenth stamp moved to noon: 10 percent off the daily grid
    df.loc[idx, "date"] = df.loc[idx, "date"] + pd.Timedelta(hours=12)
    r = _scan(df)
    found = _find(r, "time.irregular_spacing", "C")
    assert found and found[0]["severity"] == "major"
    assert found[0]["evidence"]["off_grid_stamps"] == len(idx)
    assert not _find(r, "time.irregular_spacing", "A")


def test_a_short_gap_is_reported_as_a_minor_gap_with_count_and_longest_run():
    df = _clean_frame()
    drop = df[(df.sku == "A")].index[100:130]  # 30 consecutive missing days out of 400
    r = _scan(df.drop(drop))
    found = _find(r, "gaps.missing", "A")
    assert found and found[0]["severity"] == "minor"
    ev = found[0]["evidence"]
    assert ev["runs"] == 1 and ev["longest_steps"] == 30 and ev["missing_steps"] == 30
    assert abs(ev["share"] - 30 / 400) < 1e-9


def test_a_large_gap_share_is_major():
    df = _clean_frame()
    drop = df[(df.sku == "A")].index[50:150]  # 100 of 400 steps: a quarter of the grid
    r = _scan(df.drop(drop))
    found = _find(r, "gaps.missing", "A")
    assert found and found[0]["severity"] == "major"


def test_clock_change_repeats_or_drops_a_local_hour():
    idx = pd.date_range(START, periods=60 * 24, freq="h")
    vals = _pattern(len(idx), 50, 4, season=0.3, period=24, noise=0.03)
    df = pd.DataFrame({"sku": "X", "date": idx, "units": vals})
    spring_gap = pd.Timestamp("2022-01-21 02:00")  # the local hour that does not exist on the spring-forward day
    df = df[df.date != spring_gap]
    fall_repeat = df[df.date == pd.Timestamp("2022-02-10 01:00")]  # the local hour that occurs twice on fall-back
    df = pd.concat([df, fall_repeat], ignore_index=True)
    r = _scan(df, freq="h")
    found = _find(r, "time.dst_shift", "X")
    assert found and found[0]["severity"] == "major"
    kinds = {e["kind"] for e in found[0]["evidence"]["examples"]}
    assert kinds == {"missing_hour", "repeated_hour"}
    assert _find(r, "time.duplicate_timestamps", "X")


def test_a_timezone_change_moves_the_daily_pattern():
    idx = pd.date_range(START, periods=120 * 24, freq="h")
    day = (idx - pd.Timestamp(START)).days
    hour = idx.hour
    keep = ((day < 59) & (hour >= 9) & (hour <= 17)) | ((day >= 59) & (hour >= 19))
    kept = idx[keep]
    vals = _pattern(len(kept), 50, 5, season=0.2, period=24, noise=0.03)
    df = pd.DataFrame({"sku": "W", "date": kept, "units": vals})
    r = _scan(df, freq="h")
    found = _find(r, "time.timezone_shift", "W")
    assert found and found[0]["severity"] == "major"
    shift = found[0]["evidence"]["shifts"][0]
    assert shift["shift_hours"] == 10.0 and shift["whole_hours"] is True


def test_a_timezone_aware_column_is_reported_as_info_and_read_as_local_time():
    df = _clean_frame()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("Australia/Sydney")
    r = _scan(df)
    found = _find(r, "time.timezone_aware", "all")
    assert found and found[0]["severity"] == "info"
    assert found[0]["evidence"]["timezone"].endswith("Sydney")
    assert not _blocking(r)


# ---------------------------------------------------------------------------------------------------------------------
# Values, coverage and censoring
# ---------------------------------------------------------------------------------------------------------------------

def test_missing_values_are_major_when_they_are_a_large_share():
    df = _clean_frame()
    idx = df.index[df.sku == "A"][::3][:120]  # 30 percent of entity A
    df.loc[idx, "units"] = np.nan
    r = _scan(df)
    found = _find(r, "values.missing", "A")
    assert found and found[0]["severity"] == "major"


def test_a_negative_value_is_a_minor_flag():
    df = _clean_frame()
    df.loc[df.index[df.sku == "B"][7], "units"] = -5.0
    r = _scan(df)
    found = _find(r, "values.negative", "B")
    assert found and found[0]["severity"] == "minor" and found[0]["evidence"]["negative_values"] == 1


def test_text_in_a_few_value_cells_is_major_and_the_rest_is_scanned():
    df = _clean_frame()
    df["units"] = df["units"].astype(object)
    df.loc[df.index[:5], "units"] = "n/a"  # 5 of 1200 rows: 0.4 percent, under the 1 percent blocking share
    r = _scan(df)
    found = _find(r, "schema.non_numeric_value", "all")
    assert found and found[0]["severity"] == "major"
    assert found[0]["evidence"]["rows"] == 5
    assert not _find(r, "schema.unparseable_time")


def test_text_in_most_value_cells_is_a_blocker():
    df = _clean_frame()
    df["units"] = df["units"].astype(object)
    df.loc[df.index[:800], "units"] = "see note"
    r = _scan(df)
    found = _find(r, "schema.non_numeric_value", "all")
    assert found and found[0]["severity"] == "blocker"


def test_stockout_like_zero_run_between_sales_is_a_major_censoring_finding():
    df = _clean_frame()
    idx = df.index[df.sku == "A"][100:105]  # five zeros between positive values
    df.loc[idx, "units"] = 0.0
    r = _scan(df)
    found = _find(r, "censoring.stockout_runs", "A")
    assert found and found[0]["severity"] == "major"
    assert found[0]["evidence"]["longest"] == 5


def test_a_repeated_ceiling_is_a_censoring_plateau():
    df = _clean_frame()
    t = np.arange(400)
    vals = 50 + 40 * np.sin(2 * np.pi * t / 200)  # slow cycle, so demand stays above the cap for whole runs
    vals = np.minimum(vals, 70.0)  # capped: a ceiling plateau of consecutive points at the maximum
    df = _long({"P": vals}, 400)
    r = _scan(df)
    found = _find(r, "censoring.plateau", "P")
    assert found and found[0]["severity"] == "major"


def test_a_run_of_identical_values_is_a_minor_repeat_flag():
    vals = _pattern(400, 50, 10)
    vals[200:208] = 47.0  # eight identical values in a row
    r = _scan(_long({"R": vals}, 400))
    found = _find(r, "censoring.repeated_values", "R")
    assert found and found[0]["severity"] == "minor" and found[0]["evidence"]["run_length"] == 8


def test_leading_zeros_before_launch_are_flagged_by_length():
    vals = _pattern(400, 50, 11)
    vals[:30] = 0.0
    r = _scan(_long({"L": vals}, 400))
    found = _find(r, "coverage.leading_zeros", "L")
    assert found and found[0]["severity"] == "major" and found[0]["evidence"]["zeros"] == 30


def test_trailing_zeros_below_the_dead_run_length_are_minor():
    vals = _pattern(400, 50, 12)
    vals[-3:] = 0.0
    r = _scan(_long({"T": vals}, 400))
    found = _find(r, "coverage.trailing_zeros", "T")
    assert found and found[0]["severity"] == "minor" and found[0]["evidence"]["zeros"] == 3


def test_an_interior_dead_segment_is_major():
    vals = _pattern(400, 50, 13)
    vals[150:176] = 0.0  # 26 zeros inside the series
    r = _scan(_long({"D": vals}, 400))
    found = _find(r, "coverage.dead_segment", "D")
    assert found and found[0]["severity"] == "major" and found[0]["evidence"]["longest_zeros"] == 26


def test_an_entity_with_no_demand_is_a_major_all_zero_flag():
    df = _clean_frame()
    df = pd.concat([df, _long({"Z0": np.zeros(400)}, 400)], ignore_index=True)
    r = _scan(df)
    assert _find(r, "coverage.all_zero", "Z0")[0]["severity"] == "major"


def test_too_little_active_history_is_a_major_short_history_flag():
    df = _clean_frame()
    short = _long({"SH": _pattern(8, 50, 14)}, 8)
    r = _scan(pd.concat([df, short], ignore_index=True))
    found = _find(r, "coverage.short_history", "SH")
    assert found and found[0]["severity"] == "major"
    assert found[0]["evidence"]["cycle_steps"] == 7


def test_intermittent_demand_is_classed_and_flagged_major():
    rng = np.random.default_rng(15)
    occurs = rng.random(400) < 0.4
    vals = np.where(occurs, rng.uniform(5, 15, 400), 0.0)
    r = _scan(_long({"I": vals}, 400))
    found = _find(r, "intermittency.class", "I")
    assert found and found[0]["severity"] == "major"
    assert found[0]["evidence"]["class"] in ("intermittent", "lumpy")
    assert _find(r, "intermittency.zero_inflation", "I")


# ---------------------------------------------------------------------------------------------------------------------
# Scale, concentration and heterogeneity across entities
# ---------------------------------------------------------------------------------------------------------------------

def test_a_thousand_fold_unit_group_is_flagged_as_unit_mix():
    series = {f"E{i}": _pattern(400, 50, 20 + i) for i in range(6)}
    series.update({f"M{i}": _pattern(400, 50_000, 40 + i) for i in range(4)})
    r = _scan(_long(series, 400))
    unit = _find(r, "scale.unit_mix", "all")
    assert unit and unit[0]["severity"] == "major"
    assert unit[0]["evidence"]["low_group"] == 6 and unit[0]["evidence"]["high_group"] == 4
    assert _find(r, "scale.spread", "all")


def test_a_single_large_entity_is_not_called_a_unit_mix():
    series = {f"E{i}": _pattern(400, 50, 20 + i) for i in range(6)}
    series["BIG"] = _pattern(400, 5_000, 50)
    r = _scan(_long(series, 400))
    assert not _find(r, "scale.unit_mix")


def test_one_entity_holding_most_volume_is_a_major_concentration_flag():
    series = {f"C{i}": _pattern(400, 50, 60 + i) for i in range(10)}
    series["BIG"] = _pattern(400, 5_000, 70)
    r = _scan(_long(series, 400))
    found = _find(r, "concentration.top_share", "all")
    assert found and found[0]["severity"] == "major"
    assert found[0]["evidence"]["largest"] == "BIG"


def test_entities_with_different_shapes_are_heterogeneous():
    rng = np.random.default_rng(80)
    series = {f"HA{i}": 50 * (1 + 0.9 * rng.uniform(-1, 1, 400)) for i in range(4)}
    series.update({f"HB{i}": 50 * (1 + 0.02 * rng.standard_normal(400)) for i in range(4)})
    r = _scan(_long(series, 400))
    found = _find(r, "heterogeneity.between_entities", "all")
    assert found and found[0]["severity"] == "major"
    assert found[0]["evidence"]["eta_squared"] >= sc._ETA2_MAJOR


def test_a_gappy_entity_with_a_correlated_neighbour_is_borrowable():
    df = _clean_frame()
    gap = _pattern(400, 50, 90)
    series_df = _long({"G": gap}, 400)
    rng = np.random.default_rng(91)
    series_df = series_df[rng.random(400) > 0.3]  # 30 percent of stamps absent
    r = _scan(pd.concat([df, series_df], ignore_index=True))
    found = _find(r, "neighbours.borrowable", "G")
    assert found and found[0]["severity"] == "minor"
    assert found[0]["evidence"]["neighbours"][0]["corr"] >= sc._NEIGH_CORR


# ---------------------------------------------------------------------------------------------------------------------
# Breaks, outliers and seasonality
# ---------------------------------------------------------------------------------------------------------------------

def test_a_doubling_of_level_is_a_major_level_shift_not_a_unit_change():
    vals = _pattern(400, 50, 100)
    vals[200:] *= 2.0
    r = _scan(_long({"S": vals}, 400))
    found = _find(r, "changepoints.level_shift", "S")
    assert found and found[0]["severity"] == "major"
    assert abs(found[0]["evidence"]["breaks"][0]["median_ratio"] - 2.0) < 0.2


def test_a_twentyfold_jump_in_level_is_a_blocker_unit_change():
    vals = _pattern(400, 50, 101)
    vals[200:] *= 20.0
    r = _scan(_long({"U": vals}, 400))
    found = _find(r, "changepoints.level_shift", "U")
    assert found and found[0]["severity"] == "blocker"
    assert r["summary"]["blocked"] is True


def test_a_spiky_entity_is_flagged_for_outliers():
    vals = _pattern(400, 50, 102)
    spikes = np.arange(0, 400, 20)  # 5 percent of steps
    vals[spikes] *= 1.8
    r = _scan(_long({"O": vals}, 400))
    found = _find(r, "outliers.share", "O")
    assert found and found[0]["severity"] == "major"
    assert found[0]["evidence"]["severity_uses"] == "hampel"


def test_a_cycle_that_is_not_the_grain_period_is_flagged_by_autocorrelation():
    t = np.arange(400)
    rng = np.random.default_rng(103)
    vals = 50 * (1 + 0.3 * np.sin(2 * np.pi * t / 5)) * (1 + 0.05 * rng.standard_normal(400))
    r = _scan(_long({"F5": vals}, 400))
    found = _find(r, "seasonality.acf_peak", "F5")
    assert found and found[0]["severity"] == "minor" and found[0]["evidence"]["lag"] == 5


def test_strong_weekly_season_is_summarised_once_across_entities():
    r = _scan(_clean_frame())
    found = _find(r, "seasonality.strength", "all")
    assert found and found[0]["severity"] == "info"
    assert found[0]["evidence"]["period_steps"] == 7
    assert found[0]["evidence"]["entities_seasonal"] == found[0]["evidence"]["analysed"] == 3


# ---------------------------------------------------------------------------------------------------------------------
# Entity sampling, performance and the bad-input contract
# ---------------------------------------------------------------------------------------------------------------------

def test_entities_beyond_max_entities_are_sampled_deterministically_and_said_so():
    series = {f"K{i:03d}": _pattern(200, 30 + i, 200 + i) for i in range(30)}
    df = _long(series, 200)
    a = _scan(df, max_entities=10)
    b = _scan(df, max_entities=10)
    s = a["summary"]
    assert s["sampled"] is True and s["entities_total"] == 30 and s["entities_analysed"] == 10
    assert s["sample_seed"] == sc._SAMPLE_SEED
    assert "10 of 30" in s["sample_note"]
    assert set(s["profiles"]) == set(b["summary"]["profiles"]) and len(s["profiles"]) == 10
    assert _find(a, "scan.entities_sampled", "all")


def test_the_scan_meets_the_time_ceiling_on_100k_rows_and_200_entities():
    n_ent, n_t = 200, 500
    idx = pd.date_range(START, periods=n_t, freq="D")
    frames = []
    for e in range(n_ent):
        frames.append(pd.DataFrame({"sku": f"S{e:03d}", "date": idx,
                                    "units": _pattern(n_t, 10 + e % 50, 300 + e)}))
    df = pd.concat(frames, ignore_index=True)
    assert len(df) == 100_000
    t0 = time.perf_counter()
    r = _scan(df, freq="D")
    elapsed = time.perf_counter() - t0
    assert elapsed < 20.0, f"scan took {elapsed:.1f}s"
    assert r["summary"]["entities_analysed"] == 200 and r["summary"]["sampled"] is False


def test_a_missing_column_is_a_blocker_naming_the_column():
    r = sc.scan(_clean_frame(), "date", "not_a_column", ["sku"])
    found = _find(r, "schema.missing_column", "not_a_column")
    assert found and found[0]["severity"] == "blocker"
    assert r["summary"]["blocked"] is True


def test_a_non_table_is_a_blocker_and_does_not_raise():
    r = sc.scan([1, 2, 3], "date", "units")
    assert _find(r, "schema.not_a_table")[0]["severity"] == "blocker"


def test_an_invalid_max_entities_is_a_blocker():
    r = _scan(_clean_frame(), max_entities=0)
    assert _find(r, "schema.bad_argument")[0]["severity"] == "blocker"


def test_an_unknown_frequency_is_a_blocker():
    r = _scan(_clean_frame(), freq="not-a-frequency")
    assert _find(r, "schema.bad_frequency")[0]["severity"] == "blocker"


def test_unreadable_times_are_a_blocker_and_nothing_is_analysed():
    df = _clean_frame()
    df["date"] = "not a date"
    r = _scan(df)
    assert _find(r, "schema.unparseable_time")[0]["severity"] == "blocker"
    assert _find(r, "schema.no_valid_rows")


def test_rows_with_no_entity_label_are_a_major_flag_and_kept_as_missing():
    df = _clean_frame()
    df.loc[df.index[:4], "sku"] = None
    r = _scan(df)
    assert _find(r, "schema.missing_entity", "all")[0]["severity"] == "major"
    assert _find(r, "schema.missing_entity", "all")[0]["evidence"]["rows"] == 4


def test_a_multi_column_entity_is_labelled_by_joining_its_parts():
    df = _clean_frame()
    df["store"] = np.where(np.arange(len(df)) % 2 == 0, "S1", "S2")
    r = _scan(df, entity_cols=["sku", "store"])
    assert r["summary"]["entities_total"] == 6
    assert "A|S1" in r["summary"]["profiles"]


def test_a_duplicate_column_name_is_a_blocker():
    df = _clean_frame()
    df.insert(0, "units", df["units"], allow_duplicates=True)
    r = _scan(df)
    assert _find(r, "schema.duplicate_column")[0]["severity"] == "blocker"


def test_a_weekday_only_series_is_a_business_day_calendar_not_a_gap_defect():
    import numpy as np
    import pandas as pd
    days = pd.bdate_range("2021-01-04", periods=400)
    df = pd.DataFrame({"date": days, "v": np.random.default_rng(1).normal(100, 5, len(days))})
    ids = {f["id"]: f for f in sc.scan(df, "date", "v")["findings"]}
    assert "gaps.missing" not in ids, "a clean business-day grid has no gap finding"
    holidays = days.delete([30, 95, 160, 225, 290, 350])          # holidays make pandas stop inferring a business-day grain
    df2 = pd.DataFrame({"date": holidays, "v": np.random.default_rng(3).normal(100, 5, len(holidays))})
    ids2 = {f["id"]: f for f in sc.scan(df2, "date", "v")["findings"]}
    assert ids2["calendar.business_days"]["severity"] == "info"
    assert ids2["gaps.missing"]["severity"] == "minor", "only the holidays remain as gaps"


def test_real_gaps_on_weekdays_are_still_reported_on_a_business_day_series():
    import numpy as np
    import pandas as pd
    days = pd.bdate_range("2021-01-04", periods=400)
    keep = np.ones(len(days), dtype=bool)
    keep[100:130] = False
    df = pd.DataFrame({"date": days[keep], "v": np.random.default_rng(2).normal(100, 5, int(keep.sum()))})
    ids = {f["id"]: f for f in sc.scan(df, "date", "v")["findings"]}
    assert "calendar.business_days" in ids and "gaps.missing" in ids
    assert ids["gaps.missing"]["evidence"]["longest_steps"] >= 20
