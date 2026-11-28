"""market_benchmark (round 20261128-063203 audits): the market's own forecast to compare with. Offline: the files are stubs
in each source's real layout."""
from __future__ import annotations

import math

import pytest

from prophet_agent.opendata import market_benchmark as mb
from prophet_agent.runtime import prediction_quality as pq
from prophet_agent.runtime.runner import ArgumentRejected
from prophet_agent.tools.market_benchmark_tool import _run


def _f1() -> bytes:
    head = ("﻿F1 INTEREST RATES AND YIELDS - MONEY MARKET\n"
            "Title,Cash Rate Target,EOD 1-month BABs/NCDs,EOD 3-month BABs/NCDs,EOD 6-month BABs/NCDs,1-month OIS,"
            "3-month OIS,6-month OIS\n"
            "Units,Per cent,Per cent,Per cent,Per cent,Per cent,Per cent,Per cent\n"
            "Series ID,FIRMMCRTD,FIRMMBAB30D,FIRMMBAB90D,FIRMMBAB180D,FIRMMOIS1D,FIRMMOIS3D,FIRMMOIS6D\n")
    rows = []
    import datetime as dt
    day = dt.date(2015, 1, 1)
    for k in range(300):                  # an overlap where bank bills sit 0.1, 0.2 and 0.3 above OIS
        d = (day + dt.timedelta(days=k)).strftime("%d-%b-%Y")
        rows.append(f"{d},2.00,2.10,2.20,2.30,2.00,2.00,2.00")
    rows.append("03-Nov-2026,4.60,4.70,4.80,4.90,,,")
    rows.append("04-Nov-2026,4.85,4.95,5.10,5.40,,,")
    rows.append("05-Nov-2026,4.85,,,,,,")
    return (head + "\n".join(rows) + "\n").encode()


def test_the_cash_rate_path_is_the_bank_bill_curve_less_its_spread_to_ois():
    out = mb.benchmark("au_cash_rate", get=lambda url: _f1())
    assert out["target"] == {"date": "2026-11-04", "value": 4.85} and out["as_of"] == "2026-11-04"
    assert out["spread_to_ois"]["3m"]["median"] == pytest.approx(0.2)
    assert out["implied_average"] == {"1m": pytest.approx(4.85), "3m": pytest.approx(4.9), "6m": pytest.approx(5.1)}
    fwd = {p["months"]: p for p in out["implied_forwards"]}
    assert fwd["1-3"]["rate"] == pytest.approx((3 * 4.9 - 4.85) / 2) and fwd["3-6"]["vs_target_bp"] == 45
    assert out["recent_decisions"][-1] == {"date": "2026-11-04", "from": 4.6, "to": 4.85}
    assert out["source_url"].startswith("https://www.rba.gov.au/") and "estimate" in out["method"]


def test_the_options_range_reverts_toward_the_long_run_volatility():
    body = "DATE,OPEN,HIGH,LOW,CLOSE\n" + "\n".join(f"01/{d:02d}/2025,0,0,0,{15 + (d % 3)}" for d in range(1, 29)) + \
        "\n11/26/2026,0,0,0,30.0\n"
    out = mb.benchmark("vix", level=6000, months=[1, 24], get=lambda url: body.encode())
    assert out["implied_vol_pct"] == 30.0 and out["as_of"] == "2026-11-26" and out["long_run_median_pct"] == 16.0
    one, two_years = out["implied_range"]
    assert 16.0 < two_years["vol_pct"] < one["vol_pct"] < 30.0, "high volatility today fades toward the long-run level"
    assert two_years["q10"] < 6000 < two_years["q90"]
    assert two_years["width_log"] == pytest.approx(2 * mb.Z90 * two_years["vol_pct"] / 100 * math.sqrt(2), rel=1e-3)
    gvz = mb.benchmark("gvz", level=4000, get=lambda url: b"observation_date,GVZCLS\n2026-11-24,33.6\n2026-11-25,.\n")
    assert gvz["implied_vol_pct"] == 33.6 and gvz["as_of"] == "2026-11-24", "FRED's '.' is a missing day"


def test_the_tool_refuses_what_it_cannot_answer():
    with pytest.raises(ArgumentRejected, match="kind must be"):
        _run({"kind": "bitcoin"}, "k")
    with pytest.raises(ArgumentRejected, match="level is required"):
        _run({"kind": "vix"}, "k")
    with pytest.raises(ArgumentRejected, match="months"):
        _run({"kind": "gvz", "level": 1, "months": [0]}, "k")
    with pytest.raises(ArgumentRejected, match="unknown arguments"):
        _run({"kind": "vix", "level": 1, "symbol": "x"}, "k")


def _spx_pack(lo, hi):
    return {"kind": "tool", "tool": "forecast_report", "args": {}, "result": {"ok": True, "files": ["out/f.png"]},
            "full": {"forecast": {"point": [7000.0] * 24, "quantiles": {"0.1": [lo] * 24, "0.9": [hi] * 24},
                                  "last_observed": {"date": "2026-11-24", "value": 7200.0},
                                  "future_dates": ["2028-11-30"]}}}


def _vix(vol=26.6, long_run=18.0):
    return {"kind": "tool", "tool": "market_benchmark", "args": {"kind": "vix", "level": 7200},
            "result": {"ok": True, "kind": "vix", "implied_vol_pct": vol, "long_run_median_pct": long_run,
                       "as_of": "2026-11-26", "source_url": mb.SOURCES["vix"]}}


def test_a_band_far_narrower_than_the_options_market_blocks():
    # Live SPX (round 20261128-063203): 7170 to 9135 at 24 months beside a VIX of 26.6.
    c = pq.check_market_band([_vix(), _spx_pack(7170, 9135)])
    assert not c["ok"] and c["severity"] == "block" and "VIX 26.6" in c["detail"]
    wide = pq.check_market_band([_vix(), _spx_pack(4900, 10200)])
    assert wide["ok"]
    assert pq.check_market_band([_spx_pack(7170, 9135)]) is None, "no market reading, no check"


def test_the_answer_states_the_market_forecast_it_read():
    path = {"kind": "tool", "tool": "market_benchmark", "args": {"kind": "au_cash_rate"},
            "result": {"ok": True, "kind": "au_cash_rate", "implied_average": {"1m": 4.705, "3m": 4.72, "6m": 4.87},
                       "implied_forwards": [{"months": "3-6", "rate": 5.02}], "source_url": mb.SOURCES["au_cash_rate"],
                       "as_of": "2026-11-26"}}
    silent = pq.check_market_stated("The centre rises to 5.44% by 2031.", [path])
    assert not silent["ok"] and silent["severity"] == "block"
    said = pq.check_market_stated("Bank bills imply an average cash rate of 4.72% over three months.", [path])
    assert said["ok"]
    assert pq.check_market_stated("VIX is 26.6, implying a wide range.", [_vix()])["ok"]
    assert pq.check_market_stated("anything", []) is None
