"""Round 3 gate B4, goal 2 (A6-03): a flat TimesFM band must hold the delivered pack even when the centre line is a simple rule.

Live evidence (measured): retail rep4 evidence [52] and [53]: centre naive, health flat_unexpected (period 5, shape ratio 0.14),
and the answer quoted an 80 per cent range built on that band. The band is TimesFM's whatever centre line is drawn.
"""
from prophet_agent.runtime.contracts import check_evidence, delivered_health, progress


def _pack(health, centre=None):
    rec = {"point": [10.0, 11.0, 12.0], "quantiles": {"0.1": [8.0, 9.0, 10.0], "0.9": [12.0, 13.0, 14.0]},
           "health": {"verdict": health, "causes": ["the history repeats every 5 steps"] if health == "flat_unexpected" else []}}
    if centre is not None:
        rec["centre"] = {"rule": centre}
    return rec


def _forecast_obs(rec):
    return {"kind": "tool", "tool": "forecast_report", "args": {"centre": rec.get("centre", {}).get("rule", "timesfm")},
            "result": {"forecast": rec, "run_id": "a" * 32, "files": []}}


def _inspect():
    return {"kind": "tool", "tool": "inspect_data", "args": {}, "result": {"ok": True}}


def test_a_flat_band_is_reported_as_the_delivered_health_whatever_the_centre():
    # Before the fix this returned "centre_rule:naive", so health_repaired and the checklist passed on a flat band.
    assert delivered_health(_pack("flat_unexpected", centre="naive")) == "flat_unexpected"
    assert delivered_health(_pack("ok", centre="naive")) == "ok"


def test_a_rule_centre_on_a_flat_band_fails_the_repair_check_and_blocks_the_finish():
    checks = check_evidence([_inspect(), _forecast_obs(_pack("flat_unexpected", centre="naive"))])
    repaired = next(c for c in checks if c["name"] == "health_repaired")
    assert not repaired["ok"] and repaired["severity"] == "block"


def test_the_checklist_does_not_tick_the_health_item_on_a_flat_band_under_a_rule_centre():
    item = next(i for i in progress([_forecast_obs(_pack("flat_unexpected", centre="naive"))], run_root=None)
                if i["item"].startswith("forecast health"))
    assert item["status"] == "todo"


def test_a_panel_line_with_a_flat_band_holds_the_pack_even_on_a_rule_centre():
    panel = {"kind": "tool", "tool": "forecast_report", "args": {},
             "result": {"n_series": 2, "run_id": "b" * 32, "files": [],
                        "series": {"A": {"health": "ok", "centre": "timesfm"},
                                   "B": {"health": "flat_unexpected", "centre": "naive"}}}}
    checks = check_evidence([_inspect(), panel])
    assert not next(c for c in checks if c["name"] == "health_repaired")["ok"]
