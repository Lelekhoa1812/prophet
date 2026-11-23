"""Round 6c (2026-11-23): an answer must name the global centre rule the newest pack drew.

Live electricity 29acb1c0: the last forecast_report drew global_blend for all 60 clients; the answer said the panel pack
assigned global_gbm. Every series here is synthetic.
"""
from prophet_agent.runtime import contracts


def _pack(rule, n=3):
    series = {f"c{i}": {"centre": {"rule": rule}} for i in range(n)}
    return [{"kind": "tool", "tool": "forecast_report", "args": {}, "result": {"n_series": n, "series": series}}]


def test_an_answer_naming_the_other_global_rule_is_blocked():
    text = "The panel pack assigns global_gbm as the centre for every client."
    c = contracts._centre_named_wrong(text, _pack("global_blend"))
    assert c is not None and c["ok"] is False and c["name"] == "answer_centre_named_right"
    assert "global_blend" in c["detail"]


def test_an_answer_naming_the_drawn_rule_passes():
    assert contracts._centre_named_wrong("The centre is the global_blend for every client.", _pack("global_blend")) is None


def test_a_sentence_that_negates_the_other_rule_is_not_a_claim_about_it():
    text = "The forecast uses the global_blend centre, not the requested global_gbm."
    assert contracts._centre_named_wrong(text, _pack("global_blend")) is None


def test_a_pack_with_mixed_rules_is_not_checked():
    obs = _pack("global_blend")
    obs[0]["result"]["series"]["c0"]["centre"]["rule"] = "timesfm"
    assert contracts._centre_named_wrong("The centre is global_gbm.", obs) is None


def test_a_ranking_sentence_that_names_a_rule_is_not_a_centre_claim():
    text = "TimesFM wins on 21 clients, global_gbm on 17, and the blend on 15."
    assert contracts._centre_named_wrong(text, _pack("global_blend")) is None
