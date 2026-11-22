"""C3a, A4-14 (grounding side): a method the answer says was "tested" or "computed" must have a tool call that ran it.

Live evidence (measured, audit round 3, A4): retail rep4 answered with a table row "Excluding first week / winsorizing outliers:
Tested", and "Prophet with changepoints: Tested by sub-agent". The trace has no control_lab or backtest_treatment call for either,
and the sub-agent's seven steps only read files. The claim reached the person with nothing behind it. The check is a warning.
"""
from prophet_agent.runtime import contracts, grounding

INSPECT = {"kind": "tool", "tool": "sandbox_python", "args": {"code": "print(df.head())"},
           "result": {"ok": True, "stdout": "2020-01-01 1.0"}}


def _backtest(args):
    return {"kind": "tool", "tool": "backtest_series", "args": args, "result": {"ok": True, "horizon": 14, "models": {}}}


def _failing(checks, name):
    return [c for c in checks if c["name"] == name and not c["ok"]]


ROW = "| Excluding first week / winsorizing outliers | Tested; did not resolve flat forecast shape |"


def test_a_tested_treatment_with_no_call_that_runs_it_is_unsupported():
    assert grounding.unsupported_method_claims(ROW, [INSPECT]) == [ROW]


def test_the_same_claim_is_supported_by_a_call_that_runs_the_treatment():
    history = [INSPECT, _backtest({"horizon": 14, "ignore_ranges": [["2020-02-01", "2020-03-01"]], "replace_outliers": True})]
    assert grounding.unsupported_method_claims(ROW, history) == []


def test_a_prophet_claim_backed_only_by_file_reads_in_a_sub_agent_is_unsupported():
    history = [{"kind": "subagent", "agent_type": "analyst", "answer": "Prophet gave 809.6k.", "evidence": [
        {"kind": "tool", "tool": "read_file", "args": {"path": "prophet_agent/forecast/health.py"}, "result": {"text": "..."}}]}]
    answer = "Prophet with changepoints: Tested by sub-agent."
    assert grounding.unsupported_method_claims(answer, history) == [answer]


def test_a_prophet_claim_backed_by_a_sub_agent_run_that_fits_it_is_supported():
    history = [{"kind": "subagent", "agent_type": "analyst", "answer": "done", "evidence": [
        {"kind": "tool", "tool": "sandbox_python", "args": {"code": "from prophet import Prophet\nm = Prophet()"},
         "result": {"ok": True}}]}]
    assert grounding.unsupported_method_claims("Prophet was tested on the series.", history) == []


def test_a_call_that_failed_does_not_back_the_claim():
    failed = dict(_backtest({"ignore_ranges": [["2020-02-01", "2020-03-01"]]}), error="timeout")
    assert grounding.unsupported_method_claims("Winsorising the outliers was tested.", [failed]) != []


def test_a_negated_test_claim_is_not_read_as_a_claim():
    assert grounding.unsupported_method_claims("Winsorising the outliers was not tested.", [INSPECT]) == []


def test_a_test_claim_with_no_known_treatment_is_not_read():
    assert grounding.unsupported_method_claims("The model was tested on the holdout.", [INSPECT]) == []


def test_the_answer_check_reports_the_unsupported_claim_as_a_warning():
    checks = contracts.check_answer(ROW, [INSPECT])
    bad = _failing(checks, "answer_methods_run")
    assert len(bad) == 1 and bad[0]["severity"] == "warn"


def test_the_answer_check_is_quiet_when_the_claim_is_backed():
    history = [INSPECT, _backtest({"ignore_ranges": [["2020-02-01", "2020-03-01"]], "replace_outliers": True})]
    assert _failing(contracts.check_answer(ROW, history), "answer_methods_run") == []
