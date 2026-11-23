"""Round 6l, goal audit item 5: a method claim is matched to a call's arguments by word, not by any substring.

Synthetic calls only. Before this round "symmetric" was found inside "asymmetric", so a claim that a symmetric loss was
backtested passed on a call that only used the asymmetric loss. (The "arima" in "arimax" example in the audit is not a
live false pass: ARIMA is not a named treatment in grounding.py, so no claim about it is read at all.)
"""
from __future__ import annotations

from prophet_agent.runtime.grounding import _carries, unsupported_method_claims


def _call(args):
    return [{"kind": "tool", "tool": "backtest", "args": args, "result": {"mase": 1.1}}]


def test_a_symmetric_loss_claim_does_not_pass_on_an_asymmetric_call():
    units = unsupported_method_claims("A symmetric loss was backtested.", _call({"loss": "asymmetric pinball"}))
    assert units == ["A symmetric loss was backtested."]


def test_a_symmetric_loss_claim_passes_on_a_symmetric_call():
    assert unsupported_method_claims("A symmetric loss was backtested.", _call({"loss": "symmetric"})) == []


def test_an_asymmetric_loss_claim_passes_on_an_asymmetric_call_and_fails_without_one():
    assert unsupported_method_claims("An asymmetric loss was backtested.", _call({"loss": "asymmetric pinball"})) == []
    assert unsupported_method_claims("An asymmetric loss was backtested.", _call({"loss": "pinball"})) == [
        "An asymmetric loss was backtested."]


def test_a_marker_must_start_a_word_so_a_word_containing_it_does_not_carry_it():
    assert _carries('{"loss": "symmetric"}', "symmetric") is True
    assert _carries('{"loss": "asymmetric"}', "symmetric") is False
    assert _carries('{"winsorise": true}', "winsor") is True          # a stem still matches the word it begins
    assert _carries('{"x": "nonwinsor"}', "winsor") is False


def test_a_winsorised_claim_still_passes_on_a_winsorising_call():
    units = unsupported_method_claims("Outliers were winsorised and the model was backtested.",
                                      _call({"treatment": "winsorize", "clip": 3}))
    assert units == []
