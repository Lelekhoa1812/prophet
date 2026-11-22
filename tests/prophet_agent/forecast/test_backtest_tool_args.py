"""The backtest_series tool's new arguments: combine must be a boolean, and n_origins defaults to twelve. The model is not
loaded: argument errors are raised before any engine call."""
from __future__ import annotations

import pytest

from prophet_agent.tools import data_tools as D


def test_combine_must_be_true_or_false():
    with pytest.raises(ValueError, match="combine must be true or false"):
        D._backtest({"values": [float(i % 5) for i in range(120)], "horizon": 4, "combine": "yes"}, "k")


def test_combine_and_n_origins_are_documented_in_the_tool_spec():
    assert "combine" in D.BACKTEST_SPEC["args"] and "default 12" in D.BACKTEST_SPEC["args"]["n_origins"]
