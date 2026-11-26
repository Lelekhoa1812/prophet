"""Chart digitising: the model proposes ticks (mocked here), code fits the axes and measures the series from the pixels."""
import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from prophet_agent.research.chart_vision import (READ_FROM_IMAGE, ChartReadError, ask_model, digitise_image, fit_axis)
from tests.prophet_agent.research.chart_fixtures import known_series, render, truth_range


def _median_error_share(result, x_true, y_true, y_range):
    series = result["series"][0]
    xs, ys = np.array(series["x"]), np.array(series["y"])
    order = np.argsort(xs)
    inside = (x_true >= xs.min()) & (x_true <= xs.max())
    estimate = np.interp(x_true[inside], xs[order], ys[order])
    return float(np.median(np.abs(estimate - y_true[inside])) / y_range)


class FakeClient:
    """Stands in for NimClient.complete: returns a canned proposal and records the messages it was sent."""

    def __init__(self, proposal, model="nvidia/nemotron-3.5-super-vl-120b-a12b"):
        self.text = json.dumps(proposal)
        self.model = model
        self.calls = []

    def complete(self, models, messages, **kwargs):
        self.calls.append({"models": list(models), "messages": messages, "kwargs": kwargs})
        return SimpleNamespace(text=self.text, model=self.model)


@pytest.fixture(scope="module")
def linear_chart():
    x, y = known_series()
    png, proposal = render(x, y)
    return png, proposal, x, y


def test_code_path_median_error_under_two_percent_of_axis_range(linear_chart):
    png, proposal, x, y = linear_chart
    result = digitise_image(png, proposal=proposal, model_name="offline-ticks")
    error = _median_error_share(result, x, y, truth_range()[1] - truth_range()[0])
    assert error < 0.02, error


def test_result_is_labelled_read_from_image_with_bounds_and_self_check(linear_chart):
    png, proposal, _, _ = linear_chart
    result = digitise_image(png, proposal=proposal, model_name="offline-ticks")
    assert result["exactness"] == READ_FROM_IMAGE
    s = result["series"][0]
    assert s["exactness"] == READ_FROM_IMAGE and s["method"] == "digitised"
    assert s["model"] == "offline-ticks"
    assert s["error_bound_y"] > 0 and s["error_bound_x"] > 0
    assert s["self_check"]["checked"] and s["self_check"]["precision"] > 0.8 and s["self_check"]["recall"] > 0.8
    assert result["proposal"] == proposal                     # kept for replay
    assert result["axes"]["y"]["fit"]["scale"] == "linear"


def test_gridlines_are_not_extracted_as_series(linear_chart):
    png, proposal, _, _ = linear_chart
    result = digitise_image(png, proposal=proposal, model_name="offline-ticks")
    assert len(result["series"]) == 1
    assert result["series"][0]["mask"]["gridline_pixels_removed"] >= 0


def test_log_axis_fit_is_used():
    x, y = known_series()
    y = np.clip(y, 2, 900)
    png, proposal = render(x, y, ylog=True)
    result = digitise_image(png, proposal=proposal, model_name="offline-ticks")
    assert result["axes"]["y"]["fit"]["scale"] == "log"
    s = result["series"][0]
    assert all(v > 0 for v in s["y"])


def test_model_call_sends_image_part_and_schema(linear_chart):
    png, proposal, _, _ = linear_chart
    client = FakeClient(proposal)
    result = digitise_image(png, client=client)
    assert result["model"] == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    call = client.calls[0]
    assert call["models"][0] == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    assert "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning" in call["models"]
    parts = call["messages"][1]["content"]
    assert any(p.get("type") == "image_url" and p["image_url"]["url"].startswith("data:image/png;base64,") for p in parts)
    assert call["kwargs"]["schema_name"] == "chart_axes"
    assert result["series"][0]["x"]


def test_model_data_values_are_ignored(linear_chart):
    png, proposal, x, y = linear_chart
    poisoned = copy.deepcopy(proposal)
    poisoned["series"][0]["points"] = [{"x": 0, "y": 999}]      # not in the schema: never used
    result = digitise_image(png, proposal=poisoned, model_name="offline-ticks")
    assert max(result["series"][0]["y"]) < 125


def test_fit_rejects_too_few_ticks_and_non_monotonic_ticks():
    with pytest.raises(ChartReadError, match="two"):
        fit_axis([(10, 1)], "linear")
    with pytest.raises(ChartReadError, match="monotonic"):
        fit_axis([(10, 1), (20, 5), (30, 2)], "linear")
    with pytest.raises(ChartReadError, match="distinct"):
        fit_axis([(10, 1), (10, 5)], "linear")


def test_fit_rejects_large_residual_and_accepts_exact_line():
    with pytest.raises(ChartReadError, match="residual"):
        fit_axis([(0, 0), (100, 10), (200, 20), (300, 45)], "linear")
    fit = fit_axis([(0, 0), (100, 10), (200, 20)], "linear")
    assert fit.value(150) == pytest.approx(15)


def test_fit_log_axis_requires_positive_values():
    with pytest.raises(ChartReadError, match="positive"):
        fit_axis([(0, 0), (100, 10)], "log")
    fit = fit_axis([(0, 1), (100, 10), (200, 100)], "log")
    assert fit.value(50) == pytest.approx(10 ** 0.5, rel=1e-6)


def test_refuses_a_non_image_and_a_bad_schema(linear_chart):
    with pytest.raises(ChartReadError, match="PNG and JPEG"):
        digitise_image(b"GIF89a....")
    png, proposal, _, _ = linear_chart
    bad = copy.deepcopy(proposal)
    del bad["series"]
    with pytest.raises(ChartReadError, match="schema"):
        digitise_image(png, proposal=bad)


def test_ticks_outside_the_image_are_refused(linear_chart):
    png, proposal, _, _ = linear_chart
    bad = copy.deepcopy(proposal)
    bad["y_axis"]["ticks"][0]["pixel"] = 9000
    with pytest.raises(ChartReadError):
        digitise_image(png, proposal=bad)


def test_ask_model_rejects_non_json_reply():
    client = SimpleNamespace(complete=lambda *a, **k: SimpleNamespace(text="not json", model="m"))
    with pytest.raises(ChartReadError, match="JSON"):
        ask_model(client, b"\x89PNG\r\n\x1a\n", "image/png")


@pytest.mark.skipif(os.environ.get("PROPHET_LIVE") != "1", reason="live vision test: set PROPHET_LIVE=1")
def test_live_vision_ticks_then_code_measures_series():
    """The model proposes ticks and colours; code measures the series. Reports the measured median error share."""
    from prophet_agent.nim.transport import shared_client
    x, y = known_series()
    png, _ = render(x, y)
    result = digitise_image(png, client=shared_client("nim"))
    error = _median_error_share(result, x, y, truth_range()[1] - truth_range()[0])
    print(f"\nLIVE chart digitise: model={result['model']} median abs error = {error:.4%} of y-range")
    assert error < 0.02
