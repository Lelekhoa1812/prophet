"""An image the answer names must be a file a run made.

Live evidence (measured, 2026-11-25, NVDA diagram mission): three sandbox_python cells failed with a SyntaxError and listed no
outputs. The answer still said "The diagram has been created and saved as nvda_forecast_pipeline_diagram.png", and the person
saw no image and no warning.
"""
from prophet_agent.api.outcome import outcome_for
from prophet_agent.runtime import contracts
from prophet_agent.runtime.goal import _ground_result

FAILED_CELL = {"kind": "tool", "tool": "sandbox_python", "args": {"code": "draw_box('Yahoo Finance"},
               "result": {"ok": False, "run_id": "a" * 32, "stdout": "", "outputs": [],
                          "stderr": "SyntaxError: unterminated string literal"}}
CHART_CELL = {"kind": "tool", "tool": "sandbox_python", "args": {"code": "..."},
              "result": {"ok": True, "run_id": "b" * 32, "stdout": "", "outputs": ["out/chart.png (2048 bytes)"]}}
PACK = {"kind": "tool", "tool": "forecast_report", "args": {},
        "result": {"run_id": "c" * 32, "files": [{"path": "out/forecast.png", "sha256": "x", "bytes": 1}]}}
NVDA = "The diagram has been created and saved as **nvda_forecast_pipeline_diagram.png**.\n\nFile: `nvda_forecast_pipeline_diagram.png`"


def test_an_image_no_run_made_blocks():
    check = contracts.check_answer_files(NVDA, [FAILED_CELL])
    assert check is not None and not check["ok"] and check["severity"] == "block"
    assert "nvda_forecast_pipeline_diagram.png" in check["detail"]


def test_the_check_is_part_of_the_answer_checks():
    assert [c["name"] for c in contracts.blocking(contracts.check_answer(NVDA, [FAILED_CELL]))] == ["answer_file_exists"]


def test_images_listed_by_a_cell_or_a_pack_pass():
    answer = "See ![chart](out/chart.png) and forecast.png."
    check = contracts.check_answer_files(answer, [CHART_CELL, PACK])
    assert check is not None and check["ok"]


def test_an_image_a_sub_agent_made_passes():
    history = [{"kind": "subagent", "agent_type": "analyst", "answer": "done", "evidence": [CHART_CELL]}]
    assert contracts.check_answer_files("The chart is chart.png.", history)["ok"]


def test_a_web_address_is_not_read_as_a_file_name():
    assert contracts.check_answer_files("Source: https://example.com/img/logo.png", [FAILED_CELL]) is None


def test_an_answer_that_names_no_image_adds_no_check():
    assert contracts.check_answer_files("No diagram was made, because the code failed.", [FAILED_CELL]) is None


def test_every_exit_path_records_the_failed_check_and_a_met_answer_becomes_partial():
    met = _ground_result({"status": "met", "answer": NVDA}, [FAILED_CELL], "draw a diagram")
    assert met["status"] == "partial" and met["contract_failures"] == ["answer_file_exists"]
    stalled = _ground_result({"status": "partial", "answer": NVDA, "reason": "kept repeating the same step"},
                             [FAILED_CELL], "draw a diagram")
    assert stalled["contract_failures"] == ["answer_file_exists"]
    plain = [c["plain"] for r in outcome_for("partial", stalled, {}, [])["reasons"] for c in r.get("checks", [])]
    assert plain == ["the answer names an image that was not made"]


def test_an_answer_naming_a_made_image_keeps_its_status():
    out = _ground_result({"status": "met", "answer": "The chart is chart.png."}, [CHART_CELL], "chart it")
    assert out["status"] == "met" and "contract_failures" not in out
