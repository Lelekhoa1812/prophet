"""Round 4 follow-up (A6-08c): run_lab tests the caller's ignore_ranges as a candidate, but the control_lab tool refused the
argument ("unknown arguments ['ignore_ranges']"), so the candidate could not be reached from the planner."""
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_lab import H, MeanEngine, spiky  # noqa: E402

from prophet_agent.tools import lab_tool as T  # noqa: E402


def test_control_lab_accepts_ignore_ranges_and_runs_the_candidate():
    tool = T.lab_tool(lambda: MeanEngine())
    x = spiky().tolist()
    result = tool.fn({"values": x, "horizon": H, "candidates": ["ignore_ranges"], "ignore_ranges": [[10, 14]]}, "k")
    names = [r["name"] if "name" in r else r.get("candidate") for r in result["table"]]
    assert "ignore_ranges" in names, names


def test_the_spec_documents_ignore_ranges():
    assert "ignore_ranges" in T.LAB_TOOL_SPEC["args"]
