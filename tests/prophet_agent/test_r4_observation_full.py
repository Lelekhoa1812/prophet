"""Live panels, round 4 (2026-11-23, measured): read_observation could not page back what a tool really returned.

1. The sandbox summary cuts the middle of a long stdout before the observation is stored, so the "rest" that
   read_observation paged was the already-cut text. Beijing analysts re-ran the same cell five times to see it.
2. Only the lead agent had a read_observation, bound to the lead's history; a sub-agent's call with its own step number
   paged the lead's step instead.
3. Figures printed in the cut part were real tool output, but grounding read only the cut view, so they were unmatched.

The observation now keeps the tool's full result under "full" (never shown to a model), read_observation pages it, each
sub-agent gets a pager on its own history, and grounding reads the full result."""
from prophet_agent.runtime import goal as g
from prophet_agent.runtime.grounding import unverified
from prophet_agent.runtime.runner import Tool


class _Ctx:
    def call(self, tool, args):
        return tool.fn(args, "k" * 32)


def _long_cell_tool():
    middle = "\n".join(f"client_{i:02d} MASE {1 + i / 100:.2f}" for i in range(60))
    stdout = "HEAD\n" + "x" * 3000 + "\n" + middle + "\n" + "y" * 3000 + "\nTAIL"

    def run(args, key):
        return {"ok": True, "run_id": "ab" * 16, "stdout": stdout, "outputs": []}

    def observe(result):
        text = result["stdout"]
        return {**result, "stdout": text[:200] + "\n[... cut ...]\n" + text[-200:]}

    return Tool("sandbox_python", run, idempotent=False, observe=observe), stdout


def test_the_observation_keeps_the_full_result_but_the_view_hides_it():
    tool, stdout = _long_cell_tool()
    obs = g._act(_Ctx(), tool, "sandbox_python", {"code": "print()"})
    assert "client_30 MASE 1.30" not in obs["result"]["stdout"]
    assert obs["full"]["stdout"] == stdout
    assert "full" not in g._view(obs)
    assert "client_30" not in g._evidence_lines([obs])


def test_read_observation_pages_the_full_result():
    tool, stdout = _long_cell_tool()
    obs = g._act(_Ctx(), tool, "sandbox_python", {"code": "print()"})
    obs["step"] = 1
    pager = g._observation_tool([obs])
    text = ""
    page = 1
    while True:
        out = pager.fn({"step": 1, "page": page}, "k")
        text += out["text"]
        if "next" not in out:
            break
        page = out["next"]
    assert "client_30 MASE 1.30" in text


def test_grounding_accepts_a_figure_printed_in_the_cut_part():
    tool, _ = _long_cell_tool()
    obs = g._act(_Ctx(), tool, "sandbox_python", {"code": "print()"})
    assert unverified("client_30 has a MASE of 1.30.", [obs], "forecast") == []


def test_each_sub_agent_pages_its_own_history():
    tools = {"read_observation": g._observation_tool([{"kind": "tool", "tool": "x", "result": {"lead": True}, "step": 1}])}
    sub_history = [{"kind": "tool", "tool": "x", "result": {"sub": True}, "step": 1}]
    bound = g._sub_observation_tools(tools, sub_history)
    assert '"sub"' in bound["read_observation"].fn({"step": 1}, "k")["text"]
