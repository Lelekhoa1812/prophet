"""Sandboxed Python as a runtime tool, for analytics the planner writes itself.

Motivation vs Logic
Motivation: some questions need a calculation the fixed tools do not cover. The planner may write Python, but that code
runs only in the sandbox guardrails, with a timeout and no secrets.
Logic: each call gets a fresh run folder, so it is idempotent in effect: nothing outside the folder changes. The result
returns stdout, stderr, the exit state and the hashed output files.
"""
from __future__ import annotations

from prophet_agent.runtime.runner import Tool
from prophet_agent.sandbox.runner import run_python

MAX_CODE_CHARS = 20_000


def _run(args: dict, _key: str) -> dict:
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code must be a non-empty string")
    if len(code) > MAX_CODE_CHARS:
        raise ValueError(f"code is longer than {MAX_CODE_CHARS} characters")
    res = run_python(code, timeout_s=30, mem_mb=1024)
    d = res.to_dict()
    return {"ok": d["ok"], "returncode": d["returncode"], "timed_out": d["timed_out"],
            "stdout": d["stdout"], "stderr": d["stderr"], "run_id": d["run_id"],
            "outputs": d["outputs"], "memory_limit_enforced": d["memory_limit_enforced"]}


def sandbox_python_tool() -> Tool:
    return Tool("sandbox_python", _run, idempotent=True)


SANDBOX_PYTHON_SPEC = {
    "name": "sandbox_python",
    "description": "Run Python for analytics in a guardrailed sandbox (no network, no secrets, writes only to out/). "
                   "Returns stdout, stderr and hashed output files.",
    "args": {"code": "Python source, at most 20000 characters"},
}
