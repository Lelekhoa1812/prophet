"""Sandboxed Python as a runtime tool, for analytics the planner writes itself.

Motivation vs Logic
Motivation: real data needs real analysis: reading odd files, cleaning, aggregating, looking at patterns, testing ideas.
The planner may write Python for that, but the code runs only inside the sandbox guardrails, with a timeout and no secrets.
Logic: each call gets a fresh run folder, so it is idempotent in effect. The person's files for this mission are
mounted read-only in ``in/``; ``prophet_lib`` is importable; results are written to ``out/``. The planner sees stdout,
stderr (the tail, where Python puts the error), and the list of output files; the ledger keeps the full result.
"""
from __future__ import annotations

from pathlib import Path

from prophet_agent.runtime.runner import Tool
from prophet_agent.sandbox.runner import run_python

MAX_CODE_CHARS = 20_000
SEEN_STDOUT = 3_500
SEEN_STDERR = 1_500


def make_run(datasets: dict[str, Path] | None = None):
    inputs = dict(datasets or {})

    def _run(args: dict, _key: str) -> dict:
        extra = sorted(set(args) - {"code"})
        if extra:
            raise ValueError(f"unknown arguments {extra}; allowed: code")
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be a non-empty string")
        if len(code) > MAX_CODE_CHARS:
            raise ValueError(f"code is longer than {MAX_CODE_CHARS} characters")
        res = run_python(code, timeout_s=60, mem_mb=1536, inputs=inputs)
        d = res.to_dict()
        return {"ok": d["ok"], "returncode": d["returncode"], "timed_out": d["timed_out"],
                "stdout": d["stdout"], "stderr": d["stderr"], "run_id": d["run_id"],
                "outputs": d["outputs"], "memory_limit_enforced": d["memory_limit_enforced"]}
    return _run


def observe(result: dict) -> dict:
    out = {"ok": result["ok"], "run_id": result["run_id"], "stdout": result["stdout"][:SEEN_STDOUT],
           "outputs": [f"{o['path']} ({o['bytes']} bytes)" for o in result["outputs"]]}
    if len(result["stdout"]) > SEEN_STDOUT:
        out["stdout_note"] = f"{len(result['stdout'])} characters; print less or summarise"
    if result["stderr"]:
        out["stderr_tail"] = result["stderr"][-SEEN_STDERR:]
    if result["timed_out"]:
        out["timed_out"] = True
    return out


def sandbox_python_tool(datasets: dict[str, Path] | None = None) -> Tool:
    return Tool("sandbox_python", make_run(datasets), idempotent=True, observe=observe)


SANDBOX_PYTHON_SPEC = {
    "name": "sandbox_python",
    "title": "Analysis",
    "activity": "Running an analysis",
    "description": "Run Python (pandas, numpy, scipy, statsmodels, matplotlib) in a guardrailed sandbox: no network, no "
                   "secrets, one minute. The person's uploaded files are in in/ (read-only). `import prophet_lib as pl` "
                   "gives pl.load(path, sheet) (one typed DataFrame: numbers and dates converted, blanks missing; the "
                   "report is df.attrs['report']), pl.profile(path), pl.as_number(s), pl.as_dates(s, dayfirst=None), "
                   "pl.build_series(...), pl.find_season(values) and pl.save_series(series, 'out/series.json'). "
                   "pl.read_table, pl.to_number and pl.to_datetime return (value, info) pairs: unpack both. Draw charts with pl.chart(kind, data, title=<the finding>, subtitle=<measure, unit, period>, unit=..., source=..., path=\"out/name.png\") and read the problems it returns. pl.size_report(path) and pl.query(sql, t=path) work on CSV or Parquet files too big to read whole (the query sees only the named files). Write files to out/. "
                   "Print only what you need: output you do not print, you do not see. A series written with "
                   "pl.save_series can be forecast by passing source {run_id, file}.",
    "args": {"code": "Python source, at most 20000 characters"},
}
