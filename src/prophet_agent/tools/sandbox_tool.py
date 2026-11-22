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
from prophet_agent.sandbox.catalogue import build_catalogue
from prophet_agent.sandbox.runner import OUT_DIR, SANDBOX_ROOT, run_python
from prophet_agent.tools.series_io import RUN_ID

MAX_CODE_CHARS = 20_000
SEEN_STDOUT = 3_500
SEEN_TAIL = 1_200          # of the shown characters, how many come from the end
SEEN_STDERR = 1_500


MAX_CARRY = 40          # a 12-entity panel saves a series and a scan per entity (live: 8 refused 7 calls)


def carried_inputs(carry) -> dict[str, Path]:
    """Files an earlier run wrote to its ``out/``, mounted read-only at ``in/carried/<file name>`` in this run.

    Each run starts with an empty ``out/``, so a later step that needs the scan, the forecast or a cleaned table says
    which run wrote it: ``carry=[{"run_id": "<32 hex>", "file": "out/scan.json"}]``.
    """
    if carry is None:
        return {}
    if not isinstance(carry, list) or len(carry) > MAX_CARRY:
        raise ValueError(f"carry is a list of at most {MAX_CARRY} objects {{run_id, file}}")
    mounted: dict[str, Path] = {}
    for item in carry:
        if not isinstance(item, dict) or not RUN_ID.match(str(item.get("run_id", ""))):
            raise ValueError("each carry item needs run_id (the 32-character id of the run that wrote the file) and file")
        rel = str(item.get("file", ""))
        base = (SANDBOX_ROOT / item["run_id"]).resolve()
        target = (base / rel).resolve()
        if not rel or not str(target).startswith(str((base / OUT_DIR).resolve()) + "/") or not target.is_file():
            raise ValueError(f"no file {rel!r} in the outputs of run {item['run_id']}")
        mounted[f"carried/{target.name}"] = target
    return mounted


def make_run(datasets: dict[str, Path] | None = None):
    inputs = dict(datasets or {})

    def _run(args: dict, _key: str) -> dict:
        extra = sorted(set(args) - {"code", "carry"})
        if extra:
            raise ValueError(f"unknown arguments {extra}; allowed: code, carry")
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be a non-empty string")
        if len(code) > MAX_CODE_CHARS:
            raise ValueError(f"code is longer than {MAX_CODE_CHARS} characters")
        res = run_python(code, timeout_s=60, mem_mb=1536, inputs={**inputs, **carried_inputs(args.get("carry"))})
        d = res.to_dict()
        return {"ok": d["ok"], "returncode": d["returncode"], "timed_out": d["timed_out"],
                "stdout": d["stdout"], "stderr": d["stderr"], "run_id": d["run_id"],
                "outputs": d["outputs"], "memory_limit_enforced": d["memory_limit_enforced"]}
    return _run


def observe(result: dict) -> dict:
    text = result["stdout"]
    if len(text) > SEEN_STDOUT:
        # Head and tail: a summary table usually prints last. Live (Beijing): cells printed about 9,000 characters, the
        # planner saw the first 3,500 and re-ran the same profiling about 35 times "to confirm" what was cut.
        head, tail = text[: SEEN_STDOUT - SEEN_TAIL], text[-SEEN_TAIL:]
        text = head + f"\n[... {len(result['stdout']) - SEEN_STDOUT} characters cut ...]\n" + tail
    out = {"ok": result["ok"], "run_id": result["run_id"], "stdout": text,
           "outputs": [f"{o['path']} ({o['bytes']} bytes)" for o in result["outputs"]]}
    if len(result["stdout"]) > SEEN_STDOUT:
        out["stdout_note"] = (f"{len(result['stdout'])} characters printed; the middle is cut. Do not re-run this cell to see "
                              "it: save the figures to a file under out/ (csv or json) and print only the lines the next "
                              "decision needs, one per entity")
    if result["stderr"]:
        out["stderr_tail"] = result["stderr"][-SEEN_STDERR:]
    if result["timed_out"]:
        out["timed_out"] = True
    return out


def sandbox_python_tool(datasets: dict[str, Path] | None = None) -> Tool:
    return Tool("sandbox_python", make_run(datasets), idempotent=True, observe=observe)


_SPEC_TEXT = (
    "Run Python (pandas, numpy, scipy, statsmodels, matplotlib) in a guardrailed sandbox: no network, no secrets, one "
    "minute. pandas only: `pl` is prophet_lib, not polars (df.null_count(), is_null() and sort() work, as do the helpers "
    "below). The person's uploaded files are in in/ (read-only). Import: `import prophet_lib as pl`, `import prophet_stats "
    "as ps`, `import prophet_scan as sc`. A helper that returns (value, report) is unpacked: `value, report = ...`. "
    "Entities that share dates are never summed silently: pass entity= and key=, or use pl.panel for all of them. "
    "pl.chart(kind, data, x=, y=, title=<the finding>, subtitle=<measure, unit, period>, unit=, source=, path='out/name.png') "
    "returns the problems it finds: read them. pl.size_report(path) and pl.query(sql, t=path) read files too big to load. "
    "Also: pl.profile(path), pl.as_number(s), pl.as_dates(s), pl.tidy_text(s), pl.to_number(s) and pl.to_datetime(s) "
    "(pairs: unpack them), pl.holiday_covariate(index, country, subdiv, horizon) and pl.event_covariate(index, "
    "[(first, last), ...], horizon) for known-ahead drivers, pl.detect_anomalies(values, season) (its candidates are "
    "measured facts to explain, not verdicts). "
    "Write files to out/. Every run starts with an EMPTY out/; to read an earlier run's file, pass "
    "carry=[{run_id, file: 'out/scan.json'}] and open it at in/carried/<file name>. "
    "Print only what you need: output you do not print, you do not see. A series written with pl.save_series or "
    "pl.save_panel can be forecast by passing source {run_id, file}. "
    "Signatures (generated from the code; a keyword not listed is refused with the real signature):\n"
)

SANDBOX_PYTHON_SPEC = {
    "name": "sandbox_python",
    "title": "Analysis",
    "activity": "Running an analysis",
    "description": _SPEC_TEXT + build_catalogue(),
    "args": {"code": "Python source, at most 20000 characters",
             "carry": "optional list of {run_id, file} outputs of earlier runs, mounted read-only at in/carried/<file name>"},
}
