"""Sandboxed Python as a runtime tool, for analytics the planner writes itself.

Motivation vs Logic
Motivation: real data needs real analysis: reading odd files, cleaning, aggregating, looking at patterns, testing ideas.
The planner may write Python for that, but the code runs only inside the sandbox guardrails, with a timeout and no secrets.
Logic: each call gets a fresh run folder, so it is idempotent in effect. The person's files for this mission are
mounted read-only in ``in/``; ``prophet_lib`` is importable; results are written to ``out/``. The planner sees stdout,
stderr (the tail, where Python puts the error), and the list of output files; the ledger keeps the full result.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from prophet_agent.runtime.runner import Tool
from prophet_agent.sandbox.catalogue import build_catalogue
from prophet_agent.sandbox.runner import OUT_DIR, SANDBOX_ROOT, run_python
from prophet_agent.tools.series_io import RUN_ID, missing_output

MAX_CODE_CHARS = 20_000
SEEN_STDOUT = 3_500
SEEN_TAIL = 1_200          # of the shown characters, how many come from the end
SEEN_STDERR = 1_500


MAX_CARRY = 40          # a 12-entity panel saves a series and a scan per entity (live: 8 refused 7 calls)

# Provenance guard (SciRes Mauna Loa run, steps 36 to 39): a cell that builds numbers and names an external source, and reads no
# file from in/, would present invented numbers as that source. Detection is by text, so it is a guard, not a proof.
GENERATES_NUMBERS = re.compile(
    r"np\.random\.|numpy\.random\.|\brandom\.(?:seed|random|normal|uniform|randint|choice|gauss|shuffle|sample)\b"
    r"|default_rng\(|\brng\.\w+\(|\.standard_normal\(|\.poisson\(|\.binomial\(|\.lognormal\(")
TYPED_NUMBERS = re.compile(r"\[\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){19,}\s*,?\s*\]")
NAMED_SOURCE_ACRONYM = re.compile(r"\b(?:NOAA|GML|NASA|ABS|FRED|ECB|OECD|IMF|BLS|EIA|USGS|WMO|ONS|BoM|Eurostat)\b")  # case-sensitive
NAMED_SOURCE = re.compile(
    r"https?://|mauna\s+loa|bureau of (?:statistics|meteorology)|world bank|stats nz|statistics canada|census bureau",
    re.IGNORECASE)
READS_INPUT = re.compile(r"""['"]in/|['"]in['"]|\bread_(?:csv|excel|parquet|json|table)\(|\bload_series\(|\bpl\.(?:query|size_report|profile)\(""")


def invented_source_reason(code: str) -> str | None:
    """Why this cell must not run, or None. Refused: it builds numbers (a random generator or a typed list of 20 or more
    numbers), its comments or prints name an external source, and it reads no file from in/. Plain random numbers, and a
    bootstrap of an uploaded file, are allowed."""
    if not (GENERATES_NUMBERS.search(code) or TYPED_NUMBERS.search(code)):
        return None
    named = NAMED_SOURCE_ACRONYM.search(code) or NAMED_SOURCE.search(code)
    if named is None or READS_INPUT.search(code):
        return None
    return (f"refused: this cell builds numbers (a random generator or a typed list of numbers) and names the source "
            f"{named.group(0)!r}, and it reads no file from in/. Numbers built here would be presented as that source. Read the "
            "source's own file from in/ or a carried run, or retrieve it with web research and record its id and URL. If the "
            "source was not retrieved, say so in the answer and build no stand-in values. Numbers that are only for a test or a "
            "labelled synthetic replay must not name the source: label them synthetic.")


STAND_IN_WORD = re.compile(r"\b(?:synthetic|placeholder|dummy|fake|mock|made[- ]up|stand[- ]in|fabricated)\b", re.I)


def stand_in_reason(code: str, goal: str = "") -> str | None:
    """Why a cell that builds stand-in data must not run, or None.

    Motivation: panel CPI (2026-11-26) had no data in the sandbox, and an analyst cell built "synthetic CPI data" with a
    random generator; five more cells analysed it before the run stalled. Logic: refused when the cell generates numbers
    (a random generator or a long typed list), calls them synthetic, placeholder, dummy, fake or similar, reads no file
    from in/, and the goal does not itself ask for synthetic data or a test.
    """
    if not (GENERATES_NUMBERS.search(code) or TYPED_NUMBERS.search(code)) or READS_INPUT.search(code):
        return None
    word = STAND_IN_WORD.search(code)
    if word is None or re.search(r"\bsynthetic\b|\bsimulat|\btest (?:case|data|series)\b", goal or "", re.I):
        return None
    return (f"refused: this cell builds {word.group(0)} numbers and reads no file from in/. The goal asks for the real "
            "series, so no stand-in values are built. Retrieve the data with download_web_data, download_open_data or "
            "download_market_prices (the sandbox has no network), or report what could not be found.")


def carried_inputs(carry, inputs: dict[str, Path] | None = None) -> dict[str, Path]:
    """Files an earlier run wrote to its ``out/``, mounted read-only at ``in/carried/<file name>`` in this run.

    Each run starts with an empty ``out/``, so a later step that needs the scan, the forecast or a cleaned table says
    which run wrote it: ``carry=[{"run_id": "<32 hex>", "file": "out/scan.json"}]``.
    """
    if carry is None:
        return {}
    if not isinstance(carry, list) or len(carry) > MAX_CARRY:
        raise ValueError(f"carry is a list of at most {MAX_CARRY} objects {{run_id, file}}")
    mounted: dict[str, Path] = {}
    inputs = inputs or {}
    for item in carry:
        name = str(item.get("file", "")) if isinstance(item, dict) else ""
        mounted_input = inputs.get(Path(name).name) if name else None
        if mounted_input is not None and not _run_wrote(item, name):
            # Panel round 3 NEM and GISTEMP (2026-11-26): the planner carried downloaded files with a step number as the run
            # id ("18") and read them from in/carried/. A download is already mounted at in/<name>; it is mounted at
            # in/carried/<name> too, so the cell runs as written.
            mounted[f"carried/{Path(name).name}"] = mounted_input
            continue
        if name.startswith("in/"):
            # Panel CPI (2026-11-26): the planner carried in/fredgraph.csv, a downloaded file already mounted in in/, and
            # was refused three times. Inputs are mounted at every call, so the item is not needed.
            continue
        if not isinstance(item, dict) or not RUN_ID.match(str(item.get("run_id", ""))):
            raise ValueError("each carry item needs run_id (the 32-character id of the run that wrote the file) and file")
        rel = str(item.get("file", ""))
        base = (SANDBOX_ROOT / item["run_id"]).resolve()
        target = (base / rel).resolve()
        if not rel or not str(target).startswith(str((base / OUT_DIR).resolve()) + "/") or not target.is_file():
            raise ValueError(missing_output(item["run_id"], rel))
        key = f"carried/{target.name}"
        if key in mounted and mounted[key] != target:
            # Two runs wrote the same name (each analyst saves out/scan.json). The later one is mounted under its run's first
            # eight characters, and the result lists where each file went. Live Beijing rep6: five calls failed on the old
            # "carry one" refusal while the lead tried to read the four analysts' scans together.
            key = f"carried/{item['run_id'][:8]}_{target.name}"
        mounted[key] = target
    return mounted


def _run_wrote(item, name: str) -> bool:
    """Whether the carry item names a real run whose out/ holds this file.

    Panel round 14 (AUUNEMP): the planner carried the downloaded abs_sdmx.csv with the id of the run that first read it; that
    run wrote nothing, and the refusal repeated nine times. A downloaded file is mounted for every run, so it is mounted.
    """
    if not isinstance(item, dict) or not RUN_ID.match(str(item.get("run_id", ""))):
        return False
    base = (SANDBOX_ROOT / item["run_id"]).resolve()
    target = (base / name).resolve()
    return str(target).startswith(str((base / OUT_DIR).resolve()) + "/") and target.is_file()


CARRIED_FROM = "carried_from.json"   # beside in/ and out/, outside what the program sees


def _record_carried_from(run_id: str, carried: dict[str, Path]) -> None:
    """Which earlier run each carried input came from, so a series' sources can be traced back to the download that fed it
    (panel round 5: AUUNEMP and NEM built series from carried files and cited addresses that were never fetched)."""
    if not carried or not isinstance(run_id, str) or not RUN_ID.match(run_id):
        return
    origin = {name: path.parent.parent.name for name, path in carried.items()
              if path.parent.name == "out" and RUN_ID.match(path.parent.parent.name)}
    try:
        (SANDBOX_ROOT / run_id / CARRIED_FROM).write_text(json.dumps(origin, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def make_run(datasets: dict[str, Path] | None = None):
    inputs = dict(datasets or {})

    def _run(args: dict, _key: str) -> dict:
        extra = sorted(set(args) - {"code", "carry", "name"})
        if extra:
            raise ValueError(f"unknown arguments {extra}; allowed: code, carry, name")
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be a non-empty string")
        if len(code) > MAX_CODE_CHARS:
            raise ValueError(f"code is longer than {MAX_CODE_CHARS} characters")
        reason = invented_source_reason(code)
        if reason:
            raise ValueError(reason)
        # Round 4, A5-10: the idempotency key seeds the run id, so a re-run after a crash returns the finished run.
        run_id = _key[:32] if isinstance(_key, str) and len(_key) >= 32 else None
        carried = carried_inputs(args.get("carry"), inputs)
        res = run_python(code, timeout_s=60, mem_mb=1536, inputs={**inputs, **carried}, run_id=run_id)
        d = res.to_dict()
        _record_carried_from(d["run_id"], carried)
        out = {"ok": d["ok"], "returncode": d["returncode"], "timed_out": d["timed_out"],
               "stdout": d["stdout"], "stderr": d["stderr"], "run_id": d["run_id"],
               "outputs": d["outputs"], "memory_limit_enforced": d["memory_limit_enforced"]}
        renamed = {f"in/{k}": f"{v.parent.parent.name}/out/{v.name}" for k, v in carried.items() if k != f"carried/{v.name}"}
        if renamed:
            out["carried_as"] = renamed
        return out
    return _run


SHAPE_MAX_BYTES = 5_000_000   # a larger written file is named and sized, not read
SHAPE_KEYS = 12               # how many top-level keys a written JSON file names


def _kind(value) -> str:
    if isinstance(value, dict):
        return f"dict of {len(value)} keys"
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return f"list of {len(value)} dicts, each with keys {', '.join(list(value[0])[:8])}"
        return f"list of {len(value)}"
    return type(value).__name__


def json_shape(path: Path) -> str | None:
    """What a written JSON file holds at its top level, so the next cell is written against the real keys, not a guess.

    Live Beijing e14af4bd (2026-11-23): the planner read out/scan.json as if it had a per_entity key. Its top level is findings
    and summary, and three cells failed with KeyError before the keys were printed. The shape is given when the file is written.
    """
    try:
        if path.stat().st_size > SHAPE_MAX_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict):
        return _kind(value)
    keys = list(value)
    parts = [f"{k}: {_kind(value[k])}" for k in keys[:SHAPE_KEYS]]
    more = f"; and {len(keys) - SHAPE_KEYS} more keys" if len(keys) > SHAPE_KEYS else ""
    return "top-level keys: " + "; ".join(parts) + more


PREAMBLE_ERROR = re.compile(r"Error tokenizing data\. C error: Expected \d+ fields? in line \d+, saw \d+")


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
    if result["ok"] and not result["stdout"].strip() and not result["outputs"]:
        # Live Beijing 0137871e (2026-11-23): two silent cells with the same code gave the planner nothing to act on, and the
        # run stopped as a stall. Say so, and say what to do instead of running the cell again.
        out["note"] = ("the cell printed nothing and wrote no file, so it gave you nothing to act on: print the figures the next "
                       "decision needs, or save them under out/, in the next cell")
    shapes = {}
    for o in result["outputs"]:
        if str(o.get("path", "")).endswith(".json"):
            desc = json_shape(SANDBOX_ROOT / result["run_id"] / o["path"])
            if desc:
                shapes[o["path"]] = desc
    if shapes:
        out["shapes"] = shapes
    if result.get("carried_as"):
        out["carried_as"] = result["carried_as"]      # where a file with a repeated name was mounted
    if result["stderr"]:
        err = result["stderr"]
        if len(err) > SEEN_STDERR:
            # Round 6m: the head of a long error was cut with no marker, so the planner read a traceback that began mid-line.
            out["stderr_tail"] = f"[... {len(err) - SEEN_STDERR} characters cut ...]\n" + err[-SEEN_STDERR:]
        else:
            out["stderr_tail"] = err
        if PREAMBLE_ERROR.search(err):
            # Panel round 10 (NGSTORAGE): pd.read_csv on an EIA file with title lines above its table failed four times
            # ("Expected 1 fields in line 5, saw 14"), and the run stopped as a stall.
            out["hint"] = ("the file has lines above its table (a title or notes), so pd.read_csv sees too few fields. Read "
                           "it with df = pl.load(path), which finds the table block, or print its first 10 lines "
                           "and pass skiprows")
    hint = _error_hint(result)
    if hint and "hint" not in out:
        out["hint"] = hint
    if result["ok"] and NETWORK_BLOCKED in result["stdout"]:
        # VN Unemployment (2026-11-27): a cell caught the refusal, printed it and reported ok, so the planner read a success.
        out["blocked"] = True
        out["hint"] = NETWORK_HINT
    if result["timed_out"]:
        out["timed_out"] = True
    return out


NETWORK_BLOCKED = "network access is blocked in the sandbox"
NETWORK_HINT = ("the sandbox has no network. Fetch data in a planner step instead: download_open_data (a source key; "
                "search_open_data finds the codes) or download_web_data (an https address of a CSV, JSON, page or PDF), then "
                "read the file from in/ in the next cell")
FREQ_HINT = ("pandas 3 removed the old frequency codes: use 'YE' (year end) or 'YS' (year start), 'QE' or 'QS', 'ME' or "
             "'MS', 'W', 'D', 'h', 'min'. pl.freq_alias('A') gives the current code; pl.prepare_series accepts either")


def _inputs(run_id: str) -> list[str]:
    root = SANDBOX_ROOT / run_id / "in"
    if not root.is_dir():
        return []
    return sorted(str(p.relative_to(root.parent)) for p in root.rglob("*") if p.is_file())[:30]


def _error_hint(result: dict) -> str | None:
    """The next correct action for an error the analyst keeps repeating (VN Unemployment, 2026-11-27: 23 failed cells)."""
    err = result.get("stderr") or ""
    if not err:
        return None
    if NETWORK_BLOCKED in err or "No module named 'requests'" in err or "No module named 'httpx'" in err:
        return NETWORK_HINT
    if "Invalid frequency" in err or ("is no longer supported" in err and "freq" in err.lower()):
        return FREQ_HINT
    if "FileNotFoundError" in err and ("in/" in err or "carried" in err):
        files = _inputs(str(result.get("run_id", "")))
        listing = ", ".join(files) if files else "no input files"
        return (f"that file is not mounted. This run can read: {listing}. A file another run wrote reaches this one only "
                "through carry=[{run_id, file}]. A tool's result (web_fetch, web_research) is not a file: download the "
                "address with download_web_data, or save the figures from a cell that read a mounted file")
    return None


def sandbox_python_tool(datasets: dict[str, Path] | None = None) -> Tool:
    return Tool("sandbox_python", make_run(datasets), idempotent=True, observe=observe)


_SPEC_TEXT = (
    "Run Python (pandas, numpy, scipy, statsmodels, matplotlib) in a guarded sandbox: no network, no secrets, one "
    "minute. pandas only: `pl` is prophet_lib, not polars (null_count, is_null, sort work). The person's uploaded files are in in/ (read-only). Import: `import prophet_lib as pl`, `import prophet_stats "
    "as ps`, `import prophet_scan as sc`. "
    "Entities sharing dates are never summed silently: pass entity= and key=, or use pl.panel. "
    "pl.chart(kind, data, x=, y=, title=<the finding>, subtitle=<measure, unit, period>, unit=, source=, path='out/name.png') "
    "returns the problems it finds. pl.size_report(path) and pl.query(sql, t=path) read files too big to load. "
    "Also: pl.disaggregate(values, factor, kind='mean'|'sum') (Denton; published blocks kept), "
    "pl.profile(path), pl.as_number(s), pl.as_dates(s), pl.holiday_covariate(index, country, subdiv, horizon) "
    "and pl.event_covariate(index, "
    "[(first, last), ...], horizon) for known-ahead drivers, pl.detect_anomalies(values, season) (candidates "
    "to explain, not verdicts). "
    "Write files to out/. Every run starts with an EMPTY out/; to read an earlier run's file, pass "
    "carry=[{run_id, file: 'out/scan.json'}] and open it at in/carried/<file name>. "
    "Output you do not print, you do not see. A series written with pl.save_series or "
    "pl.save_panel can be forecast by passing source {run_id, file}. "
    "Signatures (other keywords are refused):\n"
)

SANDBOX_PYTHON_SPEC = {
    "name": "sandbox_python",
    "title": "Analysis",
    "activity": "Running an analysis",
    "description": _SPEC_TEXT + build_catalogue(),
    "args": {"code": "Python source, at most 20000 characters",
             "carry": "optional list of {run_id, file} outputs of earlier runs, mounted read-only at in/carried/<file name>",
             "name": "optional 1 to 4 words saying what this code does, such as 'Backtest seasonal baselines'"},
}
