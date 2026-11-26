"""Where a tool's series comes from: numbers in the call, or a file an earlier analysis step wrote.

Motivation vs Logic
Motivation: a real series can have thousands of points. The planner should not have to copy them into a tool call, and a
copy can silently drop or alter values. Analysis code in the sandbox cleans and shapes the data, then hands the result
over by reference.
Logic: ``values`` is a list of numbers. ``source`` names a sandbox run and a file in its ``out/`` folder (written with
``prophet_lib.save_series``), and optionally one series inside it. The run id must be a hex id and the file must stay
inside ``out/``. Anything that is not a finite number is refused with a message that says what to fix.
Provenance: a series file may carry a ``provenance`` block (origin retrieved, user or synthetic). ``resolve_series`` refuses a
retrieved claim with no research fetch record, and a synthetic series that names a source. Motivation: SciRes run (2026-11-24)
generated numbers with numpy and labelled them as NOAA data.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlsplit
from typing import Any

import numpy as np

from prophet_agent.sandbox.runner import OUT_DIR, SANDBOX_ROOT

MAX_SERIES = 100_000
MAX_NAMED = 100
RUN_ID = re.compile(r"^[0-9a-f]{32}$")

VALUES_HELP = "list of numbers, oldest first (give this or source, not both)"
SOURCE_HELP = ("object {run_id: id of the analysis run that wrote the file, file: 'out/series.json', name: optional "
               "series name} to read the series from that run instead of values")


def _clean(name: str, raw: Any) -> np.ndarray:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"series {name!r} must be a non-empty list of numbers")
    if len(raw) > MAX_SERIES:
        raise ValueError(f"series {name!r} has {len(raw)} points; the limit is {MAX_SERIES}")
    # Round 6m: numpy turned a nested list into a two-dimensional array and text such as "1" into a number without a word.
    # Both are refused here, with the form the values must take.
    if any(isinstance(v, (list, tuple, dict)) for v in raw):
        raise ValueError(f"series {name!r} must be one flat list of numbers, not a list of lists or objects")
    if any(isinstance(v, (str, bool)) for v in raw):
        raise ValueError(f"series {name!r} contains values that are not numbers (text or true/false); give numbers only")
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"series {name!r} contains values that are not numbers")
    bad = int((~np.isfinite(arr)).sum())
    if bad:
        raise ValueError(f"series {name!r} has {bad} missing or non-finite values; fill or drop them in the analysis "
                         "step and save again, so the choice is explicit")
    return arr


def missing_output(run_id: str, rel: str) -> str:
    """Why a named output is not there, with what would fix it: the run does not exist (an id that no tool returned), or
    it exists and these are the files it did write.

    Live electricity rep4 (2026-11-23): three calls named files their runs never wrote, one with a run id no tool had
    returned. "no file" alone left the planner guessing which run held the file.
    """
    base = SANDBOX_ROOT / run_id
    if not base.is_dir():
        return (f"no run {run_id} exists. A run id is the run_id a sandbox_python result returned; copy it from that "
                "result, or re-run the step that writes the file")
    out = base / OUT_DIR
    wrote = sorted(str(f.relative_to(base)) for f in out.rglob("*") if f.is_file())[:12] if out.is_dir() else []
    return (f"no file {rel!r} in the outputs of run {run_id}; that run wrote "
            + (", ".join(wrote) if wrote else "no files") + ". Name one of those, or the run that saved this file")


def _out_file(run_id: str, rel: str) -> Path:
    """The resolved path of ``rel`` inside the run's ``out/`` folder. Raises ValueError when it lies elsewhere. Every reader of
    a source file goes through here: an absolute or ``../`` name used to read any file on disk (round 6l, audit item 6)."""
    base = (SANDBOX_ROOT / run_id).resolve()
    target = (base / rel).resolve()
    if not str(target).startswith(str((base / OUT_DIR).resolve()) + "/"):
        raise ValueError(missing_output(run_id, rel))
    return target


def source_path(args: dict) -> Path:
    """The file a ``source`` argument names, after the same checks as resolve_series: a hex run id and a file inside the
    run's ``out/`` folder. Raises ValueError with the fix when either check fails."""
    source = args.get("source")
    if not isinstance(source, dict) or not RUN_ID.match(str(source.get("run_id", ""))):
        raise ValueError("source needs run_id (the 32-character id of the analysis run that wrote the file)")
    rel = str(source.get("file", f"{OUT_DIR}/series.json"))
    target = _out_file(source["run_id"], rel)
    if not target.is_file():
        raise ValueError(missing_output(source["run_id"], rel))
    return target


PROVENANCE_ORIGINS = ("retrieved", "user", "synthetic")
RETRIEVED_SYNONYMS = frozenset({"market", "download", "downloaded", "official", "api", "open data", "open_data", "web",
                                "fetched", "measured", "observed", "primary"})
SOURCE_ID = re.compile(r"^S\d+$")          # the id web research gives each source it reads: S1, S2, ...
HTTP_URL = re.compile(r"^https?://\S+$")


def _fetch_record_exists(url: str) -> bool:
    """True when a fetch record with ok set exists for this URL: the entry research/fetch.py writes after it reads a page, or
    the one opendata/fetch.py writes after a download lands. A URL nothing fetched has no record, so a claim to it is refused."""
    from prophet_agent.opendata.fetch import download_record_path, endpoint_of
    from prophet_agent.research.search import cache_key, cache_root
    for path in (cache_root() / "pages" / f"{cache_key('page', url)}.json", download_record_path(url),
                 download_record_path(endpoint_of(url))):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("ok") is True:
            return True               # a page research read, or a file a download tool landed (opendata/fetch.py)
    return _stored_download(url)


def _stored_download(url: str, datasets_root: Path | None = None) -> bool:
    """True when a stored dataset was downloaded from this address (its provenance url or final_url).

    Round 6 (GISTEMP): the .txt file was landed before download records existed, and download_web_data reused the stored
    copy without fetching again, so no record was ever written; backtest_series refused the series four times.
    """
    root = datasets_root or _datasets_root()
    if not root.is_dir():
        return False
    want = url.strip()
    for i, folder in enumerate(root.iterdir()):
        if i >= MAX_DATASET_RECORDS:
            break
        try:
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        prov = meta.get("provenance") if isinstance(meta, dict) else None
        if isinstance(prov, dict) and want in (prov.get("url"), prov.get("final_url")) and meta.get("sha256"):
            return True
    return False


def check_provenance(raw: Any, name: str = "series") -> dict | None:
    """The provenance block a series file or call carries, checked. None when nothing is recorded. Raises ValueError with the fix.

    origin "retrieved": numbers copied from pages the research read. It names the source (source_name) and each page (sources:
    [{id: 'S2', url}]), and each URL must have a research fetch record. origin "user": the person's own file. origin
    "synthetic": generated for a test or a stand-in; it can never name a source, so it can never be read as one.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"provenance of {name!r} must be an object with origin, source_name and sources")
    origin = raw.get("origin")
    # Panel round 4 (NVDA): origin "market" for a download_market_prices file was refused five times. Words that mean "fetched
    # from a source" are read as retrieved, which still needs source_name and fetch-recorded urls below.
    if isinstance(origin, str) and origin.strip().lower() in RETRIEVED_SYNONYMS:
        origin = "retrieved"
    if origin not in PROVENANCE_ORIGINS:
        raise ValueError(f"provenance of {name!r} needs origin: one of {', '.join(PROVENANCE_ORIGINS)}")
    source_name = raw.get("source_name")
    if source_name is not None and (not isinstance(source_name, str) or not source_name.strip()):
        raise ValueError(f"provenance of {name!r}: source_name is the source's name, as text")
    name_text = source_name.strip() if isinstance(source_name, str) else ""
    if origin == "synthetic":
        if name_text:
            raise ValueError(f"series {name!r} is synthetic, so it cannot be labelled with the source {name_text!r}. Label it "
                             "synthetic, or retrieve the source with web research and record its id and URL")
        return {"origin": "synthetic"}
    if origin == "user":
        return {"origin": "user", **({"source_name": name_text} if name_text else {})}
    sources = raw.get("sources")
    if not name_text or not isinstance(sources, list) or not sources:
        raise ValueError(f"series {name!r} claims a retrieved source, so it needs source_name and sources: a list of "
                         "{id: 'S<n>', url} taken from the research record")
    clean = []
    for src in sources:
        if not isinstance(src, dict):
            raise ValueError(f"each source of {name!r} is an object with id and url")
        sid, url = src.get("id"), src.get("url")
        if not isinstance(sid, str) or not SOURCE_ID.match(sid):
            raise ValueError(f"a source of {name!r} needs id: the research id, such as 'S2'")
        if not isinstance(url, str) or not HTTP_URL.match(url.strip()):
            raise ValueError(f"source {sid} of {name!r} needs a url that starts with http:// or https://")
        if not _fetch_record_exists(url.strip()):
            raise ValueError(f"source {sid} ({url.strip()}) has no fetch record, so the research did not read it. Fetch the page "
                             "with web research or download it with download_web_data first, or label the series synthetic or user-supplied")
        clean.append({"id": sid, "url": url.strip()})
    return {"origin": "retrieved", "source_name": name_text, "sources": clean}


def traced_provenance(raw: Any, rel: str, run_id: str) -> dict | None:
    """The checked provenance of a series file. A retrieved claim whose cited sources fail the check, from a series traced
    (through carried files) to downloads of this mission, is kept with those downloads as its sources.

    Motivation: panel rounds 4 to 6 (AUUNEMP, NEM, GISTEMP), the series came from download_open_data files, but the cell
    cited a page address, a made-up host, or a dataset id instead of 'S1', and backtest_series refused it until the run
    stalled; the hint naming the right address (round 4) was not always followed. Logic: the downloads are known from the
    sha256 of the run's inputs and each has a download record, so the claim is supported as well as a correct citation
    would be. Synthetic and user origins pass or fail as before; with nothing traced the refusal stands, naming the fix.
    """
    try:
        return check_provenance(raw, rel)
    except ValueError as exc:
        urls = input_source_urls(run_id)
        origin = raw.get("origin") if isinstance(raw, dict) else None
        claimed = isinstance(origin, str) and (origin == "retrieved" or origin.strip().lower() in RETRIEVED_SYNONYMS)
        if not urls or not claimed:
            if urls:
                listed = "; ".join(f"{name} came from {url}" for name, url in sorted(urls.items())[:6])
                raise ValueError(f"{exc}. The files behind this series were downloaded in this mission: {listed}. Use "
                                 "that address as the source url with origin retrieved, and save the series again") from None
            raise
        name = raw.get("source_name") if isinstance(raw.get("source_name"), str) and raw["source_name"].strip() else None
        sources = [{"id": f"S{i}", "url": url} for i, url in enumerate(sorted(set(urls.values()))[:6], start=1)]
        return {"origin": "retrieved", "source_name": (name or urlsplit(sources[0]["url"]).hostname or "download").strip(),
                "sources": sources, "traced": True}


def series_provenance(args: dict) -> dict | None:
    """The checked provenance of the series a call names: the block in the source file, or one passed with values. Raises
    ValueError when the block is inconsistent (see check_provenance)."""
    source = args.get("source")
    if isinstance(source, dict) and RUN_ID.match(str(source.get("run_id", ""))):
        rel = str(source.get("file", f"{OUT_DIR}/series.json"))
        target = source_path(args)                        # a missing file or run raises, with the fix
        try:
            doc = json.loads(target.read_text(encoding="utf-8"))
        except ValueError:
            return None
        return traced_provenance(doc.get("provenance") if isinstance(doc, dict) else None, rel, source["run_id"])
    return check_provenance(args.get("provenance"), "series")


def provenance_label(prov: dict | None) -> str:
    """How an answer names where a series came from: the named source with its research ids and URLs, "user-supplied",
    "synthetic (generated, not observed)", or "source not recorded". Only a retrieved origin ever carries a source name."""
    if not prov or prov.get("origin") not in PROVENANCE_ORIGINS:
        return "source not recorded"
    if prov["origin"] == "synthetic":
        return "synthetic (generated, not observed)"
    if prov["origin"] == "user":
        return "user-supplied" + (f" ({prov['source_name']})" if prov.get("source_name") else "")
    refs = "; ".join(f"{s['id']} {s['url']}" for s in prov["sources"])
    return f"{prov['source_name']} ({refs})"


DATASETS_ROOT: Path | None = None   # None: the datasets folder beside the sandbox root in use
MAX_DATASET_RECORDS = 2000


def _datasets_root() -> Path:
    """The dataset store beside the sandbox root in use, read at call time so a test that moves one moves both."""
    return DATASETS_ROOT if DATASETS_ROOT is not None else SANDBOX_ROOT.parent / "datasets"


def input_source_urls(run_id: str, datasets_root: Path | None = None) -> dict[str, str]:
    """The download address of each file the run read from in/ (not carried outputs), matched to the dataset records by sha256.

    Motivation: panel round 4 (AUUNEMP, 2026-11-27), a series built from an abs_sdmx download named its source as "ABS SDMX flow
    ABS,LF,1.0.0 key ..." and backtest_series refused it six times; the planner never had the address to copy. The refusal now
    names it. Logic: hash each file in the run's in/ folder and look it up among the dataset records' provenance.
    """
    import hashlib
    root = datasets_root or _datasets_root()
    if not RUN_ID.match(run_id) or not root.is_dir():
        return {}
    # Round 5 (AUUNEMP, NEM): the series was built from files carried from an earlier run, whose own in/ held the download.
    # The sandbox tool records where each carried file came from (carried_from.json); follow that chain a few runs back.
    wanted: dict[str, str] = {}
    queue, seen = [run_id], set()
    while queue and len(seen) < 12:
        rid = queue.pop(0)
        if rid in seen or not RUN_ID.match(rid):
            continue
        seen.add(rid)
        inbox = SANDBOX_ROOT / rid / "in"
        if inbox.is_dir():
            for f in inbox.iterdir():
                if f.is_file() and not f.is_symlink():
                    try:
                        wanted.setdefault(hashlib.sha256(f.read_bytes()).hexdigest(), f.name)
                    except OSError:
                        continue
        try:
            origin = json.loads((SANDBOX_ROOT / rid / "carried_from.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            origin = {}
        if isinstance(origin, dict):
            queue.extend(str(v) for v in origin.values())
    if not wanted:
        return {}
    out: dict[str, str] = {}
    for i, folder in enumerate(root.iterdir()):
        if i >= MAX_DATASET_RECORDS or not wanted:
            break
        try:
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        prov = meta.get("provenance") if isinstance(meta, dict) else None
        url = prov.get("url") if isinstance(prov, dict) else None
        name = wanted.get(str(meta.get("sha256")))
        if name and isinstance(url, str) and HTTP_URL.match(url):
            out[name] = url
    return out


def resolve_series(args: dict) -> list[tuple[str, np.ndarray]]:
    values, source = args.get("values"), args.get("source")
    if (values is None) == (source is None):
        raise ValueError("give exactly one of values or source")
    if values is not None:
        return [("series", _clean("series", values))]
    if not isinstance(source, dict) or not RUN_ID.match(str(source.get("run_id", ""))):
        raise ValueError("source needs run_id (the 32-character id of the analysis run that wrote the file)")
    rel = str(source.get("file", f"{OUT_DIR}/series.json"))
    target = _out_file(source["run_id"], rel)
    if not target.is_file():
        raise ValueError(missing_output(source["run_id"], rel))
    try:
        doc = json.loads(target.read_text(encoding="utf-8"))
    except ValueError:
        raise ValueError(f"{rel} is not valid JSON; write it with prophet_lib.save_series")
    if isinstance(doc, dict):
        traced_provenance(doc.get("provenance"), rel, source["run_id"])   # refuses an unsupported claim, with the fix
    body = doc.get("series") if isinstance(doc, dict) else doc
    if isinstance(body, list):
        named = {"series": body}
    elif isinstance(body, dict):
        named = body
    else:
        raise ValueError(f"{rel} must hold a list of numbers or a dict of named lists")
    want = source.get("name")
    if want is not None and not isinstance(want, (str, list)):
        # A dict is unhashable and raised a bare TypeError in the lookup below; a number is not a series name either (round 6m).
        raise ValueError("source.name is one series name (text), or a list of series names")
    if isinstance(want, list):
        # A subset of a panel (live electricity rep5: a list of ten names failed with "unhashable type: 'list'", three times).
        if not want or not all(isinstance(n, str) for n in want):
            raise ValueError("source.name is one series name, or a list of series names")
        absent = [n for n in want if n not in named]
        if absent:
            raise ValueError(f"series {absent[:10]} not in the file; available: {sorted(named)[:20]}")
        named, want = {n: named[n] for n in dict.fromkeys(want)}, None
    if want is not None:
        if want not in named and len(named) == 1:
            want = next(iter(named))          # one series in the file: no ambiguity, so a wrong name is not an error
        if want not in named:
            raise ValueError(f"series {want!r} not in the file; available: {sorted(named)[:20]}")
        named = {want: named[want]}
    if len(named) > MAX_NAMED:
        raise ValueError(f"{len(named)} series in the file; the limit per call is {MAX_NAMED}. Pick some with source.name")
    return [(str(k), _clean(str(k), v)) for k, v in named.items()]


def source_covariates(args: dict) -> dict[str, Any]:
    """Named covariates stored with a series file (``covariates`` key written by ``prophet_lib.save_series``). Empty when
    there is no source or the file holds none; lengths are checked later against the horizon."""
    source = args.get("source")
    if not isinstance(source, dict) or not RUN_ID.match(str(source.get("run_id", ""))):
        return {}
    try:
        target = _out_file(source["run_id"], str(source.get("file", f"{OUT_DIR}/series.json")))
        doc = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    cov = doc.get("covariates") if isinstance(doc, dict) else None
    return cov if isinstance(cov, dict) else {}


def run_dates(args: dict, name: str) -> list[str] | None:
    """Dates saved next to a series, if any, for chart axes."""
    source = args.get("source")
    if not isinstance(source, dict) or not RUN_ID.match(str(source.get("run_id", ""))):
        return None
    try:
        target = _out_file(source["run_id"], str(source.get("file", f"{OUT_DIR}/series.json")))
        doc = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    d = doc.get("dates") if isinstance(doc, dict) else None
    if isinstance(d, dict):
        d = d.get(name)
    return d if isinstance(d, list) else None
