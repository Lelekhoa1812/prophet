"""Planner tool: read the series of a chart on a web page, or of a chart image the person uploaded.

Motivation vs Logic
Motivation: the numbers a forecast needs are often only drawn on a page. Embedded chart data is exact and should be read
first. A chart that is only a picture is digitised, and the planner must be able to tell the two apart.
Logic: a page is fetched raw (chart_fetch, the guarded fetch). Series come from the page's own markup when it has any
(charts.py). Only when it has none are its chart images digitised (chart_vision, at most max_images). Each series is
landed as a CSV dataset in the mission's store with its method, exactness and source in the provenance, so the next
sandbox call can read it. An uploaded image is read the same way. Nothing here returns an estimate as an observation:
every series carries its exactness label.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from bs4 import BeautifulSoup

from prophet_agent.datasets import DatasetStore, clean_name
from prophet_agent.opendata.store import preview
from prophet_agent.research import chart_fetch, chart_vision, charts
from prophet_agent.runtime.runner import ArgumentRejected, Tool

MAX_IMAGES = 4
DEFAULT_IMAGES = 2
CHART_HINT = re.compile(r"chart|graph|plot|figure|fig[-_]|visuali|histogram", re.I)
IMAGE_EXT = (".png", ".jpg", ".jpeg")
PREVIEW_ROWS = 5


@dataclass(frozen=True)
class ChartContext:
    store: DatasetStore | None
    attach: object | None          # Callable[[str], None] | None, as in OpenDataContext


def _chart_image_urls(html: str, base: str, limit: int) -> list[str]:
    """Image addresses whose alt, title, class, id or file name says they are a chart. Never more than limit."""
    soup = BeautifulSoup(html, "html.parser")
    found: list[str] = []
    for img in soup.find_all("img"):
        src = str(img.get("src") or img.get("data-src") or "").strip()
        if not src or src.startswith("data:"):
            continue
        absolute = urljoin(base, src)
        parts = urlsplit(absolute)
        if parts.scheme not in ("http", "https") or not parts.path.lower().endswith(IMAGE_EXT):
            continue
        hint = " ".join([str(img.get("alt") or ""), str(img.get("title") or ""), " ".join(img.get("class") or []),
                         str(img.get("id") or ""), unquote(parts.path.rsplit("/", 1)[-1])])
        if CHART_HINT.search(hint) and absolute not in found:
            found.append(absolute)
        if len(found) >= limit:
            break
    return found


def _csv_bytes(series: dict) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["x", "y"])
    for x, y in zip(series["x"], series["y"]):
        writer.writerow(["" if x is None else x, "" if y is None else y])
    return buf.getvalue().encode("utf-8")


def _land(ctx: ChartContext, series: dict, provenance: dict) -> dict:
    """Stores one series as a CSV dataset and records it on the mission. Returns the summary the planner reads."""
    summary = {"name": series["name"], "method": series["method"], "exactness": series["exactness"],
               "x_kind": series.get("x_kind"), "n": series["n"], "source": series.get("source")}
    if series.get("error_bound_y") is not None:
        summary["error_bound_y"] = series["error_bound_y"]
    body = _csv_bytes(series)
    summary["first_rows"] = [{"x": x, "y": y} for x, y in zip(series["x"][:PREVIEW_ROWS], series["y"][:PREVIEW_ROWS])]
    if ctx.store is None:
        summary["landed"] = False
        return summary
    name = clean_name(f"chart-{series['name'][:40]}.csv")
    meta = ctx.store.add(name, body, provenance={**provenance, "sha256_series": hashlib.sha256(body).hexdigest(),
                                                  "series_name": series["name"], "method": series["method"],
                                                  "exactness": series["exactness"], "licence_key": "unverified",
                                                  "series_bound": False})
    if ctx.attach is not None:
        ctx.attach(meta["id"])
    summary.update({"landed": True, "dataset_id": meta["id"], "file": name,
                    "row_count": preview(body, "csv")["row_count"]})
    return summary


def _read_page(url: str) -> dict:
    fetched = chart_fetch.fetch_bytes(url, max_bytes=2_000_000, media=chart_fetch.HTML_MEDIA)
    if not fetched["ok"]:
        return {"ok": False, "note": fetched["note"]}
    html = fetched["body"].decode("utf-8", errors="replace")
    ex = charts.extract_page_series(html)
    out = {"ok": True, "url": url, "final_url": fetched["final_url"], "series": [], "skipped": list(ex.skipped),
           "images": []}
    if ex.series:
        out["series"] = ex.series
        return out
    for image_url in _chart_image_urls(html, fetched["final_url"], MAX_IMAGES):
        out["images"].append(image_url)
    return out


def _digitise_into(series_list: list, image: bytes, label: str, location: dict, skipped: list) -> None:
    """Digitises one image and adds its series. A refused image is recorded in skipped, never guessed."""
    try:
        result = chart_vision.digitise_image(image)
    except chart_vision.ChartReadError as exc:
        skipped.append(f"{label}: {exc}")
        return
    for s in result["series"]:
        series_list.append({**s, "source": {**location, "image_sha256": result["image_sha256"]}})


def _run_page(args: dict, ctx: ChartContext, digitise_images: int) -> dict:
    url = args["url"].strip()
    page = _read_page(url)
    if not page["ok"]:
        return {"ok": False, "url": url, "note": page["note"]}
    found = [dict(s, source={**s["source"], "url": page["final_url"]}) for s in page["series"]]
    skipped = list(page["skipped"])
    if not found:
        for image_url in page["images"][:digitise_images]:
            try:
                image = chart_vision.fetch_image(image_url)
            except chart_vision.ChartReadError as exc:
                skipped.append(f"{image_url}: {exc}")
                continue
            _digitise_into(found, image, image_url, {"kind": "image", "url": image_url}, skipped)
    return {"ok": True, "url": url, "final_url": page["final_url"], "found": found, "skipped": skipped,
            "chart_images_seen": page["images"]}


def _run_image(args: dict, datasets: dict[str, Path]) -> dict:
    name = args["image"].strip()
    base = name[3:] if name.startswith("in/") else name
    path = datasets.get(base)
    if path is None:
        raise ArgumentRejected(f"no mission file named {base!r}; the files are {', '.join(sorted(datasets)) or 'none'}")
    skipped: list[str] = []
    found: list[dict] = []
    _digitise_into(found, path.read_bytes(), base, {"kind": "upload", "file": base}, skipped)
    return {"ok": True, "image": base, "found": found, "skipped": skipped}


def chart_tool_pair(datasets: dict[str, Path] | None, context) -> dict[str, tuple[Tool, dict]]:
    """The extract_page_charts tool, as the (tool, spec) pair default_tools expects. ``context`` is the mission's
    OpenDataContext (store and attach), or None outside a mission; the tool then reads but does not land."""
    ctx = ChartContext(store=getattr(context, "store", None), attach=getattr(context, "attach", None))
    files = dict(datasets or {})

    def run(args: dict, _key: str) -> dict:
        if not isinstance(args, dict):
            raise ArgumentRejected("the arguments must be an object")
        allowed = {"url", "image", "series", "max_images"}
        extra = sorted(set(args) - allowed)
        if extra:
            raise ArgumentRejected(f"unknown arguments {extra}; allowed: {', '.join(sorted(allowed))}")
        url, image = args.get("url"), args.get("image")
        if bool(url) == bool(image):
            raise ArgumentRejected("give exactly one of url (a page) or image (a mission file)")
        series_name = args.get("series")
        if series_name is not None and not isinstance(series_name, str):
            raise ArgumentRejected("series must be text")
        max_images = args.get("max_images", DEFAULT_IMAGES)
        if not isinstance(max_images, int) or isinstance(max_images, bool) or not 1 <= max_images <= MAX_IMAGES:
            raise ArgumentRejected(f"max_images must be a whole number from 1 to {MAX_IMAGES}")
        if url is not None:
            if not isinstance(url, str) or urlsplit(url.strip()).scheme.lower() not in ("http", "https"):
                raise ArgumentRejected("url must be an http or https address")
            result = _run_page({"url": url}, ctx, max_images)
        else:
            if not isinstance(image, str):
                raise ArgumentRejected("image must be a mission file name")
            result = _run_image({"image": image}, files)
        if not result.get("ok", True):
            return result
        found = charts.filter_series(result["found"], series_name)
        landed = []
        provenance = {"source": "chart", "url": result.get("final_url") or result.get("image"),
                      "retrieved_from": "page" if url else "upload"}
        for s in found:
            landed.append(_land(ctx, s, {**provenance, "source_location": s.get("source")}))
        note = ("Each series is a dataset. exactness embedded or svg-geometry means the values are the page's data or "
                "inverted from its SVG geometry; read_from_image means digitised from a picture, with error_bound_y. "
                "Read a landed file in sandbox_python only when it is mounted for this run.")
        return {"ok": True, "series": landed, "skipped": result.get("skipped", []),
                "chart_images_seen": result.get("chart_images_seen", []), "note": note}

    return {"extract_page_charts": (Tool("extract_page_charts", run, idempotent=True), EXTRACT_PAGE_CHARTS_SPEC)}


EXTRACT_PAGE_CHARTS_SPEC = {
    "name": "extract_page_charts",
    "title": "Chart data",
    "activity": "Reading chart data",
    "description": "Read the series of a chart. With url, read a page: its embedded chart data (Highcharts, Plotly, Chart.js, "
                   "ECharts, Vega-Lite) or its inline SVG line charts, exact where the page carries the numbers. Only when the "
                   "page has none, digitise up to max_images chart pictures with a vision model: the code measures the "
                   "pixels, and those values are labelled read_from_image with an error bound. With image, digitise a chart "
                   "picture the person uploaded (a mission file). Each series lands as a dataset with its exactness.",
    "args": {"url": "an http or https address of a page with a chart (give url or image, not both)",
             "image": "a mission file name of a PNG or JPEG chart (give url or image, not both)",
             "series": "optional; keep only series whose name contains this text",
             "max_images": f"optional; how many chart pictures to digitise when the page has no data, 1 to {MAX_IMAGES} (default {DEFAULT_IMAGES})"},
}
