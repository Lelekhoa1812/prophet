"""Forecast deliverables (PNG, SVG, XLSX, DOCX, PDF) produced inside the sandbox, with a sha256 manifest.

Motivation vs Logic
Motivation: a forecast is shared as a chart, a spreadsheet and a report. Those files must come from the same numbers,
and the reader must be able to check that the files have not changed since they were produced.
Logic: the numbers travel as a JSON input file, never as code, so the template is fixed and nothing the model wrote is
executed outside the sandbox. The template runs in the sandbox, writes into ``out/``, and the manifest records the hash
of every file. ``verify_manifest`` re-hashes them later.
"""
from __future__ import annotations

import json
from typing import Any

from prophet_agent.sandbox.runner import SandboxResult, run_python

REQUIRED_KINDS = ("png", "svg", "xlsx", "docx", "pdf")
MAX_PANEL = 50           # series per panel pack; more than this is refused, not silently cut
PANEL_PER_FIGURE = 12    # panels per grid figure: forecast_grid.png, then forecast_grid_2.png, and so on

_TEMPLATE = r'''
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Times-style serif to match the console; STIXGeneral ships with matplotlib, so it is always present.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

d = json.load(open("data/input.json", encoding="utf-8"))
hist, fc, lo, hi = d["history"], d["point"], d["q10"], d["q90"]
lo2, hi2 = d.get("q30"), d.get("q70")
title, note = d["title"], d["note"]
n, h = len(hist), len(fc)
from datetime import datetime, timedelta
dates = d.get("dates") and d.get("future_dates")
if dates:
    xs = [datetime.fromisoformat(t) for t in d["dates"] + d["future_dates"]]
    step = timedelta(seconds=float(sorted(abs((b - a).total_seconds()) for a, b in zip(xs, xs[1:]))[len(xs) // 2])) if len(xs) > 1 else timedelta(days=1)
else:
    xs, step = list(range(n + h)), 1
x, fx = xs[:n], xs[n:]

# Okabe-Ito palette: distinguishable with every common kind of colour blindness, and in greyscale print.
INK, FORECAST, EVENT_COLOURS = "#2b2b2b", "#D55E00", ["#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#E69F00", "#F0E442"]
view = max(3 * h, 60)                      # a forecast of 28 points beside 700 points of history would be a sliver
start = max(0, n - view)
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
for i, ev in enumerate(d.get("events", [])):
    colour = EVENT_COLOURS[i % len(EVENT_COLOURS)]
    first = True
    for s0, e0 in ev["spans"]:
        if e0 <= start:
            continue
        ax.axvspan(xs[max(s0, start)] - step / 2, xs[min(e0, n + h) - 1] + step / 2, color=colour, alpha=0.18, lw=0,
                   label=ev["name"] if first else None)
        first = False
ax.plot(x[start:], hist[start:], label="Your data", color=INK, linewidth=1.4)
ign = [i for s0, e0 in d.get("ignored", []) for i in range(max(s0, start), min(e0, n))]
if ign:
    ax.plot([x[i] for i in ign], [hist[i] for i in ign], "x", color="#767676", markersize=6, label="Ignored by the model")
ax.fill_between(fx, lo, hi, color=FORECAST, alpha=0.16, lw=0, label="Likely range (8 in 10)")
if lo2 and hi2:
    ax.fill_between(fx, lo2, hi2, color=FORECAST, alpha=0.28, lw=0, label="Middle range (4 in 10)")
ax.plot([x[-1]] + fx if x else fx, [hist[-1]] + fc if x else fc, label="Forecast", color=FORECAST, linewidth=2.4)
if x:
    ax.axvline(x[-1], color="#767676", linewidth=0.9, linestyle=(0, (4, 3)))
ax.set_title(title, fontsize=14, loc="left")
if dates:
    import matplotlib.dates as mdates
    loc = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
else:
    ax.set_xlabel("Period")
ax.set_ylabel(d.get("unit") or "")
if start > 0:
    fig.text(0.01, 0.01, f"Showing the last {n - start} of {n} points", fontsize=8, color="#767676", va="bottom")
ax.grid(axis="y", color="#e5dccb", linewidth=0.8)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=9, ncol=min(len(labels), 4))
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("out/forecast.png")
fig.savefig("out/forecast.svg")
plt.close(fig)

import openpyxl
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Forecast"
events = d.get("events", [])
flag = lambda ev, i: int(any(s0 <= n + i < e0 for s0, e0 in ev["spans"]))
head = (["Date"] if dates else []) + ["Periods ahead", "Forecast", "Low (P10)", "High (P90)"] + [ev["name"] for ev in events]
ws.append(head)
for k in range(len(head)):
    ws.column_dimensions[openpyxl.utils.get_column_letter(k + 1)].width = 16
for i, (p, a, b) in enumerate(zip(fc, lo, hi)):
    ws.append(([d["future_dates"][i][:10]] if dates else []) + [i + 1, p, a, b] + [flag(ev, i) for ev in events])
ws.freeze_panes = "A2"
wb.save("out/forecast.xlsx")
# The same numbers as JSON, for a later analysis cell (live retail: cells carried out/forecast.json from the pack run and
# failed five times because only the workbook existed).
json.dump({"dates": d.get("future_dates"), "point": fc, "q10": lo, "q90": hi, "title": title}, open("out/forecast.json", "w"))

from docx import Document
from docx.shared import Inches
doc = Document()
doc.add_heading(title, level=1)
if note:
    doc.add_paragraph(note)
doc.add_picture("out/forecast.png", width=Inches(6))
doc.save("out/report.docx")

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
c = canvas.Canvas("out/report.pdf", pagesize=A4)
c.setFont("Times-Bold", 16)
c.drawString(72, 780, title[:90])
import textwrap
c.setFont("Times-Roman", 10)
for k, line in enumerate(textwrap.wrap(note, 105)[:6]):
    c.drawString(72, 764 - 13 * k, line)
c.drawImage("out/forecast.png", 72, 380, width=460, preserveAspectRatio=True)
c.save()
print("built", sorted(__import__("os").listdir("out")))
'''


def build_forecast_pack(*, history: list[float], point: list[float], q10: list[float], q90: list[float],
                        title: str, note: str = "", timeout_s: int = 90, dates: list[str] | None = None,
                        future_dates: list[str] | None = None, events: list[dict] | None = None,
                        ignored: list[list[int]] | None = None, q30: list[float] | None = None,
                        q70: list[float] | None = None, unit: str = "") -> SandboxResult:
    if not (len(point) == len(q10) == len(q90) > 0):
        raise ValueError("point, q10 and q90 must be non-empty and the same length")
    payload: dict[str, Any] = {"history": [float(v) for v in history], "point": [float(v) for v in point],
                               "q10": [float(v) for v in q10], "q90": [float(v) for v in q90],
                               "title": title, "note": note, "unit": unit[:40]}
    if dates and future_dates and len(dates) == len(history) and len(future_dates) == len(point):
        payload.update(dates=list(dates), future_dates=list(future_dates))
    if events:
        payload["events"] = events
    if ignored:
        payload["ignored"] = [[int(a), int(b)] for a, b in ignored]
    if q30 and q70 and len(q30) == len(q70) == len(point):
        payload.update(q30=[float(v) for v in q30], q70=[float(v) for v in q70])
    data = json.dumps(payload).encode("utf-8")
    return run_python(_TEMPLATE, timeout_s=timeout_s, mem_mb=2048, file_mb=64, files={"data/input.json": data})


_PANEL_TEMPLATE = r'''
import json, math, os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

d = json.load(open("data/input.json", encoding="utf-8"))
series, title, note, unit = d["series"], d["title"], d["note"], d.get("unit") or ""
INK, FORECAST = "#2b2b2b", "#D55E00"            # ink for the history, Okabe-Ito orange for the forecast
PER_FIGURE, COLS = d["per_figure"], 3

def axes_x(s):
    n, h = len(s["history"]), len(s["point"])
    if "dates" in s and "future_dates" in s:
        xs = [datetime.fromisoformat(t) for t in s["dates"] + s["future_dates"]]
        return xs[:n], xs[n:], True
    return list(range(n)), list(range(n, n + h)), False

def draw(ax, s, left):
    hist, pt, lo, hi = s["history"], s["point"], s["q10"], s["q90"]
    x, fx, dated = axes_x(s)
    ax.fill_between(fx, lo, hi, color=FORECAST, alpha=0.16, lw=0, label="Likely range (8 in 10)")
    ax.plot(x, hist, color=INK, linewidth=1.2, label="Your data")
    ax.plot([x[-1]] + fx, [hist[-1]] + pt, color=FORECAST, linewidth=2.0, label="Forecast")
    ax.axvline(x[-1], color="#767676", linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_title(s["name"], fontsize=10, loc="left")
    ax.grid(axis="y", color="#e5dccb", linewidth=0.7)
    ax.tick_params(labelsize=7)
    if dated:
        loc = mdates.AutoDateLocator(minticks=3, maxticks=5)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    if left and unit:
        ax.set_ylabel(unit, fontsize=8)
    if s.get("centre_note"):
        ax.text(0.01, 0.02, s["centre_note"], transform=ax.transAxes, fontsize=6.5, color="#767676", va="bottom")

os.makedirs("out", exist_ok=True)
stems = []
for k in range(0, len(series), PER_FIGURE):
    chunk = series[k:k + PER_FIGURE]
    rows = math.ceil(len(chunk) / COLS)
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 4.4, rows * 3.3), dpi=150, squeeze=False)
    flat = list(axes.ravel())
    for i, (ax, s) in enumerate(zip(flat, chunk)):
        draw(ax, s, i % COLS == 0)
    for ax in flat[len(chunk):]:
        ax.set_visible(False)
    handles, labels = flat[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=13, x=0.01, ha="left")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    stem = "forecast_grid" if not stems else f"forecast_grid_{len(stems) + 1}"
    fig.savefig(f"out/{stem}.png")
    fig.savefig(f"out/{stem}.svg")
    plt.close(fig)
    stems.append(stem)

import openpyxl
from openpyxl.utils import get_column_letter
wb = openpyxl.Workbook()
summary = wb.active
summary.title = "Summary"
summary.append(["Series", "Last value", "Forecast at step 1", "Forecast at last step", "Low (P10) at last step",
                "High (P90) at last step"])
for s in series:
    summary.append([s["name"], s["history"][-1], s["point"][0], s["point"][-1], s["q10"][-1], s["q90"][-1]])
long = wb.create_sheet("Forecast")
long.append(["Series", "Date", "Periods ahead", "Forecast", "Low (P10)", "High (P90)"])
for s in series:
    fdates = s.get("future_dates")
    for i, (p, a, b) in enumerate(zip(s["point"], s["q10"], s["q90"])):
        long.append([s["name"], fdates[i] if fdates else None, i + 1, p, a, b])
for ws, width in ((summary, 24), (long, 18)):
    ws.freeze_panes = "A2"
    for col in range(1, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(col)].width = width
wb.save("out/forecast.xlsx")
json.dump({"series": {s["name"]: {"dates": s.get("future_dates"), "point": s["point"], "q10": s["q10"], "q90": s["q90"]}
                      for s in series}, "title": title}, open("out/forecast.json", "w"))

from docx import Document
from docx.shared import Inches
doc = Document()
doc.add_heading(title, level=1)
if note:
    doc.add_paragraph(note)
for stem in stems:
    doc.add_picture(f"out/{stem}.png", width=Inches(6.5))
doc.add_heading("Summary", level=2)
heads = ["Series", "Last value", "Step 1", "Last step", "P10, last step", "P90, last step"]
table = doc.add_table(rows=1, cols=len(heads))
table.style = "Table Grid"
for cell, text in zip(table.rows[0].cells, heads):
    cell.text = text
for s in series:
    vals = [s["name"], s["history"][-1], s["point"][0], s["point"][-1], s["q10"][-1], s["q90"][-1]]
    for cell, v in zip(table.add_row().cells, vals):
        cell.text = v if isinstance(v, str) else f"{v:,.2f}"
centre_notes = [(s["name"], s["centre_note"]) for s in series if s.get("centre_note")]
if centre_notes:
    doc.add_heading("Centre lines", level=2)
    for name, text in centre_notes:
        doc.add_paragraph(f"{name}: {text}")
doc.save("out/report.docx")

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
import textwrap
c = canvas.Canvas("out/report.pdf", pagesize=A4)
c.setFont("Times-Bold", 16)
c.drawString(72, 780, title[:90])
c.setFont("Times-Roman", 10)
lines = textwrap.wrap(note, 105)[:6]
if len(series) > PER_FIGURE:
    lines.append(f"First {PER_FIGURE} of {len(series)} series shown; the other panels are in the chart files and the workbook.")
for k, line in enumerate(lines):
    c.drawString(72, 764 - 13 * k, line)
top = 764 - 13 * len(lines) - 12
first = f"out/{stems[0]}.png"
iw, ih = ImageReader(first).getSize()
width = min(460.0, (top - 72) * iw / ih)          # fit the grid on the page, keeping its shape
height = width * ih / iw
c.drawImage(first, 72, top - height, width=width, height=height)
c.save()
print("built", sorted(os.listdir("out")))
'''


def build_panel_pack(*, series: list[dict], title: str, note: str = "", timeout_s: int = 180, unit: str = "") -> SandboxResult:
    """One pack for a panel of 1 to MAX_PANEL series, all in one sandbox run.

    Each series dict: ``name``, ``history``, ``point``, ``q10``, ``q90``, optional ``dates`` and ``future_dates`` (both
    needed for a dated axis) and optional ``centre_note``. Outputs: ``forecast_grid.png``/``.svg`` (small multiples, up to
    12 panels each; further figures are ``forecast_grid_2``, ``forecast_grid_3`` ...), ``forecast.xlsx`` (Summary and long
    Forecast sheets), ``report.docx`` (note, grids, summary table) and ``report.pdf`` (note and the first grid).
    """
    if not isinstance(series, list) or not 1 <= len(series) <= MAX_PANEL:
        got = len(series) if isinstance(series, list) else type(series).__name__
        raise ValueError(f"a panel holds 1 to {MAX_PANEL} series; got {got}")
    items: list[dict[str, Any]] = []
    names: set[str] = set()
    for s in series:
        name = str(s.get("name") or "").strip()[:60]
        if not name:
            raise ValueError("every series in a panel needs a name")
        if name in names:
            raise ValueError(f"series name {name!r} appears more than once in the panel")
        names.add(name)
        hist = [float(v) for v in (s.get("history") or [])]
        pt = [float(v) for v in (s.get("point") or [])]
        lo = [float(v) for v in (s.get("q10") or [])]
        hi = [float(v) for v in (s.get("q90") or [])]
        if not hist or not (len(pt) == len(lo) == len(hi) > 0):
            raise ValueError(f"{name}: history must be non-empty, and point, q10 and q90 must be non-empty and the same length")
        keep = max(3 * len(pt), 60)     # the chart shows this many history points; only those are sent to the sandbox
        item: dict[str, Any] = {"name": name, "history": hist[-keep:], "point": pt, "q10": lo, "q90": hi}
        dates, fdates = s.get("dates"), s.get("future_dates")
        if dates and fdates and len(dates) == len(hist) and len(fdates) == len(pt):
            item.update(dates=[str(t) for t in dates][-keep:], future_dates=[str(t) for t in fdates])
        if s.get("centre_note"):
            item["centre_note"] = str(s["centre_note"])[:200]
        items.append(item)
    payload = {"title": title, "note": note, "unit": unit[:40], "per_figure": PANEL_PER_FIGURE, "series": items}
    data = json.dumps(payload).encode("utf-8")
    return run_python(_PANEL_TEMPLATE, timeout_s=timeout_s, mem_mb=2048, file_mb=64, files={"data/input.json": data})
