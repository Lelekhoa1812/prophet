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
