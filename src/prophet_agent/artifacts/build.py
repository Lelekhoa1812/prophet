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

d = json.load(open("data/input.json", encoding="utf-8"))
hist, fc, lo, hi = d["history"], d["point"], d["q10"], d["q90"]
title, note = d["title"], d["note"]
x = list(range(len(hist)))
fx = list(range(len(hist), len(hist) + len(fc)))

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
ax.plot(x, hist, label="History", color="#1f2937", linewidth=1.5)
ax.plot(fx, fc, label="TimesFM 3 point forecast", color="#7c3aed", linewidth=2)
ax.fill_between(fx, lo, hi, color="#7c3aed", alpha=0.25, label="P10 to P90")
ax.set_title(title)
ax.set_xlabel("Period")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("out/forecast.png")
fig.savefig("out/forecast.svg")
plt.close(fig)

import openpyxl
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Forecast"
ws.append(["Period ahead", "Point", "P10", "P90"])
for i, (p, a, b) in enumerate(zip(fc, lo, hi), start=1):
    ws.append([i, p, a, b])
wb.save("out/forecast.xlsx")

from docx import Document
from docx.shared import Inches
doc = Document()
doc.add_heading(title, level=1)
doc.add_paragraph(note)
doc.add_picture("out/forecast.png", width=Inches(6))
doc.save("out/report.docx")

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
c = canvas.Canvas("out/report.pdf", pagesize=A4)
c.setFont("Helvetica-Bold", 14)
c.drawString(72, 780, title[:90])
c.setFont("Helvetica", 9)
c.drawString(72, 764, note[:140])
c.drawImage("out/forecast.png", 72, 380, width=460, preserveAspectRatio=True)
c.save()
print("built", sorted(__import__("os").listdir("out")))
'''


def build_forecast_pack(*, history: list[float], point: list[float], q10: list[float], q90: list[float],
                        title: str, note: str, timeout_s: int = 90) -> SandboxResult:
    if not (len(point) == len(q10) == len(q90) > 0):
        raise ValueError("point, q10 and q90 must be non-empty and the same length")
    payload: dict[str, Any] = {"history": [float(v) for v in history], "point": [float(v) for v in point],
                               "q10": [float(v) for v in q10], "q90": [float(v) for v in q90],
                               "title": title, "note": note}
    data = json.dumps(payload).encode("utf-8")
    return run_python(_TEMPLATE, timeout_s=timeout_s, mem_mb=2048, file_mb=64, files={"data/input.json": data})
