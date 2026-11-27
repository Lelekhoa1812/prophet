"""Charts and tables for the fundamentals layer: the driver map and the scenario-weighted forecast, built in the sandbox.

Motivation vs Logic
Motivation (owner review, 2026-11-27): a forecast has to show the factors behind it and the political, geopolitical and
economic scenarios that could move it, as a picture and a table a person opens, not as words in an answer.
Logic: each builder writes a JSON input and runs a fixed template with matplotlib, so the files come from the same numbers
the tool returns and are listed in the run's sha256 manifest.
"""
from __future__ import annotations

import json

from prophet_agent.sandbox.runner import SandboxResult, run_python

_DRIVER_TEMPLATE = r"""
import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["svg.fonttype"] = "none"

d = json.load(open("data/input.json", encoding="utf-8"))
fams, findings, scen, bench = d["families"], d["findings"], d["scenarios"], d["benchmarks"]
os.makedirs("out", exist_ok=True)
with open("out/driver_map.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["family", "signal", "kind", "direction", "effect", "evidence"])
    for f in findings:
        w.writerow([f["family"], f["signal"], f["kind"], f["direction"], f["effect"], f["evidence"]])
    for b in bench:
        w.writerow(["benchmark", b["source"], "benchmark", "", f"{b['value']} ({b.get('what', '')})", b["evidence"]])
    for s in scen:
        w.writerow(["scenario", s["name"], "scenario", f"p={s['probability']}", s["impact"], s.get("evidence", "")])
json.dump(d, open("out/driver_map.json", "w", encoding="utf-8"), indent=1)

KIND_COL = {"tested": "#D55E00", "researched": "#E69F00", "benchmark": "#0072B2"}
DIR_MARK = {"up": "▲", "down": "▼", "mixed": "◆", "none": "○"}
rows = fams
n = len(rows)
fig, axes = plt.subplots(1, 2 if scen else 1, figsize=(14 if scen else 9.5, max(4.5, 0.42 * n + 2.2)), dpi=144,
                         gridspec_kw={"width_ratios": [3, 2]} if scen else None)
ax = axes[0] if scen else axes
for i, fam in enumerate(rows):
    y = n - 1 - i
    got = [f for f in findings if f["family"] == fam]
    tested = any(f["kind"] == "tested" for f in got)
    col = "#D55E00" if tested else "#E69F00" if got else "#cccccc"
    ax.barh(y, 1, color=col, alpha=0.85 if got else 0.5, height=0.7)
    label = "; ".join(f"{DIR_MARK.get(f['direction'], '')} {f['signal']}" for f in got)[:95] or "not covered"
    ax.text(1.03, y, label, va="center", fontsize=8.5, color="#2b2b2b" if got else "#888888")
ax.set_yticks(range(n))
ax.set_yticklabels(list(reversed(rows)), fontsize=9.5)
ax.set_xlim(0, 4.2)
ax.set_xticks([])
ax.spines["bottom"].set_visible(False)
ax.set_title("Driver families: tested against the target (orange), researched (amber), not covered (grey)",
             fontsize=10, loc="left")
if scen:
    ax2 = axes[1]
    names = [s["name"][:34] for s in scen]
    probs = [100 * float(s["probability"]) for s in scen]
    ax2.barh(range(len(scen))[::-1], probs, color="#0072B2", alpha=0.8, height=0.55)
    ax2.set_ylim(-0.8, max(len(scen), 4) - 0.2)
    for j, s in enumerate(scen):
        ax2.text(probs[j] + 1, len(scen) - 1 - j, f"{probs[j]:.0f}%  {s['impact'][:48]}", va="center", fontsize=8.3)
    ax2.set_yticks(range(len(scen))[::-1])
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlim(0, max(probs + [10]) * 2.6)
    ax2.set_xlabel("probability (%)")
    ax2.set_title("Scenarios and their impact", fontsize=10, loc="left")
fig.suptitle(d["title"], fontsize=13.5, fontweight="bold", x=0.01, ha="left")
if bench:
    fig.text(0.01, -0.01, "Benchmarks: " + "; ".join(f"{b['source']}: {b['value']}" for b in bench)[:220],
             fontsize=8.5, color="#555555", ha="left", va="top")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("out/driver_map.png", bbox_inches="tight")
fig.savefig("out/driver_map.svg", bbox_inches="tight")
"""


def build_driver_map(payload: dict, timeout_s: int = 90) -> SandboxResult:
    """driver_map.csv, .json, .png and .svg from {title, families, findings, scenarios, benchmarks}."""
    return run_python(_DRIVER_TEMPLATE, timeout_s=timeout_s, mem_mb=2048, file_mb=64,
                      files={"data/input.json": json.dumps(payload).encode("utf-8")})


_SCENARIO_TEMPLATE = r"""
import csv, json, os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["svg.fonttype"] = "none"

d = json.load(open("data/input.json", encoding="utf-8"))
hist, hd, fd = d["history"], d.get("dates"), d.get("future_dates")
h = len(d["point"])
dated = bool(hd and fd and len(hd) == len(hist) and len(fd) == h)
if dated:
    hx = [datetime.fromisoformat(t[:10]) for t in hd]
    fx = [datetime.fromisoformat(t[:10]) for t in fd]
else:
    hx = list(range(len(hist)))
    fx = list(range(len(hist), len(hist) + h))
keep = max(36, 3 * h)
hx, hist_k = hx[-keep:], hist[-keep:]
INK, ORANGE, GREY = "#2b2b2b", "#D55E00", "#8a8a8a"
pal = ["#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=144)
ax.plot(hx, hist_k, color=INK, lw=1.4, label="History")
ax.fill_between(fx, d["base_q10"], d["base_q90"], color=GREY, alpha=0.12, lw=0, label="Base pack range (10 to 90)")
ax.plot(fx, d["base_point"], color=GREY, lw=1.3, ls="--", label="Base pack centre")
ax.fill_between(fx, d["q10"], d["q90"], color=ORANGE, alpha=0.18, lw=0, label="Scenario-weighted range (10 to 90)")
ax.plot([hx[-1]] + fx, [hist_k[-1]] + d["point"], color=ORANGE, lw=2.3, label="Scenario-weighted median")
for j, s in enumerate(d["scenarios"]):
    ax.plot(fx, s["median"], color=pal[j % len(pal)], lw=1.1, ls=":",
            label=f"{s['name'][:38]} ({100 * s['probability']:.0f}%)")
ax.grid(color="#e6e6e6", lw=0.6)
ax.set_ylabel(d.get("unit") or "")
ax.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
ax.set_title("Each scenario shifts the base forecast from its start; the orange line and band weigh them by probability",
             fontsize=10, loc="left")
fig.suptitle(d["title"], fontsize=13.5, fontweight="bold", x=0.01, ha="left")
fig.text(0.01, -0.01, d["note"], fontsize=8.5, color="#555555", ha="left", va="top", wrap=True)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs("out", exist_ok=True)
fig.savefig("out/scenario_forecast.png", bbox_inches="tight")
fig.savefig("out/scenario_forecast.svg", bbox_inches="tight")
with open("out/scenario_forecast.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["step", "date", "median", "q10", "q90", "base_point", "base_q10", "base_q90"]
               + [f"median_{s['name'][:30]}" for s in d["scenarios"]])
    for i in range(h):
        w.writerow([i + 1, fd[i] if dated else "", d["point"][i], d["q10"][i], d["q90"][i], d["base_point"][i],
                    d["base_q10"][i], d["base_q90"][i]] + [s["median"][i] for s in d["scenarios"]])
json.dump({k: d[k] for k in ("point", "q10", "q90", "quantiles", "scenarios", "title")} | {"future_dates": fd},
          open("out/scenario_forecast.json", "w", encoding="utf-8"), indent=1)
"""


def build_scenario_chart(payload: dict, timeout_s: int = 90) -> SandboxResult:
    """scenario_forecast.png, .svg, .csv and .json from the mixed forecast and its scenarios."""
    return run_python(_SCENARIO_TEMPLATE, timeout_s=timeout_s, mem_mb=2048, file_mb=64,
                      files={"data/input.json": json.dumps(payload).encode("utf-8")})
