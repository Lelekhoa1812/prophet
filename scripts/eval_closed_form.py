"""Closed-form evaluation: generated files whose answers are computed here, then asked of the live agent.

Each case writes its own files, states the true answer(s) computed directly from the generated rows, asks a plain
question, and passes only if every expected number appears in the answer text within tolerance. The expected values
are never shown to the agent. Needs the API running (python -m prophet_agent.api) and NIM keys.

    PYTHONPATH=src .venv/bin/python scripts/eval_closed_form.py            # all cases
    PYTHONPATH=src .venv/bin/python scripts/eval_closed_form.py messy dupes # chosen cases
"""
from __future__ import annotations

import csv
import io
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta

BASE = "http://127.0.0.1:8010"
FINAL = {"completed", "met", "failed", "stalled", "impossible", "iteration_budget", "budget_exhausted",
         "outcome_unknown", "cancelled"}


@dataclass
class Case:
    name: str
    files: dict[str, bytes]
    question: str
    expect: list[float]                    # every one must appear in the answer
    tol: float = 0.005                     # relative tolerance
    words: list[str] = field(default_factory=list)  # lowercase phrases that must appear


def _csv(rows: list[list], delimiter: str = ",") -> bytes:
    buf = io.StringIO()
    csv.writer(buf, delimiter=delimiter).writerows(rows)
    return buf.getvalue().encode()


def case_regions() -> Case:
    rng = random.Random(1)
    rows, tot = [["Week", "Region", "Units", "Price"]], {"North": 0.0, "South": 0.0, "East": 0.0}
    d0 = date(2026, 1, 5)
    for w in range(26):
        for r in tot:
            u, p = rng.randint(20, 90), round(rng.uniform(8, 14), 2)
            rows.append([(d0 + timedelta(weeks=w)).isoformat(), r, u, p])
            tot[r] += u * p
    best = max(tot, key=tot.get)
    return Case("regions", {"sales.csv": _csv(rows)}, "What is the total revenue (Units x Price) for each region?",
                [round(v, 2) for v in tot.values()], words=[best.lower()])


def case_messy() -> Case:
    rng = random.Random(2)
    body, total = [], 0.0
    for i in range(30):
        v = round(rng.uniform(900, 25000), 2)
        total += v
        body.append([f"INV-{1000 + i}", f"${v:,.2f}"])
    rows = [["ACME Pty Ltd"], ["Invoice register, FY26"], [""], ["Invoice", "Amount"]] + body + [["", ""], ["Printed by finance"]]
    return Case("messy", {"register.csv": _csv(rows)}, "What is the total of all invoice amounts, and how many invoices are there?",
                [round(total, 2), 30.0])


def case_join() -> Case:
    rng = random.Random(3)
    customers = [[f"C{i}", rng.choice(["AU", "NZ", "US"])] for i in range(1, 41)]
    orders, per = [["order", "customer", "amount"]], {"AU": 0, "NZ": 0, "US": 0}
    unmatched = 0
    for o in range(120):
        cid = rng.randint(1, 48)             # C41..C48 do not exist
        orders.append([o, f"C{cid}", rng.randint(10, 200)])
        if cid > 40:
            unmatched += 1
        else:
            per[customers[cid - 1][1]] += 1
    return Case("join", {"orders.csv": _csv(orders), "customers.csv": _csv([["customer", "country"]] + customers)},
                "How many orders came from each country, and how many orders could not be matched to a customer?",
                [float(v) for v in per.values()] + [float(unmatched)])


def case_dupes() -> Case:
    rng = random.Random(4)
    rows = [["id", "name", "score"]]
    for i in range(60):
        rows.append([i, f"n{i}", "" if i % 9 == 0 else rng.randint(40, 100)])
    dup = 7
    rows += [list(rows[k]) for k in range(1, dup + 1)]
    seen, vals = set(), []
    for r in rows[1:]:
        if r[0] in seen:
            continue
        seen.add(r[0])
        if r[2] != "":
            vals.append(r[2])
    return Case("dupes", {"scores.csv": _csv(rows)},
                "How many rows are exact duplicates of an earlier row, and what is the average score once duplicates and blanks are removed?",
                [float(dup), round(sum(vals) / len(vals), 2)], tol=0.01)


def case_dates() -> Case:
    rng = random.Random(5)
    rows, by = [["Date", "Sales"]], {}
    for _ in range(90):
        d = date(2025, 1, 1) + timedelta(days=rng.randint(0, 364))
        v = rng.randint(50, 500)
        rows.append([d.strftime("%d/%m/%Y"), v])
        by[d.month] = by.get(d.month, 0) + v
    best = max(by, key=by.get)
    name = date(2025, best, 1).strftime("%B").lower()
    return Case("dates", {"daily.csv": _csv(rows)}, "Which month had the highest total sales, and what was that total? Dates are day/month/year.",
                [float(by[best])], words=[name])


def case_trend() -> Case:
    d0 = date(2025, 1, 6)
    rows = [["Week", "Orders"]] + [[(d0 + timedelta(weeks=i)).isoformat(), 100 + 5 * i] for i in range(52)]
    return Case("trend", {"weekly.csv": _csv(rows)}, "Forecast the next 4 weeks of Orders and say what the last forecast week is expected to be.",
                [100 + 5 * 55], tol=0.06)


CASES = {c.__name__[5:]: c for c in (case_regions, case_messy, case_join, case_dupes, case_dates, case_trend)}


def call(method: str, path: str, body=None, raw: bytes | None = None, headers: dict | None = None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = dict(headers or {})
    if body is not None:
        h["content-type"] = "application/json"
    with urllib.request.urlopen(urllib.request.Request(BASE + path, data=data, method=method, headers=h), timeout=120) as r:
        return json.loads(r.read())


def numbers_in(text: str) -> list[float]:
    out = []
    for m in re.finditer(r"-?\d[\d,]*\.?\d*", text):
        s = m.group().replace(",", "").rstrip(".")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def run_case(c: Case, timeout: float = 420.0) -> dict:
    nb = call("POST", "/api/notebooks", {"title": f"eval {c.name}"})
    for name, data in c.files.items():
        call("POST", f"/api/notebooks/{nb['id']}/sources", raw=data, headers={"X-File-Name": urllib.parse.quote(name)})
    for _ in range(60):
        if all(s["status"] != "profiling" for s in call("GET", f"/api/notebooks/{nb['id']}")["sources"]):
            break
        time.sleep(1)
    turn = call("POST", f"/api/notebooks/{nb['id']}/turns", {"goal": c.question})
    start, m = time.time(), {}
    while time.time() - start < timeout:
        m = call("GET", f"/api/missions/{turn['id']}")
        if m["status"] in FINAL or m["status"] == "waiting_user":
            break
        time.sleep(3)
    ans = m.get("answer") if isinstance(m.get("answer"), dict) else {}
    text = ans.get("answer", "") if ans.get("status") == "met" else ""
    nums, low = numbers_in(text), text.lower()
    missing = [e for e in c.expect if not any(abs(n - e) <= c.tol * max(1.0, abs(e)) for n in nums)]
    missing_words = [w for w in c.words if w not in low]
    ok = m.get("status") in {"met", "completed"} and bool(text) and not missing and not missing_words
    return {"case": c.name, "ok": ok, "status": m.get("status"), "seconds": round(time.time() - start),
            "missing_numbers": missing, "missing_words": missing_words, "notebook": nb["id"], "answer": text[:600]}


def main(argv: list[str]) -> int:
    chosen = argv or list(CASES)
    results = [run_case(CASES[n]()) for n in chosen]
    for r in results:
        flag = "PASS" if r["ok"] else "FAIL"
        print(f"{flag} {r['case']:<8} {r['status']:<10} {r['seconds']:>4}s", "" if r["ok"] else f"missing={r['missing_numbers']} {r['missing_words']}")
        if not r["ok"]:
            print("     ", r["answer"].replace("\n", " ")[:300])
    passed = sum(r["ok"] for r in results)
    print(f"{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
