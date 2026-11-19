"""TimesFM ERP lab API. Everything the console shows comes from here."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..agents import META, REGISTRY
from ..agents.base import AgentContext
from ..config import BEDROCK_HARD_CAP_USD, BEDROCK_MODEL, DATA, DATASETS, REVEAL_DB, TIMESFM_CHECKPOINT
from ..engine import baselines
from ..engine.backtest import run_backtest
from ..engine.forecast import TimesFMEngine
from ..engine.state import ForecastService, get_world, week_start, weekly_matrix
from ..narration import bedrock, templated
from ..policy import alerts as alert_policy, rollout
from ..policy.governor import LADDER, LADDER_LABEL, Governor, Lesson, ladder_status
from ..sim.replay import replay, scored_window
from ..store import open_dataset, open_lab
from . import jobs

app = FastAPI(title="TimesFM ERP lab", version="0.1.0")

# Motivation vs Logic
# Motivation: the API has no login, and `/api/reveal` returns real customer names. Any web page open in the same browser
# could POST to localhost (decisions, tenant settings, replays), and a rebound hostname could read the reveal map.
# Logic: refuse requests whose Host is not a loopback name (blocks DNS rebinding), and refuse writes that a browser marks
# as coming from another site (Origin not loopback, or Sec-Fetch-Site cross-site). Requests with no Origin (curl, tests,
# the same-origin console) pass. TIMESFM_ERP_ALLOWED_HOSTS adds hostnames for a deliberate shared deployment, which
# still needs real authentication before it holds client data.
_LOOPBACK = {"localhost", "127.0.0.1", "[::1]", "::1"}


def _hostname(value: str) -> str:
    from urllib.parse import urlsplit
    return (urlsplit(value if "//" in value else f"//{value}").hostname or "").lower()


def _netloc(value: str) -> str:
    from urllib.parse import urlsplit
    return urlsplit(value if "//" in value else f"//{value}").netloc.lower()


@app.middleware("http")
async def _same_origin_guard(request, call_next):
    from fastapi.responses import JSONResponse
    import os
    extra = {_hostname(h) for h in os.environ.get("TIMESFM_ERP_ALLOWED_HOSTS", "").split(",") if h.strip()}
    allowed = _LOOPBACK | extra
    host = request.headers.get("host", "")
    if _hostname(host) not in allowed:
        return JSONResponse({"detail": "host not allowed"}, status_code=400)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        # Same origin means same host AND port as the request itself. Another page on another local port is another site.
        if (origin and _netloc(origin) != _netloc(host)) or request.headers.get("sec-fetch-site") in ("cross-site", "same-site"):
            return JSONResponse({"detail": "cross-site write refused"}, status_code=403)
    return await call_next(request)
_lock = threading.Lock()
_governors: dict[str, Governor] = {}


def _lab():
    return open_lab()


def _available() -> list[str]:
    return [d for d, p in DATASETS.items() if Path(p).exists()]


def _ds(ds: str) -> str:
    if ds not in _available():
        raise HTTPException(404, f"dataset {ds} not built yet")
    return ds


def _df(con, sql, params=None) -> list[dict]:
    d = con.execute(sql, params or []).df()
    return json.loads(d.to_json(orient="records", date_format="iso"))


def _reveal_map() -> dict:
    if not Path(REVEAL_DB).exists():
        return {}
    c = duckdb.connect(str(REVEAL_DB), read_only=True)
    m = {code: name for _, code, name in c.execute("select * from reveal_map").fetchall()}
    c.close()
    return m


# ---------------------------------------------------------------- meta
@app.get("/api/meta")
def meta():
    return {
        "model": {"name": "TimesFM 3.0", "checkpoint": TIMESFM_CHECKPOINT, "backend": "MLX (Apple silicon)", "loaded": TimesFMEngine._inst is not None,
                  "licence": "TimesFM 3.0 weights are non-commercial and non-production. Client production use is blocked until a licence answer.",
                  "licence_status": "unresolved"},
        "datasets": {d: _dataset_card(d) for d in _available()},
        "agents": {k: {**v, "key": k} for k, v in META.items()},
        "bedrock": {"model": BEDROCK_MODEL, "spent_usd": round(bedrock.spent(), 4), "cap_usd": BEDROCK_HARD_CAP_USD},
        "ladder": [{"key": k, "label": LADDER_LABEL[k]} for k in LADDER],
        "ship_bar": {"improvement_vs_best_baseline": 0.15, "band_coverage": [0.70, 0.90], "useful_share": 0.70, "invented_numbers": 0},
    }


def _dataset_card(ds: str) -> dict:
    con = open_dataset(ds, read_only=True)
    meta = dict(con.execute("select key,value from meta").fetchall())
    n = {t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in ["products", "customers", "order_lines", "lots", "truth_events"]}
    rev = con.execute("select coalesce(sum(qty*coalesce(unit_price,0)),0) from order_lines where not coalesce(cancelled,false)").fetchone()[0]
    con.close()
    return {"key": ds, "source": meta.get("source"), "first": meta.get("first_date"), "last": meta.get("last_date"), "counts": n, "revenue": float(rev),
            "has_truth": n["truth_events"] > 0, "extracted_at": meta.get("extracted_at"),
            "role_audit": json.loads(meta["role_audit"]) if meta.get("role_audit") else None}


@app.get("/api/datasets/{ds}/summary")
def summary(ds: str):
    _ds(ds)
    con = open_dataset(ds, read_only=True)
    out = {"card": _dataset_card(ds)}
    out["weekly_sales"] = _df(con, """select date_trunc('week', delivery_date)::date as week, sum(qty*coalesce(unit_price,0)) as revenue,
                                      count(distinct order_id) as orders from order_lines where not coalesce(cancelled,false) group by 1 order by 1""")
    out["by_category"] = _df(con, """select p.category, sum(l.qty*coalesce(l.unit_price,0)) as revenue, count(*) as lines
                                      from order_lines l join products p using(product_id) group by 1 order by 2 desc""")
    if out["card"]["has_truth"]:
        o = con.execute("""select sum(expired_value), sum(demand*p.unit_price), sum(fulfilled), sum(demand) from daily_outcomes d join products p using(product_id)""").fetchone()
        out["operations"] = {"waste_pct": float(o[0] / o[1]) if o[1] else None, "fill_rate": float(o[2] / o[3]) if o[3] else None}
    con.close()
    return out


# ---------------------------------------------------------------- agents
class AgentConfigIn(BaseModel):
    enabled: bool | None = None
    owner: str | None = None
    quiet_start: str | None = None
    quiet_end: str | None = None
    cooldown_days: int | None = None
    autonomy_level: str | None = None
    params: dict | None = None


def _config(ds: str, agent: str) -> dict:
    lab = _lab()
    r = lab.execute("select enabled, owner, quiet_start, quiet_end, cooldown_days, params, autonomy_level from agent_config where dataset=? and agent=?", [ds, agent]).fetchone()
    lab.close()
    base = {"enabled": True, "owner": META[agent]["owner"].split(" / ")[0], "quiet_start": "20:00", "quiet_end": "05:00", "cooldown_days": 3,
            "params": {}, "autonomy_level": "suggests"}
    if r:
        base = {"enabled": r[0], "owner": r[1], "quiet_start": r[2], "quiet_end": r[3], "cooldown_days": r[4], "params": json.loads(r[5] or "{}"), "autonomy_level": r[6]}
    return base


@app.get("/api/agents")
def agents(dataset: str = "twin"):
    _ds(dataset)
    sc = _scorecard(dataset)
    out = []
    for k, m in META.items():
        cfg = _config(dataset, k)
        stats = (sc.get("agents") or {}).get(k, {})
        stats = {**stats, "proposals": stats.get("proposals", 0)}
        lab = _lab()
        d = lab.execute("""select decision, count(*) from (select d.decision from decisions d join proposals p on p.proposal_id=d.proposal_id where p.dataset=? and p.agent=?
                           qualify row_number() over (partition by d.proposal_id order by d.decided_at desc) = 1) group by 1""", [dataset, k]).fetchall()
        lab.close()
        out.append({"key": k, **m, "config": cfg, "stats": stats, "operator_decisions": dict(d),
                    "ladder": ladder_status(k, cfg["autonomy_level"], stats), "defaults": getattr(REGISTRY[k], "DEFAULTS", {})})
    return out


@app.put("/api/agents/{agent}")
def put_agent(agent: str, body: AgentConfigIn, dataset: str = "twin"):
    if agent not in REGISTRY:
        raise HTTPException(404, "unknown agent")
    cfg = {**_config(dataset, agent), **{k: v for k, v in body.model_dump().items() if v is not None}}
    if cfg["autonomy_level"] not in LADDER:
        raise HTTPException(400, "unknown autonomy level")
    with _lock:
        lab = _lab()
        lab.execute("delete from agent_config where dataset=? and agent=?", [dataset, agent])
        lab.execute("insert into agent_config values (?,?,?,?,?,?,?,?,?)",
                    [dataset, agent, cfg["enabled"], cfg["owner"], cfg["quiet_start"], cfg["quiet_end"], cfg["cooldown_days"], json.dumps(cfg["params"]), cfg["autonomy_level"]])
        lab.close()
    return cfg


# ---------------------------------------------------------------- proposals and decisions
def _prop_row(r, reveal: dict | None = None) -> dict:
    d = {"proposal_id": r[0], "agent": r[1], "as_of": str(r[2]), "entity_type": r[3], "entity_id": r[4], "title": r[5], "severity": r[6],
         "dollars_at_stake": r[7], "confidence": r[8], "autonomy_level": r[9], "owner": r[10], "evidence": json.loads(r[11]), "action": json.loads(r[12]),
         "status": r[13], "narration_templated": r[14], "narration_model": r[15]}
    d["narration_templated"] = d["narration_templated"] or templated.narrate(d)
    return d


COLS = "proposal_id, agent, as_of, entity_type, entity_id, title, severity, dollars_at_stake, confidence, autonomy_level, owner, evidence, action, status, narration_templated, narration_model"


@app.get("/api/proposals")
def proposals(dataset: str = "twin", agent: str | None = None, status: str | None = None, as_of: str | None = None, until: str | None = None,
              limit: int = 100, sort: str = "dollars"):
    _ds(dataset)
    q, p = f"select {COLS} from proposals where dataset=?", [dataset]
    if agent: q += " and agent=?"; p.append(agent)
    if status: q += " and status=?"; p.append(status)
    if as_of: q += " and as_of=?"; p.append(as_of)
    if until: q += " and as_of<=?"; p.append(until)
    q += " order by " + ("dollars_at_stake desc" if sort == "dollars" else "as_of desc, dollars_at_stake desc") + " limit ?"
    p.append(limit)
    lab = _lab(); rows = lab.execute(q, p).fetchall(); lab.close()
    return [_prop_row(r) for r in rows]


def lab_ds(pid: str) -> str | None:
    lab = _lab(); r = lab.execute("select dataset from proposals where proposal_id=?", [pid]).fetchone(); lab.close()
    return r[0] if r else None


@app.get("/api/proposals/{pid}")
def proposal(pid: str):
    lab = _lab(); r = lab.execute(f"select {COLS} from proposals where proposal_id=?", [pid]).fetchone()
    decs = lab.execute("select decided_at, decided_by, decision, reason, note from decisions where proposal_id=? order by decided_at", [pid]).fetchall()
    lab.close()
    if not r:
        raise HTTPException(404, "proposal not found")
    d = _prop_row(r)
    ds = lab_ds(pid)
    d["useful_rate"] = rollout.useful_rate(rollout.load(ds), d["agent"], (d.get("evidence") or {}).get("p_event")) if ds else None
    d["decisions"] = [{"at": str(a), "by": b, "decision": c, "reason": e, "note": f} for a, b, c, e, f in decs]
    return d


class DecisionIn(BaseModel):
    decision: str            # approve | adjust | dismiss
    reason: str | None = None  # for dismiss: one_off | known_promo | wrong_alert | not_useful
    note: str | None = None
    decided_by: str = "liam"


_OPERATOR = __import__("re").compile(r"^[A-Za-z0-9][A-Za-z0-9 ._@'-]{0,63}$")


def _operator(header: str | None, body_name: str) -> str:
    """Who is recorded as deciding. There is no login, so this is asserted, never verified: it is stored as 'unverified:<name>'.

    Motivation vs Logic
    Motivation: decided_by was any string from the request body, including markup and unbounded text, shown later in the
    decision history. Logic: the X-Operator header (set by whatever fronts the lab) beats the body; the name must be 1 to 64
    plain characters; the stored value says plainly that it was not authenticated.
    """
    name = (header or body_name or "").strip()
    if not _OPERATOR.match(name):
        raise HTTPException(400, "operator name must be 1 to 64 letters, digits, spaces or . _ @ ' -")
    return f"unverified:{name}"


@app.post("/api/proposals/{pid}/decision")
def decide(pid: str, body: DecisionIn, x_operator: str | None = Header(default=None)):
    if body.decision not in ("approve", "adjust", "dismiss"):
        raise HTTPException(400, "decision must be approve, adjust or dismiss")
    operator = _operator(x_operator, body.decided_by)
    with _lock:
        lab = _lab()
        r = lab.execute("select dataset, agent, entity_id, dedupe_key, as_of from proposals where proposal_id=?", [pid]).fetchone()
        if not r:
            lab.close(); raise HTTPException(404, "proposal not found")
        ds, agent, entity, dk, as_of = r
        did = f"D-{uuid.uuid4().hex[:8]}"
        # Bugs vs Fixes
        # Bug: a second decision on the same proposal left the first dismissal's lesson active (or stacked a duplicate), so a
        # reviewer changing their mind kept suppressing the entity.
        # Fix: the latest decision wins. Active lessons born from earlier decisions on this proposal are retired first.
        lab.execute("""update lessons set active=false, undone_at=? where active and source_decision_id in
                       (select decision_id from decisions where proposal_id=?)""", [datetime.now(timezone.utc), pid])
        lab.execute("insert into decisions values (?,?,?,?,?,?,?)", [did, pid, datetime.now(timezone.utc), operator, body.decision, body.reason, body.note])
        lab.execute("update proposals set status=? where proposal_id=?", [{"approve": "approved", "adjust": "adjusted", "dismiss": "dismissed"}[body.decision], pid])
        lesson = None
        if body.decision == "dismiss" and body.reason in ("one_off", "known_promo", "wrong_alert", "not_useful"):
            days = {"one_off": 14, "known_promo": 28, "wrong_alert": 45, "not_useful": 30}[body.reason]
            lid = f"LS-{uuid.uuid4().hex[:6]}"
            lab.execute("insert into lessons values (?,?,?,?,?,?,?,?,?,?)",
                        [lid, ds, agent, "entity", entity, json.dumps({"suppress_days": days, "reason": body.reason}), did, True, datetime.now(timezone.utc), None])
            lesson = {"lesson_id": lid, "agent": agent, "scope": entity, "suppress_days": days, "reason": body.reason}
        _governors.pop(ds, None)               # lessons may have changed (added or retired): rebuild governor on next use
        lab.close()
    return {"decision_id": did, "lesson": lesson}


@app.get("/api/lessons")
def lessons(dataset: str = "twin"):
    lab = _lab()
    rows = lab.execute("select lesson_id, agent, scope_type, scope_id, effect, source_decision_id, active, created_at, undone_at from lessons where dataset=? order by created_at desc", [dataset]).fetchall()
    lab.close()
    return [{"lesson_id": a, "agent": b, "scope_type": c, "scope_id": d, "effect": json.loads(e), "source_decision_id": f, "active": g, "created_at": str(h), "undone_at": None if i is None else str(i)}
            for a, b, c, d, e, f, g, h, i in rows]


@app.post("/api/lessons/{lid}/undo")
def undo_lesson(lid: str):
    with _lock:
        lab = _lab()
        r = lab.execute("select dataset from lessons where lesson_id=?", [lid]).fetchone()
        if not r:
            lab.close(); raise HTTPException(404, "lesson not found")
        lab.execute("update lessons set active=false, undone_at=? where lesson_id=?", [datetime.now(timezone.utc), lid])
        lab.close()
        _governors.pop(r[0], None)
    return {"ok": True}


class NarrateIn(BaseModel):
    mode: str = "bedrock"


@app.post("/api/proposals/{pid}/narrate")
def narrate(pid: str, body: NarrateIn):
    p = proposal(pid)
    t = templated.narrate(p)
    out = {"templated": t, "bedrock": None}
    if body.mode == "bedrock":
        try:
            r = bedrock.narrate(p)
        except Exception as e:  # noqa: BLE001
            r = {"text": None, "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}", "cost_usd": 0.0}
        out["bedrock"] = r
        if r.get("ok") and r.get("text"):
            lab = _lab()
            lab.execute("update proposals set narration_model=? where proposal_id=?", [r["text"], pid])
            lab.close()
    out["spent_usd"] = round(bedrock.spent(), 4)
    return out


# ---------------------------------------------------------------- scorecard, backtests, replay
def _scorecard(ds: str) -> dict:
    f = DATA / f"scorecard_{ds}.json"
    return json.loads(f.read_text()) if f.exists() else {}


@app.get("/api/scorecard")
def scorecard(dataset: str = "twin"):
    sc = _scorecard(dataset)
    if "events" in sc:
        sc = {**sc, "events": sc["events"][:2000]}
    return sc


@app.get("/api/backtests")
def backtests(dataset: str = "twin"):
    lab = _lab()
    rows = lab.execute("select backtest_id, created_at, config, results from backtests where dataset=? order by created_at desc", [dataset]).fetchall()
    lab.close()
    return [{"backtest_id": a, "created_at": str(b), "config": json.loads(c), "results": json.loads(d)} for a, b, c, d in rows]


class BacktestIn(BaseModel):
    dataset: str = "twin"
    freq: str = "W"
    horizon: int = 4
    n_origins: int = 26


@app.post("/api/backtests/run")
def run_bt(body: BacktestIn):
    _ds(body.dataset)
    return {"job": jobs.submit("backtest", lambda progress, **kw: run_backtest(progress=progress, **kw), dataset=body.dataset, freq=body.freq,
                               horizon=body.horizon, n_origins=body.n_origins)}


class ReplayIn(BaseModel):
    dataset: str = "twin"
    start: str | None = None
    end: str | None = None
    agents: list[str] | None = None
    params: dict | None = None


@app.post("/api/replay/run")
def run_replay(body: ReplayIn):
    _ds(body.dataset)
    pol = rollout.economics_of(rollout.load(body.dataset)).describe()      # replays run under the client's own policy
    return {"job": jobs.submit("replay", lambda progress, **kw: {k: v for k, v in replay(progress=progress, **kw).items() if k not in ("_rows", "events")},
                               dataset=body.dataset, start=body.start, end=body.end, agents=body.agents, params=body.params, policy=pol)}


@app.get("/api/jobs")
def list_jobs():
    return jobs.list_jobs()


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = jobs.get(jid)
    if not j:
        raise HTTPException(404, "job not found")
    return j


# ---------------------------------------------------------------- explorer
@app.get("/api/products")
def products(dataset: str = "twin", q: str | None = None, limit: int = 50):
    _ds(dataset)
    w = get_world(dataset)
    df = w.products
    if q:
        df = df[df.display_name.str.contains(q, case=False) | df.product_id.str.contains(q, case=False) | df.category.str.contains(q, case=False)]
    rev = (w.lines.assign(v=w.lines.qty * w.lines.unit_price.fillna(0)).groupby("product_id").v.sum())
    df = df.assign(revenue=df.product_id.map(rev).fillna(0)).sort_values("revenue", ascending=False).head(limit)
    return json.loads(df[["product_id", "display_name", "category", "uom", "shelf_life_days", "unit_price", "revenue"]].to_json(orient="records"))


@app.get("/api/forecast")
def forecast(dataset: str, product: str, as_of: str, freq: str = "W", horizon: int = 4, history: int = 400):
    """History, TimesFM forecast with bands, baselines, what really happened, and the true underlying demand (twin).

    history is in weeks for freq=W and in days for freq=D. Daily values are derived from the weekly forecast, the
    same way the briefing does it, so the two views agree.
    """
    _ds(dataset)
    w = get_world(dataset)
    if product not in w.pidx:
        raise HTTPException(404, "unknown product")
    if freq not in ("W", "D"):
        raise HTTPException(400, "freq must be W or D")
    if not 1 <= horizon <= (12 if freq == "W" else 28):
        raise HTTPException(400, "horizon out of range (weekly 1 to 12, daily 1 to 28)")
    pi = w.pidx[product]
    a = pd.Timestamp(as_of)
    season = 52 if freq == "W" else 7
    actual_weekly = _weekly_full(w, pi)
    actual_daily = pd.Series(w.demand[pi], index=w.days)
    base: dict = {}
    if freq == "W":
        # Motivation vs Logic
        # Motivation: the week containing as_of is only partly known, so a forecast starting there mixes known and
        # unknown days, and the first point is not comparable with the history.
        # Logic: forecast one extra week, then drop it, so the forecast starts at the next full week.
        mat, weeks = weekly_matrix(w, a)
        series, idx = mat[pi][-history:], weeks[-history:]
        if len(series) < 8:
            raise HTTPException(400, "not enough history before this date")
        fc = TimesFMEngine.get().predict([series], horizon + 1)
        fut_start = week_start(a) + pd.Timedelta(days=7)
        fut_days = [fut_start + pd.Timedelta(days=7 * i) for i in range(horizon)]
        point = fc.point[0, 1:]
        qmap = {f"q{int(q * 100)}": fc.q[0, 1:, j] for j, q in enumerate(np.arange(1, 10) / 10)}
        for m in ["ma4", "ma8", "seasonal_year"]:
            b = baselines.run(m, [series], horizon + 1, season)
            base[m] = {"point": [round(float(x), 2) for x in b.point[0, 1:]], "q10": [round(float(x), 2) for x in b.q[0, 1:, 0]],
                       "q90": [round(float(x), 2) for x in b.q[0, 1:, 8]]}
        known_from = week_start(a)
        known = w.demand[pi, w.days.searchsorted(known_from):w.days.searchsorted(a)]
        partial = {"week_start": str(known_from.date()), "known_days": int(len(known)), "known_units": round(float(known.sum()), 2)}
        hist_x = [str(x.date()) for x in idx]
        hist_y = series
        # Motivation vs Logic
        # Motivation: the model's answer depends on how much history it sees, and users need to know how much.
        # Logic: forecast again on the last 52 weeks and report the average percentage change in the next weeks.
        sensitivity = None
        if len(series) > 52:
            alt = TimesFMEngine.get().predict([series[-52:]], horizon).point[0]
            sensitivity = round(float(np.mean(np.abs(alt - point) / np.maximum(point, 1e-6))), 3)
        model_saw = {"periods": int(len(series)), "from": str(idx[0].date()), "to": str(idx[-1].date()), "unit": "weeks",
                     "full_year": bool(len(series) >= 52), "seasonal_years": round(len(series) / 52, 2), "derived": False,
                     "one_year_sensitivity": sensitivity}
        actual_future = [None if d not in actual_weekly.index else round(float(actual_weekly[d]), 2) for d in fut_days]
    else:
        # Daily derived: the weekly forecast for the window, split by the product's weekday pattern.
        mat, weeks = weekly_matrix(w, a)
        wseries, widx = mat[pi][-max(8, history // 7):], weeks[-max(8, history // 7):]
        if len(wseries) < 8:
            raise HTTPException(400, "not enough history before this date")
        fc = TimesFMEngine.get().predict([wseries], 5)  # 28 days from a Sunday touch five week-starts
        shape = ForecastService(w).weekday_profile(a, pi)
        ws = week_start(a)
        fut_days = [a + pd.Timedelta(days=j) for j in range(horizon)]

        def per_day(row: np.ndarray) -> list[float]:
            return [round(float(row[(d - ws).days // 7] * shape[d.dayofweek] / 7.0), 2) for d in fut_days]

        point = np.array(per_day(fc.point[0]))
        qmap = {f"q{int(q * 100)}": np.array(per_day(fc.q[0, :, j])) for j, q in enumerate(np.arange(1, 10) / 10)}
        dser = w.demand[pi, :w.days.searchsorted(a)][-history:]
        didx = w.days[:w.days.searchsorted(a)][-history:]
        b = baselines.run("ma8", [dser], horizon, season)
        base["ma8"] = {"point": [round(float(x), 2) for x in b.point[0]], "q10": [round(float(x), 2) for x in b.q[0, :, 0]],
                       "q90": [round(float(x), 2) for x in b.q[0, :, 8]]}
        partial = None
        hist_x = [str(x.date()) for x in didx]
        hist_y = dser
        model_saw = {"periods": int(len(wseries)), "from": str(widx[0].date()), "to": str(widx[-1].date()), "unit": "weeks",
                     "full_year": bool(len(wseries) >= 52), "seasonal_years": round(len(wseries) / 52, 2), "derived": True,
                     "one_year_sensitivity": None}
        actual_future = [None if d not in actual_daily.index else round(float(actual_daily[d]), 2) for d in fut_days]
    fut = [str(d.date()) for d in fut_days]
    expected = _expected_series(dataset, product, freq)
    return {
        "product": product, "name": w.products.set_index("product_id").display_name[product], "freq": freq, "as_of": as_of,
        "history": {"x": hist_x, "y": [round(float(v), 2) for v in hist_y]},
        "forecast": {"x": fut, "point": [round(float(v), 2) for v in point],
                     "q": {k: [round(float(v), 2) for v in arr] for k, arr in qmap.items()}},
        "baselines": base, "actual_future": actual_future, "partial_week": partial,
        "truth_expected": None if expected is None else {"x": [str(x) for x in expected.index], "y": [round(float(v), 2) for v in expected.values]},
        "events": _product_events(dataset, product),
        "model_saw": model_saw,
    }


def _weekly_full(w, pi) -> pd.Series:
    s = pd.Series(w.demand[pi], index=w.days)
    wk = s.groupby(s.index - pd.to_timedelta(s.index.dayofweek, unit="D")).sum()
    return wk


def _expected_series(dataset: str, product: str, freq: str):
    con = open_dataset(dataset, read_only=True)
    n = con.execute("select count(*) from truth_expected").fetchone()[0]
    if not n:
        con.close(); return None
    df = con.execute("select day, expected_qty from truth_expected where product_id=? order by day", [product]).df(); con.close()
    s = pd.Series(df.expected_qty.values, index=pd.to_datetime(df.day))
    if freq == "W":
        s = s.groupby(s.index - pd.to_timedelta(s.index.dayofweek, unit="D")).sum()
    s.index = s.index.date
    return s


def _product_events(dataset: str, product: str) -> list[dict]:
    con = open_dataset(dataset, read_only=True)
    rows = _df(con, "select event_id, kind, start_date, end_date, magnitude, note from truth_events where entity_id=? or entity_id like ?", [product, f"%|{product}"])
    con.close()
    return rows


@app.get("/api/truth/events")
def truth_events(dataset: str = "twin"):
    con = open_dataset(_ds(dataset), read_only=True)
    rows = _df(con, "select * from truth_events order by start_date")
    con.close()
    return rows


# ---------------------------------------------------------------- briefing, time machine
@app.get("/api/briefing")
def briefing(dataset: str = "twin", as_of: str | None = None, owner: str | None = None, limit: int = 12):
    _ds(dataset)
    w = get_world(dataset)
    lo, hi = scored_window(w)
    d = as_of or str((lo + (hi - lo) / 2).date())
    lab = _lab()
    rows = lab.execute(f"select {COLS} from proposals where dataset=? and as_of=? order by dollars_at_stake desc", [dataset, d]).fetchall()
    open_older = lab.execute(f"select {COLS} from proposals where dataset=? and status='open' and as_of<? and as_of>=? order by dollars_at_stake desc limit 20",
                             [dataset, d, str((pd.Timestamp(d) - pd.Timedelta(days=7)).date())]).fetchall()
    lab.close()
    props = [_prop_row(r) for r in rows]
    if owner:
        props = [p for p in props if owner in p["owner"]]
    cats = {pid: str(c) for pid, c in w.products.set_index("product_id").category.items()}
    for p in props:                                        # the segment an alert belongs to, for learned priorities
        if p["entity_type"] == "product":
            p["segment"] = cats.get(p["entity_id"])
    cfg = rollout.load(dataset)
    for p in props:
        pe = (p.get("evidence") or {}).get("p_event")
        p["useful_rate"] = rollout.useful_rate(cfg, p["agent"], pe)
    econ = rollout.economics_of(cfg)
    ph = rollout.phase(cfg, d)
    shown, digest_items, below = alert_policy.select(props, econ)
    digest_items = digest_items + shown[limit:]          # a screen limit moves overflow into the digest, never drops it
    shown = shown[:limit]
    digest = _digest_groups(digest_items) or _stored_digest(dataset, d, owner)
    if ph["phase"] in ("shadow", "setup"):
        hidden = len(shown)
        return {"as_of": d, "headline": f"Shadow mode: {hidden} alert{'s' if hidden != 1 else ''} recorded but not shown.", "needs_you": [],
                "digest": [], "still_open": [], "by_agent": {}, "dollars_in_play": 0.0, "phase": ph, "policy": econ.describe(),
                "shadow_hidden": hidden, "below_floor": len(below)}
    return {"as_of": d, "headline": _headline(shown), "needs_you": shown, "digest": digest, "still_open": [_prop_row(r) for r in open_older][:6],
            "by_agent": {k: sum(1 for p in shown if p["agent"] == k) for k in REGISTRY},
            "dollars_in_play": round(sum(p["dollars_at_stake"] for p in shown), 2),
            "phase": ph, "policy": econ.describe(), "below_floor": len(below)}


def _stored_digest(dataset: str, d: str, owner: str | None) -> list[dict]:
    """Digest recorded by the replay for a past day (the digested alerts themselves are not stored)."""
    f = DATA / f"digest_{dataset}.json"
    if not f.exists():
        return []
    rows = json.loads(f.read_text()).get(d, [])
    return [r for r in rows if not owner or owner in r["owner"]]


def _digest_groups(rest: list[dict]) -> list[dict]:
    """Group alerts that are above the score floor but over the owner's budget, by owner and agent."""
    return alert_policy.digest_groups(rest)


def _headline(props: list[dict]) -> str:
    if not props:
        return "Nothing needs you this morning."
    top = props[0]
    n = len(props)
    return f"{n} thing{'s' if n != 1 else ''} need{'' if n != 1 else 's'} you today, about ${sum(p['dollars_at_stake'] for p in props):,.0f} in play. Biggest: {top['title']}."


def _governor(ds: str) -> Governor:
    if ds not in _governors:
        g = Governor()
        lab = _lab()
        for a, b, c, d, e, f in lab.execute("select lesson_id, agent, scope_type, scope_id, effect, created_at from lessons where dataset=? and active", [ds]).fetchall():
            g.lessons.append(Lesson(a, b, c, d, json.loads(e), pd.Timestamp(f).date()))
        lab.close()
        _governors[ds] = g
    return _governors[ds]


class StepIn(BaseModel):
    dataset: str = "twin"
    to: str


class TenantIn(BaseModel):
    review_cost: float | None = None
    effectiveness: float | None = None
    budget_per_owner: int | None = None
    false_alarm_cost: float | None = None
    go_live: str | None = None
    shadow_days: int | None = None


@app.get("/api/tenant")
def tenant(dataset: str = "twin", as_of: str | None = None):
    _ds(dataset)
    cfg = rollout.load(dataset)
    d = as_of or str(w_default_date(dataset))
    return {"dataset": dataset, "economics": rollout.economics_of(cfg).describe(),
            "go_live": cfg["go_live"], "shadow_days": cfg["shadow_days"], "phase": rollout.phase(cfg, d), "learning_log": cfg["learning_log"][-20:],
            "retune_log": cfg["retune_log"][-20:]}


def w_default_date(dataset: str):
    lo, hi = scored_window(get_world(dataset))
    return (lo + (hi - lo) / 2).date()


@app.post("/api/tenant")
def tenant_update(body: TenantIn, dataset: str = "twin"):
    _ds(dataset)
    cfg = rollout.load(dataset)
    e = cfg["economics"]
    for k in ("review_cost", "effectiveness", "budget_per_owner", "false_alarm_cost"):
        v = getattr(body, k)
        if v is not None:
            e[k] = v
    if body.go_live is not None:
        cfg["go_live"] = body.go_live or None
    if body.shadow_days is not None:
        cfg["shadow_days"] = body.shadow_days
    cfg["economics"] = rollout.economics_of(cfg).describe()
    rollout.save(dataset, cfg)
    return tenant(dataset)


_CANDS: dict = {}


@app.get("/api/policy/frontier")
def policy_frontier(dataset: str = "twin", budgets: str = "3,5,8,12,20"):
    """What each alerts-per-owner budget buys, under the client's economics, on recorded candidates (no forecasting)."""
    _ds(dataset)
    from ..sim.policy_sim import load_candidates, simulate
    if dataset not in _CANDS:
        try:
            _CANDS[dataset] = load_candidates(f"cand_{dataset}.pkl")
        except FileNotFoundError:
            raise HTTPException(404, "No recorded candidates for this dataset yet. Run a candidate replay first.")
    econ = rollout.economics_of(rollout.load(dataset))
    w = get_world(dataset)
    rows = []
    for b in [int(x) for x in budgets.split(",") if x.strip()][:8]:
        e = alert_policy.Economics(review_cost=econ.review_cost, effectiveness=econ.effectiveness, budget_per_owner=b,
                                   false_alarm_cost=econ.false_alarm_cost, ranking_false_alarm_cost=econ.ranking_false_alarm_cost, agent_weight=econ.agent_weight)
        r = simulate(_CANDS[dataset], e, dataset, scored_window(w))
        rows.append({"budget_per_owner": b, "shown_per_day": r["shown_per_day"], "digest_per_day": r["digest_per_day"], "agents": r["agents"]})
    return {"dataset": dataset, "policy": econ.describe(), "rows": rows,
            "caveat": "Scored against stockouts and write-offs recorded in the data. Flagged is not the same as prevented."}


@app.get("/api/policy/whatif")
def policy_whatif(dataset: str = "twin", acceptance: float = 0.5, effectiveness: float | None = None):
    """What the alerts of the stored replay could be worth if people act on a share of them. Expected, not observed."""
    from ..policy.whatif import whatif
    _ds(dataset)
    if not 0.0 <= acceptance <= 1.0 or (effectiveness is not None and not 0.0 <= effectiveness <= 1.0):
        raise HTTPException(400, "acceptance and effectiveness are shares between 0 and 1")
    sc = _scorecard(dataset)
    if not sc.get("agents"):
        raise HTTPException(404, "No replay stored for this dataset yet.")
    econ = rollout.economics_of(rollout.load(dataset))
    out = whatif(sc["agents"], econ, acceptance, effectiveness)
    from ..policy.whatif import problem_level
    lab = _lab()
    rows = lab.execute("select agent, entity_id, cast(as_of as varchar), dollars_at_stake from proposals where dataset=?", [dataset]).fetchall()
    lab.close()
    pl = problem_level(rows)
    for a in out["agents"]:                      # distinct problems and their claimed dollars, beside the alert-level sum
        if a["agent"] in pl:
            a["problems_alerted"], a["claimed_dollars_per_problem"] = pl[a["agent"]]["problems"], pl[a["agent"]]["claimed_dollars"]
    return {"dataset": dataset, "window": sc.get("window"), **out}


@app.get("/api/readiness")
def readiness(agent: str, dataset: str = "twin", weeks: int = 13):
    _ds(dataset)
    if agent not in REGISTRY:
        raise HTTPException(404, "unknown agent")
    return rollout.readiness(dataset, agent, weeks)


@app.get("/api/drift")
def drift(dataset: str = "twin", as_of: str | None = None):
    _ds(dataset)
    return rollout.drift_report(dataset, as_of)


@app.post("/api/tenant/calibrate")
def tenant_calibrate(dataset: str = "twin"):
    _ds(dataset)
    try:
        return {"calibration": rollout.calibrate(dataset)}
    except FileNotFoundError:
        raise HTTPException(404, "No recorded candidates for this dataset yet. Run a candidate replay first.")


@app.post("/api/tenant/learn")
def tenant_learn(dataset: str = "twin"):
    _ds(dataset)
    lab = _lab()
    rows = lab.execute("select p.agent, d.decision, d.reason, p.entity_id, d.decision_id from decisions d join proposals p using(proposal_id) where p.dataset=? qualify row_number() over (partition by d.proposal_id order by d.decided_at desc) = 1", [dataset]).fetchall()
    lab.close()
    return {"changes": rollout.learn(dataset, rows), "decisions_seen": len(rows)}


@app.post("/api/time/step")
def time_step(body: StepIn):
    """Advance the clock to a date, running every enabled agent for each day passed and storing new proposals."""
    _ds(body.dataset)
    w = get_world(body.dataset); fs = ForecastService(w)
    lab = _lab()
    cur = lab.execute("select cursor_date from replay_state where dataset=?", [body.dataset]).fetchone()
    lab.close()
    start = pd.Timestamp(cur[0]) + pd.Timedelta(days=1) if cur else scored_window(w)[0]
    to = pd.Timestamp(body.to)
    if to < start:
        return {"cursor": str(start.date() - pd.Timedelta(days=1)), "new": []}
    if (to - start).days > 14:
        raise HTTPException(400, "step at most 14 days at a time")
    with _lock:
        gov = _governor(body.dataset)
        econ = rollout.economics_of(rollout.load(body.dataset))
        new = []
        for d in pd.date_range(start, to, freq="D"):
            day: list = []
            opened = alert_policy.agent_params(econ)
            for name in REGISTRY:
                cfg = _config(body.dataset, name)
                if not cfg["enabled"]:
                    continue
                params = {**cfg["params"], **opened.get(name, {})} if name in opened else cfg["params"]
                ps = gov.filter(REGISTRY[name].run(AgentContext(w, d, fs, params, gov.lessons)), d.date(), commit=False)
                for p in ps:
                    p.autonomy_level, p.owner = cfg["autonomy_level"], cfg["owner"] or p.owner
                day += ps
            shown, digest, below = alert_policy.select(day, econ)
            gov.commit(shown, d.date())
            new += shown + digest             # digested alerts are stored too: the briefing re-applies the budget when it reads
        lab = _lab()
        for p in new:
            lab.execute("delete from proposals where proposal_id=?", [p.proposal_id])
            lab.execute("insert into proposals values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        [p.proposal_id, "TT-live", body.dataset, p.agent, p.as_of, p.entity_type, p.entity_id, p.title, p.severity, p.dollars_at_stake,
                         p.confidence, p.autonomy_level, p.owner, json.dumps(p.evidence, default=str), json.dumps(p.action, default=str), "open",
                         p.dedupe_key, None, None, datetime.now(timezone.utc)])
        lab.execute("delete from replay_state where dataset=?", [body.dataset])
        lab.execute("insert into replay_state values (?,?,?,?)", [body.dataset, to.date(), 1.0, False])
        lab.close()
    return {"cursor": str(to.date()), "new": [_prop_row_from(p) for p in new]}


def _prop_row_from(p) -> dict:
    d = {"proposal_id": p.proposal_id, "agent": p.agent, "as_of": p.as_of, "entity_type": p.entity_type, "entity_id": p.entity_id, "title": p.title,
         "severity": p.severity, "dollars_at_stake": p.dollars_at_stake, "confidence": p.confidence, "autonomy_level": p.autonomy_level, "owner": p.owner,
         "evidence": p.evidence, "action": p.action, "status": "open"}
    d["narration_templated"] = templated.narrate(d)
    return d


@app.post("/api/time/reset")
def time_reset(dataset: str = "twin"):
    lab = _lab()
    lab.execute("delete from replay_state where dataset=?", [dataset])
    lab.execute("delete from proposals where dataset=? and run_id='TT-live'", [dataset])
    lab.close()
    _governors.pop(dataset, None)
    return {"ok": True}


@app.get("/api/time/state")
def time_state(dataset: str = "twin"):
    w = get_world(_ds(dataset))
    lo, hi = scored_window(w)
    lab = _lab()
    cur = lab.execute("select cursor_date from replay_state where dataset=?", [dataset]).fetchone()
    lab.close()
    return {"start": str(lo.date()), "end": str(hi.date()), "cursor": None if not cur else str(cur[0]), "first": str(w.first.date()), "last": str(w.last.date())}


# ---------------------------------------------------------------- reveal
@app.get("/api/reveal")
def reveal(codes: str):
    m = _reveal_map()
    return {c: m.get(c) for c in codes.split(",") if c}


# ---------------------------------------------------------------- static console
_DIST = Path(__file__).resolve().parents[3] / "console" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = _DIST / path
        return FileResponse(f if path and f.is_file() else _DIST / "index.html")
