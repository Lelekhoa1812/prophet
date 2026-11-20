"""HTTP API for the Prophet console. Bind to 127.0.0.1 only; this is a local R&D tool with no login.

Motivation vs Logic
Motivation: the console starts missions, follows their progress, answers questions and downloads the files they made.
Logic: thin routes over MissionService. Writes must come from the same origin (or no origin, for scripts), hosts are
restricted to loopback names, and artefact downloads are served only from a run's ``out/`` folder, after the run id is
checked to be a hex id and the file name is checked to stay inside that folder. Without NIM keys the planner cannot
run, so mission creation returns 503 rather than running a stub that would look like real output.
"""
from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from prophet_agent.config import ROOT
from prophet_agent.nim.errors import NotConfigured
from prophet_agent.runtime.store import LeaseHeld
from prophet_agent.sandbox.runner import SANDBOX_ROOT, OUT_DIR

from .service import MissionService

ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]
LOOPBACK_NAMES = {"127.0.0.1", "localhost"}
RUN_ID = re.compile(r"^[0-9a-f]{32}$")


def _loopback_origin(origin: str) -> bool:
    """True for an http origin on a loopback name, on any port (the dev server may move off its default port)."""
    parts = urlsplit(origin)
    return parts.scheme == "http" and parts.hostname in LOOPBACK_NAMES
WEB_DIST = ROOT / "web" / "dist"


class MissionIn(BaseModel):
    goal: str = Field(min_length=3, max_length=2000)


class AnswerIn(BaseModel):
    answers: dict[str, Any]


def create_app(service: MissionService) -> FastAPI:
    app = FastAPI(title="Prophet", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
    app.state.service = service

    @app.middleware("http")
    async def same_origin_writes(request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            origin = request.headers.get("origin")
            if origin is not None and not _loopback_origin(origin):
                return JSONResponse({"detail": "cross-origin write refused"}, status_code=403)
        return await call_next(request)

    @app.get("/api/health")
    def health() -> dict:
        from prophet_agent.nim.transport import load_keys
        return {"ok": True, "nim_keys_configured": len(load_keys()) > 0}

    @app.get("/api/tools")
    def list_tools() -> list[dict]:
        """What the agent can do, in words a person (or another agent) can read. Code names stay out of the UI."""
        return [{"name": t.name, "title": t.title or t.name, "activity": t.activity or t.title or t.name,
                 "description": t.doc} for t in service.tools.values()]

    @app.get("/api/missions")
    def list_missions() -> list[dict]:
        return service.list()

    @app.post("/api/missions", status_code=201)
    def create_mission(body: MissionIn) -> dict:
        from prophet_agent.nim.transport import load_keys
        if service.brain_factory is default_brain_factory and not load_keys():
            # Refuse before writing a mission row, so no orphan "created" mission is left behind.
            raise HTTPException(503, "NVIDIA NIM keys are not configured; the planner cannot run")
        mission_id = service.create(body.goal.strip())
        try:
            service.start(mission_id)
        except NotConfigured:
            raise HTTPException(503, "NVIDIA NIM keys are not configured; the planner cannot run")
        return {"id": mission_id}

    @app.get("/api/missions/{mission_id}")
    def get_mission(mission_id: str, after: int = 0) -> dict:
        try:
            return service.snapshot(mission_id, after)
        except KeyError:
            raise HTTPException(404, "no such mission")

    @app.post("/api/questions/{question_id}/answer", status_code=202)
    def answer_question(question_id: str, body: AnswerIn) -> dict:
        if not body.answers or not all(isinstance(k, str) and len(k) <= 12 for k in body.answers):
            raise HTTPException(422, "answers must be a non-empty object keyed by question header")
        try:
            mission_id = service.answer(question_id, body.answers)
        except KeyError:
            raise HTTPException(404, "no such question")
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        except LeaseHeld:
            raise HTTPException(409, "mission is already running")
        except NotConfigured:
            raise HTTPException(503, "NVIDIA NIM keys are not configured; the planner cannot run")
        return {"mission_id": mission_id}

    @app.get("/api/missions/{mission_id}/artifacts")
    def list_artifacts(mission_id: str) -> list[dict]:
        rows = service.store.conn.execute(
            "SELECT result_json, args_json FROM tool_calls "
            "WHERE mission_id=? AND tool='forecast_report' AND status='done' ORDER BY step_idx", (mission_id,)).fetchall()
        out = []
        for r in rows:
            res = json.loads(r["result_json"] or "null") or {}
            run_id = res.get("run_id")
            if run_id:
                fc = res.get("forecast") or {}
                q = fc.get("quantiles") or {}
                args = json.loads(r["args_json"] or "null") or {}
                values = args.get("values")
                history = [float(v) for v in values[-400:]] if isinstance(values, list) else []
                out.append({"run_id": run_id, "files": res.get("files", []), "history": history,
                            "forecast": {"config": fc.get("config"), "point": fc.get("point"),
                                         "q10": q.get("0.1"), "q50": q.get("0.5"), "q90": q.get("0.9")}})
        return out

    @app.get("/api/artifacts/{run_id}/{name:path}")
    def get_artifact(run_id: str, name: str) -> FileResponse:
        if not RUN_ID.match(run_id):
            raise HTTPException(404)
        base = (SANDBOX_ROOT / run_id / OUT_DIR).resolve()
        target = (base / name).resolve()
        if not str(target).startswith(str(base) + "/") or not target.is_file():
            raise HTTPException(404)
        return FileResponse(target, filename=target.name)

    if WEB_DIST.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")

    return app


def default_brain_factory():
    """Build the NIM-backed planner, or raise NotConfigured when no keys are present."""
    from prophet_agent.nim.transport import NimClient
    from prophet_agent.runtime.goal import NimBrain
    return NimBrain(NimClient.from_env())


__all__ = ["create_app", "default_brain_factory"]
