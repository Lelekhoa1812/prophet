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
from prophet_agent.datasets import MAX_BYTES
from prophet_agent.nim.errors import NotConfigured
from prophet_agent.runtime.store import LeaseHeld, NotebookBusy
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
    dataset_ids: list[str] = Field(default_factory=list, max_length=10)


class AnswerIn(BaseModel):
    answers: dict[str, Any]


class NotebookIn(BaseModel):
    title: str = Field(default="New notebook", max_length=120)


class NotebookPatch(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    instructions: str | None = Field(default=None, max_length=2000)
    archived: bool | None = None


class SourcePatch(BaseModel):
    included: bool | None = None
    removed: bool | None = None


class TurnIn(BaseModel):
    goal: str = Field(min_length=3, max_length=2000)


def _history(args: dict) -> list[float]:
    """The series behind a forecast, for the chart: from the call's own numbers or from the analysis run it named."""
    from prophet_agent.tools.series_io import resolve_series
    try:
        _, arr = resolve_series({k: args[k] for k in ("values", "source") if k in args})[0]
    except (ValueError, KeyError, IndexError):
        return []
    return [float(v) for v in arr[-400:]]


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

    @app.post("/api/datasets", status_code=201)
    async def upload_dataset(request: Request, name: str = "data") -> dict:
        """Raw file bytes in the body, the file name in ?name=. Parsing happens later, in the sandbox."""
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BYTES:
            raise HTTPException(413, f"the file is larger than {MAX_BYTES // (1024 * 1024)} MB")
        data = await request.body()
        try:
            return service.datasets.add(name, data)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.get("/api/datasets")
    def list_datasets() -> list[dict]:
        return service.datasets.list()

    @app.get("/api/datasets/{dataset_id}")
    def get_dataset(dataset_id: str) -> dict:
        try:
            return service.datasets.get(dataset_id)
        except KeyError:
            raise HTTPException(404, "no such file")

    @app.get("/api/missions")
    def list_missions() -> list[dict]:
        return service.list()

    @app.post("/api/missions", status_code=201)
    def create_mission(body: MissionIn) -> dict:
        from prophet_agent.nim.transport import load_keys
        if service.brain_factory is default_brain_factory and not load_keys():
            # Refuse before writing a mission row, so no orphan "created" mission is left behind.
            raise HTTPException(503, "NVIDIA NIM keys are not configured; the planner cannot run")
        try:
            mission_id = service.create(body.goal.strip(), dataset_ids=body.dataset_ids)
        except KeyError:
            raise HTTPException(422, "one of the uploaded files no longer exists; upload it again")
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
                history = _history(args)
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

    # -- notebooks ----------------------------------------------------------------------------------------------

    def _nb(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError:
            raise HTTPException(404, "no such notebook item")

    @app.get("/api/notebooks")
    def list_notebooks() -> list[dict]:
        return service.notebooks.list()

    @app.post("/api/notebooks", status_code=201)
    def create_notebook(body: NotebookIn) -> dict:
        return service.notebooks.create(body.title.strip())

    @app.get("/api/notebooks/{nid}")
    def get_notebook(nid: str) -> dict:
        nb = _nb(service.notebooks.get, nid)
        return {**nb, "sources": service.notebooks.sources(nid), "tables": service.notebooks.tables(nid),
                "turns": service.notebooks.turns(nid)}

    @app.patch("/api/notebooks/{nid}")
    def patch_notebook(nid: str, body: NotebookPatch) -> dict:
        return _nb(service.notebooks.update, nid, title=body.title, instructions=body.instructions,
                   archived=body.archived)

    @app.post("/api/notebooks/{nid}/sources", status_code=201)
    async def add_source(nid: str, request: Request, name: str = "data") -> dict:
        """Raw file bytes in the body, the name in ?name=. Profiling runs in the background; poll the source."""
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BYTES:
            raise HTTPException(413, f"the file is larger than {MAX_BYTES // (1024 * 1024)} MB")
        data = await request.body()
        try:
            _nb(service.notebooks.get, nid)
            return service.notebooks.add_source(nid, name, data)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.patch("/api/notebooks/{nid}/sources/{sid}")
    def patch_source(nid: str, sid: str, body: SourcePatch) -> dict:
        return _nb(service.notebooks.set_source, nid, sid, included=body.included, removed=body.removed)

    @app.get("/api/notebooks/{nid}/tables")
    def list_tables(nid: str) -> list[dict]:
        _nb(service.notebooks.get, nid)
        return service.notebooks.tables(nid)

    @app.get("/api/notebooks/{nid}/tables/{name}/versions")
    def table_versions(nid: str, name: str) -> list[dict]:
        _nb(service.notebooks.get, nid)
        return service.notebooks.table_versions(nid, name)

    @app.get("/api/notebooks/{nid}/tables/{name}/download")
    def download_table(nid: str, name: str, version: int | None = None) -> FileResponse:
        try:
            path = service.notebooks.table_path(nid, name, version)
        except KeyError:
            raise HTTPException(404, "no such table or version")
        return FileResponse(path, media_type="text/csv", filename=f"{name}{'' if version is None else f'_v{version}'}.csv")

    @app.post("/api/notebooks/{nid}/turns", status_code=201)
    def create_turn(nid: str, body: TurnIn) -> dict:
        from prophet_agent.nim.transport import load_keys
        if service.brain_factory is default_brain_factory and not load_keys():
            raise HTTPException(503, "NVIDIA NIM keys are not configured; the planner cannot run")
        try:
            mission_id = service.create_turn(nid, body.goal.strip())
        except KeyError:
            raise HTTPException(404, "no such notebook")
        except NotebookBusy as exc:
            raise HTTPException(409, str(exc))
        try:
            service.start(mission_id)
        except NotConfigured:
            raise HTTPException(503, "NVIDIA NIM keys are not configured; the planner cannot run")
        service.notebooks.touch(nid)
        return {"id": mission_id, "seq": service.store.mission(mission_id)["seq"]}

    @app.get("/api/notebooks/{nid}/turns")
    def list_turns(nid: str) -> list[dict]:
        _nb(service.notebooks.get, nid)
        return service.notebooks.turns(nid)

    @app.post("/api/missions/{mission_id}/cancel")
    def cancel_mission(mission_id: str) -> dict:
        try:
            return {"status": service.cancel(mission_id)}
        except KeyError:
            raise HTTPException(404, "no such mission")

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
