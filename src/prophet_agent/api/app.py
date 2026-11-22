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
from urllib.parse import unquote, urlsplit

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from prophet_agent import mentions
from prophet_agent.config import ROOT
from prophet_agent.datasets import MAX_BYTES
from prophet_agent.nim.errors import NotConfigured
from prophet_agent.nim.providers import CATALOG, PROVIDERS, planner_problem
from prophet_agent.nim.transport import probe_provider, shared_client
from prophet_agent.runtime.store import ACTIVE, LeaseHeld, NotebookBusy
from prophet_agent.settings import view as settings_view
from prophet_agent.sandbox.runner import SANDBOX_ROOT, OUT_DIR
from prophet_agent.usage import ledger_for

from .bundle import build_bundle
from .panels import entity_page, panel_summary
from .service import MissionService
from .sharing import PolicyRefused, normalise_patch

ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]
LOOPBACK_NAMES = {"127.0.0.1", "localhost"}
RUN_ID = re.compile(r"^[0-9a-f]{32}$")


def _loopback_origin(origin: str) -> bool:
    """True for an http origin on a loopback name, on any port (the dev server may move off its default port)."""
    parts = urlsplit(origin)
    return parts.scheme == "http" and parts.hostname in LOOPBACK_NAMES
WEB_DIST = ROOT / "web" / "dist"


# The message text the person sees is capped at 2000 characters. A skill, tool or file picked with / or @ is stored as a
# link (see prophet_agent.mentions), which is longer than the words it shows, so the stored text may be longer.
GOAL_VISIBLE_MAX = 2000
GOAL_STORED_MAX = 6000


def _checked_goal(goal: str) -> str:
    text = goal.strip()
    shown = mentions.plain(text).strip()
    if len(shown) < 3:
        raise HTTPException(422, "the message is too short")
    if len(shown) > GOAL_VISIBLE_MAX:
        raise HTTPException(422, f"the message is {len(shown)} characters; the limit is {GOAL_VISIBLE_MAX}")
    return text


class MissionIn(BaseModel):
    goal: str = Field(min_length=3, max_length=GOAL_STORED_MAX)
    dataset_ids: list[str] = Field(default_factory=list, max_length=10)
    model: str | None = Field(default=None, max_length=120)     # a catalog model id; omitted means the default chain


class MissionPatch(BaseModel):
    archived: bool | None = None
    title: str | None = Field(default=None, max_length=60)
    pinned: bool | None = None


class AnswerIn(BaseModel):
    answers: dict[str, Any]


class NotebookIn(BaseModel):
    title: str = Field(default="New notebook", max_length=120)


class NotebookPatch(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    instructions: str | None = Field(default=None, max_length=2000)
    archived: bool | None = None
    pinned: bool | None = None


class SourcePatch(BaseModel):
    included: bool | None = None
    removed: bool | None = None


class TurnIn(BaseModel):
    goal: str = Field(min_length=3, max_length=GOAL_STORED_MAX)
    model: str | None = Field(default=None, max_length=120)     # a catalog model id; omitted means the default chain


DISPLAY_EXTS = (".png", ".svg", ".xlsx", ".docx", ".pdf")


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

    @app.get("/api/skills")
    def list_skill_playbooks() -> list[dict]:
        """The analysis playbooks a person can invoke with "/": name, a readable title and the one-line summary."""
        from prophet_agent.skills import list_skills

        return [{"name": s["name"], "title": mentions.humanise(s["name"]), "summary": s["summary"]} for s in list_skills()]

    @app.get("/api/skills/{name}")
    def get_skill_playbook(name: str) -> dict:
        from prophet_agent.skills import list_skills, read_skill

        summary = {s["name"]: s["summary"] for s in list_skills()}
        if name not in summary:
            raise HTTPException(404, "no such skill")
        return {"name": name, "title": mentions.humanise(name), "summary": summary[name], "body": read_skill(name)}

    @app.post("/api/datasets", status_code=201)
    async def upload_dataset(request: Request, name: str = "data") -> dict:
        """Raw file bytes in the body; the name in the X-File-Name header (or ?name= for scripts). Parsing happens later."""
        from urllib.parse import unquote
        if request.headers.get("x-file-name"):
            name = unquote(request.headers["x-file-name"])
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

    @app.patch("/api/missions/{mission_id}")
    def patch_mission(mission_id: str, body: MissionPatch) -> dict:
        try:
            if body.archived is not None:
                service.archive(mission_id, body.archived)
            if body.title is not None:
                service.set_title(mission_id, body.title)
            if body.pinned is not None:
                service.set_pinned(mission_id, body.pinned)
        except KeyError:
            raise HTTPException(404, "no such quick forecast")
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        changed = body.model_dump(exclude_none=True)
        return {"id": mission_id, **changed}

    @app.delete("/api/missions/{mission_id}")
    def delete_mission(mission_id: str) -> dict:
        try:
            service.delete(mission_id)
        except KeyError:
            raise HTTPException(404, "no such quick forecast")
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        return {"id": mission_id, "deleted": True}

    def _model_gate(model: str | None) -> None:
        """422 for a model id that is not in the catalog, or whose model or provider is switched off in Settings."""
        if model is None:
            return
        try:
            service.settings.check_model_choice(model)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    def _planner_gate(model: str | None) -> None:
        """503 while the planner cannot run (no usable key, no endpoint). Checked before any row is written."""
        if service.brain_factory is default_brain_factory:
            problem = service.settings.planner_problem(model=model)
            if problem:
                raise HTTPException(503, problem)

    @app.post("/api/missions", status_code=201)
    def create_mission(body: MissionIn) -> dict:
        _model_gate(body.model)
        _planner_gate(body.model)     # refused before a mission row exists, so no orphan "created" mission is left behind
        try:
            mission_id = service.create(_checked_goal(body.goal), dataset_ids=body.dataset_ids, model=body.model)
        except KeyError:
            raise HTTPException(422, "one of the uploaded files no longer exists; upload it again")
        try:
            service.start(mission_id)
        except NotConfigured:
            raise HTTPException(503, planner_problem() or "the planner is not configured")
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
            raise HTTPException(503, planner_problem() or "the planner is not configured")
        return {"mission_id": mission_id}

    @app.get("/api/missions/{mission_id}/artifacts")
    def list_artifacts(mission_id: str) -> list[dict]:
        rows = service.store.conn.execute(
            "SELECT tool, result_json, args_json FROM tool_calls "
            "WHERE mission_id=? AND tool IN ('forecast_report','sandbox_python') AND status='done' ORDER BY step_idx", (mission_id,)).fetchall()
        out = []
        for r in rows:
            res = json.loads(r["result_json"] or "null") or {}
            run_id = res.get("run_id")
            if not run_id:
                continue
            if r["tool"] == "sandbox_python":
                # Analysis runs show up only when they made something a person opens: a chart or a document.
                files = [{"path": o["path"], "sha256": o["sha256"], "bytes": o["bytes"]} for o in res.get("outputs", [])
                         if isinstance(o, dict) and o.get("path", "").startswith("out/")]
                if any(f["path"].lower().endswith(DISPLAY_EXTS) for f in files):
                    out.append({"run_id": run_id, "files": files, "history": [], "forecast": None})
                continue
            fc = res.get("forecast") or {}
            q = fc.get("quantiles") or {}
            args = json.loads(r["args_json"] or "null") or {}
            out.append({"run_id": run_id, "files": res.get("files", []), "history": _history(args),
                        "forecast": {"config": fc.get("config"), "point": fc.get("point"),
                                     "q10": q.get("0.1"), "q50": q.get("0.5"), "q90": q.get("0.9")}})
        return out

    def _panel_runs(mission_id: str) -> list[tuple[int, dict]]:
        try:
            service.store.mission(mission_id)
        except KeyError:
            raise HTTPException(404, "no such mission")
        with service.store.lock:
            rows = service.store.conn.execute(
                "SELECT step_idx, result_json FROM tool_calls WHERE mission_id=? AND tool='panel_backtest' AND status='done' "
                "ORDER BY step_idx", (mission_id,)).fetchall()
        return [(int(r["step_idx"]), json.loads(r["result_json"] or "null") or {}) for r in rows]

    @app.get("/api/missions/{mission_id}/panels")
    def list_panels(mission_id: str) -> list[dict]:
        """The panel backtests a mission ran: size, models and overall scores. The entities are paged separately."""
        return [panel_summary(step, res) for step, res in _panel_runs(mission_id)]

    @app.get("/api/missions/{mission_id}/panels/{step}/entities")
    def panel_entities(mission_id: str, step: int, page: int = 1, page_size: int = 25, sort: str = "entity",
                       order: str = "asc") -> dict:
        """One page of a panel's entities: best model and MASE per model. Sort by entity, best, or mase:<model>."""
        match = [res for s_idx, res in _panel_runs(mission_id) if s_idx == step]
        if not match:
            raise HTTPException(404, "no such panel run in this mission")
        try:
            return entity_page(match[0], page=page, page_size=page_size, sort=sort, order=order)
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.get("/api/missions/{mission_id}/repro")
    def reproducibility_bundle(mission_id: str) -> Response:
        """A zip a person can keep and check: code cells, inputs and outputs (sha256), ledger, configuration and answer.
        Read-only. Never includes .env or key-shaped text."""
        try:
            data = build_bundle(service.store, mission_id, SANDBOX_ROOT)
        except KeyError:
            raise HTTPException(404, "no such mission")
        return Response(data, media_type="application/zip",
                        headers={"content-disposition": f'attachment; filename="prophet-{mission_id[:12]}-repro.zip"'})

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
        if body.pinned is not None:
            _nb(service.notebooks.set_pinned, nid, body.pinned)
        return _nb(service.notebooks.update, nid, title=body.title, instructions=body.instructions,
                   archived=body.archived)

    @app.delete("/api/notebooks/{nid}")
    def delete_notebook(nid: str) -> dict:
        try:
            _nb(service.notebooks.delete, nid)
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        return {"id": nid, "deleted": True}

    @app.post("/api/notebooks/{nid}/sources", status_code=201)
    async def add_source(nid: str, request: Request, name: str = "data") -> dict:
        """Raw file bytes in the body; the name in the X-File-Name header (or ?name= for scripts). Profiling runs in the
        background; poll the source."""
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BYTES:
            raise HTTPException(413, f"the file is larger than {MAX_BYTES // (1024 * 1024)} MB")
        data = await request.body()
        header_name = request.headers.get("x-file-name")
        if header_name:
            name = unquote(header_name)
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

    def _provider_states() -> list[dict]:
        return [{"name": name, "label": p.label, "enabled": service.settings.provider_enabled(name),
                 "ready": service.settings.provider_ready(name)} for name, p in PROVIDERS.items()]

    @app.get("/api/notebooks/{nid}/sharing")
    def get_sharing(nid: str) -> dict:
        """What this notebook may share: web research on or off, and the providers that may receive its data (null: all).
        ``available`` lists every provider with whether it is turned on and ready. Defaults keep everything on."""
        try:
            sharing = service.get_sharing(nid)
        except KeyError:
            raise HTTPException(404, "no such notebook")
        return {**sharing, "available": _provider_states()}

    @app.put("/api/notebooks/{nid}/sharing")
    def put_sharing(nid: str, body: dict = Body(...)) -> dict:
        """A partial update of the notebook's sharing. Only the fields sent change; ``providers: null`` allows every provider."""
        try:
            patch = normalise_patch(body)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        try:
            sharing = service.set_sharing(nid, patch)
        except KeyError:
            raise HTTPException(404, "no such notebook")
        return {**sharing, "available": _provider_states()}

    @app.post("/api/notebooks/{nid}/turns", status_code=201)
    def create_turn(nid: str, body: TurnIn) -> dict:
        _model_gate(body.model)
        _planner_gate(body.model)
        try:
            mission_id = service.create_turn(nid, _checked_goal(body.goal), model=body.model)
        except KeyError:
            raise HTTPException(404, "no such notebook")
        except NotebookBusy as exc:
            raise HTTPException(409, str(exc))
        except PolicyRefused as exc:
            raise HTTPException(409, str(exc))
        try:
            service.start(mission_id)
        except NotConfigured:
            raise HTTPException(503, planner_problem() or "the planner is not configured")
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

    @app.post("/api/missions/{mission_id}/resume", status_code=202)
    def resume_mission(mission_id: str) -> dict:
        """Run an interrupted mission again. Completed steps and tool calls are read back from the ledger, not repeated;
        a side-effecting call left in progress still stops the mission for review rather than running twice."""
        try:
            mission = service.store.mission(mission_id)
        except KeyError:
            raise HTTPException(404, "no such mission")
        if mission["status"] != "interrupted":
            raise HTTPException(409, "only an interrupted mission can be resumed")
        try:
            service.start(mission_id)
        except LeaseHeld:
            raise HTTPException(409, "this mission is already running")
        except PolicyRefused as exc:
            raise HTTPException(409, str(exc))
        except NotConfigured:
            raise HTTPException(503, planner_problem() or "the planner is not configured")
        return {"id": mission_id, "status": "running"}

    def _checkpoint_action(mission_id: str, action) -> dict:
        try:
            action(mission_id)
        except KeyError:
            raise HTTPException(404, "no such mission")
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        except LeaseHeld:
            raise HTTPException(409, "this mission is already running")
        except NotConfigured:
            raise HTTPException(503, planner_problem() or "the planner is not configured")
        return {"id": mission_id, "status": "running"}

    @app.post("/api/missions/{mission_id}/continue", status_code=202)
    def continue_mission(mission_id: str) -> dict:
        """Start the next part of a checkpointed mission with a fresh tool allowance. Completed steps replay from the ledger."""
        return _checkpoint_action(mission_id, service.continue_mission)

    @app.post("/api/missions/{mission_id}/answer-now", status_code=202)
    def answer_now(mission_id: str) -> dict:
        """Write the best answer from the evidence gathered so far, labelled partial."""
        return _checkpoint_action(mission_id, service.answer_now)

    # -- settings ------------------------------------------------------------------------------------------------

    def _pool_status() -> dict | None:
        try:
            return shared_client().status()
        except NotConfigured:
            return None

    @app.get("/api/settings")
    def get_settings() -> dict:
        return settings_view(service.settings, service.store, service.datasets, _pool_status())

    @app.put("/api/settings")
    def put_settings(body: dict = Body(...)) -> dict:
        """A partial update. ``nim.keys`` replaces the key list; a masked entry keeps its stored key. Limits are bounded."""
        patch = dict(body)
        nim = patch.pop("nim", None)
        providers = patch.pop("providers", None)
        if nim is not None and (not isinstance(nim, dict) or set(nim) - {"keys"}):
            raise HTTPException(422, "nim accepts only keys")
        if providers is not None and not isinstance(providers, dict):
            raise HTTPException(422, "providers must map a provider name to {keys: [...]}")
        try:
            service.settings.update(patch)
            if nim is not None and "keys" in nim:
                service.settings.set_keys(nim["keys"])
            for name, entry in (providers or {}).items():
                if name not in PROVIDERS or not isinstance(entry, dict) or set(entry) - {"keys"}:
                    raise ValueError(f"providers.{name} accepts only keys for a known provider")
                if "keys" in entry:
                    service.settings.set_keys(entry["keys"], provider=name)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return get_settings()

    @app.post("/api/settings/test-keys")
    def test_keys(provider: str = "nim") -> dict:
        """One 1-token call per key of one provider. Each key's status (valid, invalid or untested) and HTTP code are
        saved in settings.json, so the planner stops using an invalid key. No key material and no generated text."""
        if provider not in PROVIDERS:
            raise HTTPException(422, f"unknown provider {provider!r}")
        try:
            rows = probe_provider(provider)
        except NotConfigured as exc:
            raise HTTPException(503, str(exc))
        return {"provider": provider, "keys": service.settings.save_key_checks(provider, rows)}

    @app.get("/api/usage")
    def get_usage() -> dict:
        """Tokens used per model over the last 1, 7 and 30 days, with the ceilings set in Settings and when each window
        frees up again. Every catalog model is listed, whether or not it is switched on."""
        ledger = ledger_for(service.settings.data_root)
        budgets = service.settings.budgets()
        return {"models": ledger.summary(CATALOG, budgets)}

    @app.get("/api/settings/export")
    def export_data() -> JSONResponse:
        data = service.settings.export(service.store, service.datasets)
        return JSONResponse(data, headers={"content-disposition": 'attachment; filename="prophet-export.json"'})

    @app.post("/api/settings/clear-data")
    def clear_data(body: dict = Body(...)) -> dict:
        """Delete missions, notebooks, uploads and sandbox output. Settings and keys are kept. Needs the exact phrase."""
        try:
            removed = service.settings.clear(service.store, ACTIVE, str(body.get("confirm", "")))
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {"removed": removed}

    if WEB_DIST.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")

    return app


def default_brain_factory(model: str | None = None):
    """Build the planner for the chosen model, or for the default chain (the planner's provider and its enabled models).
    Raises NotConfigured when nothing can plan, and ValueError for an unknown or switched-off model."""
    from prophet_agent.nim.transport import shared_client
    from prophet_agent.runtime.goal import NimBrain
    from prophet_agent.settings import Settings
    provider, models = Settings().planner_choice(model=model)
    return NimBrain(shared_client(provider), models=list(models))


__all__ = ["create_app", "default_brain_factory"]
