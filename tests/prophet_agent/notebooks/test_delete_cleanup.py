"""Deleting a notebook or quick forecast removes the files it owned: run folders, notebook folder, and datasets nothing else uses.

The sandbox runs for real (a child process); nothing calls NIM or TimesFM.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.sandbox.runner import SANDBOX_ROOT, run_python
from prophet_agent.tools import default_tools

CSV = b"date,units,region\n2026-01-05,10,North\n2026-01-12,12,North\n2026-01-19,9,South\n"


class NoBrain:
    def decide(self, messages, schema, name):
        raise AssertionError("no planner call expected in these tests")


@pytest.fixture()
def service(tmp_path):
    svc = MissionService(tmp_path / "p.sqlite", lambda: NoBrain(), default_tools(), tools_factory=default_tools)
    yield svc
    svc.store.close()


def _run_with_output(name: str, text: str) -> str:
    res = run_python(f"import pathlib; pathlib.Path('out/{name}').write_text({text!r}, encoding='utf-8')", timeout_s=30)
    assert res.ok, res.stderr
    return res.run_id


def _finish(service, mission_id: str) -> None:
    """A finished mission (cancelled here, as a person would stop one) can be deleted; a created one cannot."""
    service.store.set_status(mission_id, "cancelled")


def _dataset_exists(service, did: str) -> bool:
    try:
        service.datasets.get(did)
        return True
    except KeyError:
        return False


def test_notebook_delete_removes_its_files_and_orphan_dataset(service):
    nid = service.notebooks.create("Sales")["id"]
    src = service.notebooks.add_source(nid, "sales.csv", CSV, profile_async=False)
    service.notebooks.profile_source(src["id"])
    profile_run = service.notebooks.source(src["id"])["run_id"]
    assert profile_run and (SANDBOX_ROOT / profile_run).is_dir()

    turn = service.create_turn(nid, "forecast units")
    _finish(service, turn)
    turn_run = _run_with_output("table.csv", "a,b\n1,2\n")
    service.store.append_event(turn, "tool.result", {"run_id": turn_run})

    service.notebooks.promote(nid, turn_run, "table.csv", "clean")
    assert (service.notebooks.root / nid).is_dir()

    service.notebooks.delete(nid)

    assert not (service.notebooks.root / nid).exists()
    assert not (SANDBOX_ROOT / profile_run).exists()
    assert not (SANDBOX_ROOT / turn_run).exists()
    assert not _dataset_exists(service, src["dataset_id"])


def test_notebook_delete_keeps_dataset_a_forecast_still_uses(service):
    nid = service.notebooks.create("Sales")["id"]
    src = service.notebooks.add_source(nid, "sales.csv", CSV, profile_async=False)
    shared = src["dataset_id"]
    forecast = service.create("forecast units", dataset_ids=[shared])
    _finish(service, forecast)

    service.notebooks.delete(nid)
    assert _dataset_exists(service, shared)

    service.delete(forecast)
    assert not _dataset_exists(service, shared)


def test_forecast_delete_removes_run_folder_and_unused_attachment(service):
    attached = service.datasets.add("notes.csv", CSV)["id"]
    mission = service.create("forecast units", dataset_ids=[attached])
    _finish(service, mission)
    run = _run_with_output("series.json", "{}")
    service.store.append_event(mission, "tool.result", {"run_id": run})
    assert (SANDBOX_ROOT / run).is_dir()

    service.delete(mission)

    assert not (SANDBOX_ROOT / run).exists()
    assert not _dataset_exists(service, attached)
    with pytest.raises(KeyError):
        service.store.mission(mission)


def test_deleted_forecast_outputs_are_no_longer_served(service):
    mission = service.create("forecast units")
    _finish(service, mission)
    run = _run_with_output("report.txt", "figures")
    service.store.append_event(mission, "tool.result", {"run_id": run})
    client = TestClient(create_app(service))
    url = f"/api/artifacts/{run}/report.txt"
    assert client.get(url).status_code == 200

    service.delete(mission)

    assert client.get(url).status_code == 404
