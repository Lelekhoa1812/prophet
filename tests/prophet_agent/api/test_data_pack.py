"""Downloads for a session and a notebook: the uploaded file, every image, and one zip of all of it."""
from __future__ import annotations

import io
import zipfile

from fastapi.testclient import TestClient

from prophet_agent.api.app import create_app
from prophet_agent.api.data_pack import build_mission_pack, build_notebook_pack
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools

CSV = b"Date,Close\n2015-02-17,22.0\n"
PNG = b"\x89PNG\r\n\x1a\n"
RUN = "ab" * 16


class NoBrain:
    def decide(self, messages, schema, name):
        raise AssertionError("no planner call expected")


def _service(tmp_path):
    svc = MissionService(tmp_path / "p.sqlite", lambda: NoBrain(), default_tools())
    return svc


def _client(svc):
    return TestClient(create_app(svc), base_url="http://127.0.0.1:8000")


def _run_folder(root, run_id=RUN):
    run = root / run_id
    (run / "in").mkdir(parents=True)
    (run / "out").mkdir()
    (run / "in" / "NVDA_daily.csv").write_bytes(CSV)
    (run / "out" / "chart.png").write_bytes(PNG)
    (run / "out" / "chart.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    (run / "lib").mkdir()
    (run / "lib" / "prophet_lib.py").write_text("x = 1\n", encoding="utf-8")
    return run


def test_a_session_zip_holds_the_upload_and_every_image(tmp_path):
    svc = _service(tmp_path)
    meta = svc.datasets.add("NVDA_daily.csv", CSV)
    mid = svc.create("forecast the close", dataset_ids=[meta["id"]])
    svc.store.set_status(mid, "completed")
    sandbox = tmp_path / "sandbox"
    _run_folder(sandbox)
    svc.store.append_event(mid, "tool.result", {"run_id": RUN})

    data, name = build_mission_pack(svc, mid, sandbox)
    assert name.endswith(".zip")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
    assert "files/NVDA_daily.csv" in names
    assert f"runs/{RUN}/in/NVDA_daily.csv" in names
    assert f"runs/{RUN}/out/chart.png" in names
    assert f"runs/{RUN}/out/chart.svg" in names
    assert f"runs/{RUN}/lib/prophet_lib.py" not in names
    svc.store.close()


def test_a_notebook_zip_and_the_file_route_return_the_upload(tmp_path):
    svc = _service(tmp_path)
    client = _client(svc)
    nid = svc.notebooks.create("Prices")["id"]
    src = svc.notebooks.add_source(nid, "NVDA_daily.csv", CSV, profile_async=False)
    sandbox = tmp_path / "sandbox"
    _run_folder(sandbox)
    turn = svc.create_turn(nid, "what is the close")
    svc.store.set_status(turn, "completed")
    svc.store.append_event(turn, "tool.result", {"run_id": RUN})

    packed, _ = build_notebook_pack(svc, nid, sandbox)
    with zipfile.ZipFile(io.BytesIO(packed)) as zf:
        names = set(zf.namelist())
        assert zf.read("files/NVDA_daily.csv") == CSV
    assert "files/NVDA_daily.csv" in names
    assert any(n.endswith("chart.png") for n in names)
    assert any(n.endswith("chart.svg") for n in names)

    got = client.get(f"/api/datasets/{src['dataset_id']}/file")
    assert got.status_code == 200
    assert got.content == CSV
    http_zip = client.get(f"/api/notebooks/{nid}/data")
    assert http_zip.status_code == 200
    assert http_zip.headers["content-type"].startswith("application/zip")
    session = client.get(f"/api/missions/{turn}/data")
    assert session.status_code == 200
    with zipfile.ZipFile(io.BytesIO(session.content)) as zf:
        assert "files/NVDA_daily.csv" in zf.namelist()
    assert client.get("/api/datasets/nope/file").status_code == 404
    assert client.get("/api/notebooks/nope/data").status_code == 404
    svc.store.close()
