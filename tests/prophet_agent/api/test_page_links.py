"""The app's own addresses (/n/<id> notebook, /c/<id> quick forecast) serve the built page; other missing paths stay 404."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from prophet_agent.api import app as app_module, titles
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(titles, "name_in_background", lambda *a, **k: None)
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>page</title>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    monkeypatch.setattr(app_module, "WEB_DIST", dist)
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    yield TestClient(app_module.create_app(service), base_url="http://127.0.0.1:8000")
    service.close()


@pytest.mark.parametrize("path", ["/", "/n/" + "a" * 32, "/c/" + "b" * 32, "/n/" + "a" * 32 + "/"])
def test_the_page_opens_at_every_address_of_the_app(client, path):
    r = client.get(path)
    assert r.status_code == 200 and "<title>page</title>" in r.text


def test_files_still_come_from_the_bundle_and_other_paths_are_404(client):
    assert client.get("/assets/app.js").text == "console.log(1)"
    assert client.get("/assets/missing.js").status_code == 404
    assert client.get("/x/" + "a" * 32).status_code == 404
    assert client.get("/n/bad id").status_code == 404
    assert client.get("/api/nope").status_code == 404
