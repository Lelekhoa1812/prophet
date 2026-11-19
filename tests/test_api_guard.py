"""The lab API must not be driven by a web page from another site, nor answer to a rebound hostname."""
import pytest
from fastapi.testclient import TestClient

from lab.api.main import app

c = TestClient(app, base_url="http://localhost:8000")


def test_foreign_origin_cannot_write():
    r = c.post("/api/time/reset", params={"dataset": "guard-test"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_cross_site_fetch_metadata_cannot_write():
    r = c.post("/api/time/reset", params={"dataset": "guard-test"}, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403


def test_rebound_host_is_refused_even_for_reads():
    r = TestClient(app, base_url="http://attacker.example").get("/api/reveal", params={"codes": "X"})
    assert r.status_code == 400


@pytest.mark.parametrize("origin", ["http://localhost:5173", "http://127.0.0.1:8000", "http://localhost:8000"])
def test_console_origins_still_work(origin):
    r = c.post("/api/time/reset", params={"dataset": "nope"}, headers={"Origin": origin})
    assert r.status_code not in (400, 403)


def test_no_origin_header_is_allowed():                       # curl and the proxy-less same-origin console
    r = c.get("/api/meta")
    assert r.status_code not in (400, 403)
