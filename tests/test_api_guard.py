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


@pytest.mark.parametrize("base", ["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:5173"])
def test_same_origin_console_still_works(base):                 # dev proxy keeps Host = the page's own host:port
    r = TestClient(app, base_url=base).post("/api/time/reset", params={"dataset": "guard-test"}, headers={"Origin": base, "Sec-Fetch-Site": "same-origin"})
    assert r.status_code == 200


def test_another_local_port_is_another_site():
    r = c.post("/api/time/reset", params={"dataset": "guard-test"}, headers={"Origin": "http://localhost:9999"})
    assert r.status_code == 403
    r = c.post("/api/time/reset", params={"dataset": "guard-test"}, headers={"Sec-Fetch-Site": "same-site"})
    assert r.status_code == 403


def test_allowed_hosts_env_accepts_ports_and_schemes(monkeypatch):
    monkeypatch.setenv("TIMESFM_ERP_ALLOWED_HOSTS", "erp.internal:8443, https://erp.example.com")
    assert TestClient(app, base_url="http://erp.internal:8443").get("/api/meta").status_code == 200
    assert TestClient(app, base_url="http://erp.example.com").get("/api/meta").status_code == 200


def test_no_origin_header_is_allowed():                       # curl and the proxy-less same-origin console
    r = c.get("/api/meta")
    assert r.status_code == 200
