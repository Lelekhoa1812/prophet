"""Panel GISTEMP round 2 and NEM round 1 (2026-11-26): a series landed by a download tool cited its URL, and the provenance
check refused it because only web research wrote fetch records. A successful download now writes one."""
import httpx
import pytest

from prophet_agent.opendata import fetch
from prophet_agent.research import netguard
from prophet_agent.tools.series_io import check_provenance

URL = "https://data.example.org/gistemp/GLB.Ts+dSST.csv"


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(netguard, "_default_resolver", lambda host, port: ["93.184.216.34"])


def _serve(monkeypatch, status=200):
    handler = lambda request: httpx.Response(status, headers={"content-type": "text/csv"}, content=b"year,v\n2026,1.2\n")  # noqa: E731
    monkeypatch.setattr(fetch, "open_client", lambda timeout_s: httpx.Client(transport=httpx.MockTransport(handler),
                                                                             follow_redirects=False))


def _prov():
    return {"origin": "retrieved", "source_name": "NASA GISTEMP v4", "sources": [{"id": "S1", "url": URL}]}


def test_a_downloaded_url_can_be_cited_as_a_source(monkeypatch):
    with pytest.raises(ValueError, match="no fetch record"):
        check_provenance(_prov())
    _serve(monkeypatch)
    fetch.download(URL)
    assert check_provenance(_prov())["origin"] == "retrieved"


def test_a_failed_download_writes_no_record(monkeypatch):
    _serve(monkeypatch, status=404)
    with pytest.raises(fetch.DownloadError):
        fetch.download(URL)
    with pytest.raises(ValueError, match="download_web_data"):
        check_provenance(_prov())


def test_a_source_url_with_part_of_its_query_dropped_still_finds_the_download(monkeypatch):
    """Panel round 3 NVDA (2026-11-26): the series cited the Nasdaq history URL without its date and limit parameters."""
    full = "https://api.nasdaq.com/api/quote/NVDA/historical?assetclass=stocks&fromdate=2024-01-01&limit=9999"
    fetch.record_download(full, full, b"date,close\n", "2026-11-26T00:00:00Z")
    short = {"origin": "retrieved", "source_name": "Nasdaq", "sources": [
        {"id": "S1", "url": "https://api.nasdaq.com/api/quote/NVDA/historical?assetclass=stocks"}]}
    assert check_provenance(short)["origin"] == "retrieved"
    other = {"origin": "retrieved", "source_name": "Nasdaq", "sources": [
        {"id": "S1", "url": "https://api.nasdaq.com/api/quote/AMD/historical?assetclass=stocks"}]}
    with pytest.raises(ValueError, match="no fetch record"):
        check_provenance(other)
