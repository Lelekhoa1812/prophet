"""CSV is text and is read as text. In the FinRes-20261124-100411 record the FRED, CBOE and datahub CSV files answered
HTTP 200 with text/csv or application/csv and were refused as "not text", so no daily close series reached the planner.
The body is kept as written, not extracted as an article."""
import httpx
import pytest

from prophet_agent.research.fetch import fetch_page
from prophet_agent.research.netguard import HostThrottle, Robots

PUBLIC = "93.184.216.34"
CSV = b"DATE,SP500\n2026-11-20,6800.10\n2026-11-23,6850.40\n"


def public_resolver(host, port):
    return [PUBLIC]


@pytest.mark.parametrize("content_type", ["text/csv", "text/csv; charset=utf-8", "application/csv"])
def test_a_csv_response_is_read_as_plain_text(tmp_path, monkeypatch, content_type):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, headers={"content-type": content_type}, content=CSV)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    page = fetch_page("https://data.example.com/fredgraph.csv", client=client, resolver=public_resolver,
                      robots=Robots(client=client, resolver=public_resolver), throttle=HostThrottle(0.0),
                      use_cache=False)
    assert page.ok is True, page.note
    assert page.text == CSV.decode().strip()
