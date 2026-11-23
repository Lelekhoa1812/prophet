"""Open-data planner tools: listing, guarded downloads, the mission's files and the sandbox reading them in the same run.
The network is replaced (fetch.open_client returns an httpx.MockTransport, names resolve to a public address), and the
dataset store lives under tmp_path, so nothing is sent or resolved."""
import hashlib
import re
from pathlib import Path

import httpx
import pytest

from prophet_agent.api.service import MissionService
from prophet_agent.datasets import DatasetStore
from prophet_agent.opendata import fetch
from prophet_agent.research import netguard
from prophet_agent.runtime.runner import ArgumentRejected
from prophet_agent.tools import default_tools
from prophet_agent.tools import opendata_tools as od
from prophet_agent.tools.opendata_tools import LiveMounts, OpenDataContext

FIXTURES = Path(__file__).resolve().parents[1] / "opendata" / "fixtures"
CSV_BYTES = (FIXTURES / "open_meteo_daily.csv").read_bytes()
PUBLIC = "93.184.216.34"
FORECAST = {"latitude": 52.52, "longitude": 13.41, "daily": "temperature_2m_max,precipitation_sum",
            "timezone": "Europe/Berlin", "forecast_days": 7}
ARCHIVE = {"latitude": 52.52, "longitude": 13.41, "start_date": "2024-01-01", "end_date": "2024-01-07"}
SOURCE_KEYS = {"open_meteo_forecast", "open_meteo_archive", "ecb_sdmx"}
WEB_URL = "https://data.example.org/files/daily.csv"


@pytest.fixture
def served(monkeypatch):
    """Serve ``body`` with ``content_type`` for every request, and count the requests."""
    state = {"calls": 0, "body": CSV_BYTES, "content_type": "text/csv", "status": 200}

    def handler(request):
        state["calls"] += 1
        return httpx.Response(state["status"], headers={"content-type": state["content_type"]}, content=state["body"])

    monkeypatch.setattr(fetch, "open_client", lambda timeout_s: httpx.Client(transport=httpx.MockTransport(handler),
                                                                              follow_redirects=False))
    monkeypatch.setattr(netguard, "_default_resolver", lambda host, port: [PUBLIC])
    return state


class Mission:
    """The part of a mission the download tools see: its dataset store, and the ids recorded on it."""

    def __init__(self, root: Path):
        self.store = DatasetStore(root / "datasets")
        self.ids: list[str] = []

    def context(self) -> OpenDataContext:
        return OpenDataContext(store=self.store, attach=self.ids.append)


@pytest.fixture
def mission(tmp_path):
    return Mission(tmp_path)


# --- listing ----------------------------------------------------------------------------------------------------------

def test_list_open_data_sources_returns_the_three_source_keys():
    tool = default_tools()["list_open_data_sources"]
    out = tool.fn({}, "k")
    assert {source["key"] for source in out["sources"]} == SOURCE_KEYS
    assert tool.idempotent is True


def test_downloads_are_offered_only_with_a_mission_context(mission):
    assert "download_open_data" not in default_tools()
    assert "download_web_data" not in default_tools()
    tools = default_tools(opendata=mission.context())
    assert {"download_open_data", "download_web_data", "list_open_data_sources"} <= set(tools)
    assert all(tools[name].doc and tools[name].title and tools[name].activity
               for name in ("download_open_data", "download_web_data", "list_open_data_sources"))


def test_planner_text_for_the_download_tools_says_nothing_about_licensing():
    for tool in default_tools(opendata=OpenDataContext(store=None, attach=None)).values():
        if tool.name in ("list_open_data_sources", "download_open_data", "download_web_data"):
            assert not re.search(r"licen[cs]e(?!_key)|terms of use|non-commercial|research use", tool.doc, re.IGNORECASE), tool.name


# --- source downloads -------------------------------------------------------------------------------------------------

def test_download_open_data_stores_the_preview_and_the_second_call_returns_the_stored_copy(served, mission):
    tool = default_tools(opendata=mission.context())["download_open_data"]
    first = tool.fn({"source": "open_meteo_forecast", "params": FORECAST}, "k1")
    assert served["calls"] == 1
    assert first["source"] == "open_meteo_forecast"
    assert first["row_count"] == 7
    assert first["columns"] == ["time", "temperature_2m_max", "precipitation_sum"]
    assert len(first["first_rows"]) == 5 and first["first_rows"][0] == ["2024-01-01", "4.5", "0.0"]
    assert first["sandbox_file"] == f"in/{first['name']}"
    assert mission.ids == [first["dataset_id"]]

    second = tool.fn({"source": "open_meteo_forecast", "params": FORECAST}, "k2")
    assert served["calls"] == 1, "the same source and parameters must not download a second time"
    assert second["dataset_id"] == first["dataset_id"]
    assert mission.ids == [first["dataset_id"]]


def test_invalid_source_key_and_parameters_are_clear_tool_errors_before_any_request(served, mission):
    tool = default_tools(opendata=mission.context())["download_open_data"]
    with pytest.raises(ValueError, match="unknown source key"):
        tool.fn({"source": "no_such_source", "params": {}}, "k")
    with pytest.raises(ValueError, match="latitude"):
        tool.fn({"source": "open_meteo_forecast", "params": {**FORECAST, "latitude": 200}}, "k")
    with pytest.raises(ValueError, match="missing required parameter"):
        tool.fn({"source": "open_meteo_archive", "params": {"latitude": 1, "longitude": 1}}, "k")
    with pytest.raises(ArgumentRejected, match="unknown arguments"):
        tool.fn({"source": "ecb_sdmx", "params": {}, "extra": 1}, "k")
    with pytest.raises(ArgumentRejected, match="params must be an object"):
        tool.fn({"source": "ecb_sdmx", "params": "flow=EXR"}, "k")
    assert served["calls"] == 0
    assert mission.ids == []


def test_the_file_limit_refuses_a_download_once_the_mission_is_full(served, mission, monkeypatch):
    monkeypatch.setattr(od, "MAX_MISSION_FILES", 1)
    tool = default_tools(opendata=mission.context())["download_open_data"]
    tool.fn({"source": "open_meteo_forecast", "params": FORECAST}, "k1")
    with pytest.raises(ValueError, match="already holds 1 files"):
        tool.fn({"source": "open_meteo_archive", "params": ARCHIVE}, "k2")
    assert served["calls"] == 1 and len(mission.ids) == 1


def test_mounted_names_take_a_numeric_suffix_when_a_name_is_taken(tmp_path):
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    live = LiveMounts({"a.csv": first})
    assert live.add("a.csv", second) == "a_2.csv"
    assert live.name_of(second) == "a_2.csv" and live.name_of(tmp_path / "x") is None
    assert len(live) == 2


# --- web downloads ----------------------------------------------------------------------------------------------------

def test_download_web_data_stores_a_csv_with_an_unverified_licence_key_and_provenance(served, mission):
    tool = default_tools(opendata=mission.context())["download_web_data"]
    out = tool.fn({"url": WEB_URL}, "k1")
    assert out["source"] == "web" and out["licence_key"] == "unverified"
    assert out["name"] == "daily.csv" and out["sandbox_file"] == "in/daily.csv"
    assert out["row_count"] == 7 and out["columns"][0] == "time"
    stored = mission.store.get(out["dataset_id"])
    provenance = stored["provenance"]
    assert provenance["url"] == WEB_URL and provenance["final_url"] == WEB_URL
    assert provenance["content_type"] == "text/csv" and provenance["format"] == "csv"
    assert provenance["sha256"] == hashlib.sha256(CSV_BYTES).hexdigest() and provenance["bytes"] == len(CSV_BYTES)
    assert provenance["retrieved_at"] and provenance["licence_key"] == "unverified"
    assert mission.store.path(out["dataset_id"]).read_bytes() == CSV_BYTES
    assert mission.ids == [out["dataset_id"]]


def test_download_web_data_returns_the_stored_copy_for_the_same_url(served, mission):
    tool = default_tools(opendata=mission.context())["download_web_data"]
    first = tool.fn({"url": WEB_URL}, "k1")
    second = tool.fn({"url": WEB_URL}, "k2")
    assert served["calls"] == 1 and second["dataset_id"] == first["dataset_id"]


def test_a_known_licence_key_is_recorded_and_an_unknown_one_is_refused(served, mission):
    tool = default_tools(opendata=mission.context())["download_web_data"]
    out = tool.fn({"url": "https://data.example.org/a.csv", "licence_key": "ecb"}, "k1")
    assert out["licence_key"] == "ecb"
    with pytest.raises(ValueError, match="licence_key must be"):
        tool.fn({"url": "https://data.example.org/b.csv", "licence_key": "made-up"}, "k2")
    assert served["calls"] == 1


@pytest.mark.parametrize("content_type, body, message", [
    ("text/html", b"<html><body>a,b</body></html>", "content type"),
    ("text/plain", b"<!doctype html><html>a,b</html>", "HTML"),
    ("text/plain", b"just words\nmore words\n", "at least two columns"),
    ("text/csv", b"a,b\n", "no data rows"),
    ("application/json", b'{"a": 1}', "does not hold a table"),
    ("application/octet-stream", b"\x00\x01\x02", "parses as CSV or JSON"),
])
def test_download_web_data_refuses_a_body_that_is_not_a_csv_or_json_table(served, mission, content_type, body, message):
    served["content_type"], served["body"] = content_type, body
    tool = default_tools(opendata=mission.context())["download_web_data"]
    with pytest.raises(ValueError, match=message):
        tool.fn({"url": WEB_URL}, "k")
    assert mission.ids == []
    assert list(mission.store.root.glob("*")) == [], "a refused body must not leave a dataset folder behind"


def test_download_web_data_refuses_a_private_address_without_a_request(served, mission):
    tool = default_tools(opendata=mission.context())["download_web_data"]
    with pytest.raises(ValueError, match="refused"):
        tool.fn({"url": "https://10.0.0.5/data.csv"}, "k")
    with pytest.raises(ValueError, match="must start with https"):
        tool.fn({"url": "http://data.example.org/data.csv"}, "k")
    assert served["calls"] == 0


# --- the mission's files reach the sandbox ----------------------------------------------------------------------------

def test_a_downloaded_file_is_read_by_sandbox_python_in_the_same_run(served, mission):
    tools = default_tools(opendata=mission.context())
    got = tools["download_open_data"].fn({"source": "open_meteo_forecast", "params": FORECAST}, "k1")
    code = (f"import os\nprint(sorted(os.listdir('in')))\n"
            f"print(len(open({got['sandbox_file']!r}).read().splitlines()))\n")
    out = tools["sandbox_python"].fn({"code": code}, "run-key-" + "0" * 22)
    assert out["ok"], out["stderr"]
    lines = out["stdout"].splitlines()
    assert got["name"] in lines[0]
    assert lines[1] == str(len(CSV_BYTES.decode().splitlines()))


def test_the_mission_records_the_download_and_a_rebuilt_tool_set_mounts_it(served, tmp_path):
    service = MissionService(tmp_path / "m.sqlite", lambda: None, default_tools(), tools_factory=default_tools)
    mid = service.create("forecast the weather")
    tools = service.tools_for(service.store.mission(mid))
    got = tools["download_web_data"].fn({"url": WEB_URL}, "k1")
    assert service.store.mission(mid)["budget"]["dataset_ids"] == [got["dataset_id"]]

    rebuilt = service.tools_for(service.store.mission(mid))
    out = rebuilt["sandbox_python"].fn({"code": "import os\nprint(os.listdir('in'))"}, "run-key-" + "1" * 22)
    assert out["ok"] and got["name"] in out["stdout"]
