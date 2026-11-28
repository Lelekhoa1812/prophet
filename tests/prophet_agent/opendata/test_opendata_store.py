"""download_to_dataset: preview parsing, provenance, idempotency and no-store-on-bad-body. Stores live under tmp_path. The
network is replaced: fetch.open_client returns an httpx.MockTransport, and the default name resolver is pinned to a public
address, so nothing is resolved or sent."""
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import httpx
import pytest
from openpyxl import Workbook

from prophet_agent.datasets import DatasetStore
from prophet_agent.opendata import fetch, sources, store
from prophet_agent.research import netguard

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CSV_BYTES = (FIXTURES / "open_meteo_daily.csv").read_bytes()
CSV_TABLE = CSV_BYTES[CSV_BYTES.index(b"\ntime,") + 1:]        # the table alone: what the store keeps on disk
JSON_BYTES = (FIXTURES / "open_meteo_daily.json").read_bytes()
# A real capture of the Open-Meteo CSV response shape, taken live on 2026-11-23 (location block, blank line, table).
LIVE_BYTES = (FIXTURES / "open_meteo_daily_live.csv").read_bytes()
PUBLIC = "93.184.216.34"
FORECAST = {"latitude": 52.52, "longitude": 13.41, "daily": "temperature_2m_max,precipitation_sum",
            "timezone": "Europe/Berlin", "forecast_days": 7}
PROVENANCE_FIELDS = {"source", "url", "params", "params_sha256", "retrieved_at", "final_url", "content_type", "sha256",
                     "bytes", "raw_sha256", "raw_bytes", "preamble", "licence_key", "landing"}


@pytest.fixture
def served(monkeypatch):
    """Serve ``body`` (with ``content_type``) for every request and count the requests."""
    state = {"calls": 0, "body": CSV_BYTES, "content_type": "text/csv", "status": 200}

    def handler(request):
        state["calls"] += 1
        return httpx.Response(state["status"], headers={"content-type": state["content_type"]}, content=state["body"])

    monkeypatch.setattr(fetch, "open_client", lambda timeout_s: httpx.Client(transport=httpx.MockTransport(handler),
                                                                              follow_redirects=False))
    monkeypatch.setattr(netguard, "_default_resolver", lambda host, port: [PUBLIC])
    return state


def store_at(tmp_path) -> DatasetStore:
    return DatasetStore(tmp_path / "datasets")


# --- preview parsing -------------------------------------------------------------------------------------------------

def test_csv_preview_skips_the_comment_and_location_block_and_finds_the_table():
    table = store.preview(CSV_BYTES, "csv")
    assert table["columns"] == ["time", "temperature_2m_max", "precipitation_sum"]
    assert table["row_count"] == 7
    assert table["rows"][0] == ["2024-01-01", "4.5", "0.0"]
    assert len(table["rows"]) == store.PREVIEW_ROWS


def test_json_preview_takes_the_daily_block():
    table = store.preview(JSON_BYTES, "json")
    assert table["columns"] == ["time", "temperature_2m_max", "precipitation_sum"]
    assert table["row_count"] == 7
    assert table["rows"][0] == ["2024-01-01", 4.5, 0.0]


def test_json_list_of_records_and_plain_csv_are_previewed():
    records = json.dumps([{"date": "2024-01-01", "units": 3}, {"date": "2024-01-02", "units": 5, "note": "x"}]).encode()
    table = store.preview(records, "json")
    assert table["columns"] == ["date", "units", "note"] and table["row_count"] == 2
    plain = store.preview(b"week,units\n1,2\n3,4\n5,6\n7,8\n9,10\n11,12\n", "csv")
    assert plain == {"columns": ["week", "units"], "rows": [["1", "2"], ["3", "4"], ["5", "6"], ["7", "8"], ["9", "10"]],
                     "row_count": 6}


@pytest.mark.parametrize("body, fmt, message", [
    (b"", "csv", "no rows"),
    (b"\x89PNG\r\n\x1a\n", "csv", "not UTF-8 text"),
    (b'{"error": true, "reason": "bad"}', "json", "does not hold a table"),
    (b"[1, 2, 3]", "json", "does not hold a table"),
    (b"latitude,longitude\n52,13\n", "csv", "no time or date header"),
    (b"time,value\n", "csv", None),                  # a header with no rows is a valid empty table
])
def test_bodies_that_are_not_tables_are_refused(body, fmt, message):
    if message is None:
        assert store.preview(body, fmt)["row_count"] == 0
        return
    with pytest.raises(ValueError, match=message):
        store.preview(body, fmt)


# --- download_to_dataset: provenance -----------------------------------------------------------------------------------

def test_download_stores_the_bytes_with_full_provenance_and_returns_the_preview(served, tmp_path):
    datasets = store_at(tmp_path)
    meta = store.download_to_dataset("open_meteo_forecast", FORECAST, datasets)
    url = sources.build_url("open_meteo_forecast", FORECAST)
    provenance = meta["provenance"]

    assert set(provenance) == PROVENANCE_FIELDS
    assert provenance["source"] == "open_meteo_forecast"
    assert provenance["url"] == url
    assert provenance["url"].startswith("https://api.open-meteo.com/v1/forecast?")
    assert provenance["params"] == sources.normalise("open_meteo_forecast", FORECAST)
    assert provenance["final_url"] == url
    assert provenance["content_type"] == "text/csv"
    assert provenance["sha256"] == hashlib.sha256(CSV_TABLE).hexdigest() == meta["sha256"]
    assert provenance["bytes"] == len(CSV_TABLE) == meta["bytes"]
    assert provenance["raw_sha256"] == hashlib.sha256(CSV_BYTES).hexdigest()
    assert provenance["raw_bytes"] == len(CSV_BYTES)
    assert provenance["preamble"].startswith("# SYNTHETIC") and "latitude" in provenance["preamble"]
    assert provenance["licence_key"] == "open_meteo"
    assert provenance["retrieved_at"].endswith("+00:00")

    assert meta["name"] == "open_meteo_forecast.csv"
    assert meta["preview"]["row_count"] == 7
    assert datasets.path(meta["id"]).read_bytes() == CSV_TABLE
    on_disk = json.loads((datasets.root / meta["id"] / "meta.json").read_text(encoding="utf-8"))
    assert on_disk["provenance"] == provenance
    assert served["calls"] == 1


def test_json_format_is_stored_as_json_and_previewed_as_json(served, tmp_path):
    served.update(body=JSON_BYTES, content_type="application/json")
    meta = store.download_to_dataset("open_meteo_forecast", {**FORECAST, "format": "json"}, store_at(tmp_path))
    assert meta["name"] == "open_meteo_forecast.json"
    assert meta["preview"]["columns"][0] == "time"


def test_a_name_can_be_given(served, tmp_path):
    meta = store.download_to_dataset("ecb_sdmx", {"flow": "EXR", "key": "D.USD.EUR.SP00.A"}, store_at(tmp_path),
                                     name="usd_eur.csv")
    assert meta["name"] == "usd_eur.csv"
    assert meta["provenance"]["url"] == "https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=csvdata"
    assert meta["provenance"]["licence_key"] == "ecb"


def test_the_live_capture_is_stored_as_the_table_alone(served, tmp_path):
    """The location block that Open-Meteo prints before its table is kept in provenance, not in the stored file, so a
    reader of the file (the sandbox) sees the time header on its first line."""
    served.update(body=LIVE_BYTES, content_type="text/csv")
    datasets = store_at(tmp_path)
    meta = store.download_to_dataset("open_meteo_forecast", FORECAST, datasets)
    stored = datasets.path(meta["id"]).read_bytes()

    assert stored.decode("utf-8").splitlines()[0] == "time,temperature_2m_max (°C),temperature_2m_min (°C),precipitation_sum (mm)"
    assert b"latitude" not in stored
    assert meta["provenance"]["preamble"].splitlines()[0].startswith("latitude,longitude")
    assert meta["provenance"]["raw_sha256"] == hashlib.sha256(LIVE_BYTES).hexdigest()
    assert meta["preview"]["columns"] == ["time", "temperature_2m_max (°C)", "temperature_2m_min (°C)",
                                          "precipitation_sum (mm)"]
    assert meta["preview"]["row_count"] == 3 and meta["preview"]["rows"][0] == ["2026-11-23", "31.3", "22.6", "1.30"]


def test_the_stored_table_previews_the_same_as_the_raw_body():
    preamble, table = store.split_csv(LIVE_BYTES.decode("utf-8"))
    assert store.preview(table.encode("utf-8"), "csv") == store.preview(LIVE_BYTES, "csv")
    assert preamble.endswith("\n\n")                       # the blank line between the block and the table stays there


def test_a_csv_without_a_location_block_is_stored_unchanged(served, tmp_path):
    plain = b"week,units\n1,2\n3,4\n"
    served.update(body=plain, content_type="text/csv")
    meta = store.download_to_dataset("ecb_sdmx", {"flow": "EXR", "key": "D.USD.EUR.SP00.A"}, store_at(tmp_path))
    assert meta["provenance"]["preamble"] == ""
    assert meta["provenance"]["sha256"] == meta["provenance"]["raw_sha256"] == hashlib.sha256(plain).hexdigest()


def test_a_location_block_with_no_table_header_is_refused_and_not_stored(served, tmp_path):
    served.update(body=b"latitude,longitude\n52,13\n", content_type="text/csv")
    datasets = store_at(tmp_path)
    with pytest.raises(ValueError, match="no time or date header"):
        store.download_to_dataset("open_meteo_forecast", FORECAST, datasets)
    assert not datasets.root.exists() or list(datasets.root.iterdir()) == []


# --- idempotency ------------------------------------------------------------------------------------------------------

def test_a_second_call_returns_the_stored_dataset_without_downloading(served, tmp_path):
    datasets = store_at(tmp_path)
    first = store.download_to_dataset("open_meteo_forecast", FORECAST, datasets)
    second = store.download_to_dataset("open_meteo_forecast", FORECAST, datasets)
    assert served["calls"] == 1
    assert second["id"] == first["id"]
    assert second["provenance"] == first["provenance"]
    assert second["preview"] == first["preview"]
    assert len(list(datasets.root.iterdir())) == 1


def test_equal_values_spelled_differently_are_the_same_download(served, tmp_path):
    datasets = store_at(tmp_path)
    first = store.download_to_dataset("open_meteo_forecast", {"latitude": 52.52, "longitude": 13.41}, datasets)
    second = store.download_to_dataset("open_meteo_forecast", {"latitude": 52.520, "longitude": 13.41}, datasets)
    assert second["id"] == first["id"] and served["calls"] == 1


def test_different_parameters_are_a_different_download(served, tmp_path):
    datasets = store_at(tmp_path)
    store.download_to_dataset("open_meteo_forecast", FORECAST, datasets)
    other = store.download_to_dataset("open_meteo_forecast", {**FORECAST, "forecast_days": 8}, datasets)
    assert served["calls"] == 2
    assert other["provenance"]["params"]["forecast_days"] == "8"
    assert len(list(datasets.root.iterdir())) == 2


def test_the_same_parameters_from_another_source_are_not_a_match(served, tmp_path):
    datasets = store_at(tmp_path)
    store.download_to_dataset("open_meteo_forecast", {"latitude": 52, "longitude": 13}, datasets)
    store.download_to_dataset("open_meteo_archive", {"latitude": 52, "longitude": 13, "start_date": "2024-01-01",
                                                     "end_date": "2024-01-07"}, datasets)
    assert served["calls"] == 2


def test_a_download_with_no_format_asks_for_csv_and_records_it(served, tmp_path):
    """Open-Meteo answers JSON when no format is named. The request, the provenance and the stored type must agree."""
    meta = store.download_to_dataset("open_meteo_forecast", FORECAST, store_at(tmp_path))
    assert meta["provenance"]["params"]["format"] == "csv"
    assert "format=csv" in meta["provenance"]["url"]
    assert meta["name"] == "open_meteo_forecast.csv"


def test_a_json_body_returned_for_a_csv_request_is_not_stored_as_a_table(served, tmp_path):
    served.update(body=JSON_BYTES, content_type="application/json")
    datasets = store_at(tmp_path)
    with pytest.raises(ValueError, match="is JSON, not CSV"):
        store.download_to_dataset("open_meteo_forecast", FORECAST, datasets)
    assert not datasets.root.exists() or list(datasets.root.iterdir()) == []


# --- refusals store nothing --------------------------------------------------------------------------------------------

def test_invalid_parameters_are_refused_before_any_request(served, tmp_path):
    with pytest.raises(ValueError, match="parameter 'latitude' must be between -90 and 90"):
        store.download_to_dataset("open_meteo_forecast", {**FORECAST, "latitude": 95}, store_at(tmp_path))
    assert served["calls"] == 0


def test_a_body_that_is_not_the_requested_format_is_not_stored(served, tmp_path):
    served.update(body=CSV_BYTES, content_type="text/csv")                  # JSON was requested, CSV came back
    datasets = store_at(tmp_path)
    with pytest.raises(ValueError, match="does not parse as a JSON table"):
        store.download_to_dataset("open_meteo_forecast", {**FORECAST, "format": "json"}, datasets)
    assert not datasets.root.exists() or list(datasets.root.iterdir()) == []


def test_a_title_row_is_left_out_of_the_stored_csv():
    text = "Cash rate table\ndate,rate\n2024-01-01,4.35\n2024-02-01,4.35\n"
    table = store.preview(text.encode(), "csv")
    assert table["columns"] == ["date", "rate"] and table["row_count"] == 2
    preamble, kept = store.split_csv(text)
    assert preamble.startswith("Cash rate table") and kept.startswith("date,rate")


def test_a_world_bank_envelope_becomes_a_table_of_records():
    payload = [{"page": 1, "pages": 1, "total": 1},
               [{"country": {"id": "US", "value": "United States"}, "date": "2023", "value": 4.1}]]
    table = store.preview(json.dumps(payload).encode(), "json")
    assert table["columns"] == ["country", "date", "value"]
    assert table["rows"] == [["US", "2023", 4.1]]


def test_a_zip_download_keeps_one_series_and_the_archive_hash(served, tmp_path):
    buf = io.BytesIO()
    with ZipFile(buf, "w") as archive:
        archive.writestr("ELEC.txt", "\n".join([
            json.dumps({"series_id": "OTHER", "data": [["2020", 1]]}),
            json.dumps({"series_id": "ELEC.A", "data": [["2020", 9], ["2021", None]]}),
        ]))
    served.update(body=buf.getvalue(), content_type="application/zip")
    datasets = store_at(tmp_path)
    meta = store.download_to_dataset("eia_bulk", {"dataset": "ELEC", "series_id": "ELEC.A"}, datasets)
    assert meta["preview"]["columns"] == ["period", "value"]
    assert meta["preview"]["row_count"] == 2
    assert meta["provenance"]["raw_sha256"] == hashlib.sha256(buf.getvalue()).hexdigest()
    assert meta["provenance"]["url"] == "https://www.eia.gov/opendata/bulk/ELEC.zip"
    assert datasets.path(meta["id"]).read_text().splitlines() == ["period,value", "2020,9", "2021,"]


def test_an_xlsx_download_stores_the_first_sheet(served, tmp_path):
    book = Workbook()
    sheet = book.active
    sheet.append(["Monthly prices"])
    sheet.append(["date", "oil"])
    sheet.append(["2020-01", 50])
    buf = io.BytesIO()
    book.save(buf)
    served.update(body=buf.getvalue(),
                  content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    meta = store.download_to_dataset("world_bank_pink_sheet", {}, store_at(tmp_path))
    assert meta["preview"]["columns"] == ["date", "oil"]
    assert meta["preview"]["rows"] == [["2020-01", "50"]]


def test_a_plain_text_source_is_stored_as_text(served, tmp_path):
    body = b"# comment\nSEAS YR TOTAL ANOM\nDJF 1950 1.2 -0.1\nJFM 1950 1.3 -0.2\n"
    served.update(body=body, content_type="text/plain")
    meta = store.download_to_dataset("cpc_oni", {}, store_at(tmp_path))
    assert meta["name"] == "cpc_oni.txt"
    assert meta["preview"]["columns"] == ["SEAS", "YR", "TOTAL", "ANOM"]
    assert meta["preview"]["row_count"] == 2


def test_a_refused_download_is_not_stored(served, tmp_path):
    served.update(status=400, body=b"{}", content_type="application/json")
    datasets = store_at(tmp_path)
    with pytest.raises(fetch.DownloadError, match="HTTP 400"):
        store.download_to_dataset("open_meteo_forecast", FORECAST, datasets)
    assert not datasets.root.exists() or list(datasets.root.iterdir()) == []


def test_a_stored_download_older_than_the_reuse_window_is_fetched_again(served, tmp_path, monkeypatch):
    """Round 20261128-125459 (GOLD): a Pink Sheet stored days earlier ended in July while the live one ran to October."""
    datasets = store_at(tmp_path)
    first = store.download_to_dataset("open_meteo_forecast", FORECAST, datasets)
    real = store.time.time
    monkeypatch.setattr(store.time, "time", lambda: real() + store.REUSE_S + 60)
    again = store.download_to_dataset("open_meteo_forecast", FORECAST, datasets)
    assert served["calls"] == 2 and again["id"] != first["id"]


def test_a_download_landed_by_an_older_landing_is_not_reused(served, tmp_path, monkeypatch):
    datasets = store_at(tmp_path)
    first = store.download_to_dataset("open_meteo_forecast", FORECAST, datasets)
    assert first["provenance"]["landing"] == store.LANDING_VERSION
    monkeypatch.setattr(store, "LANDING_VERSION", store.LANDING_VERSION + 1)
    store.download_to_dataset("open_meteo_forecast", FORECAST, datasets)
    assert served["calls"] == 2
