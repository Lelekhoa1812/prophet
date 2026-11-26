"""Extraction, fusion, calibration, and diagnosis. The questions are inputs. The checks are not."""
from prophet_agent.source_table.calibrate import calibrate_overlap
from prophet_agent.source_table.claim import build_claim
from prophet_agent.source_table.diagnose import diagnose_failure
from prophet_agent.source_table.evaluate import process_score
from prophet_agent.source_table.extract import extract_html
from prophet_agent.source_table.fuse import fuse_sources
from prophet_agent.source_table.publish import bind_series
from prophet_agent.source_table.validate import validate_table
from prophet_agent.datasets import DatasetStore


HTML = """
<html><body>
<table><tr><td>Home</td><td>About</td></tr></table>
<table>
<tr><th>date</th><th>name</th><th>close</th></tr>
<tr><td>2024-01-01</td><td>Alpha</td><td>3</td></tr>
<tr><td>2024-01-02</td><td>Alpha</td><td>4</td></tr>
</table>
<a href="https://example.test/alpha.csv">series</a>
</body></html>
"""


def test_an_html_page_yields_the_data_table_and_the_file_link():
    extracted = extract_html(HTML, final_url="https://example.test/alpha")
    assert extracted["skipped_layout"] >= 1
    assert len(extracted["candidates"]) == 1
    assert extracted["candidates"][0]["columns"] == ["date", "name", "close"]
    assert extracted["file_links"][0]["url"].endswith("/alpha.csv")


def test_a_matching_table_binds_and_a_disagreement_stays_in_two_columns(tmp_path):
    claim = build_claim({"series_name": "alpha_daily", "entity": "Alpha", "aliases": ["Alpha"], "measure": "close"})
    csv_bytes = extract_html(HTML, final_url="https://example.test/alpha")["candidates"][0]["csv"].encode()
    verdict = validate_table(claim, csv_bytes, final_url="https://example.test/alpha")
    assert verdict["status"] == "accepted" and verdict["bound_name"] == "alpha_daily"
    published = bind_series(DatasetStore(tmp_path / "datasets"), claim, csv_bytes, final_url="https://example.test/alpha")
    assert published["name"] == "alpha_daily.csv"
    fused = fuse_sources(
        [{"source_id": "a", "rows": [{"date": "2024-01-01", "close": "3"}]},
         {"source_id": "b", "rows": [{"date": "2024-01-01", "close": "9"}]}],
        time_column="date", measure_column="close")
    assert fused["conflict_times"] == ["2024-01-01"]
    assert fused["rows"][0]["a__close"] == "3" and fused["rows"][0]["b__close"] == "9"
    report = calibrate_overlap(
        [{"date": "2024-01-01", "close": "10"}], [{"date": "2024-01-01", "close": "12"}],
        time_column="date", measure_column="close", source_left="a", source_right="b")
    assert report["rows"][0]["close"] == "10"
    assert report["rows"][0]["calibrated_close"] != 10
    assert report["rows"][0]["calibrated_origin"] == "derived"


def test_download_errors_are_closed_signals():
    assert diagnose_failure("HTTP 401; only a 200 response is used")["signal"] == "auth_denied"
    assert diagnose_failure("HTTP 404; only a 200 response is used")["signal"] == "not_found"
    assert diagnose_failure("content type text/html is not allowed; use CSV, plain text, JSON, ZIP or XLSX")["signal"] == "html_page"


def test_the_process_score_ignores_the_question_and_rejects_a_false_bind():
    events = [
        {"tool": "state_table_claim", "series_name": "alpha_daily"},
        {"tool": "diagnose_source", "signal": "html_page"},
        {"tool": "validate_table", "status": "wrong_entity", "series_name": "alpha_daily", "bound_name": None},
        {"tool": "bind_series", "status": "wrong_entity", "series_name": "alpha_daily", "bound_name": None},
    ]
    assert process_score(events)["passed"] is True
    events[-1]["bound_name"] = "alpha_daily"
    events[-1]["status"] = "wrong_entity"
    assert process_score(events)["passed"] is False
