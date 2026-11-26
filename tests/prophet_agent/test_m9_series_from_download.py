"""Panel NEM (2026-11-26): a series filtered in code from a long downloaded file is not a pasted preview."""
from prophet_agent.runtime import contracts as c

DOWNLOAD = {"kind": "tool", "tool": "download_web_data", "args": {"url": "https://example.org/data.csv"},
            "result": {"row_count": 604, "sandbox_file": "in/data.csv"}}


def _cell(run_id, code, outputs):
    return {"kind": "tool", "tool": "sandbox_python", "args": {"code": code},
            "result": {"ok": True, "run_id": run_id, "outputs": outputs}}


def _forecast(run_id, points=120):
    return {"kind": "tool", "tool": "forecast_report", "args": {"source": {"run_id": run_id, "file": "out/series.json"}},
            "result": {"forecast": {"config": {"history_points": points}}}}


def test_a_region_filtered_from_the_download_passes():
    obs = [DOWNLOAD,
           _cell("r1", "df = pd.read_csv('in/data.csv')\nnsw = df[df.key == 'NSW1']\nnsw.to_csv('out/nsw.csv')",
                 ["out/nsw.csv (900 bytes)"]),
           _cell("r2", "s = pd.read_csv('in/carried/nsw.csv')\npl.save_series(s)", ["out/series.json (400 bytes)"]),
           _forecast("r2")]
    assert c._series_shorter_than_file(obs) is None


def test_a_typed_paste_is_still_blocked():
    obs = [DOWNLOAD,
           _cell("r1", "values = [7000, 7100, 7050]\npl.save_series(values)", ["out/series.json (100 bytes)"]),
           _forecast("r1")]
    assert c._series_shorter_than_file(obs)["name"] == "series_covers_download"
