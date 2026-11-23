"""One live check of the open-data path: daily maximum temperature for one city from the Open-Meteo archive, stored with
its provenance, then scored by the backtest and TimesFM tools against the simple rules.

Usage (from the repository root):
    PYTHONPATH=src .venv/bin/python scripts/weather_open_data_check.py <store folder>

The script makes one request to the archive API, stores the table, and writes <store folder>/weather_check.json. It prints
a short summary. Every figure it prints is measured on the downloaded data; nothing is synthetic. Attribution and any
use-restriction wording live in docs/prophet/, not here, so the script refers to the source by its key only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from prophet_agent.datasets import DatasetStore
from prophet_agent.opendata import store as opendata_store
from prophet_agent.tools.data_tools import backtest_tool
from prophet_agent.tools.timesfm_tool import timesfm_tool

SOURCE = "open_meteo_archive"
PARAMS = {"latitude": 52.52, "longitude": 13.41, "timezone": "Europe/Berlin", "daily": "temperature_2m_max",
          "start_date": "2022-01-01", "end_date": "2026-11-15"}
HORIZON = 14
N_ORIGINS = 12
ANNUAL_SEASON = 365


def series_from_table(path: Path, value_column: str) -> tuple[list[str], list[float]]:
    """Read the stored table (time header first, as stored) and return the dates and the values. Refuses missing values:
    the check should not quietly fill a gap in measured data."""
    import csv

    dates, values = [], []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get(value_column) or "").strip()
            if raw == "":
                raise ValueError(f"missing value on {row.get('time')}; the check refuses to fill gaps")
            dates.append(row["time"])
            values.append(float(raw))
    return dates, values


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    root = Path(argv[1]).resolve()
    store = DatasetStore(root)
    meta = opendata_store.download_to_dataset(SOURCE, PARAMS, store, name="berlin_tmax.csv")
    value_column = meta["preview"]["columns"][1]
    dates, values = series_from_table(store.path(meta["id"]), value_column)

    # Two backtests. The automatic season found no cycle for daily data (the tool tests short lags only), which leaves the
    # simple rules weak, so the fair comparison sets the annual season explicitly. Both are recorded.
    backtest_auto = backtest_tool().fn({"values": values, "horizon": HORIZON, "n_origins": N_ORIGINS}, "weather-backtest-auto")
    backtest = backtest_tool().fn({"values": values, "horizon": HORIZON, "n_origins": N_ORIGINS, "season": ANNUAL_SEASON},
                                  "weather-backtest-annual")
    forecast = timesfm_tool().fn({"values": values, "horizon": HORIZON}, "weather-forecast")

    provenance = meta["provenance"]
    summary = {
        "source_key": provenance["source"],
        "url": provenance["url"],
        "retrieved_at": provenance["retrieved_at"],
        "stored_sha256": provenance["sha256"],
        "raw_sha256": provenance["raw_sha256"],
        "licence_key": provenance["licence_key"],
        "variable": value_column,
        "points": len(values),
        "first_date": dates[0],
        "last_date": dates[-1],
        "horizon_days": HORIZON,
        "backtest_automatic_season": backtest_auto,
        "backtest_annual_season": backtest,
        "forecast": forecast,
    }
    (root / "weather_check.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({
        "points": summary["points"], "first_date": summary["first_date"], "last_date": summary["last_date"],
        "stored_sha256": summary["stored_sha256"], "verdict": backtest.get("verdict"),
        "best_baseline": backtest.get("best_baseline"), "forecast_health": forecast.get("health"),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
