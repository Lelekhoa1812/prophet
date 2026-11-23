#!/usr/bin/env python3
"""Fetch and normalise real-world time-series datasets for Prophet stress tests.

Downloads are streamed with httpx into data/datasets/_raw/<name>/ (resumable via .part
files), hashed with sha256 and recorded in data/datasets/MANIFEST.json. Each dataset is then
normalised to one tidy long Parquet table (entity, timestamp, value, plus covariates) with
every defect kept as found in the source, and a README.md with pandas-measured defects is
written next to it.

Examples:
  PYTHONPATH=src .venv/bin/python scripts/fetch_datasets.py --list
  PYTHONPATH=src .venv/bin/python scripts/fetch_datasets.py --only fred_dexuseu,ett_etth1
  PYTHONPATH=src .venv/bin/python scripts/fetch_datasets.py --max-mb 300
  PYTHONPATH=src .venv/bin/python scripts/fetch_datasets.py --download-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "datasets"
RAW = DATA / "_raw"
MANIFEST = DATA / "MANIFEST.json"
USER_AGENT = "prophet-dataset-fetch/1.0"

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
ECB_FX_ZIP = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
UCI_ELEC = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
UCI_HOUSEHOLD = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
UCI_BEIJING = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
UCI_RETAIL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ETT_RAW = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{}.csv"
M4_RAW = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/{}/Daily-{}.csv"
OWID_CSV = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
ZENODO_CAR_PARTS = "https://zenodo.org/api/records/4656022/files/car_parts_dataset_with_missing_values.zip/content"
ZENODO_LONDON = "https://zenodo.org/api/records/4656072/files/london_smart_meters_dataset_with_missing_values.zip/content"
HF_GIFT = "https://huggingface.co/datasets/Salesforce/GiftEval/resolve/main/{}/data-00000-of-00001.arrow"
HF_CHRONOS = "https://huggingface.co/datasets/autogluon/chronos_datasets/resolve/main/{}"

# Dataset terms are kept in docs/prophet/DATASET_LICENCES.md (owner decision, 2026-11-23). Sources name a key from that
# table; licence_text() reads the text when a README is written, and fails if the key is missing.
LICENCE_DOC = ROOT / "docs" / "prophet" / "DATASET_LICENCES.md"
_LICENCE_ROW = re.compile(r"^\|\s*`([a-z_]+)`\s*\|\s*(.+?)\s*\|\s*$")


def licence_text(key: str) -> str:
    for line in LICENCE_DOC.read_text(encoding="utf-8").splitlines():
        row = _LICENCE_ROW.match(line)
        if row and row.group(1) == key:
            return row.group(2)
    raise KeyError(f"no licence entry '{key}' in {LICENCE_DOC.relative_to(ROOT)}")


@dataclass(frozen=True)
class Source:
    name: str
    title: str
    files: tuple  # ((raw file name, url), ...)
    licence_key: str
    domain: str
    freq: str | None  # pandas alias of the native step, None if irregular
    entities: str
    normalise: Callable  # (dict raw_name -> Path, Source) -> pd.DataFrame
    notes: str = ""
    regular: bool = True  # False skips the gap measurement (transaction logs)
    covariates: str = "none"


# ---------------------------------------------------------------------------
# Normalisers. Each returns long data with columns entity, timestamp, value and
# optional covariates. Missing, zero and duplicate values are NOT removed.
# ---------------------------------------------------------------------------

def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def norm_fred(paths, src, series: str):
    raw = pd.read_csv(paths[f"{series}.csv"], dtype=str)
    date_col, val_col = raw.columns[0], raw.columns[1]
    return pd.DataFrame({
        "entity": series,
        "timestamp": pd.to_datetime(raw[date_col]),
        "value": _num(raw[val_col].replace(".", np.nan)),
    })


def norm_ecb(paths, src):
    with zipfile.ZipFile(paths["eurofxref-hist.zip"]) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = pd.read_csv(zf.open(name), dtype=str)
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
    long = raw.melt(id_vars="Date", var_name="entity", value_name="value")
    out = pd.DataFrame({
        "entity": long["entity"],
        "timestamp": pd.to_datetime(long["Date"]),
        "value": _num(long["value"]),
    })
    return out.sort_values(["entity", "timestamp"], ignore_index=True)


def norm_uci_electricity(paths, src):
    with zipfile.ZipFile(paths["electricityloaddiagrams20112014.zip"]) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
        wide = pd.read_csv(zf.open(name), sep=";", decimal=",", index_col=0)
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "timestamp"
    long = wide.reset_index().melt(id_vars="timestamp", var_name="entity", value_name="value")
    long["entity"] = long["entity"].astype("category")
    long["value"] = long["value"].astype("float32")
    return long


def norm_household(paths, src):
    with zipfile.ZipFile(paths["individual+household+electric+power+consumption.zip"]) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
        raw = pd.read_csv(zf.open(name), sep=";", na_values=["?"], dtype=str)
    ts = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
    out = pd.DataFrame({"entity": "household_1", "timestamp": ts,
                        "value": _num(raw["Global_active_power"])})
    for col in ["Global_reactive_power", "Voltage", "Global_intensity",
                "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
        out[col] = _num(raw[col])
    return out


def _beijing_station(handle, base: str) -> pd.DataFrame:
    station = re.search(r"PRSA_Data_(.+?)_2013", base).group(1)
    raw = pd.read_csv(handle, na_values=["NA"])
    frame = pd.DataFrame({
        "entity": station,
        "timestamp": pd.to_datetime(raw[["year", "month", "day", "hour"]]),
        "value": _num(raw["PM2.5"]),
    })
    for col in ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]:
        frame[col] = _num(raw[col])
    frame["wd"] = raw["wd"].astype(str)
    return frame


def norm_beijing(paths, src):
    frames = []
    with zipfile.ZipFile(paths["beijing+multi+site+air+quality+data.zip"]) as zf:
        for n in sorted(zf.namelist()):
            base = os.path.basename(n)
            if n.startswith("__MACOSX"):
                continue
            if base.endswith(".zip"):
                # The UCI bundle nests the per-station CSVs in an inner zip.
                with zipfile.ZipFile(zf.open(n)) as inner:
                    for m in sorted(inner.namelist()):
                        mb = os.path.basename(m)
                        if mb.startswith("PRSA_Data_") and mb.endswith(".csv"):
                            frames.append(_beijing_station(inner.open(m), mb))
            elif base.startswith("PRSA_Data_") and base.endswith(".csv"):
                frames.append(_beijing_station(zf.open(n), base))
    return pd.concat(frames, ignore_index=True).sort_values(["entity", "timestamp"], ignore_index=True)


def norm_retail(paths, src):
    with zipfile.ZipFile(paths["online+retail+ii.zip"]) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
        sheets = pd.read_excel(zf.open(name), sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    raw = pd.concat(sheets.values(), ignore_index=True)
    return pd.DataFrame({
        "entity": raw["StockCode"].astype(str),
        "timestamp": pd.to_datetime(raw["InvoiceDate"]),
        "value": _num(raw["Quantity"]),
        "price": _num(raw["Price"]),
        "invoice": raw["Invoice"].astype(str),
        "customer_id": raw["Customer ID"],
        "country": raw["Country"],
        "description": raw["Description"].astype(str),
    })


FREQ_MAP = {
    "minutely": "min", "10_minutes": "10min", "half_hourly": "30min", "hourly": "h",
    "daily": "D", "weekly": "W-SUN", "monthly": "MS", "quarterly": "QS", "yearly": "YS",
}


def parse_tsf(text: str):
    """Monash .tsf parser: header attributes, then 'name:start:v1,v2,...' lines."""
    attrs, rows, in_data = {}, [], False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if not in_data:
            if s.lower().startswith("@data"):
                in_data = True
            elif s.lower().startswith("@frequency"):
                attrs["frequency"] = s.split(None, 1)[1].strip()
            continue
        name, start, values = s.split(":", 2)
        rows.append((name, start, values))
    return attrs, rows


def tsf_to_long(text: str) -> pd.DataFrame:
    attrs, rows = parse_tsf(text)
    freq = FREQ_MAP[attrs["frequency"]]
    entities, stamps, values = [], [], []
    for name, start, vals in rows:
        start = re.sub(r"(\d{2})-(\d{2})-(\d{2})$", r"\1:\2:\3", start.strip())
        arr = np.array([np.nan if v.strip() == "?" else float(v) for v in vals.split(",")])
        ts = pd.date_range(start=pd.Timestamp(start), periods=len(arr), freq=freq)
        entities.append(np.repeat(name, len(arr)))
        stamps.append(ts.values)
        values.append(arr)
    return pd.DataFrame({
        "entity": np.concatenate(entities),
        "timestamp": pd.to_datetime(np.concatenate(stamps)),
        "value": np.concatenate(values),
    })


def norm_monash_zip(paths, src):
    with zipfile.ZipFile(next(iter(paths.values()))) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".tsf"))
        text = zf.read(name).decode("utf-8", errors="replace")
    return tsf_to_long(text)


def norm_ett(paths, src, unit: str):
    raw = pd.read_csv(paths[f"{unit}.csv"])
    out = pd.DataFrame({"entity": unit, "timestamp": pd.to_datetime(raw["date"]), "value": raw["OT"]})
    for col in ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]:
        out[col] = raw[col]
    return out


def norm_m4(paths, src):
    train = pd.read_csv(paths["Daily-train.csv"], index_col=0)
    test = pd.read_csv(paths["Daily-test.csv"], index_col=0)
    assert list(train.index) == list(test.index), "M4 train and test series do not align"
    tr = train.to_numpy(dtype=float)
    te = test.to_numpy(dtype=float)
    tr_mask, te_mask = ~np.isnan(tr), ~np.isnan(te)
    tr_len = tr_mask.sum(axis=1)
    # Interior NaN count (gaps inside a series, excluding trailing padding) is recorded, not removed.
    last_pos = tr.shape[1] - np.argmax(tr_mask[:, ::-1], axis=1)
    interior = int((last_pos - tr_len).sum())
    src_notes[src.name] = {"m4_interior_nan_train": interior}
    r1, c1 = np.nonzero(tr_mask)
    r2, c2 = np.nonzero(te_mask)
    ids = train.index.to_numpy(dtype=object)
    long = pd.DataFrame({
        "entity": np.concatenate([ids[r1], ids[r2]]),
        "timestamp": np.concatenate([c1 + 1, tr_len[r2] + c2 + 1]).astype("int32"),
        "value": np.concatenate([tr[r1, c1], te[r2, c2]]),
        "split": np.concatenate([np.repeat("train", len(r1)), np.repeat("test", len(r2))]),
    })
    return long


def norm_owid(paths, src):
    raw = pd.read_csv(paths["owid-covid-data.csv"], low_memory=False)
    out = pd.DataFrame({"entity": raw["location"], "timestamp": pd.to_datetime(raw["date"]),
                        "value": _num(raw["new_cases"])})
    for col in ["iso_code", "continent", "total_cases", "new_deaths", "total_deaths",
                "new_tests", "stringency_index"]:
        if col in raw.columns:
            out[col] = raw[col]
    return out


def norm_gift_arrow(paths, src, freq: str):
    import pyarrow as pa

    path = paths["data-00000-of-00001.arrow"]
    with pa.memory_map(str(path)) as mm:
        try:
            table = pa.ipc.open_stream(mm).read_all()
        except pa.ArrowInvalid:
            table = pa.ipc.open_file(mm).read_all()
    df = table.to_pandas()
    return explode_hf_series(df, freq)


def explode_hf_series(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Explode one-row-per-series tables (item_id, start, target[, freq]).

    A 2-D target (channels x steps) becomes one entity per channel: '<item_id>:v<k>'.
    The row's own freq column wins over the default when present.
    """
    frames = []
    for _, row in df.iterrows():
        raw_t = row["target"]
        if len(raw_t) and np.ndim(raw_t[0]) == 0:
            target = np.asarray(raw_t, dtype=float)
        else:
            target = np.stack([np.asarray(c, dtype=float) for c in raw_t])
        step = str(row["freq"]) if "freq" in df.columns and row["freq"] else freq
        step = re.sub(r"^(\d*)T$", r"\1min", step)  # legacy pandas alias, e.g. 10T
        channels = target if target.ndim == 2 else target[None, :]
        for k, values in enumerate(channels):
            ts = pd.date_range(start=pd.Timestamp(row["start"]), periods=len(values), freq=step)
            name = str(row["item_id"]) if target.ndim == 1 else f"{row['item_id']}:v{k:02d}"
            frames.append(pd.DataFrame({"entity": name, "timestamp": ts, "value": values}))
    return pd.concat(frames, ignore_index=True)


def norm_chronos_parquet(paths, src, freq: str):
    df = pd.read_parquet(paths["train-00000-of-00001.parquet"])
    id_col = "id" if "id" in df.columns else "item_id"
    rows = []
    for _, row in df.iterrows():
        target = np.asarray(row["target"], dtype=float)
        if "timestamp" in df.columns and isinstance(row["timestamp"], (list, np.ndarray)):
            ts = pd.to_datetime(np.asarray(row["timestamp"]))
        else:
            ts = pd.date_range(start=pd.Timestamp(row["start"]), periods=len(target), freq=freq)
        rows.append(pd.DataFrame({"entity": str(row[id_col]), "timestamp": ts, "value": target}))
    return pd.concat(rows, ignore_index=True)


# Shared scratch for normalisers that report extra facts (keyed by dataset name).
src_notes: dict = {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def fred(series, name, title, freq, domain, notes=""):
    return Source(
        name=name, title=title,
        files=((f"{series}.csv", FRED_CSV.format(series)),),
        licence_key="fred", domain=domain, freq=freq,
        entities=f"1 series ({series})",
        normalise=lambda paths, src, s=series: norm_fred(paths, src, s),
        notes=notes,
    )


SOURCES: list[Source] = [
    # FX (FRED daily reference rates, business days; holidays are missing rows)
    fred("DEXUSEU", "fred_dexuseu", "USD per EUR (FRED DEXUSEU)", "D", "FX",
         "Daily noon buying rates in New York. Holidays appear as NA ('.') in FRED."),
    fred("DEXJPUS", "fred_dexjpus", "JPY per USD (FRED DEXJPUS)", "D", "FX",
         "Holidays appear as NA ('.') in FRED."),
    fred("DEXUSUK", "fred_dexusuk", "USD per GBP (FRED DEXUSUK)", "D", "FX",
         "Holidays appear as NA ('.') in FRED."),
    fred("DEXCHUS", "fred_dexchus", "CNY per USD (FRED DEXCHUS)", "D", "FX",
         "Regime shift in July 2005 (managed float); holidays as NA."),
    fred("DEXCAUS", "fred_dexcaus", "CAD per USD (FRED DEXCAUS)", "D", "FX",
         "Holidays appear as NA ('.') in FRED."),
    # Commodities
    fred("DCOILWTICO", "fred_dcoilwtico", "WTI crude oil spot, daily (FRED)", "D", "commodities",
         "Includes the April 2020 negative-price episode if present in the source."),
    fred("DCOILBRENTEU", "fred_dcoilbrenteu", "Brent crude oil spot, daily (FRED)", "D", "commodities"),
    fred("GASREGW", "fred_gasregw", "US regular gasoline retail, weekly (FRED)", "W-MON", "commodities"),
    fred("DHHNGSP", "fred_dhhngsp", "Henry Hub natural gas spot, daily (FRED)", "D", "commodities"),
    # Rates and macro
    fred("DGS10", "fred_dgs10", "10-year Treasury constant maturity, daily (FRED)", "D", "rates"),
    fred("T10Y2Y", "fred_t10y2y", "10-year minus 2-year Treasury spread, daily (FRED)", "D", "rates",
         "Negative spells are real inversions, not defects."),
    fred("UNRATE", "fred_unrate", "US unemployment rate, monthly (FRED)", "MS", "macro"),
    fred("CPIAUCSL", "fred_cpiaucsl", "US CPI all urban consumers, monthly (FRED)", "MS", "macro",
         "Index level. Revised by BLS; the vintage here is whatever FRED served at fetch time."),
    fred("VIXCLS", "fred_vixcls", "CBOE VIX, daily (FRED)", "D", "markets",
         "Holidays appear as NA ('.') in FRED."),
    fred("SP500", "fred_sp500", "S&P 500 index, daily (FRED, recent years only)", "D", "markets",
         "FRED serves only roughly the last 10 years of this series."),
    fred("GOLDAMGBD228NLBM", "fred_gold_lbma", "Gold London PM fix (FRED GOLDAMGBD228NLBM)", "D", "commodities",
         "Expected to fail: FRED returned 404 when checked. Kept in the registry so the failure is recorded."),

    Source(
        name="ecb_fx_hist", title="ECB euro foreign exchange reference rates, historical",
        files=(("eurofxref-hist.zip", ECB_FX_ZIP),), licence_key="ecb", domain="FX", freq="B",
        entities="about 40 currencies quoted against EUR",
        normalise=norm_ecb,
        notes="Business-day reference rates. 'N/A' marks non-publication days and is kept as NaN. Data is served newest first; rows are sorted here.",
    ),

    Source(
        name="uci_electricity_15min", title="UCI Electricity Load Diagrams 2011-2014",
        files=(("electricityloaddiagrams20112014.zip", UCI_ELEC),), licence_key="uci",
        domain="electricity", freq="15min", entities="370 client meters (MT_001 to MT_370)",
        normalise=norm_uci_electricity,
        notes="Kilowatt-hours per 15 minutes, source timezone Portugal (no tz information). Values stored as float32. Many clients have zero runs before they were connected.",
    ),
    Source(
        name="uci_household_power", title="UCI Individual household electric power consumption",
        files=(("individual+household+electric+power+consumption.zip", UCI_HOUSEHOLD),), licence_key="uci",
        domain="electricity", freq="min", entities="1 household",
        normalise=norm_household,
        notes="Source marks missing values with '?'. value = Global_active_power (kW); sub-metering and voltage kept as covariates.",
        covariates="Global_reactive_power, Voltage, Global_intensity, Sub_metering_1..3",
    ),
    Source(
        name="uci_beijing_air_quality", title="UCI Beijing multi-site air quality (12 stations)",
        files=(("beijing+multi+site+air+quality+data.zip", UCI_BEIJING),), licence_key="uci",
        domain="air quality / weather", freq="h", entities="12 monitoring stations",
        normalise=norm_beijing,
        notes="value = PM2.5 (ug/m3). Covariates include meteorology (TEMP, PRES, DEWP, RAIN, WSPM, wd). 'NA' kept as NaN.",
        covariates="PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM, wd",
    ),
    Source(
        name="uci_online_retail_ii", title="UCI Online Retail II (transaction lines)",
        files=(("online+retail+ii.zip", UCI_RETAIL),), licence_key="uci",
        domain="retail", freq=None, regular=False, entities="about 4,600 StockCodes (product lines)",
        normalise=norm_retail,
        notes="One row per invoice line, not a sales series. value = Quantity (negative = returns). Cancelled invoices start with 'C'. Aggregation to daily demand is left to the agent.",
        covariates="price, invoice, customer_id, country, description",
    ),

    Source(
        name="monash_car_parts_missing", title="Monash car parts with missing values (monthly)",
        files=(("car_parts_dataset_with_missing_values.zip", ZENODO_CAR_PARTS),),
        licence_key="zenodo_monash", domain="retail / intermittent demand", freq="MS",
        entities="2,674 part series (as recorded)", normalise=norm_monash_zip,
        notes="Monash .tsf format parsed here. '?' marks missing values, kept as NaN. Very intermittent: many zeros.",
    ),
    Source(
        name="monash_london_smart_meters_missing", title="Monash London smart meters with missing values",
        files=(("london_smart_meters_dataset_with_missing_values.zip", ZENODO_LONDON),),
        licence_key="zenodo_monash", domain="electricity (households)", freq="30min",
        entities="about 5,560 households (as recorded)", normalise=norm_monash_zip,
        notes="Monash .tsf format parsed here. '?' marks missing values, kept as NaN. Households start and stop at different dates.",
    ),

    Source(
        name="ett_etth1", title="ETT transformer, ETTh1 (hourly)",
        files=(("ETTh1.csv", ETT_RAW.format("ETTh1")),), licence_key="unverified",
        domain="electricity transformer", freq="h", entities="1 transformer (ETTh1)",
        normalise=lambda paths, src: norm_ett(paths, src, "ETTh1"),
        notes="value = OT (oil temperature, target). Six load covariates kept.",
        covariates="HUFL, HULL, MUFL, MULL, LUFL, LULL",
    ),
    Source(
        name="ett_ettm1", title="ETT transformer, ETTm1 (15-minute)",
        files=(("ETTm1.csv", ETT_RAW.format("ETTm1")),), licence_key="unverified",
        domain="electricity transformer", freq="15min", entities="1 transformer (ETTm1)",
        normalise=lambda paths, src: norm_ett(paths, src, "ETTm1"),
        notes="value = OT (oil temperature, target). Six load covariates kept.",
        covariates="HUFL, HULL, MUFL, MULL, LUFL, LULL",
    ),

    Source(
        name="m4_daily", title="M4 competition, Daily subset (train plus 14-step test)",
        files=(("Daily-train.csv", M4_RAW.format("Train", "train")),
               ("Daily-test.csv", M4_RAW.format("Test", "test"))),
        licence_key="unverified", domain="mixed (competition)", freq="D",
        entities="4,227 daily series", normalise=norm_m4,
        notes="M4 has no calendar dates: timestamp is an integer step index (test steps continue after each series' train length). Trailing padding is dropped; interior NaN are kept (count in README). split column marks train vs test.",
        covariates="split",
    ),

    Source(
        name="owid_covid_daily", title="Our World in Data COVID-19 (daily, all locations)",
        files=(("owid-covid-data.csv", OWID_CSV),), licence_key="owid",
        domain="epidemiology", freq="D", entities="about 250 locations including aggregates",
        normalise=norm_owid,
        notes="value = new_cases as reported. Reporting artifacts, weekday effects, backfills and negative corrections are in the source.",
        covariates="iso_code, continent, total_cases, new_deaths, total_deaths, new_tests, stringency_index",
    ),

    Source(
        name="gift_hospital", title="GIFT-Eval hospital (Hugging Face Salesforce/GiftEval arrow)",
        files=(("data-00000-of-00001.arrow", HF_GIFT.format("hospital")),), licence_key="gift_card",
        domain="healthcare counts", freq="MS", entities="767 series (as recorded)",
        normalise=lambda paths, src: norm_gift_arrow(paths, src, "MS"),
        notes="Arrow IPC file from the Hugging Face repo. Frequency taken from the row freq column.",
    ),
    Source(
        name="gift_jena_weather", title="GIFT-Eval Jena climate (Hugging Face arrow, 10-minute file)",
        files=(("data-00000-of-00001.arrow", HF_GIFT.format("jena_weather")),), licence_key="gift_card",
        domain="weather", freq="10min", entities="weather station series (as recorded)",
        normalise=lambda paths, src: norm_gift_arrow(paths, src, "10min"),
        notes="Top-level arrow file of the jena_weather folder (one row, 21 channels x 52,704 steps, freq 10T). Each channel is exported as its own entity <item_id>:vNN; channel meanings are not labelled in the file.",
    ),
    Source(
        name="gift_covid_deaths", title="GIFT-Eval COVID deaths (Hugging Face arrow)",
        files=(("data-00000-of-00001.arrow", HF_GIFT.format("covid_deaths")),), licence_key="gift_card",
        domain="epidemiology", freq="D", entities="about 266 countries (as recorded)",
        normalise=lambda paths, src: norm_gift_arrow(paths, src, "D"),
        notes="Cumulative deaths per country in the GIFT copy (confirm). Daily frequency assumed from the GIFT-Eval folder name.",
    ),
    Source(
        name="chronos_monash_electricity_hourly", title="Chronos corpus, Monash electricity hourly (Hugging Face parquet)",
        files=(("train-00000-of-00001.parquet", HF_CHRONOS.format("monash_electricity_hourly/train-00000-of-00001.parquet")),),
        licence_key="chronos_card",
        domain="electricity", freq="h", entities="series as recorded in the Chronos copy",
        normalise=lambda paths, src: norm_chronos_parquet(paths, src, "h"),
        notes="Chronos-style corpus copy. Start timestamps and frequency assumed hourly; confirm against the source Monash file.",
    ),
]

BY_NAME = {s.name: s for s in SOURCES}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fetch_file(client: httpx.Client, url: str, dest: Path, max_bytes: int | None) -> dict:
    """Stream url into dest. Resumes from dest.part. Returns a manifest-ready record."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(3):
        start = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code == 416 and start:
                    break  # the partial file is already complete
                if resp.status_code not in (200, 206):
                    return {"url": url, "status": "failed", "http_status": resp.status_code, "fetched_at": now()}
                resuming = resp.status_code == 206
                length = resp.headers.get("content-length")
                total = (int(length) + (start if resuming else 0)) if length else None
                if total is not None and max_bytes is not None and total > max_bytes:
                    return {"url": url, "status": "skipped_too_large", "http_status": resp.status_code,
                            "bytes_expected": total, "fetched_at": now()}
                with part.open("ab" if resuming else "wb") as fh:
                    for chunk in resp.iter_bytes(1 << 20):
                        fh.write(chunk)
            if total is not None and part.stat().st_size != total:
                return {"url": url, "status": "failed", "error": "incomplete download; rerun to resume",
                        "bytes_expected": total, "fetched_at": now()}
            break
        except httpx.TransportError as exc:
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    else:
        return {"url": url, "status": "failed", "error": f"transport error: {last_exc}", "fetched_at": now()}

    part.replace(dest)
    return {"url": url, "status": "ok", "http_status": 200, "bytes": dest.stat().st_size,
            "sha256": sha256_of(dest), "fetched_at": now()}


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"description": "Downloads and normalised outputs for scripts/fetch_datasets.py. Paths are relative to the repo root.",
            "datasets": {}}


def save_manifest(manifest: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_name(MANIFEST.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    tmp.replace(MANIFEST)


def download_source(client, src: Source, manifest: dict, max_bytes, force: bool) -> dict:
    entry = manifest["datasets"].setdefault(src.name, {})
    prev = {f["name"]: f for f in entry.get("files", [])}
    files_out = []
    status = "ok"
    for raw_name, url in src.files:
        dest = RAW / src.name / raw_name
        old = prev.get(raw_name)
        if (not force and old and old.get("status") == "ok" and old.get("url") == url
                and dest.exists() and dest.stat().st_size == old.get("bytes")):
            rec = dict(old, cached=True)
        else:
            print(f"  GET {url}", flush=True)
            rec = fetch_file(client, url, dest, max_bytes)
        rec["name"] = raw_name
        rec["path"] = str(dest.relative_to(ROOT)) if rec.get("status") == "ok" else None
        files_out.append(rec)
        if rec["status"] != "ok":
            status = rec["status"] if status == "ok" else status
    entry.update({
        "title": src.title, "licence": licence_text(src.licence_key), "domain": src.domain,
        "native_freq": src.freq, "files": files_out, "download_status": status,
    })
    return entry


# ---------------------------------------------------------------------------
# Measurement and output
# ---------------------------------------------------------------------------

def measure(df: pd.DataFrame, regular: bool) -> dict:
    m = {
        "rows": int(len(df)),
        "entities": int(df["entity"].nunique()),
        "timestamp_min": str(df["timestamp"].min()),
        "timestamp_max": str(df["timestamp"].max()),
        "missing_values": int(df["value"].isna().sum()),
        "missing_share_pct": round(100 * float(df["value"].isna().mean()), 3),
        "zeros": int((df["value"] == 0).sum()),
        "negative_values": int((df["value"] < 0).sum()),
        "duplicate_entity_timestamp": int(df.duplicated(["entity", "timestamp"]).sum()),
    }
    s = df.sort_values(["entity", "timestamp"], ignore_index=True)
    jumps = s.groupby("entity", observed=True)["value"].diff().abs()
    if jumps.notna().any():
        i = int(jumps.idxmax())
        m["largest_jump"] = {
            "abs_change": float(jumps.iloc[i]), "entity": str(s.loc[i, "entity"]),
            "at": str(s.loc[i, "timestamp"]), "value_before": float(s["value"].iloc[i - 1]),
            "value_after": float(s.loc[i, "value"]),
        }
    if regular:
        dt = s.groupby("entity", observed=True)["timestamp"].diff()
        if dt.notna().any():
            step = dt.median()
            gap_mask = dt > 1.5 * step
            m["median_step"] = str(step)
            m["step_gaps"] = int(gap_mask.sum())
            if gap_mask.any():
                m["longest_gap"] = str(dt[gap_mask].max())
                m["missing_steps_implied"] = int(np.round(dt[gap_mask] / step - 1).sum())
    return m


def write_readme(src: Source, out_dir: Path, metrics: dict, files: list, rel_out: str, extra: dict) -> None:
    lines = [
        f"# {src.name}", "",
        src.title, "",
        f"- Domain: {src.domain}",
        f"- Native frequency: {src.freq or 'irregular (event or transaction log)'}",
        f"- Entities: {src.entities}",
        f"- Covariates: {src.covariates}",
        f"- Licence (see docs/prophet/DATASET_LICENCES.md, key {src.licence_key}): {licence_text(src.licence_key)}",
        f"- Output: `{rel_out}` (Parquet, long format; columns entity, timestamp, value" +
        (f", {src.covariates}" if src.covariates != "none" else "") + ")",
        "", "## Source files", "",
    ]
    for f in files:
        lines.append(f"- {f['url']}")
        lines.append(f"  - bytes: {f.get('bytes')}, sha256: {f.get('sha256')}, fetched: {f.get('fetched_at')}")
    lines += ["", "## Measured with pandas (defects kept, nothing removed)", ""]
    lines.append(f"- Rows: {metrics['rows']:,}")
    lines.append(f"- Entities: {metrics['entities']:,}")
    lines.append(f"- Timestamp range: {metrics['timestamp_min']} to {metrics['timestamp_max']}")
    lines.append(f"- Missing values (NaN in value): {metrics['missing_values']:,} ({metrics['missing_share_pct']}%)")
    lines.append(f"- Zeros: {metrics['zeros']:,}")
    lines.append(f"- Negative values: {metrics['negative_values']:,}")
    lines.append(f"- Duplicate (entity, timestamp) pairs: {metrics['duplicate_entity_timestamp']:,}")
    if "median_step" in metrics:
        lines.append(f"- Median step between observations: {metrics['median_step']}")
        lines.append(f"- Gaps (step > 1.5 x median): {metrics['step_gaps']:,}")
        if "longest_gap" in metrics:
            lines.append(f"- Longest gap: {metrics['longest_gap']}; implied missing steps: {metrics.get('missing_steps_implied', 0):,}")
    if "largest_jump" in metrics:
        j = metrics["largest_jump"]
        lines.append(f"- Largest consecutive jump in value: {j['abs_change']:.6g} "
                     f"(entity {j['entity']}, at {j['at']}, {j['value_before']:.6g} to {j['value_after']:.6g})")
    for k, v in extra.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Notes", "", src.notes or "None.", "",
              "## Reproduce", "",
              "```", f"PYTHONPATH=src .venv/bin/python scripts/fetch_datasets.py --only {src.name}", "```", ""]
    (out_dir / "README.md").write_text("\n".join(lines))


def normalise_source(src: Source, manifest: dict) -> dict:
    entry = manifest["datasets"][src.name]
    paths = {f["name"]: ROOT / f["path"] for f in entry["files"] if f.get("status") == "ok"}
    src_notes.pop(src.name, None)
    df = src.normalise(paths, src)
    df = df.reset_index(drop=True)
    out_dir = DATA / src.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{src.name}.parquet"
    df.to_parquet(out_file, index=False, compression="zstd")
    metrics = measure(df, src.regular)
    extra = dict(src_notes.pop(src.name, {}))
    write_readme(src, out_dir, metrics, entry["files"], str(out_file.relative_to(ROOT)), extra)
    entry["output"] = str(out_file.relative_to(ROOT))
    entry["output_bytes"] = out_file.stat().st_size
    entry["normalised"] = dict(metrics, **extra)
    entry["normalised_at"] = now()
    entry.pop("normalise_error", None)
    return metrics


def select(only: str | None) -> list[Source]:
    if not only:
        return list(SOURCES)
    names = [n.strip() for n in only.split(",") if n.strip()]
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        sys.exit(f"Unknown dataset name(s): {', '.join(unknown)}. Use --list.")
    return [BY_NAME[n] for n in names]


def print_list() -> None:
    print(f"{'name':36} {'freq':8} {'files':5} domain")
    for s in SOURCES:
        print(f"{s.name:36} {str(s.freq):8} {len(s.files):<5} {s.domain}")
    print(f"\n{len(SOURCES)} datasets")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list the registry and exit")
    ap.add_argument("--only", help="comma-separated dataset names")
    ap.add_argument("--max-mb", type=float, default=300.0,
                    help="skip any file whose Content-Length exceeds this many MB (default 300)")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--download-only", action="store_true", help="fetch raw files, skip normalising")
    args = ap.parse_args(argv)

    if args.list:
        print_list()
        return 0

    selected = select(args.only)
    max_bytes = int(args.max_mb * 1_000_000) if args.max_mb else None
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    with httpx.Client(headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}, follow_redirects=True,
                      timeout=httpx.Timeout(60.0, read=300.0)) as client:
        for src in selected:
            print(f"[{src.name}]", flush=True)
            download_source(client, src, manifest, max_bytes, args.force)
            save_manifest(manifest)

    if not args.download_only:
        for src in selected:
            entry = manifest["datasets"][src.name]
            if entry["download_status"] != "ok":
                print(f"[{src.name}] not normalised: {entry['download_status']}")
                continue
            print(f"[{src.name}] normalising", flush=True)
            try:
                m = normalise_source(src, manifest)
                print(f"  rows={m['rows']:,} entities={m['entities']:,} missing={m['missing_values']:,}")
            except Exception as exc:  # keep going; the failure goes into the manifest
                entry["normalise_error"] = f"{type(exc).__name__}: {exc}"
                print(f"  normalise failed: {entry['normalise_error']}")
            save_manifest(manifest)

    print("\nSummary")
    for src in selected:
        e = manifest["datasets"][src.name]
        codes = ",".join(str(f.get("http_status", f.get("error", "?"))) for f in e["files"])
        print(f"  {src.name:36} download={e['download_status']:20} http={codes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
