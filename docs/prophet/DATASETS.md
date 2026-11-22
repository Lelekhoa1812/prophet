# Real-world time-series datasets for Prophet stress tests

Status: downloaded and normalised on 2026-11-22. Every number below was measured with pandas from the files in `data/datasets/`. Nothing here is cleaned: missing values, zeros, duplicates, gaps and outliers are as the source published them.

Layout:

- Raw downloads: `data/datasets/_raw/<name>/` (git-ignored, with sha256 in `data/datasets/MANIFEST.json`).
- Normalised tables: `data/datasets/<name>/<name>.parquet`. Long format: `entity`, `timestamp`, `value`, plus covariates named in the table.
- Per-dataset measurements: `data/datasets/<name>/README.md`.
- Downloader and normaliser: `scripts/fetch_datasets.py`.

Nothing was written to `examples/` or outside `data/datasets/`.

## What was downloaded

Sizes are the Parquet files. Raw sizes are in `MANIFEST.json`.

| Name | Domain | Freq | Entities | Rows | Date range (measured) | Parquet | Covariates | Measured defects | Licence |
|---|---|---|---|---|---|---|---|---|---|
| fred_dexuseu | FX, USD per EUR | Daily (business) | 1 | 7,270 | 1999-01-04 to 2026-11-13 | 84 KB | none | 282 NA (3.9%); 1,453 step gaps (weekends and holidays); largest jump 0.062 (2009-03-19) | FRED terms |
| fred_dexjpus | FX, JPY per USD | Daily (business) | 1 | 14,575 | 1971-01-04 to 2026-11-13 | 159 KB | none | 572 NA (3.9%); largest jump 18.21 (1974-01-07) | FRED terms |
| fred_dexusuk | FX, USD per GBP | Daily (business) | 1 | 14,575 | 1971-01-04 to 2026-11-13 | 165 KB | none | 566 NA (3.9%); largest jump 0.116 (2016-06-24) | FRED terms |
| fred_dexchus | FX, CNY per USD | Daily (business) | 1 | 11,966 | 1981-01-02 to 2026-11-13 | 137 KB | none | 517 NA (4.3%); largest jump 2.91 (1994-01-03) | FRED terms |
| fred_dexcaus | FX, CAD per USD | Daily (business) | 1 | 14,575 | 1971-01-04 to 2026-11-13 | 158 KB | none | 560 NA (3.8%); largest jump 0.064 (2008-10-29) | FRED terms |
| fred_dcoilwtico | Commodity, WTI crude | Daily (business) | 1 | 10,664 | 1986-01-02 to 2026-11-17 | 120 KB | none | 1,117 NA (10.5%); 1 negative price; largest jump 55.29 (2020-04-20) | FRED terms |
| fred_dcoilbrenteu | Commodity, Brent crude | Daily (business) | 1 | 10,305 | 1987-05-20 to 2026-11-17 | 116 KB | none | 1,173 NA (11.4%); largest jump 20.69 (2026-10-02) | FRED terms |
| fred_gasregw | Commodity, US gasoline | Weekly | 1 | 1,892 | 1990-08-20 to 2026-11-16 | 23 KB | none | 6 NA (0.3%); largest jump 0.494 (2022-03-07) | FRED terms |
| fred_dhhngsp | Commodity, Henry Hub gas | Daily (business) | 1 | 7,791 | 1997-01-07 to 2026-11-17 | 73 KB | none | 2,059 NA (26.4%); largest jump 22.3 (2026-01-23) | FRED terms |
| fred_dgs10 | Rates, 10Y Treasury | Daily (business) | 1 | 16,928 | 1962-01-02 to 2026-11-19 | 165 KB | none | 722 NA (4.3%); largest jump 0.75 (1987-10-20) | FRED terms |
| fred_t10y2y | Rates, 10Y minus 2Y spread | Daily (business) | 1 | 13,169 | 1976-06-01 to 2026-11-20 | 123 KB | none | 554 NA (4.2%); 2,049 negative (real inversions); 30 zeros; largest jump 0.55 (1981-01-05) | FRED terms |
| fred_unrate | Macro, US unemployment | Monthly | 1 | 946 | 1948-01-01 to 2026-10-01 | 11 KB | none | 1 NA; largest jump 10.4 (2020-04-01) | FRED terms |
| fred_cpiaucsl | Macro, US CPI index | Monthly | 1 | 958 | 1947-01-01 to 2026-10-01 | 14 KB | none | 1 NA; largest jump 3.84 (2008-11-01) | FRED terms; BLS revisions |
| fred_vixcls | Markets, VIX | Daily (business) | 1 | 9,623 | 1990-01-02 to 2026-11-19 | 100 KB | none | 302 NA (3.1%); largest jump 24.86 (2020-03-16) | FRED terms |
| fred_sp500 | Markets, S&P 500 | Daily (business) | 1 | 2,609 | 2016-11-21 to 2026-11-19 | 35 KB | none | 96 NA (3.7%); largest jump 474.13 (2025-04-09) | FRED terms; about 10 years only |
| ecb_fx_hist | FX, about 40 currencies vs EUR | Daily (business) | 41 | 292,781 | 1999-01-04 to 2026-11-20 | 1.5 MB | none | 70,644 N/A (24.1%, non-publication days); largest jump 425,633 (TRL, 2001-02-22) | ECB reuse policy |
| uci_electricity_15min | Electricity, 370 clients | 15 min | 370 | 51,894,720 | 2011-01-01 00:15 to 2015-01-01 00:00 | 379 MB | none | 0 NA; 10,457,342 zeros (20.2%); largest jump 110,000 kWh (MT_362, 2012-09-21 09:30) | CC BY 4.0 |
| uci_household_power | Electricity, one household | 1 min | 1 | 2,075,259 | 2006-12-16 17:24 to 2010-11-26 21:02 | 20 MB | Reactive power, voltage, intensity, sub-metering 1 to 3 | 25,979 NA (1.25%, source '?'); largest jump 5.06 kW (2007-09-05 06:38) | CC BY 4.0 |
| uci_beijing_air_quality | Air quality (PM2.5), 12 stations | Hourly | 12 | 420,768 | 2013-03-01 00:00 to 2017-02-28 23:00 | 6 MB | PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, WSPM, wd | 8,739 NA in PM2.5 (2.1%); largest jump 853 (Dingling, 2016-02-08 01:00) | CC BY 4.0 |
| uci_online_retail_ii | Retail transaction lines | Irregular | 5,305 StockCodes | 1,067,371 | 2009-12-01 07:45 to 2011-12-09 12:50 | 6 MB | price, invoice, customer_id, country, description | value is Quantity: 22,950 negative (returns); 49,775 duplicate (entity, timestamp) pairs; largest jump 161,990 (StockCode 23843, 2011-12-09) | CC BY 4.0 |
| monash_car_parts_missing | Retail, intermittent parts sales | Monthly | 2,674 | 136,374 | 1998-01 to 2002-03 | 42 KB | none | 6,122 NA (4.5%); 97,398 zeros (71.4%); largest jump 52 (2000-05) | CC BY 4.0 (Zenodo, as checked) |
| monash_london_smart_meters_missing | Electricity, London households | 30 min | 5,560 | 166,528,896 | 2011-11-24 00:00:01 to 2014-02-27 23:30:01 | 573 MB | none | 5,486 NA (0.003%, far less than the name suggests); 1,986,614 zeros (1.2%); largest jump 8.66 kWh (2013-08-24 14:00:01) | CC BY 4.0 (Zenodo, as checked) |
| ett_etth1 | Electricity transformer, ETTh1 | Hourly | 1 | 17,420 | 2016-07-01 00:00 to 2018-06-26 19:00 | 287 KB | HUFL, HULL, MUFL, MULL, LUFL, LULL | 0 NA; 111 zeros; 154 negatives; largest jump 9.92 (2017-05-22) | Not verified |
| ett_ettm1 | Electricity transformer, ETTm1 | 15 min | 1 | 69,680 | 2016-07-01 00:00 to 2018-06-26 19:45 | 1.1 MB | as ETTh1 | 0 NA; 432 zeros; 623 negatives; largest jump 9.50 (2016-07-14 17:30) | Not verified |
| m4_daily | M4 competition, Daily (train plus 14-step test) | Daily (integer step index, no dates) | 4,227 | 10,023,836 | steps 1 to 9,933 | 48 MB | split (train or test) | 0 NA; 0 interior NaN in train (trailing padding dropped); largest jump 347,814 (D4225, step 2) | Not verified |
| owid_covid_daily | COVID-19 new cases, daily | Daily | 255 | 429,435 | 2020-01-01 to 2024-08-14 | 1.5 MB | iso_code, continent, total_cases, new_deaths, total_deaths, new_tests, stringency_index | 19,276 NA (4.5%); 368,385 zeros (85.8%); 7,770 duplicate (entity, date) pairs; largest jump 44,236,227 (World, 2022-12-25) | CC BY 4.0 (sources vary) |
| gift_hospital | Hospital admissions (monthly counts) | Monthly | 767 | 64,428 | 2000-01 to 2006-12 | 92 KB | none | 0 NA; 0 zeros; largest jump 3,333 (2000-02) | Not verified |
| gift_covid_deaths | COVID deaths by country (GIFT copy) | Daily | 266 | 56,392 | 2020-01-22 to 2020-08-20 | 42 KB | none | 0 NA; 21,522 zeros (38.2%); largest jump 4,143 (T203, 2020-08-14) | Not verified |
| gift_jena_weather | Weather, Jena climate (GIFT copy, 10 min) | 10 min | 21 channels (item_0:v00 to v20) | 1,106,784 | 2020-01-01 00:10 to 2021-01-01 00:00 | 4.5 MB | the other 20 channels of the same series, as separate entities | 189 NA (0.02%); 176,744 zeros; 12,650 negatives; largest jump 12,489 (item_0:v18, 2020-07-05 12:00) | Not verified |
| chronos_monash_electricity_hourly | Electricity, Monash copy (Chronos corpus) | Hourly | 321 | 8,443,584 | 2012-01-01 01:00 to 2015-01-01 00:00 | 29 MB | none | 0 NA; 91,817 zeros; largest jump 376,300 (T000313, 2013-07-10 11:00) | Not verified |

Where a row says "Not verified", the licence was not read from the source. Read the source's README or dataset card before any external use.

## Failures

| Dataset | Source | Result |
|---|---|---|
| fred_gold_lbma (FRED GOLDAMGBD228NLBM) | FRED CSV | HTTP 404. The series is not served at that ID. Not replaced. |

Four sources first failed with "incomplete download". The cause was gzip: the server compressed the body, so the Content-Length did not match the decoded bytes. The downloader now sends `Accept-Encoding: identity`, and all four were re-fetched and pass.

Other failures from the earlier research table were not attempted in this run: the Chronos M5 and monash_traffic parquet files, the Monash FRED-MD copy, the Max Planck Jena zip, Kaggle sources, PeMS, Citi Bike, NYC taxi and EIA. Several need login or returned errors earlier.

## Things the agent should know

- **Jena has 21 channels, not 14, and covers 2020, not 2009 to 2016.** The GIFT-Eval copy's dates and channel count differ from the original Max Planck release as recalled in the research notes. Channel labels are not in the file, so treat them as unlabelled.
- **OWID stops at 2024-08-14 in this copy.** The file is from the GitHub mirror. It has no 2025 or 2026 observations, so it cannot serve as a post-cutoff holdout.
- **London smart meters has far fewer NaN than its name suggests.** Only 5,486 NaN appear, from '?' markers. Check the README before writing any missingness test.
- **Business-day gaps are weekends and holidays, not defects.** The step-gap counts for FRED and ECB series include them. Read the README note before counting them as defects.
- **GIFT-Eval start dates and frequencies are assumed from folder names.** The hospital start (2000-01, monthly) and COVID start (2020-01-22, daily) were inferred. Confirm against `dataset_info.json` before results depend on them.
- **M4 has no calendar.** Its `timestamp` is an integer step index. Test rows continue after each series' train length.

## Leakage and evaluation

No dataset here is verified as leakage-free against any model's pretraining data. Public benchmark series, including M4, Monash-derived sets, GIFT-Eval and the Chronos corpus, are probably inside TimesFM and Chronos pretraining data. Treat them as risk-flagged, not clean.

For leakage-free evaluation, prefer the last months of FRED, ECB and OWID series beyond the model's training cutoff. Those series keep updating, and the FRED and ECB files here run to November 2026. OWID does not (see above).

The cutoff dates for TimesFM and Chronos are not verified in this work. Take them from each model's own documentation before drawing any conclusion from a holdout.

## Reproduce

Run from the repo root:

```bash
# one-time: pyarrow and openpyxl were installed into .venv for this work
uv pip install --python .venv/bin/python pyarrow openpyxl

PYTHONPATH=src .venv/bin/python scripts/fetch_datasets.py --list
PYTHONPATH=src .venv/bin/python scripts/fetch_datasets.py                  # all, max 300 MB per file
PYTHONPATH=src .venv/bin/python scripts/fetch_datasets.py --only fred_dexuseu,ett_etth1
PYTHONPATH=src .venv/bin/python scripts/fetch_datasets.py --download-only  # raw files only
PYTHONPATH=src .venv/bin/python scripts/fetch_datasets.py --force          # re-download everything
```

Reruns skip files already in the manifest with matching size. A failed source is recorded in the manifest with its HTTP status.

## Disk

- Raw downloads (`data/datasets/_raw/`): about 792 MB.
- Normalised Parquet (30 datasets): about 1.0 GB, of which the London and UCI electricity tables are about 950 MB.
- Total for this work: about 1.8 GB, under the 2 GB limit. Those figures are the measured sums of this work's folders.
- `data/datasets/` also holds hash-named folders from earlier work. They were not touched.
