# Example time series

Real, publicly published, long-span time series for exercising Prophet's forecasting, backtesting and analysis tools.
Every value is an observed or published figure copied from its source. Nothing here is simulated, imputed or
interpolated. Missing values are empty cells, not zeros, except where the source itself reports a zero.

Downloaded 2026-11-21. Each file is a UTF-8 CSV with a header row and no index column. The first column is `date`
(ISO 8601: `YYYY-MM-DD`, or `YYYY-MM-DD HH:MM:SS` for hourly data). Upload them as they are; the app accepts CSV and
XLSX up to 50 MB.

## Summary

| Area | File | Frequency | Span | Rows | Value columns | Empty cells |
|---|---|---|---|---|---|---|
| Finance | `finance/ust_par_yield_curve_daily_2016_2026.csv` | business daily | 2016-11-21 to 2026-11-20 | 2,500 | 14 | 4,013 |
| Finance | `finance/ecb_eur_reference_rates_daily_2016_2026.csv` | business daily | 2016-11-21 to 2026-11-20 | 2,560 | 4 | 0 |
| Finance | `finance/eia_wti_spot_daily_2016_2026.csv` | business daily | 2016-11-21 to 2026-11-17 | 2,498 | 1 | 0 |
| Sales | `sales/treasury_dts_daily_cash_flows_2019_2021.csv` | business daily | 2019-01-02 to 2021-09-30 | 693 | 4 | 53 |
| Sales | `sales/treasury_dts_daily_cash_flows_2022_2026.csv` | business daily | 2022-04-18 to 2026-11-19 | 1,151 | 7 | 206 |
| Sales | `sales/uci_online_retail_ii_daily_sales_2009_2011.csv` | calendar daily | 2009-12-01 to 2011-12-08 | 738 | 8 | 1,080 |
| Science (biology) | `science/biology_turku_daily_airborne_pollen_2000_2009.csv` | daily | 2000-01-01 to 2009-12-31 | 3,653 | 1 | 43 |
| Science (medical) | `science/medical_ukhsa_england_daily_covid_2020_2023.csv` | daily | 2020-08-01 to 2023-12-31 | 1,248 | 3 | 2 |
| Science (nature / climate) | `science/mauna_loa_co2_daily_2014_2023.csv` | daily | 2014-01-01 to 2023-12-31 | 3,652 | 1 | 405 |
| Science (physics / energy) | `science/uci_electricity_hourly_total_load_2012_2014.csv` | hourly (UTC as supplied) | 2012-01-01 00:00 to 2014-12-31 23:00 | 26,304 | 2 | 0 |
| Science (geophysics) | `science/usgs_global_quakes_m4_5plus_daily_2014_2023.csv` | calendar daily | 2014-01-01 to 2023-12-31 | 3,652 | 2 | 0 |

Total size is about 1.4 MB. Row counts exclude the header.

## Finance

### `finance/ust_par_yield_curve_daily_2016_2026.csv`

- **What it is:** US Treasury Daily Par Yield Curve Rates, one column per maturity.
- **Columns:** `date`, then 14 yields in percent: `ust_1mo_yield_percent`, `ust_1_5mo_yield_percent`,
  `ust_2mo_yield_percent`, `ust_3mo_yield_percent`, `ust_4mo_yield_percent`, `ust_6mo_yield_percent`,
  `ust_1y_yield_percent`, `ust_2y_yield_percent`, `ust_3y_yield_percent`, `ust_5y_yield_percent`,
  `ust_7y_yield_percent`, `ust_10y_yield_percent`, `ust_20y_yield_percent`, `ust_30y_yield_percent`.
- **Source:** U.S. Department of the Treasury, Daily Treasury Par Yield Curve Rates (`home.treasury.gov`, CSV per year, merged and filtered).
- **Licence:** US government work, public domain.
- **Attribution:** Source: U.S. Department of the Treasury, Daily Treasury Par Yield Curve Rates, retrieved 2026-11-21.
- **Complexity:**
  - Regime change: the 10-year yearly mean moves from 2.45% (2016) to 0.89% (2020), 2.95% (2022), 4.21% (2024) and 4.56% (2026).
  - Shock: the 10-year yield falls from 1.10% to 0.54% in the week of 2020-03-02 to 2020-03-09, with a low of 0.52% on 2020-08-04.
  - Hiking cycle: the 2-year yield rises from 0.78% (2022-01-03) to 5.19% (2023-10-17).
  - Volatility clustering: autocorrelation of absolute daily 10-year changes is 0.19 at lag 1 and 0.12 at lag 10.
  - Calendar gaps: weekends, plus 4-day and 2-day steps in holiday weeks.
  - Zero-bound values: the 1-month and 3-month bills print 0.00 on 2020-03-25, kept as published.
- **Caveats:** Several short-maturity columns did not exist early in the window (1.5-month, 2-month and 4-month), so there are large empty blocks. The Treasury website terms are broader than the data's public-domain status. The file was retained with attribution.

### `finance/ecb_eur_reference_rates_daily_2016_2026.csv`

- **What it is:** ECB euro foreign exchange reference rates, euro-area currency per 1 EUR.
- **Columns:** `date`, `eur_ref_rate_usd_per_eur`, `eur_ref_rate_gbp_per_eur`, `eur_ref_rate_jpy_per_eur`, `eur_ref_rate_chf_per_eur`.
- **Source:** European Central Bank, Statistical Data Warehouse, EXR daily reference rates (`data-api.ecb.europa.eu`).
- **Licence:** ECB free reuse on condition the source is quoted.
- **Attribution:** Source: ECB statistics.
- **Complexity:**
  - Regimes: USD per EUR yearly means are 1.06 (2016), 1.18 (2018 and 2021), 1.05 (2022) and 1.15 (2026).
  - Shock: USD per EUR reaches 0.9565 on 2022-09-28, below parity, inside the 2022 regime.
  - JPY per EUR ranges from 114.65 (2020-05-06) to 187.72 (2026-04-17), with a trend and a break in 2022.
  - GBP per EUR has a low of 0.82388 (2022-03-04) and a high of 0.92985 (2020-03-19, the COVID shock).
  - Volatility clustering is weak for this pair: autocorrelation of squared returns is 0.03 at lag 1.
  - Holiday gaps are 4 and 5 days long (e.g. 2023-12-22 to 2023-12-27).
- **Caveats:** The SDMX metadata columns were dropped. The values and dates are unchanged. Rates are reference rates, not tradeable quotes.

### `finance/eia_wti_spot_daily_2016_2026.csv`

- **What it is:** WTI Cushing crude oil spot price, US dollars per barrel.
- **Columns:** `date`, `wti_cushing_spot_price_usd_per_barrel`.
- **Source:** U.S. Energy Information Administration, Cushing WTI spot price FOB (`eia.gov/dnav/pet/hist_xls/RWTCd.xls`). Excel serial dates were converted to ISO dates.
- **Licence:** US government work, public domain.
- **Attribution:** Source: U.S. Energy Information Administration (Nov 2026).
- **Complexity:**
  - Regimes: yearly means are $50.7 (2016), $39.2 (2020), $94.9 (2022), $65.4 (2025) and $87.3 (2026 year to date).
  - Negative price: -36.98 on 2020-04-20, the only negative day. This is genuine and must not be clipped.
  - Peak: 123.64 on 2022-03-08.
  - Strong volatility clustering: annualised volatility of log returns is about 50%, and autocorrelation of squared log returns is 0.19 at lag 1.
- **Caveats:** This is spot FOB Cushing. It differs from front-month futures settlement. The last date is 2026-11-17, three days before the window end.

## Sales and cash flows

### `sales/treasury_dts_daily_cash_flows_2019_2021.csv`

- **What it is:** Daily Treasury Statement, federal operating cash flows, excluding transfers, for the earlier table format.
- **Columns:** `date`, `fra_deposits_excl_transfers_usd_millions`, `fra_withdrawals_excl_transfers_usd_millions`,
  `individual_income_employment_taxes_not_withheld_deposits_usd_millions`,
  `social_security_benefits_eft_withdrawals_usd_millions`.
- **Source:** U.S. Treasury Fiscal Data, Daily Treasury Statement, "Deposits and Withdrawals of Operating Cash" (`api.fiscaldata.treasury.gov`, no account or key needed).
- **Licence:** US government data, free to copy, adapt and redistribute.
- **Attribution:** Source: U.S. Department of the Treasury, Bureau of the Fiscal Service, Fiscal Data, Daily Treasury Statement.
- **Units:** USD millions. Figures are rounded to the nearest million.
- **Complexity:**
  - Level shift around COVID-19: the monthly mean of withdrawals rises from 57,510 (2019-03) to 97,424 (2020-04) and 115,511 (2021-03). The data do not establish a cause.
  - Month-end and quarter-end spikes: the largest withdrawal days include 2020-12-31 (377,911) and 2021-06-01 (376,571).
  - No rows on weekends or federal holidays.
- **Caveats:** Blanks in the Social Security column (53 days) are missing, not zero. Rows end 2021-09-30 because the publisher changed the table format in October 2021. The 2021-10-01 to 2022-04-15 window is omitted because the source has line items but no published totals.

### `sales/treasury_dts_daily_cash_flows_2022_2026.csv`

- **What it is:** Daily Treasury Statement, Treasury General Account (TGA) deposit and withdrawal totals, closing balance, and selected payment lines, in the newer table format.
- **Columns:** `date`, `tga_total_deposits_usd_millions`, `tga_total_withdrawals_usd_millions`,
  `tga_closing_balance_usd_millions`, `taxes_withheld_individual_fica_deposits_usd_millions`,
  `federal_salaries_eft_withdrawals_usd_millions`, `ssa_benefits_payments_withdrawals_usd_millions`,
  `interest_on_treasury_securities_withdrawals_usd_millions`.
- **Source:** U.S. Treasury Fiscal Data, Daily Treasury Statement, "Deposits and Withdrawals of Operating Cash" and "Operating Cash Balance".
- **Licence:** US government data, free to copy, adapt and redistribute.
- **Attribution:** Source: U.S. Department of the Treasury, Bureau of the Fiscal Service, Fiscal Data, Daily Treasury Statement.
- **Units:** USD millions.
- **Complexity:**
  - Balance trough and rebuild: the closing balance falls to 22,892 on 2023-06-01, then reaches 1,052,530 on 2026-10-27. The trough is consistent with the 2023 debt-limit period, but the data do not establish the cause.
  - Largest deposit day in the window is 571,150 on 2024-10-31.
  - Tax-date and payment-date effects on the individual tax and benefit columns.
- **Caveats:** Taxes-withheld values are blank before February 2023 (206 rows). This is missing, not zero. Columns are published totals and selected lines, not a complete cash-flow breakdown.

### `sales/uci_online_retail_ii_daily_sales_2009_2011.csv`

- **What it is:** Daily sales for a UK-based online gift-ware retailer, aggregated from invoice lines.
- **Columns:** `date`, `orders_count`, `cancelled_invoices_count`, `units_sold`, `units_returned`, `gross_sales_gbp`,
  `returns_and_adjustments_gbp`, `net_sales_gbp`, `active_customers_count`.
- **Source:** UCI Machine Learning Repository, Online Retail II (Chen, 2012), `archive.ics.uci.edu/dataset/502/online+retail+ii`.
- **Aggregation:** Invoice lines are summed per calendar day. Cancellations and adjustments are kept as separate columns. Negative lines are kept.
- **Licence:** CC BY 4.0, per the UCI donation policy. The dataset page itself has no separate licence field.
- **Attribution:** Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
- **Complexity:**
  - Weekday pattern: mean net sales are highest Thursday (38,348) and Tuesday (35,864). Saturday is the weakest day (9,803) and has the most empty days.
  - Intermittent demand: 135 days have no source lines, 104 of them Saturdays. These are blank, not zero.
  - Yearly and seasonal pattern: monthly net sales peak in October to November (November 2010: 1,422,655; November 2011: 1,461,756).
  - Credit days: negative net days on 2010-08-27 (-1,683.50) and 2011-01-05 (-1,566.23).
  - Large single days include 2011-11-14 (112,141).
  - Closures over Christmas and New Year, and UK bank holidays, appear as blanks.
- **Caveats:** The final day, 2011-12-09, is excluded because the source ends at 12:50 that day. The two Treasury files come from one publisher, so they are not independent sources. No third daily sales series with a clean licence and no login was found.

## Science

### `science/biology_turku_daily_airborne_pollen_2000_2009.csv`

- **What it is:** Daily total airborne pollen count, Turku, Finland.
- **Columns:** `date`, `pollen_total_count_per_m3` (grains per cubic metre per day).
- **Source:** Turku Aerobiology Unit, University of Turku, Zenodo record 53963 (DOI 10.5281/zenodo.53963).
- **Licence:** CC BY 4.0.
- **Attribution:** Turku Aerobiology Unit, University of Turku, daily airborne pollen counts for Turku 2000–2009, Zenodo record 53963 (DOI 10.5281/zenodo.53963), CC BY 4.0.
- **Processing:** The raw per-year files are semicolon-delimited with sentinels (`.` and `-1`). Sentinels became blanks. One 2004 day-index typo was corrected from its month-day code. This is the only timestamp change.
- **Complexity:**
  - Strong annual seasonality: the monthly mean is near zero in winter and peaks in May (683.5 grains/m³).
  - Intermittency: 1,890 of 3,610 non-missing days (52%) are exactly zero, and the median is 0.
  - Heavy-tailed peaks: the maximum is 8,900 on 2000-04-29.
  - Strong inter-annual variability: annual totals range from 24,895 (2001) to 78,889 (2002).
  - Missing blocks: runs in 2004 (October to December) and 2000 (October to November).
- **Caveats:** The series is one city's pollen count, so it reflects local vegetation and weather.

### `science/medical_ukhsa_england_daily_covid_2020_2023.csv`

- **What it is:** Daily COVID-19 surveillance for England: new lab-confirmed cases, hospital admissions, and occupied beds.
- **Columns:** `date`, `new_cases_count`, `hospital_admissions_count`, `hospital_occupied_beds_count`.
- **Source:** UK Health Security Agency, UKHSA Data Dashboard, COVID-19 England metrics `casesByDay`, `healthcare_admissionByDay`, and `healthcare_occupiedBedsByDay` (`api.ukhsa-dashboard.data.gov.uk`).
- **Licence:** Open Government Licence v3.0.
- **Attribution:** Contains public sector information licensed under the Open Government Licence v3.0. Source: UK Health Security Agency, UKHSA Data Dashboard (COVID-19, England).
- **Complexity:**
  - Epidemic waves at different scales: the peak is 234,874 daily cases on 2022-01-04. Median daily cases are 12,488 (2020), 23,457 (2021), 8,592 (2022) and 1,263 (2023).
  - Regime shifts between the 2020–21 waves and the 2022–23 period.
  - Admissions range from 25 (2020-08-22) to 4,134 (2021-01-12). Occupied beds range from 451 (2020-09-02) to 34,336 (2021-01-18).
  - Weekday effect in reported cases: weekend dips are a reporting-process artefact.
- **Caveats:** Missing values are two admissions days, 2023-05-30 and 2023-05-31. The API returns current values, not data vintages, so later revisions are not captured.

### `science/mauna_loa_co2_daily_2014_2023.csv`

- **What it is:** Daily mean atmospheric CO2 concentration at Mauna Loa, Hawaii.
- **Columns:** `date`, `co2_ppm_daily_mean` (parts per million).
- **Source:** NOAA Global Monitoring Laboratory, Mauna Loa CO2 daily file (`gml.noaa.gov/webdata/ccgg/trends/co2/co2_daily_mlo.csv`), 2014 to 2023 rows.
- **Licence:** Freely available to the public. Citation requested. No formal licence identifier.
- **Attribution:** NOAA Global Monitoring Laboratory, Mauna Loa CO2 daily mean. Please cite NOAA GML.
- **Complexity:**
  - Trend: annual means rise from 398.86 ppm (2014) to 421.21 ppm (2023), about 2.5 ppm per year.
  - Annual cycle: the monthly climatology peaks in May (413.53 ppm) and troughs in September (407.25 ppm), a swing of about 6.3 ppm.
  - Missing blocks: 289 gaps in total, including a 9-day block.
- **Caveats:** Measurements were suspended from late November 2022 to July 2023 after the Mauna Loa eruption. From December 2022 to 4 July 2023 the data come from a site about 21 miles away (Maunakea Observatories), so a small offset is possible. The source header flags recent months as preliminary.

### `science/uci_electricity_hourly_total_load_2012_2014.csv`

- **What it is:** Hourly total electricity load, summed across the 321 consumer series in the UCI electricity dataset.
- **Columns:** `date` (hourly timestamp), `total_load_kw`, `active_series_count` (how many consumer series reported load in that hour).
- **Source:** UCI Machine Learning Repository, ElectricityLoadDiagrams20112014 (dataset 321, DOI 10.24432/C58C86), via the Monash / Zenodo hourly aggregate, Zenodo record 4656140.
- **Licence:** CC BY 4.0.
- **Attribution:** UCI (2020), "Electricityloaddiagrams20112014 data set, UCI Machine Learning Repository"; Lai, G., Chang, W., Yang, Y., Liu, H. (2017), arXiv:1703.07015.
- **Processing:** The Monash file has timestamps offset by one second (`:01`). They were set to `:00`. This is the only change. Source timezone and unit were not independently verified.
- **Complexity:**
  - Daily cycle: the hour-of-day mean ranges from about 412,021 kW (lowest hour) to 1,107,877 kW (highest hour), a swing of about 2.7 times.
  - Level shifts: yearly means are 848,189 (2012), 802,912 (2013) and 793,664 (2014). These are confounded by changes in the number of active series.
  - Active-series count varies from 102 to 321, showing client-composition changes and zero-load periods.
  - Weekly seasonality is weak in the aggregate: weekday mean 815,696 vs weekend mean 813,090.
  - Zero-load readings are present in the source and are kept as zero, not blank.
- **Caveats:** This is an aggregated derivative. The original UCI data are at 15-minute resolution. The hourly aggregation was done by the Monash/Lai source, not by this project.

### `science/usgs_global_quakes_m4_5plus_daily_2014_2023.csv`

- **What it is:** Daily count of earthquakes with magnitude 4.5 or more worldwide, and the largest magnitude that day.
- **Columns:** `date`, `quake_count_m4_5plus`, `quake_max_mag`.
- **Source:** U.S. Geological Survey, Earthquake Hazards Program, ComCat FDSN event service (`earthquake.usgs.gov/fdsnws/event/1/query`, `minmagnitude=4.5`), one request per calendar year.
- **Aggregation:** Events are counted per UTC calendar day.
- **Licence:** USGS data. Public domain status is assumed under USGS policy but was not verified against the copyright and credits page before publication.
- **Attribution:** U.S. Geological Survey, Earthquake Hazards Program, ComCat.
- **Complexity:**
  - Heavy-tailed daily counts: median 18 events per day, 99th percentile 62, maximum 186 (2021-03-05).
  - Burst structure: sustained high-count runs follow large events and their aftershock sequences (e.g. 2021-08-12 and 2023-12-02).
  - Yearly totals range from 6,372 (2017) to 8,959 (2021).
  - No weekly or annual seasonality is expected. This is not a seasonal test case.
- **Caveats:** The catalogue is a mix of reviewed and automatic events, and recent events may be revised. Detection completeness at M4.5 is high globally but not uniform over time or region, and this is not corrected.

## Not included

These were checked and left out on licence, frequency or availability grounds.

- **S&P 500 and VIX:** S&P Dow Jones Indices and Cboe restrict reproduction of their index data.
- **FRED series:** FRED terms prohibit storing or compiling FRED content, so the finance files come from the primary publishers.
- **Kaggle sales sets** (Walmart, Rossmann, Favorita, store-item-demand): need a Kaggle account.
- **Iowa Liquor Sales:** the endpoint returned HTTP 404.
- **UK ONS Retail Sales Index:** monthly only, kept as a fallback.
- **SILSO sunspot number and GFZ Kp/Ap/F10.7 file:** the sunspot column is CC BY-NC (non-commercial), and the whole file is over 5 MB.
- **NYC GHCN-Daily:** no gaps in the window, so it did not add the test value the brief asked for.
- **OWID COVID-19 by country:** weekly reporting for some countries is zero-filled, so a daily series would contain synthetic zeros. The licence is also ambiguous.
- **NEON ecological data:** now requires a login.
- **Portal rodents:** monthly sampling, so it is not a continuous daily or weekly series.
- **Columbia River DART fish passage and USDA APHIS bird flu:** no clear open licence or stable endpoint.

## Usage notes

- Prophet's forecasting and analysis tools can read these files directly. Pick the `date` column and any value column.
- Blank cells are missing values. Zeros are real observations, except where a column is documented as sparse
  (intermittent demand, pollen season).
- Label any result from these files as measured on real public data. They are not a ground truth for the
  synthetic benchmark in `prophet_agent/synth/`.
- Keep attribution text with any redistribution. Licence terms were read from source pages and metadata. They have
  not been legally reviewed.
- Raw source downloads and build scripts are kept in the git-ignored `data/` folder, not here.
