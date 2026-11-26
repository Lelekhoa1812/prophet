# Dataset licence wording

Owner decision (Liam, 2026-11-23): dataset licence wording lives in this Markdown file, not in code. `scripts/fetch_datasets.py`
refers to each dataset by its key in the table below and reads the wording from here when it writes a dataset's README.
The licence-wording guard (`tests/prophet_agent/test_licence_wording_guard.py`) therefore finds no licence text in the script.

To add a dataset, add a row with a new key, then set `licence_key` to that key on its `Source`. A key that is missing from
this table stops the script with an error rather than writing a blank licence.

| Key | Licence wording |
|-----|-----------------|
| `uci` | CC BY 4.0 (UCI Machine Learning Repository) |
| `fred` | FRED terms of use (St. Louis Fed). Some series carry third-party restrictions; check the series page. |
| `ecb` | ECB reuse policy: free reuse with source attribution |
| `zenodo_monash` | CC BY 4.0 (Monash Time Series Forecasting archive, Zenodo record, as checked) |
| `owid` | CC BY 4.0 (Our World in Data; underlying sources vary by country) |
| `gift_card` | Not verified: check the Salesforce/GiftEval dataset card |
| `chronos_card` | Not verified: check the autogluon/chronos_datasets dataset card |
| `unverified` | Not verified: read the source repository or dataset card before any external use |
| `open_meteo` | CC BY 4.0 for the data (Open-Meteo terms, checked 2026-11-23). The free API is for non-commercial use only; a paid plan and key are needed for commercial use. Research use only in this project. |
| `bis` | BIS statistics may be used if the BIS is cited, a translation is marked unofficial, and a commercial product does not add a charge for the statistics. The BIS may suspend an address without notice. |
| `oecd` | OECD data may be extracted and shared for any purpose when the dataset is cited, unless that dataset's metadata says otherwise. At most 60 downloads an hour. |
| `eurostat` | Eurostat statistical data may be reused with source acknowledgement. Commercial reuse is refused for countries other than EU Member States, EFTA members, and official EU acceding and candidate countries. |
| `abs` | CC BY 4.0 for ABS website statistics, except the logo, microdata, and third-party content. Attribute the Australian Bureau of Statistics. |
| `boe` | UK Open Government Licence for Bank of England Database series the Bank owns. SONIA needs the Bank's attribution sentence. Exchange-rate series the Bank holds from another publisher are not included. |
| `boc` | Bank of Canada website content may be used with attribution. This source is the overnight-rate target, series V39079. Exchange rates on the same API are supplied by LSEG and are not this row. |
| `rba` | Reserve Bank of Australia cash-rate table. Financial data and the cash rate may be reused with attribution. |
| `fhfa` | FHFA House Price Index may be downloaded and reused. Attribute the index type and whether it is seasonally adjusted. |
| `fao` | CC BY 4.0. FAO statistical databases may be accessed and re-disseminated. Do not use the dataset to promote a commercial enterprise. |
| `nws` | US National Weather Service pages are in the public domain unless a page says otherwise. Do not imply NOAA endorsement. |
| `gfz` | CC BY 4.0 for GFZ Kp and the derived indices GFZ names. Cite GFZ. The sunspot number and the 10.7 cm flux in a combined file are not this grant. |
| `nasa_giss` | Not verified: read the GISTEMP pages at data.giss.nasa.gov/gistemp for the use and citation terms before any external use. |
| `aemo` | Not verified: read AEMO's copyright and permissions page at aemo.com.au before any external use of the price and demand files. |
| `hadcrut` | Crown copyright under the Open Government Licence. The acknowledgement must name the analysis version, the download address, the download date, and British Crown Copyright, Met Office. |
| `snb` | Swiss National Bank data may be saved and transmitted for non-commercial purposes with a source reference. |
| `world_bank` | CC BY 4.0 for World Bank datasets unless the indicator metadata says otherwise. Some indicators are supplied by another publisher and may not be redistributed. Attribute the World Bank. |
| `eia` | US Energy Information Administration publications on its site are in the public domain and may be used and distributed. Acknowledge the EIA. |

Research-use labelling (owner decision, Liam, 2026-11-23): this project is research use. Outputs built from a dataset whose
key is `open_meteo` carry the research-use label and its attribution in documents and reports only. The label is never
written into agent prompts, contracts, tools or the UI, and the code refers to the dataset by its key alone.

Stock data (owner decision, 2026-11-26, replaces the 2026-11-23 decision): the keyless price connector
(`download_market_prices`) is on by default and downloads daily prices from Yahoo's chart endpoint, Nasdaq's historical
endpoint and Stooq. The owner has read and accepts their terms, as checked in the research report
`reports/Multi signal data acquisition for forecasting.md`: Yahoo's terms forbid automated collection without permission,
Stooq limits its files to personal use, and Nasdaq treats API redistribution as a data feed needing approval.

FRED (owner decision, 2026-11-26): FRED series stay as they are, including the local cache, although FRED's legal page
forbids storing, caching or archiving its content in a database.

Where terms live (owner decision, 2026-11-26): data-source terms are recorded in this file and other Markdown docs only.
They are never written into tool outputs, provenance fields, agent prompts, sub-agent briefs, the UI or chat reports, so
they do not take space in the research context.
