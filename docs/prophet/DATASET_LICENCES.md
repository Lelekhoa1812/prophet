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

Research-use labelling (owner decision, Liam, 2026-11-23): this project is research use. Outputs built from a dataset whose
key is `open_meteo` carry the research-use label and its attribution in documents and reports only. The label is never
written into agent prompts, contracts, tools or the UI, and the code refers to the dataset by its key alone.

Stock data (owner decision, 2026-11-23): no stock source with a no-key download and clear terms was found. Yahoo Finance and
FRED terms prohibit automated extraction, and Stooq's terms could not be verified, so none of them is downloaded automatically.
Stock series come from a user upload until a source with a verified licence is added as a row above.
