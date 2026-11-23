# Weather forecast from open data: one measured run (2026-11-24)

Research use only. This is a research run of the Prophet open-data path, not a client deliverable and not a basis for any
commercial decision.

**Attribution and licence.** Weather data: Open-Meteo historical weather API (archive endpoint), CC BY 4.0 for the data.
The free Open-Meteo API is for non-commercial use only; this project uses it for research only. Terms as checked on
2026-11-23 and recorded in `DATASET_LICENCES.md` under the key `open_meteo`. Forecasts: TimesFM 3 (`google/timesfm-3.0-pytorch`),
whose weights are non-commercial research use only (see `ARCHITECTURE.md` section 5).

Every figure below is **measured** on the downloaded open data. None is synthetic, and none is an expected value.

## What was run

| Item | Value |
|---|---|
| Place | Berlin, latitude 52.52, longitude 13.41 (timezone Europe/Berlin) |
| Variable | Daily maximum 2 m temperature, degrees Celsius (`temperature_2m_max`) |
| Period downloaded | 2022-01-01 to 2026-11-15, 1,780 daily points, no missing values |
| Download | `open_meteo_archive`, one request, retrieved 2026-11-23 13:06 UTC |
| Stored file SHA-256 | `d4997454694691e30895ad0b0001e7cf79920a17a06769f8f0246a06d9c859b0` (the table as stored) |
| Raw response SHA-256 | `a89bdbc9d151bf5b1d47d499f704c121a3844841ec61c16bad4a6030043d4eed` (the body as fetched, with its location block) |
| Horizon | 14 days (2026-11-16 to 2026-11-29) |
| Backtest | 12 rolling origins, 14-period horizon, taken from the end of the history |
| Engine | TimesFM 3 (MLX backend), full context of 1,780 points |

Reproduce with:

```bash
PYTHONPATH=src .venv/bin/python scripts/weather_open_data_check.py <store folder>
```

The script writes `weather_check.json` (every number here) into the store folder.

## Accuracy against simple rules (measured)

MASE is the mean absolute error scaled by the average seasonal change, so 1.0 means the error equals the average change.
WAPE is the total absolute error as a share of the total actual. Lower is better for both.

**Fair comparison: annual season set to 365 (the correct cycle for daily temperature).**

| Method | MASE | WAPE |
|---|---|---|
| **TimesFM 3** | **0.752** | **0.161** |
| Seasonal naive (same day last year) | 0.951 | 0.203 |
| Naive (last value) | 1.255 | 0.268 |
| Drift | 1.260 | 0.269 |
| TSB (expected demand) | 1.211 | 0.258 |
| Mean of last season | 1.894 | 0.405 |

- Best simple rule: seasonal naive.
- TimesFM beat it in 10 of 12 origins.
- Gain over the best rule: 20.9% in MASE, 90% interval 9.2% to 33.2%. One-sided sign-flip test, p = 0.0095 that TimesFM is better.
- The 80% band covered 76.8% of actual values (target about 80%).

**Automatic season (what the tool chose without being told): not a fair comparison.**

The tool found no season and used 1. Its search tests lags up to 14 only, so the annual cycle was missed. The "best
simple rule" then became a 4-point mean, which is much weaker. Its headline (MASE 1.50 against 2.14, a 29.8% gain) overstates
the advantage and must not be quoted. The annual-season result above is the one to use.

## Forecast

Median and 10% to 90% band, degrees Celsius, from TimesFM 3 with the same history.

| Date | Median | 10% | 90% |
|---|---|---|---|
| 2026-11-16 (day 1) | 6.8 | 4.5 | 9.4 |
| 2026-11-29 (day 14) | 5.5 | 1.0 | 10.5 |

The tool's own health check gave **check**, not ok. Its reasons: the median stays within 0.3 typical one-step moves of the
last value, so it carries little directional view, and the shape of the annual cycle is too short a window to judge at a
14-day horizon. Present this as a range with the direction uncertain, not as a confident call.

## Limits of this run

- One city, one variable, one 14-day horizon, 12 origins. The tool's error-correlation check gives a variance ratio of 5.2,
  so the errors move together and the 12 origins are far from 12 independent checks; the p-value is indicative only.
- The annual season had to be set by hand. Automatic season detection tests short lags only, so daily series with an
  annual cycle get weak simple rules and an inflated apparent gain. This is a known gap in the tool, recorded here, not fixed
  in this run.
- Open-Meteo data is for research use in this project. Any use outside research needs a paid plan and its own terms.
- TimesFM weights are non-commercial. Do not use these outputs for commercial decisions, client deliverables or paid services.
