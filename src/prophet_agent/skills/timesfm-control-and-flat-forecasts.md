name: timesfm-control-and-flat-forecasts
summary: Control TimesFM 3 inputs and horizon, check for flat or repeated output, and keep every result research-only.

# TimesFM 3 control and flat forecasts

Project rule first: TimesFM 3 weights are non-commercial. Outputs are for research use only, never for client deliverables or commercial decisions.

1. Pass a regular grid. TimesFM 3 takes no frequency argument, so resample to the target frequency and mark missing periods explicitly.
2. Context: the model keeps only the most recent 15,360 points. Test shorter windows (for example 128, 512 and 2,048) in the backtest, and report the one used.
3. Horizon: keep production horizons inside the range backtested. Above 128 steps the model stitches output patches, so check those horizons separately (proposed default).
4. Quantiles: the nine outputs are the point plus the 0.1 to 0.9 levels. The 10 to 90 band is a nominal 80 percent interval. Check each row is monotonic; sort or flag it if not.
5. Flat output check: before using a forecast, check the context has variety. Route to naive or seasonal naive when it has fewer than about five distinct values or near-zero variance (proposed default).
6. Determinism check: run the same input in two fresh processes. Different results, or one value repeated across the horizon, is a failed run, not a forecast. An open upstream issue reports flat repeated outputs on one hardware setup, with no recorded fix.
7. Flat is honest when history is flat. Flat where history clearly trends is a fault. Compare with drift and seasonal naive on the same origins.
8. Covariates: past-only for values known up to the origin; past-future for values known over the horizon. Assert the length equals context plus horizon.
9. Do not apply a global multiplicative correction to the median unless it wins on pinball in backtest. A repo test found one made results worse.

```python
import numpy as np
x = np.asarray(history, dtype=float)[-15360:]    # the model keeps only the latest points anyway
if np.unique(x).size < 5 or np.std(x) < 1e-9:
    route = "naive or seasonal naive"            # too little variety for a model
else:
    route = "timesfm_forecast"
```

Decision rules:
- If the determinism check fails, do not use any forecast from that setup until it passes.
- If the horizon is outside the tested range, report the forecast as untested at that horizon.
- If the series is intermittent, use the route in the zeros skill, not the median.

Do not present a flat repeated forecast as real. Do not present the median as the expected value for lumpy demand. Do not quote horizons beyond the tested range. Do not move TimesFM output into a commercial or client deliverable. Do not treat the 80 percent band as calibrated without a coverage check.
