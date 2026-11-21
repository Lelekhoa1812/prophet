name: smoothing-without-leakage
summary: Clean history causally and origin by origin; never let a model see later values.

# Smoothing without leakage

Anything a model or a backtest sees must be a function of history up to its origin. Most leaks come from one convenient line that looks harmless.

1. Use causal tools for anything a model sees. `ps.smooth(x, method, window, causal=True)` is causal by default. Centred windows, two-sided Kalman smoothing, and STL fitted on the full series all use future points.
2. Refit every cleaning step at each backtest origin: detection, imputation, smoothing and scaling. Write one function of history up to the origin and call it inside the loop.
3. Use centred smoothing only for charts and descriptive reports, never as a model input or a label feature.
4. Do not smooth the model input by default. Smoothing lowers the noise the model's quantiles were not built for, which can make intervals too narrow. This is an inference: test coverage before accepting it.
5. Build derivative features (slopes, accelerations) with `ps.derivative_features(x, windows=(3, 7, 28))`. Check that every feature uses past values only.
6. Run a leakage canary: change the future part of the series and check that the forecast at each origin does not move. Any change means future information reached the forecast.
7. Accept a cleaning step only if it improves held-out origins, not the training fit. Keep a no-cleaning arm in every comparison.

```python
import numpy as np
import prophet_stats as ps
def clean_history(y, origin):
    hist = y.iloc[:origin]                       # history only, refit at every origin
    flags, rep = ps.hampel(hist, window=7, causal=True)
    masked = hist.mask(np.asarray(flags, dtype=bool))
    filled, rep2 = ps.impute(masked, method="linear", max_gap=3, causal=True)
    return filled
```

Decision rules:
- If a cleaning step cannot be expressed as a function of history, it cannot be used in a backtest.
- If a smoothed series scores better in-sample but worse on held-out origins, reject it.
- If the leakage canary fails, stop and fix the pipeline before reading any metric.

Do not centre a window in a model input. Do not fit a scaler, imputer or decomposition on the full table. Do not back-fill in a backtest. Do not compare cleaned and raw series on the training fit. Do not report intervals from a smoothed series as calibrated.
