name: choose-model-by-backtest
summary: Pick TimesFM, a classical model or a naive baseline by rolling backtest and a pre-set gate.

# Choose a model by backtest

A model earns its place by beating simple baselines on the same origins, not by looking good on one split.

1. Choose origins by calendar (every month for monthly data). Use at least about 20 origins per series group for seasonal data (proposed default). Use the same origins for every model.
2. Always include naive and seasonal naive, plus drift and a statsmodels ETS model. A forecast without a baseline is not a result.
3. Evaluate every horizon step you will report, and show a table by horizon. Do not average a long horizon into a short one.
4. Give every compared model the same context length, and state it.
5. Score point accuracy with MASE and bias (mean error). Score probabilistic accuracy with mean pinball loss across the 0.1 to 0.9 levels and empirical coverage of the 10 to 90 band. Do not use MAPE or sMAPE as headline numbers.
6. Test each treatment (cleaning, covariate, control setting) with `ps.backtest_treatment(x, treat, horizon, origins=5, period=None, baseline="seasonal_naive")`. Read its helped flag and the paired figures it returns.
7. Gate: choose a candidate only if its mean loss beats the best baseline and the bootstrap interval of the paired difference over series lies below zero (proposed default). Otherwise keep the baseline and log the rejection.
8. Select per segment (product family, frequency, length band), not per series, unless each series has many origins. Per-series choice on about 20 origins overfits.
9. If a project bar is set (for example 15 percent better than the best baseline), fix it before the run, report against it, and do not move it when it is missed.
10. Re-run the gate on the newest data each cycle, and check it on a locked final block never used for choice.

```python
import prophet_stats as ps
r = ps.backtest_treatment(x, treat=treatment, horizon=12, origins=8, period=7,
                          baseline="seasonal_naive")
print(r)                                     # helped flag, baseline and paired difference
```

Decision rules:
- If the winner changes between origins, report the gate as unstable, not as a win.
- If the naive baseline beats the model on the primary metric, say so in the first line of the result.
- If the seasonal period is uncertain, backtest both candidates rather than picking one.

Do not pick a winner from one origin. Do not tune thresholds until the backtest looks good and then report it as validation. Do not change the baseline set between runs. Do not use the locked block to choose. Do not claim a win without the interval.
