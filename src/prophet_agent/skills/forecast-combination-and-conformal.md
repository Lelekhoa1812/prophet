name: forecast-combination-and-conformal
summary: Equal-weight forecast combination and conformal intervals from past residuals, both gated by a backtest.

# Forecast combination and conformal intervals

Start with the equal-weight combination. Weighted schemes and tuned intervals must beat it in a backtest.

1. Combine forecasts, one row per model:

```python
combined, rep = ps.combine_forecasts(np.vstack([f_model, f_naive]), weights="mean")
```

Use weights="median" when one member is occasionally wild. Use weights="inverse_error" only with errors from a backtest on history before the origin, passed as `errors=[mse_model, mse_naive]`.
2. Intervals from past out-of-sample residuals only:

```python
lo, hi, rep = ps.conformal_interval(past_residuals, combined, alpha=0.1, method="rolling", window=60)
print(rep["half_width"], rep["empirical_coverage"])
```

Use "split" for a stable series and "rolling" when volatility moves. A 90 percent interval needs at least 9 residuals.
3. What it can destroy: inverse-error weights overfit short holdouts (the forecast combination puzzle). Conformal intervals assume exchangeable errors, so after a level shift they are too narrow until the window refills. One width ignores changing volatility.
4. Reporting: empirical_coverage is measured on the calibration residuals, so it is close to 1 minus alpha by construction. It is not evidence of calibration. Only held-out coverage counts.
5. Gate before use: at least 20 held-out origins, or the comparison is too thin to read (proposed default).
6. Mandatory check: at each walk-forward origin, compute held-out coverage of the intervals and the error of the combination against each member alone and against the untreated forecast. Keep the combination and intervals only if held-out coverage is near the target and the combined error is no worse than the best member. Record the number of origins.

Keep the chosen weighting, the error inputs, the interval method and the number of origins in the Ledger, so the combination can be traced to the backtest that justified it. A combination without that record is an opinion about the members.

Do not fit weights on the test period. Do not report empirical_coverage as calibration evidence. Do not combine forecasts built on different target windows.
