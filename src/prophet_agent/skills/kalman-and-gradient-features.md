name: kalman-and-gradient-features
summary: Causal Kalman level and trailing slope or curvature features; diagnostic smoothers never become model inputs.

# Kalman level and gradient features

Use the causal filter for a noisy level and trailing slopes for momentum. The two-sided smoother only describes history.

1. Apply when the level is noisy and a feature needs the current level or its rate of change. Gate: at least 60 observed points, and a first half that is not constant, because the causal filter fits its noise on that half (proposed default for the length).
2. Causal level and its report:

```python
level, rep = ps.kalman_smooth(y)
print(rep["q_ratio"], rep["outlier_count"], rep["outliers"][:3])
```

3. Trailing slope, curvature and change z-score over several windows:

```python
feats = ps.rolling_slope_features(y, windows=(3, 7, 28))
```

Read diff1, slope_7 and curvature_7. Each change_z_w column is the current change measured against the previous w changes.
4. Diagnostic only: `ps.savgol_smooth(y, 7, 2, causal=False)` and `ps.kalman_smooth(y, causal=False)` report diagnostic_only True. Use them to describe history in a chart, never as an input.
5. What it can destroy: a local level has no trend, so it lags a ramp unless q_ratio is large. A quadratic fit lags the slope by about half a window. A spike moves slopes and curvature for a whole window, and the Kalman report keeps flagged outliers in the output, so decide them separately.
6. Mandatory check: `ps.causal_canary` on each feature set must report causal True. Then a walk-forward backtest compares the forecast with these features against the same forecast without them. Keep the features only if held-out MASE improves at most origins.

The outlier list is a prompt to look, not a verdict. A flagged point with a large influence moved the level, so check its cause before trusting the feature built on it. Causal filter output is safe to store as a feature only with its canary result attached.

Do not feed causal=False output to a model. Do not refit q_ratio on the full series. Do not treat one canary pass as proof of causality at every origin.
