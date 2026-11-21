name: changepoints-and-regimes
summary: Find persistent mean shifts on history only; a backtest decides between a regime flag and truncation.

# Changepoints and regimes

A changepoint is a persistent shift in level, not a spike. Find it on history, then let a backtest decide how to use it.

1. Gate: a shift that lasts at least two seasonal cycles and moves the mean by about two standard deviations (proposed default, as in the outlier skill). Remove spikes first with `ps.hampel`, so they are not read as breaks.
2. Find the shifts on the history given:

```python
shifts, rep = ps.changepoints(history, min_size=20, max_changes=5)
for c in shifts:
    print(c["index"], c["shift_in_sd"])
```

Run it again at every origin with only the data before that origin. Indexes refer to the series passed in.
3. Options to compare: a regime indicator after the last confirmed shift, truncation of training to the post-shift window, or no change. Choose by backtest, never by the shift position on the full series.
4. What it can destroy: at a fixed penalty a long series shows noise as breaks, and truncation discards history a pretrained model could use. A break found only later than an origin was unknown at that origin, so a backtest must not credit it earlier.
5. Causality: `ps.changepoints` uses the whole history it is given and fails `ps.causal_canary`. Never use its output as a live flag without re-running it at each origin.
6. Mandatory check: a walk-forward backtest of the three options, each refit at each origin. Use `ps.backtest_treatment` for the treatment arm and keep a shift treatment only if held-out MASE is lower at more than half the origins.

Report each shift in standard deviations next to its index, so a small move is not presented as a regime. A break that the next origin cannot confirm should be listed as unconfirmed, not dropped silently.

Do not treat a changepoint from the full series as known at an earlier origin. Do not truncate before the shift is confirmed. Do not count a spike as a regime.
