name: outliers-and-level-shifts
summary: Separate spikes from level shifts, flag before changing, keep raw values, stop above 2 percent.

# Outliers and level shifts

Flag first, decide by cause, and never overwrite the raw data.

1. Flag on a robust basis. For spikes in a trending or seasonal series use `ps.hampel(x, window=7, n_sigmas=3.0, causal=True)`. When seasonality is clear, use `ps.stl_outliers(x, period, k=3.0)` on the remainder. Both return flags and a report.
2. Check the flagged share. Above about 2 percent of points in one series, stop: that suggests a noisy process or a broken feed, not a few outliers (proposed default).
3. Decide each flagged point by cause:
   - Repeats on a known event (promotion, holiday, weather day): keep it and add an event covariate.
   - Single point, no cause in the metadata: keep the raw value. Forecast with it and with a winsorised copy clipped at the flag bound. Keep the copy only if the backtest difference beats noise.
   - Recording error with evidence (for example a value off by a factor of 1000): replace it, and log the reason.
4. Check for level shifts before calling anything an outlier. Run `ps.changepoints(x, min_size=20, penalty=None, max_changes=5)`. A shift that persists for at least two seasonal cycles and moves the mean by about two standard deviations is a regime (proposed default). Truncate training to the post-shift window or add a regime indicator, and choose by backtest.
5. A transient spike that reverts within one cycle is an outlier, not a break.
6. Write treatments to a derived column with a reason code. Keep the raw column unchanged.

```python
import prophet_stats as ps
x = s.to_numpy(dtype=float)
flags, rep = ps.hampel(x, window=7, n_sigmas=3.0, causal=True)
print(rep, int(flags.sum()) / len(x))           # flagged share; stop above about 0.02
shifts, rep2 = ps.changepoints(x, min_size=20, max_changes=5)
for c in shifts:
    print(c["index"], c["shift_in_sd"])
```

Decision rules:
- If a flagged spike is a real peak (launch, peak season), keep it. Do not winsorise real signal.
- If the detector flags a whole run, look for a level shift before any outlier treatment.
- If the thresholds were chosen after seeing backtest results, say so and report the tuning count.

Do not Hampel-clip a level shift. Do not use centred windows for model inputs. Do not delete outliers without a recorded cause. Do not winsorise launches or peak-season spikes. Do not treat a flagged share above 2 percent as routine cleaning. Do not set thresholds on the full series before the backtest.
