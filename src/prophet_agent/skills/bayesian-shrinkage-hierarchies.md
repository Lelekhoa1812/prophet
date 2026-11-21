name: bayesian-shrinkage-hierarchies
summary: Pull noisy per-series means or rates toward the group when each series has little history.

# Bayesian shrinkage for many short series

Apply when many series each have few observations and their estimates scatter more than their noise explains.

1. Gate: at least five series. The between-series variance must be well above zero. If `between_series_variance` is 0, every series collapses to the group; treat that as a warning about the data, not a result.
2. Rates (counts over exposure, Poisson):

```python
shrunk, rep = ps.shrink_to_group(rates, n_obs=exposure)
print(rep["group_mean"], rep["between_series_variance"], rep["shrinkage_factors"][:5])
```

3. Continuous means: pass the within-series variance as `within_var=sigma2`, so the sampling variance is known.
4. Read the factors. 0 keeps the series' own estimate and 1 uses the group mean. A factor near 1 for a series with plenty of data means the variance is being estimated badly.
5. What it can destroy: a genuinely different series (another product, store or market) pulled toward the group, and a heterogeneous population shrunk to a mean that fits none of its members. Zero-demand series shrink toward the group rate, which can hide stockouts. Check `ps.censoring_check` before using rates.
6. Mandatory check: compute the group mean and tau squared from history only at each walk-forward origin. Compare the held-out error of the shrunk estimates with the raw estimates, and keep shrinkage only if held-out error falls at most origins. Report how many origins were tested.

Shrinkage trades a little bias for a large cut in noise, and it helps only when the series really resemble one another. The shrinkage factors are the audit trail: store the group mean, tau squared and each factor with the forecast, so a reviewer can reproduce the pull toward the group. A factor report that is all zeros or all ones is a sign the group is wrong, not that the method works.

Do not compute the group mean from the test period. Do not shrink intermittent series with the continuous rule. Do not present shrunk values as observed.
