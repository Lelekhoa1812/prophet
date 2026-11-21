name: intervals-and-trust-statement
summary: State how far intervals can be trusted, check coverage on backtest origins, label every figure.

# Intervals and the trust statement

An interval is a claim. Check it on past origins before you print it, and say what would make it wrong.

1. Measure coverage on rolling origins, per horizon step and per segment. For the 10 to 90 band (nominal 80 percent), report the share of actuals inside it, the number of origins, and an interval around that share.
2. Compare with nominal. Coverage within about 5 points of nominal is acceptable (proposed default). Outside that, the band is not calibrated.
3. Apply a correction only if it is gated on held-out origins and improves pinball loss:
   - Split conformal on scaled backtest residuals, per horizon and segment. It assumes the calibration windows look like the forecast window, so expect under-coverage after a regime change.
   - Adaptive conformal for slow drift, with its running coverage logged.
   Never fit a correction on data that overlaps the test window.
4. Aim for at least about 100 calibration residuals per horizon band before claiming a calibrated band (proposed default).
5. Give each series a trust status: fit-for-use, use-with-caveat, or not forecastable. Put the reason next to the status.
6. Label every figure as observed, derived, imputed or synthetic. Mark synthetic figures in the same sentence.
7. List the assumptions that would change the interval: an unconfirmed driver, a short history, a regime not yet seen.
8. Record each decision in the ledger with `ps.Ledger().add(stage, finding, decision, evidence=None)`.

```python
import numpy as np
inside = (y_true >= q10) & (y_true <= q90)
cov = float(np.mean(inside))
print(f"10-90 band covers {cov:.2f} of {len(y_true)} actuals (nominal 0.80)")
```

Decision rules:
- If coverage is off by more than the tolerance and no gated correction helps, say the band is uncalibrated and widen the caveat.
- If there are too few origins for a horizon band, report that band as insufficient rather than estimating it.
- If the trust status is use-with-caveat, the caveat goes in the first paragraph of the result.

Do not call an 80 percent band calibrated without a coverage check. Do not report a forecast without its origins and horizon. Do not let a correction see its test window. Do not present a use-with-caveat result as fit-for-use. Do not lower the trust status to make a deliverable look clean.
