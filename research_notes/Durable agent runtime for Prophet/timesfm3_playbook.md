# TimesFM 3 playbook: public record, configuration, covariates, adaptation and evaluation

As of 20 Nov 2026. Latest public sources I could reach are dated up to Sep 2026. Labels: **[Verified: source]** = read in a fetched primary or official page; **[Verified: secondary]** = read in a search snippet or third-party page, not checked against the primary; **[Hypothesis]** = my inference; **[General practice]** = standard forecasting-evaluation method, not fetched in this run.

Caveat on the fetch tool: where I quote licence text, it came through a summarising fetch, not a raw copy. Treat the wording as close paraphrase and confirm against the file before relying on it.

---

## Q1. What is "TimesFM 3" publicly, and how does it differ from 1.0, 2.0 and 2.5?

### Takeaway
TimesFM 3.0 is a 330M-parameter, zero-shot, decoder-only model released by Google Research around 31 Aug 2026, with native multivariate forecasting and past-only and past-future covariates. Its weights are under a non-commercial licence. Google's own public page gives no max context, no max horizon and no numeric benchmark table, so the repo's 15,360-token context figure is a library default, not a published model limit.

### Cited Findings
- Model card: "TimesFM 3.0 (PyTorch)", 0.3B params; "Stacked Mixing Transformer with Variate Attention and CPM Iterative RevIN"; 20 layers, model dim 1280, 16 heads; context patch length 32, horizon patch length 64; median at index 4; the page names a licence "TimesFM Non-Commercial License v1.0". Max context, max horizon, covariate support and limitations are not stated on the page — [Verified: source](https://huggingface.co/google/timesfm-3.0-pytorch)
- Google blog, dated 31 Aug 2026: "TimesFM-3 has 330 million parameters"; pre-trained on "more than 1 trillion time points" of real and synthetic series; decoder-only; "patches of 32 time steps", per-series normalisation; layers alternate causal temporal attention and full variate attention; decoding is single-pass and non-autoregressive ("the entire forecasting horizon in a single forward pass"), using Contiguous Patch Masking (cites arXiv 2505.23719); outputs 9 quantiles from the 10th to the 90th percentile — [Verified: source](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
- The blog says TimesFM 3 supports three input types: multiple targets (joint forecasting), past covariates (historical only), and past-future covariates ("known future signals like holidays or scheduled events"). It says the model is used "without requiring task-specific fine-tuning" — [Verified: source](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
- Release timeline: PyPI shows `timesfm` 3.0.0 on 28 Aug 2026, 3.0.1 on 2 Sep, 3.0.2 on 9 Sep 2026; 2.0.0 on 5 Jun 2026, 2.0.2 on 2 Jul 2026, 1.3.0 on 6 Jul 2025. Extras: `torch`, `mlx`, `flax`, `xreg`; Requires-Python >=3.10 — [Verified: source](https://pypi.org/project/timesfm/)
- GitHub README timeline: TimesFM 2.5 on 15 Sep 2025 (200M params, up to 16k context, optional 30M quantile head for horizons up to 1k); XReg covariate support for 2.5 re-added 29 Oct 2025; a LoRA fine-tuning example (Transformers + PEFT) added 9 Apr 2026; "TimesFM 3.0 is out!" August 2026; BigQuery ML rollout September 2026 — [Verified: source](https://github.com/google-research/timesfm)
- Original paper (TimesFM 1.0, ICML 2024): arXiv 2310.10688 — [Verified: source](https://arxiv.org/abs/2310.10688) (cited by the HF card as the paper; I did not re-read it in this run)
- Installed package in the repo: `timesfm` 3.0.2 with `timesfm3` (see repo investigation). Its version and date match the PyPI record above — [Verified: source](https://pypi.org/project/timesfm/) plus [repo file](/Users/liamle/Downloads/prophet/docs/prophet/investigation/01-timesfm.md)

### Inferences
- The public architecture facts (20 layers, 1280 dim, 16 heads, patch 32 in / 64 out, nine quantiles 0.1 to 0.9, median index 4) match the repo's `config.json` reading in `01-timesfm.md`. The repo's architecture section is consistent with the public record.
- Repo claim "TimesFM 3 weights are non-commercial" (`CLAUDE.md`, `PLAN.md:10`) is consistent with every source I found. See Q2.
- Repo library defaults that are not public: `max_context_length=15360` and `per_core_batch_size=4` come from the installed package, not from any published model limit. Treat them as implementation defaults, not model guarantees.
- Repo says the MLX path uses "single decode" with iterative CPM RevIN refinement; the public "CPM Iterative RevIN" name is consistent with that. No contradiction with "single forward pass" in the blog, which describes horizon generation, not normalisation.

### Discrepancies and corrections
- **Release date.** The blog is dated 31 Aug 2026 and PyPI's first 3.0 release is 28 Aug 2026. Which one counts as "the release" is unclear. Use "late Aug 2026".
- **Early secondary claim retracted.** A search summary said arXiv 2609.24559 "argues that the benchmarks do not isolate covariate skill". When I fetched that PDF, its title is "t_0: A Time-Series Foundation Model for Forecasting with Context" (Meyer et al.), not a TimesFM 3 paper. I cannot link it to TimesFM 3. Do not cite it for TimesFM 3 covariate claims. [Verified: source](https://arxiv.org/pdf/2609.24559)
- **Unverified secondary URL.** A search result listed `alphaxiv.org/abs/2608.timesfm-3.md`. It is not a real arXiv id format, and I did not read it. Ignore it.
- **No formal TimesFM 3 paper found.** Searches found no arXiv paper for TimesFM 3.0. Architecture details therefore rest on the model card and the Google blog. [Verified: secondary, search returned none]

### Gaps
- Max context length and max horizon for TimesFM 3 are not published on the model card, blog or README. The library default (15,360 context) is the only number, and I have not verified the horizon limit. Gap.
- Quantile calibration of TimesFM 3 is not published.
- Whether TimesFM 3 is really the "2.0" family's successor, or a separate architecture, is not stated. The blog says it is a new model. Gap.

---

## Q2. Licence: what does it actually say, and does the repo's claim hold?

### Takeaway
The repo's claim is correct. Code is Apache-2.0. TimesFM 3.0 **weights** are under the "TimesFM Non-Commercial License v1.0". Permitted use is research and testing "not tied to commercial gain, production deployment, or revenue generation". Using the results "in commercial decision-making, client deliverables, or paid products/services" is outside the licence. For Prophet, which runs on arbitrary user datasets, this is a hard gate, not a footnote.

### Cited Findings
- Licence title: "TimesFM Non-Commercial License v1.0" — [Verified: source](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE) (via summarising fetch)
- Grant: "access, use and create Derivatives of the TimesFM Model solely for your Non-Commercial Purposes" — [Verified: source](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)
- Definition: Non-Commercial Purpose "means use for testing, evaluation, or research not tied to commercial gain, production deployment, or revenue generation." "This includes internal benchmarking" "provided the results are not used in commercial decision-making, client deliverables, or paid products/services." — [Verified: source](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)
- Excluded: use "(a) for any revenue-generating activity, (b) in direct or indirect interactions with end users or production systems"; commercial or production use requires "a commercial license from Company, which Company may grant in its sole discretion" — [Verified: source](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)
- Outputs: "Outputs do not include any components of the TimesFM Model" and "Outputs are not considered Derivatives under this License." — [Verified: source](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)
- Revocable grant; assignment or sublicence void and terminates the licence. No governing-law clause in the text summarised — [Verified: source](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)
- PyPI: "The TimesFM source code in this repository is licensed under Apache-2.0"; "model weights up to version 2.5 remain Apache-2.0"; "TimesFM 3.0 pretrained weights are distributed under the separate timesfm-non-commercial-license-v1.0 license"; "Commercial or production use of the default pretrained weights is not permitted." — [Verified: source](https://pypi.org/project/timesfm/)
- GitHub: same split; "Commercial and production use of TimesFM 3.0 is fully permitted through authorized Google Cloud services" — [Verified: source](https://github.com/google-research/timesfm). The BigQuery route is the commercial path (blog: "directly integrated into BigQuery", `AI.FORECAST`) — [Verified: source](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
- Weights up to 2.5 are Apache-2.0 — [Verified: source](https://pypi.org/project/timesfm/); TimesFM 2.5 can be self-hosted commercially on that basis [Verified: secondary, search summary]

### Inferences
- **Prophet is not licence-clean for client use if it runs TimesFM 3 weights on client data.** Even "internal benchmarking" is allowed only if results do not feed commercial decisions. A client-facing agent that forecasts for paying customers is outside the licence. [Hypothesis, needs legal read]
- The lab's R&D on the synthetic twin is closer to the permitted "research" side, but the `CLAUDE.md` rule "no client production use until cleared" is the right gate.
- Options that avoid the gate: (a) use TimesFM 2.5 weights (Apache-2.0) for anything commercial, with a separate benchmark; (b) use the Google Cloud/BigQuery route for 3.0; (c) seek a commercial licence from Google.
- Cached weights do not change the licence. Forecasts produced from them in client work are the risk.

### Gaps
- I did not verify the licence text against the raw file. The summary tool quoted it. Confirm the exact wording before any client decision.
- Whether Google's BigQuery or Vertex terms differ from the licence for customer data: not checked.
- Whether a commercial licence from Google is obtainable: not checked.
- **Action:** escalate this to Bill or Ly, per the AXION governance instructions, before any client-facing use of TimesFM 3 weights.

---

## Q3. Official configuration, knobs and recommended settings

### Takeaway
The public record names almost no knobs for TimesFM 3. The installed package exposes the flags listed in `01-timesfm.md`, but none has a published recommended value. Context length, horizon, normalisation, NaN handling and batch sizing for TimesFM 3 are open questions. Treat every setting as a tunable to validate on your own data, not as a recommended default.

### Cited Findings
- Installed `predict_batch` flags: `return_quantiles`, `use_symmetric_averaging`, `make_positive`, `sort_quantiles`, `use_znorm`, `padding_mode` ("none" or "edge"), `past_only_covariates`, `past_future_covariates` — [Verified: repo file](/Users/liamle/Downloads/prophet/docs/prophet/investigation/01-timesfm.md) (read from the installed package, not public docs)
- GitHub README: univariate `predict_batch(..., horizon=12, return_quantiles=True)` returns (12,) point and (12, 9) quantiles; multivariate example takes `past_only_covariates` and `past_future_covariates` — [Verified: source](https://github.com/google-research/timesfm)
- MLX example: `TimesFM3Forecaster.from_pretrained(...)`, `predict(context, horizon=128, return_quantiles=True)` — [Verified: source](https://github.com/google-research/timesfm)
- Backends: PyTorch (`timesfm[torch]`), MLX (`timesfm[mlx]`, "numerically matched" to PyTorch per README), Flax (2.5 only) — [Verified: source](https://github.com/google-research/timesfm). Backend parity is a README claim, not independently checked.
- The README import paths are unclear: install commands use `timesfm`, examples use `timesfm3`. Confirm the import before use — [Verified: source](https://github.com/google-research/timesfm)

### Inferences
- Missing values, zeros and outliers: no official guidance found. The repo's own handling (clipping at zero, sorting quantiles in code, `make_positive` not used) is the only practice on record.
- Context length: no official accuracy-versus-context study for TimesFM 3. The repo caps at 512 (`forecast.py:41`) and asks, untested, whether up to 15,360 helps (H5 in the repo audit).
- Horizon: no published limit. The repo's MLX code has no horizon cap, but accuracy at long horizons is untested [Hypothesis].

### Gaps
- No official guidance on context length, horizon limit, chunking/autoregressive extension, frequency hints, NaN/outlier handling, or memory use on MLX/MPS/CPU.
- No official guidance on conformal adjustment or quantile calibration. The 2.5 README mentions a quantile head but gives no calibration method.

---

## Q4. Covariates and events

### Takeaway
TimesFM 3 is the first TimesFM version with covariates in the model itself (the 2.5 covariate route was a separate linear XReg model). Known-future covariates reach the model as conditioning rows, and the blog says they carry "known future signals like holidays or scheduled events". There is no independent evidence yet that they help on real data. The repo's own calendar result was negative on the twin.

### Cited Findings
- Past-future covariates "remain visible in the horizon", using a "lookahead" token design — [Verified: source](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
- TimesFM 2.5 used XReg for covariates ("xreg + timesfm" or "timesfm + xreg") — [Verified: source](https://github.com/google-research/timesfm); the repo reads the same modes in the installed `timesfm` 2.5 package
- Chronos-2 (arXiv 2510.15821) supports past-only and known-future covariates, real and categorical — [Verified: secondary, search summary](https://arxiv.org/abs/2510.15821). Its abstract claims a lead on fev-bench, which covers covariate tasks — [Verified: secondary, search summary](https://arxiv.org/html/2510.15821v1). Vendor-reported.
- Repo results (synthetic twin, own evidence): calendar covariate WAPE 21.5% against 21.0% without it over 52 origins; no gain — [Verified: repo doc](/Users/liamle/Downloads/prophet/docs/PLAN.md) (`PLAN.md:32`). Repo audit lists promo covariate as "no result recorded" (`01-timesfm.md` 4.2).
- Repo covariate design: one channel per series, conditioning only, raw scale unless `use_znorm` is set — [Verified: repo file](/Users/liamle/Downloads/prophet/docs/prophet/investigation/01-timesfm.md)

### Inferences
- How to encode events for scenario what-if:
  - **Holidays and calendar:** a 0/1 or fractional flag per period across the full context plus horizon. The repo's `calendar_vic.py` is an example of how it was done, but the public record has no recommended encoding.
  - **Promotions and price:** a known-future magnitude channel (for example, uplift over the window). Only known-in-advance values are valid. The repo's promo covariate reads the twin's truth table, so it has no real-data source yet [Verified: repo file].
  - **Intervention scenarios:** construct the covariate row for the scenario, run the forecast under each, and compare. This is a counterfactual-style use of conditioning, not a validated causal method [Hypothesis].
- The lab's calendar result says more channels or per-segment effects may be needed (H1 in the repo audit). That is consistent with a conditioning-only design, but untested.
- Covariate scale: the blog does not say whether scale normalisation is applied to covariates. The repo says the internal RevIN partly absorbs scale, but this is unverified [Hypothesis].

### Gaps
- No independent benchmark result for TimesFM 3 covariate skill that I could verify. Google's own claim is that TimesFM 3 is best on fev-bench and GIFT-Eval, including covariate tasks — [Verified: source (vendor claim)](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/), but no numbers were readable.
- No public guidance on whether to use past-only or past-future for each event type.

---

## Q5. Fine-tuning and adaptation

### Takeaway
Google presents TimesFM 3 as zero-shot. The package I checked exposes no training hook. Google's GitHub README lists a LoRA fine-tuning example added 9 Apr 2026, but it is unclear which TimesFM version it covers, and I did not open the example. There is no official fine-tuning guidance for TimesFM 3, and the weights carry the non-commercial licence, so fine-tuned derivatives carry it too.

### Cited Findings
- Blog: TimesFM 3 works "without requiring task-specific fine-tuning" — [Verified: source](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
- GitHub README: "Added a fine-tuning example using HuggingFace Transformers and PEFT (LoRA)" dated 9 Apr 2026, path `timesfm-forecasting/examples/finetuning/`; version not stated — [Verified: source](https://github.com/google-research/timesfm)
- The licence grants rights to "create Derivatives", but restricts all use of the derivatives to non-commercial purposes [Verified: source](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)
- Repo: "no training or fine-tuning hook" in the 3.0 package — [Verified: repo file](/Users/liamle/Downloads/prophet/docs/prophet/investigation/01-timesfm.md)

### Inferences
- Fine-tuning is not an option the public record supports for TimesFM 3. Any adaptation the agent does is limited to covariates, context selection, output post-processing and ensembling.
- Adaptation that does not touch weights is the safer route under the licence (for example, residual correction on top of the output). Whether the output counts as a "Derivative" is for legal to decide; the licence says outputs are not Derivatives.
- Residual correction and stacking with statistical baselines (ma8, seasonal naive, ETS) are the practical adaptation routes. The repo already tests a TimesFM+ma8 ensemble on the twin (`models.py:91-105`, backtest only).

### Gaps
- Data volumes and gains for LoRA on TimesFM 2.x or 3.x: not checked.
- Whether the LoRA example applies to 3.0 weights: not checked.

---

## Q6. Evaluation: benchmarks, rankings and where TimesFM sits

### Takeaway
Vendor benchmark claims put TimesFM 3 first on GIFT-Eval, fev-bench and TIME. I could not read any leaderboard table: the GIFT-Eval space did not render, and the fev-bench CSV returned 404. Independent figures are secondary and mixed. Chronos-2 is the main competitor on covariate-informed tasks. Do not quote rankings without opening the live leaderboard.

### Cited Findings
- Google claims TimesFM 3 is "the top-ranked model in terms of both point and probabilistic forecasting metrics" on GIFT-Eval, FEV-Bench and TIME, and "the best average rank" in multivariate mode. Comparators named: Chronos-2, Toto 2.0 family, TimesFM-2.5. No numbers on the page — [Verified: source (vendor claim)](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
- GitHub README makes the same first-place claims for fev-bench, TIME and GIFT-Eval — [Verified: source (vendor claim)](https://github.com/google-research/timesfm)
- Secondary (search summary): a 2026 paper lists TimesFM-3.0 at CRPS 0.4557 and MASE 0.6668 on GIFT-Eval, ahead of "t0-beta" — [Verified: secondary, search summary, source not opened]. I could not confirm which paper.
- Secondary: GIFT-Eval added a test-data leakage column in Aug 2025 because some models had pretraining data overlapping the test split. Check the flag before comparing rows — [Verified: secondary, search summary]
- Secondary: TimesFM 2.5 was reported top among zero-shot models on GIFT-Eval at launch, by MASE — [Verified: secondary, search summary]
- Chronos-2 fev-bench: a slide deck reports Chronos-2 90.7% win rate and 47.3% skill score, TiRex 80.8%/42.6%, TimesFM-2.5 75.9%/42.3% — [Verified: secondary, search summary, deck not opened]. A third-party blog gives Chronos-2 47.28 and Timestral 50.39 — [Verified: secondary, conflicts with the deck on ranking metric].
- Mixed independent results: one third-party comparison has TimesFM 2.5 winning on an electricity dataset and Chronos-2 on air quality — [Verified: secondary, search summary]

### Inferences
- Rankings on fev-bench and GIFT-Eval depend on the metric (win rate, skill score, CRPS, MASE, average rank). Sources disagree on which Chronos-2 and TimesFM-2.5 numbers are right. Do not pick one.
- Google's first-place claims are vendor claims. Independent checks are the only evidence the agent should trust.
- Statistical baselines (AutoETS, AutoARIMA, Theta, seasonal naive) are standard fev-bench and GIFT-Eval baselines; I did not verify their numbers.

### Gaps
- No readable leaderboard data for TimesFM 3, Chronos-2, Chronos-Bolt, Moirai 2, TiRex, Toto, Sundial or statistical baselines. Next step: open `https://huggingface.co/spaces/Salesforce/GIFT-Eval` and `https://huggingface.co/spaces/autogluon/fev-bench` in a browser and copy the current table with the date.
- Monash and TIME results are not checked.

### Standard evaluation practice (General practice, not fetched)
- Rolling-origin (walk-forward) backtest, with the same origins for every model, and a frozen forecast cache.
- Report MASE (scale-free) and a probabilistic score such as CRPS or weighted quantile loss (WQL). Use the same metric the leaderboard uses so the comparison is fair.
- Report pinball loss per quantile, WAPE only for point comparisons, and empirical coverage of the 10 to 90 band (target 80%).
- Pitfalls: leakage from future covariates or from normalisation computed on the whole series; very short horizons that favour persistence; scale-dependent metrics (MAE, RMSE, WAPE) across series of different sizes; comparing models on different origins or different baseline sets (the repo's own 15% bar conflict is an example of this).

---

## Q7. Known weaknesses and mitigations an agent can apply

### Takeaway
No official document lists TimesFM 3 weaknesses. Public-record gaps are: intermittent demand, short history, regime shifts and out-of-distribution scale. The repo's own evidence points to intermittent and lumpy series being worse under TimesFM than under TSB-style models (`PLAN.md:66`: lumpy pinball 0.341 TimesFM vs 0.348 TSB, a small gap). Mitigations below are [Hypothesis] unless stated.

### Cited Findings
- Repo: lumpy series TimesFM pinball 0.341 vs TSB 0.348; steady TimesFM 0.114 vs TSB 0.130 — [Verified: repo doc](/Users/liamle/Downloads/prophet/docs/PLAN.md) (`PLAN.md:66`)
- Repo calibration on the twin: median under-forecasts big sellers (42% of actuals at or below median), over-forecasts small (57%) — [Verified: repo doc](/Users/liamle/Downloads/prophet/docs/DESIGN_REVIEW.md) (`DESIGN_REVIEW.md:9`)
- Secondary: one commentary says teams using TimesFM-3 for important decisions should evaluate interval calibration on their own data — [Verified: secondary, search summary, source not opened]

### Inferences (agent-applicable mitigations)
- **Intermittent or lumpy series:** route to a TSB or Croston-type model, or use the TimesFM quantiles only as a check. Use a classifier on the share of zero periods (the repo's class definitions can be reused) [Hypothesis, supported by repo results].
- **Short history:** fall back to a moving average or seasonal naive when history is below a threshold (the repo uses `min_history` and falls back to ma4 — `models.py:144-159`) [Hypothesis].
- **Regime shifts:** shorten the context window, or add a regime flag as a past-future covariate. Not tested [Hypothesis].
- **Miscalibration:** check coverage on a rolling backtest and widen or narrow the band with a per-segment correction. The repo found a multiplicative trailing bias correction worsened WAPE and pinball (`DESIGN_REVIEW.md:10`), so test before adopting any such correction.
- **Scale:** normalise on a trailing window before calling the model, and un-scale after. `use_znorm` does this inside the library; test both.

### Gaps
- No public evidence on TimesFM 3 for intermittent series, short histories or regime change. Repo tests only cover the twin.

---

## Q8. Decision table: data situation to recommended configuration

These rows are a starting playbook, not a validated recommendation. Every row needs a rolling-origin backtest on the target data, and any row that touches client data needs the licence decision in Q2 first.

| Data situation | Suggested model or config | Basis |
|---|---|---|
| Client or commercial use, any data | Do not use TimesFM 3 weights. Use TimesFM 2.5 (Apache-2.0) or statistical baselines, or the Google Cloud/BigQuery route, after legal sign-off | [Verified: licence] (Q2) |
| Research or benchmarking on synthetic or internal data, non-commercial | TimesFM 3 with quantiles (return_quantiles=True), clip to zero, sort quantiles, compare to ma8 and seasonal naive on the same origins | [Verified: licence permits research] + [repo results] |
| Steady, high-volume series, long history | TimesFM 3, univariate, context 512 to start, test longer contexts (untested) | [Hypothesis] (repo H5) |
| Intermittent or lumpy series | Route to TSB or Croston; use TimesFM only if it wins on pinball in backtest | [Verified: repo, small gap]; [Hypothesis] |
| Short history (under a few seasonal cycles) | Seasonal naive or moving average; avoid seasonal claims | [Hypothesis] |
| Known future events (holidays, promos) | Past-future covariate rows of the same length as context plus horizon; test with and without | [Verified: blog for capability]; [repo: no gain on calendar, promo untested] |
| Regime shift or structural break | Shorter context; add a regime flag; monitor coverage | [Hypothesis] |
| Need calibrated intervals | Measure empirical coverage per segment; apply correction only if backtest improves pinball | [General practice]; [repo DESIGN_REVIEW] |

---

## Corrections to what the repo says

- `CLAUDE.md` and `PLAN.md` say "weights are non-commercial". The public record confirms this for TimesFM 3.0 weights. It does **not** apply to TimesFM 2.5 or earlier, which are Apache-2.0 per PyPI and the GitHub README.
- The repo's MLX defaults (context 15,360, batch 4) are implementation defaults in the installed package, not published model limits.
- The repo's `01-timesfm.md` section 5 item 1 says "no fine-tuning hook" in the installed 3.0 package. The public repo has a LoRA example dated Apr 2026 (not covered for 3.0). That is an example in the source repo, not in the package the lab installs. Consistent, but both are worth noting.

---

## Sources

- TimesFM 3.0 model card: https://huggingface.co/google/timesfm-3.0-pytorch
- TimesFM 3.0 licence file: https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE
- Google Research blog, TimesFM-3 (31 Aug 2026): https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/
- google-research/timesfm GitHub: https://github.com/google-research/timesfm
- PyPI timesfm: https://pypi.org/project/timesfm/
- TimesFM 1.0 paper (arXiv 2310.10688): https://arxiv.org/abs/2310.10688
- Chronos-2 paper (arXiv 2510.15821): https://arxiv.org/abs/2510.15821
- Chronos-2 HTML: https://arxiv.org/html/2510.15821v1
- GIFT-Eval leaderboard (did not render): https://huggingface.co/spaces/Salesforce/GIFT-Eval
- fev-bench (leaderboard CSV returned 404): https://huggingface.co/spaces/autogluon/fev-bench
- arXiv 2609.24559, "t_0" (not TimesFM 3; retracted as a source for TimesFM 3): https://arxiv.org/pdf/2609.24559
- Repo audit: /Users/liamle/Downloads/prophet/docs/prophet/investigation/01-timesfm.md
- Repo plan: /Users/liamle/Downloads/prophet/docs/PLAN.md
- Repo design review: /Users/liamle/Downloads/prophet/docs/DESIGN_REVIEW.md

## Next steps (for a human, before relying on this)
1. Escalate the Q2 licence question to Bill or Ly before any client-facing use of TimesFM 3 weights.
2. Open the GIFT-Eval and fev-bench leaderboards in a browser and record the current table with its date.
3. Read the raw licence file, not the summary, before quoting it.
4. Open the GitHub fine-tuning example and check which TimesFM version it targets.
5. Confirm the `timesfm3` import path from the README (the README install and import names differ).
