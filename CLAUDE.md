# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An R&D lab (not a deployed product) that tests where TimesFM 3 forecasting plus "agents" that raise alerts add value to FoodFlow/AXIS (food distribution ERP). Read-only against FoodFlow. Numbers in the docs come from a synthetic twin (method demo, not client results). Locked product decisions are in `docs/PLAN.md`; the standing improvement brief is `docs/CONTINUOUS_GOAL.md`; ranked work is `docs/GOAL_BACKLOG.md`, history in `docs/GOAL_CHANGELOG.md`. Keep decisions there consistent with code changes, and label every figure as synthetic replay vs real staging vs measured.

## Commands

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -e ".[dev]"   # timesfm[mlx] needs Apple silicon
scripts/check.sh                       # local gate: pytest + console typecheck + build (CI was removed; Actions billing blocked)
scripts/check.sh --no-console          # pytest only
.venv/bin/python -m pytest -q tests/test_rollout.py::test_phase_demo_shadow_live   # single test (109 tests, ~40 s total)
PYTHONPATH=src .venv/bin/uvicorn lab.api.main:app --host 127.0.0.1 --port 8000   # API + built console (serves console/dist)
cd console && npm install && npm run dev      # Vite dev server; `npm run build` = tsc --noEmit && vite build
```

- Always run python as `.venv/bin/python` (or with `PYTHONPATH=src`); the package is `lab` under `src/`.
- There is no linter configured.
- TimesFM loads in ~1.5 s when weights are cached (first-ever load downloads `google/timesfm-3.0-pytorch` from Hugging Face; the runbook's "80 s" is cold). Forecasts are cached in `data/cache` (~2000 pickles); stale cache is a common source of confusing replay results.
- `TIMESFM_ERP_LAB_DB` points the proposals/decisions DB at a throwaway file; use it when testing UI or API writes.

## Data and git-ignore

`data/` is git-ignored and holds everything stateful: `twin.duckdb`, `twin_censored.duckdb`, `staging.duckdb` (pseudonymised real snapshot), `lab.duckdb` (runs, proposals, decisions, lessons), `reveal.duckdb` (code-to-real-name map, local only), and the `exp_*.py` experiment scripts with their logs/pickles. Experiment scripts are therefore not in git; `docs/RUNBOOK.md` §4 describes them. Never commit anything from `data/`, real names, or credentials. Staging credentials are read from a file outside the repo (`config.ENV_FILE`, default `../foodflow/axis/env3.local`) via `read_env_value`; never print them.

Tests that need `data/twin.duckdb` skip themselves when it is absent; most tests use the `mini` fixture (`tests/conftest.py`): a 60-product seeded twin built per module with a cheap router (moving average + TSB, no TimesFM).

## Architecture

Data flows: **source → DuckDB dataset → `WorldData` → forecast → agents → policy/governor → store → API/console**, and `sim/` replays this day by day.

- **Datasets** (`config.DATASETS`, `store.py` schema): the same schema for the real staging snapshot (`source/staging.py`, SELECT-only guard as a second line of defence; the DB role is the real limit, accepted risk B12) and the synthetic twin (`twin/generator.py`, `twin/supply.py`, `twin/censor.py`). Only the twin has `truth_events` and `daily_outcomes`, which is how agents get scored. Names are pseudonymised to codes (`source/pseudo.py`).
- **Engine** (`engine/`): `state.py` builds an in-memory `WorldData` per dataset (`get_world`, cached in `_WORLDS`) and `ForecastService` (disk-cached forecasts). `forecast.py` wraps TimesFM 3 (MLX) as a process-wide singleton and returns `Forecast(point, q)` with 9 quantiles (0.1–0.9), clipped at 0 and sorted. `models.py` is a model-agnostic adapter: every model returns the same `Forecast`; `Router` sends each series class (short / intermittent / steady) to a model; `fit_routing` keeps TimesFM unless a challenger beats it by ≥2% pinball loss; `allow_non_commercial=False` swaps in free models for licence reasons. `distribution.py` turns quantiles into sampled demand paths used for expiry/shortfall maths; `profile.py` derives client-relative thresholds; `backtest.py`/`baselines.py`/`metrics.py` are walk-forward evaluation.
- **Agents** (`agents/`): shortfall, freshness, customer, margin. Each is a pure function `AgentContext(world, as_of, fs, params, lessons) → [Proposal]` with evidence, dollars at stake, confidence, `dedupe_key`, and `autonomy_level` (capped at "prepares": the lab cannot claim acts/runs). Registered in `agents/__init__.py` `REGISTRY`.
- **Policy** (`policy/`): `alerts.py` is the one-score/one-budget rule (net value = share saved × expected loss − review cost − P(false) × false-alarm cost; per-owner daily budget, rest to a digest; order-entry stops exempt). `governor.py` dedupes, applies cooldowns, dismissal lessons and the autonomy ladder; `learning.py`/`calibration.py` adjust priorities within bounds; `rollout.py` owns the tenant file, readiness backtest, 14-day shadow phase and drift check; `whatif.py` estimates dollars protected (expected, never "observed").
- **Sim** (`sim/`): `replay.py` steps days, runs agents, governs, stores, scores (`scorer.py` against twin truth, reporting count and dollar recall/precision). `policy_sim.py` re-scores recorded candidates under new economics in seconds, which is why changing policy settings does not require re-forecasting.
- **API** (`api/main.py`, ~900 lines, FastAPI): same-origin guard middleware (Host allow-list, Origin check on writes, token on `/api/reveal`); operator identity is a labelled `unverified:` header, no real login. `api/jobs.py` runs backtests/replays on threads with file-per-job persistence. Narration (`narration/`) is templated by default, Bedrock (Haiku, hard $30 cap) optional.
- **Console** (`console/`, React 18 + Vite + Tailwind 4 + Radix, pages under `src/pages`): talks to the API through `src/lib/api.ts`; the built output `console/dist` is served by FastAPI. Rebuild after UI changes.

## Invariants worth preserving

- No future leakage: every agent/forecast at `as_of` may only see data up to that date (`tests/test_no_future_leak*.py`, `test_decision_integrity.py`, `test_eval_gate.py` guard this and decision-quality floors).
- "Flagged", "useful", "accepted", "acted" and "loss prevented" are different outcomes; never present synthetic or expected dollars as observed savings.
- The 15% improvement-over-best-baseline bar for TimesFM is documented and currently **not met** (7–10% on the twin); do not move it to claim a win.
- TimesFM 3 weights are non-commercial: no client production use until cleared. The lab never writes to FoodFlow or connects to production.
- Weekly forecasts start at the next full week; the default history window is all history on every page (briefing and Explorer must agree).
- Several modules use "Motivation vs Logic" / "Bugs vs Fixes" docstring blocks; follow that style when adding non-obvious logic.
