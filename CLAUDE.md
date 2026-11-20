# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Prophet is an R&D forecasting agent. A durable goal loop plans, calls tools (including TimesFM 3), asks the user
structured questions, runs sandboxed analytics, and produces charts and documents. It is not a deployed product and
it is not connected to any ERP or production system. Locked decisions live in `docs/prophet/ARCHITECTURE.md`; the
completion criteria and their evidence live in `docs/prophet/GOAL.md`.

## Commands

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -e ".[dev]"   # timesfm[mlx] needs Apple silicon
scripts/check.sh                 # offline gate: pytest without the real model
scripts/check.sh --real          # also the real TimesFM 3 tests (cached weights)
PYTHONPATH=src .venv/bin/python -m prophet_agent.nim.probe keys     # live NIM check; prints status only, never keys
PYTHONPATH=src .venv/bin/python -m prophet_agent.nim.probe models   # live schema check per chain model
```

- Run Python as `.venv/bin/python` with `PYTHONPATH=src`. The package is `prophet_agent` under `src/`.
- Do not name the package `prophet`: that collides with Meta's forecasting library on PyPI.

## Architecture

- `prophet_agent/nim/`: NVIDIA NIM transport following `NIM.md` section 4.2 (key pool, per-(key, model) cooldowns,
  failover before the first token, schema-bound output, local validation, redaction).
- `prophet_agent/runtime/`: durable mission runtime on SQLite WAL. `store.py` holds the event log, step ledger,
  idempotent tool-call ledger, leases and questions. `runner.py` replays a deterministic program and pauses on
  questions. `goal.py` is the plan, act, evaluate loop with stall detection and budgets. `questions.py` validates the
  AskUserQuestion shape.
- `prophet_agent/forecast/`: forecasting maths (metrics, baselines, TSB, distributions, walk-forward backtest) and the
  TimesFM 3 engine (`timesfm.py`, MLX, full official context).
- `prophet_agent/tools/`: tools the goal loop calls. Each declares whether it is idempotent.
- `prophet_agent/sandbox/`: child-process runner for analytics Python. A guardrail, not isolation: `-I`, scrubbed
  environment, rlimits (memory cap only on Linux), an audit hook blocking writes outside the run folder, subprocesses
  and sockets, and a hard timeout. Outputs are listed in a sha256 manifest.
- `prophet_agent/artifacts/`: chart and document pack (PNG, SVG, XLSX, DOCX, PDF) built inside the sandbox.
- `prophet_agent/synth/`: synthetic demand generator with censoring for backtests. Numbers from it are synthetic.

## Invariants

- Idempotency: a tool call that finished is never re-run. A non-idempotent call left in "started" by a crash is not
  retried; the mission stops with `outcome_unknown`.
- Replay: model decisions and tool results are stored before the run moves on. A resumed run reads them; it does not
  ask the model again for completed work.
- Never present synthetic or expected numbers as observed results. Label every figure as synthetic replay, real
  staging, or measured.
- TimesFM 3 weights are non-commercial: research use only. Do not use outputs for commercial decisions, client
  deliverables or paid services. The licence gate is in `docs/prophet/ARCHITECTURE.md` section 5.
- Keys: `nvapi-` values are never printed, logged, stored in the ledger or committed. `.env` is git-ignored.
- `data/` is git-ignored working data (ledgers, caches, sandbox runs). Do not commit it.
- The 15 % improvement bar over the best baseline is not met on the synthetic twin. Do not move it to claim a win.
