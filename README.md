# Prophet

A durable forecasting agent. It plans, calls tools (including TimesFM 3), asks structured questions, runs sandboxed
analytics and produces charts and documents. R&D use only.

- Plan and status: [docs/prophet/GOAL.md](docs/prophet/GOAL.md)
- Architecture decisions: [docs/prophet/ARCHITECTURE.md](docs/prophet/ARCHITECTURE.md)
- Working guidance for Claude Code: [CLAUDE.md](CLAUDE.md)

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -e ".[dev]"
scripts/check.sh
```

TimesFM 3 weights are under a non-commercial licence. Read `docs/prophet/ARCHITECTURE.md` section 5 before any use.
