#!/usr/bin/env bash
# Local gate for Prophet. CI is not used (Actions billing blocked).
#   scripts/check.sh            offline tests (no model, no network)
#   scripts/check.sh --real     also runs the real TimesFM 3 tests (needs cached weights, Apple silicon)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
export PYTHONPATH=src

"$PY" -m pytest -q -m "not timesfm"

if [[ "${1:-}" == "--real" ]]; then
  "$PY" -m pytest -q -m timesfm
fi
