#!/usr/bin/env bash
# Local test gate. Replaces the GitHub Actions workflow (removed while Actions billing is blocked).
# Usage: scripts/check.sh [--no-console]
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== pytest =="
.venv/bin/python -m pytest -q -W ignore tests
if [[ "${1:-}" != "--no-console" ]]; then
  echo "== console typecheck + build =="
  (cd console && npm run build)
fi
echo "OK: all gates passed"
