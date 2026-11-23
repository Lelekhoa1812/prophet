"""Paths and settings. Secrets are read from the environment or the git-ignored `.env`, never printed or stored.

Motivation vs Logic
Motivation: the legacy lab hard-coded personal paths and an ERP env file. Prophet needs one obvious place for state
(`data/`), one obvious place for keys (`.env` at the repo root) and no personal defaults.
Logic: `PROPHET_HOME` overrides the root (tests use a temp dir); `data/` holds the SQLite event store, uploaded files,
mission workspaces and artefacts. `load_dotenv_values` is a tiny parser so no extra dependency is needed and values are
returned, not exported into `os.environ`.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("PROPHET_HOME") or Path(__file__).resolve().parents[2])
DATA = ROOT / "data"
DB_PATH = Path(os.environ.get("PROPHET_DB", DATA / "prophet.sqlite"))
MISSIONS_DIR = DATA / "missions"          # one workspace folder per mission: uploads/, work/, artifacts/
DOTENV = ROOT / ".env"

TIMESFM_CHECKPOINT = os.environ.get("PROPHET_TIMESFM_CHECKPOINT", "google/timesfm-3.0-pytorch")


def ensure_dirs() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    MISSIONS_DIR.mkdir(parents=True, exist_ok=True)


def load_dotenv_values(path: Path | None = None) -> dict[str, str]:
    """Parse KEY=VALUE lines. Ignores comments and blanks; strips one pair of quotes. Never raises on a missing file."""
    p = path or DOTENV
    try:
        text = p.read_text()
    except OSError:
        return {}
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        out[k] = _value(v.strip())
    return out


def _value(raw: str) -> str:
    """A quoted value is what is between its quotes. An unquoted value ends at a space and "#", as in python-dotenv, so a
    trailing note is not read as part of a key."""
    if raw[:1] in ("'", '"'):
        end = raw.find(raw[0], 1)
        if end != -1:
            return raw[1:end]
    return re.split(r"\s+#", raw, maxsplit=1)[0].rstrip()
