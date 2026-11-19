"""Lab configuration. Secrets are read from files, never printed or stored in the repo."""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# One DuckDB file per dataset so the real snapshot and the synthetic twin never mix.
DATASETS = {
    "staging": DATA / "staging.duckdb",
    "twin": DATA / "twin.duckdb",
    "twin_censored": DATA / "twin_censored.duckdb",   # same world as twin, orders drop after stockouts (twin/censor.py)
}
LAB_DB = DATA / "lab.duckdb"          # runs, proposals, decisions, lessons, ledgers
REVEAL_DB = DATA / "reveal.duckdb"    # local-only map from codes back to real names

# Liam, 19 Nov 2026: staging connection lives in foodflow/axis/env3.local.
DEFAULT_ENV_FILE = Path.home() / "Downloads/Axion/foodflow/axis/env3.local"
ENV_FILE = Path(os.environ.get("TIMESFM_ERP_ENV_FILE", DEFAULT_ENV_FILE))

BEDROCK_REGION = "ap-southeast-2"
BEDROCK_MODEL = os.environ.get("TIMESFM_ERP_BEDROCK_MODEL", "au.anthropic.claude-haiku-5-5")
BEDROCK_HARD_CAP_USD = 30.0           # Liam, 19 Nov 2026: hard stop, lab-only ledger

TIMESFM_CHECKPOINT = "google/timesfm-3.0-pytorch"


def read_env_value(key: str, env_file: Path | None = None) -> str:
    """Read one KEY=value from an env file without exporting or logging it."""
    path = env_file or ENV_FILE
    text = path.read_text()
    m = re.search(rf"^{re.escape(key)}=(.+)$", text, re.M)
    if not m:
        raise KeyError(f"{key} not found in {path.name}")
    return m.group(1).strip().strip('"').strip("'")
