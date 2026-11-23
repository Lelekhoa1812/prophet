"""Every source's licence key must be a row in docs/prophet/DATASET_LICENCES.md. The code names keys only; the wording is
read from that table (the same table scripts/fetch_datasets.py reads), so a missing row fails here, not in a download."""
import re
from pathlib import Path

from prophet_agent.opendata import sources

ROOT = Path(__file__).resolve().parents[3]
LICENCE_DOC = ROOT / "docs" / "prophet" / "DATASET_LICENCES.md"
# The same row pattern as scripts/fetch_datasets.py (_LICENCE_ROW).
LICENCE_ROW = re.compile(r"^\|\s*`([a-z_]+)`\s*\|\s*(.+?)\s*\|\s*$")


def _table_keys() -> dict:
    rows = {}
    for line in LICENCE_DOC.read_text(encoding="utf-8").splitlines():
        row = LICENCE_ROW.match(line)
        if row:
            rows[row.group(1)] = row.group(2)
    return rows


def test_every_source_licence_key_exists_in_the_licence_table():
    table = _table_keys()
    missing = {source.key: source.licence_key for source in sources.SOURCES.values()
               if source.licence_key not in table}
    assert not missing, f"licence keys missing from {LICENCE_DOC.name}: {missing}"


def test_the_table_rows_used_by_opendata_have_wording():
    table = _table_keys()
    for key in ("open_meteo", "ecb"):
        assert table[key].strip(), key
