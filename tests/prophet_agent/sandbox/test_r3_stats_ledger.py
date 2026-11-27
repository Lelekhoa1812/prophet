"""Round 3 regression (A4-05): a later ps.Ledger() cell must never shrink the saved mission ledger.

Motivation: WTI rep3 cell plan:d0:9 created `ledger = ps.Ledger()` and saved it with no entries, so out/ledger.json went
from five stage entries to []. The contract reads the saved ledgers, so the loss was silent. Logic: each sandbox cell is
a fresh process, so a Ledger cannot see the earlier cells' entries; save() must therefore read the file it is about to
replace and keep everything already on disk.
"""
from __future__ import annotations

import json

import pytest

from prophet_agent.sandbox import prophet_stats as ps


def _add_five(ledger: ps.Ledger) -> None:
    for i in range(1, 6):
        ledger.add(f"stage {i}", f"finding {i}", f"decision {i}", evidence={"i": i})


def _on_disk(path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_empty_later_ledger_does_not_erase_earlier_entries(tmp_path):
    # Cell 1 saves five entries; cell 2 creates an empty Ledger and saves it, as WTI rep3 did.
    target = tmp_path / "out" / "ledger.json"
    first = ps.Ledger()
    _add_five(first)
    first.save(str(target))
    assert len(_on_disk(target)) == 5

    ps.Ledger().save(str(target))
    kept = _on_disk(target)
    assert len(kept) == 5, "an empty later ledger must not replace five saved entries"
    assert [e["finding"] for e in kept] == [f"finding {i}" for i in range(1, 6)]


def test_later_cell_extends_the_saved_ledger(tmp_path):
    target = tmp_path / "out" / "ledger.json"
    first = ps.Ledger()
    _add_five(first)
    first.save(str(target))

    later = ps.Ledger()  # a new process: knows nothing about the five entries
    later.add("stage 6", "finding 6", "decision 6")
    later.save(str(target))
    saved = _on_disk(target)
    assert len(saved) == 6
    assert [e["index"] for e in saved] == [1, 2, 3, 4, 5, 6]
    assert saved[-1]["finding"] == "finding 6"


def test_repeated_save_on_one_ledger_does_not_duplicate(tmp_path):
    target = tmp_path / "ledger.json"
    led = ps.Ledger()
    _add_five(led)
    led.save(str(target))
    led.save(str(target))
    assert len(_on_disk(target)) == 5


def test_open_loads_the_existing_file_and_save_keeps_it(tmp_path):
    target = tmp_path / "ledger.json"
    first = ps.Ledger()
    _add_five(first)
    first.save(str(target))

    opened = ps.Ledger.open(str(target))
    assert len(opened.entries) == 5
    opened.add("stage 6", "finding 6", "decision 6")
    opened.save(str(target))
    assert len(_on_disk(target)) == 6


def test_unreadable_existing_file_is_not_overwritten(tmp_path):
    target = tmp_path / "ledger.json"
    target.write_text("{not a list", encoding="utf-8")
    led = ps.Ledger()
    led.add("stage 1", "finding", "keep")
    with pytest.raises(ValueError, match="cannot read"):
        led.save(str(target))
    assert target.read_text(encoding="utf-8") == "{not a list"


def test_opening_a_scan_as_a_ledger_says_what_the_file_is(tmp_path):
    """Round 20 (NVDA): the same opaque error three times on in/carried/scan.json stalled the mission."""
    target = tmp_path / "scan.json"
    target.write_text(json.dumps({"level_shifts": [], "outliers": []}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"not a ledger.*level_shifts.*json.load.*ps.Ledger\(\)"):
        ps.Ledger.open(str(target))
