"""Mission 9 (M9-7): the stall rule catches a near-duplicate re-parse loop within 4 repeats.

Live, EPL 56465803 (2026-11-26): about 60 sandbox cells re-read the same carried files and printed the same standings table with
small differences, and the stall rule fired only at step 63 of 196. The fixture replays that mission's main-planner calls (the
act steps at depth 0, read read-only from data/prophet.sqlite). Sub-agent calls have their own history, so they are not replayed.
Synthetic cells cover the progress cases. Nothing here runs the sandbox or the model.
"""
from __future__ import annotations

import json
from pathlib import Path

from prophet_agent.runtime import goal as G

FIXTURE = Path(__file__).parent / "fixtures" / "m9_epl_sandbox_replay.json"


def _history(entries: list[dict]) -> list[dict]:
    out = []
    for e in entries:
        obs = {"kind": "tool", "tool": e["tool"], "args": e["args"], "result": e["result"], "step": e["step"]}
        if "error" in e:
            obs["error"] = e["error"]
        out.append(obs)
    return out


def _main_calls() -> list[dict]:
    doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [e for e in doc["calls"] if e["step"].startswith("act:d0:")]


def _is_standings(e: dict) -> bool:
    out = str(e["result"].get("stdout") or "")
    return e["tool"] == "sandbox_python" and "Arsenal" in out and "(11, 9)" in out


def _cell(step: str, stdout: str, *, code: str = "open('in/carried/pl_standings_2026_27.csv')", outputs=(), ok=True) -> dict:
    return {"kind": "tool", "tool": "sandbox_python", "args": {"code": code}, "step": step,
            "result": {"ok": ok, "returncode": 0 if ok else 1, "timed_out": False, "stdout": stdout, "stderr": "",
                       "outputs": list(outputs)}}


TABLE = "Standings shape: (11, 9)\n" + "\n".join(
    f"{i} {club} {p} {w} {d} {l} {gd} {pts} {form}"
    for i, (club, p, w, d, l, gd, pts, form) in enumerate([
        ("Arsenal", 11, 9, 1, 1, 10, 28, "WWWWD"), ("Brighton", 11, 7, 2, 2, 5, 23, "WDWLW"),
        ("Aston_Villa", 11, 6, 3, 2, 4, 21, "DWLWW"), ("Chelsea", 11, 6, 2, 3, 3, 20, "LWWDL"),
        ("Liverpool", 11, 5, 4, 2, 6, 19, "DDWWL"), ("Bournemouth", 11, 5, 3, 3, 2, 18, "WLLWD"),
        ("Sunderland", 11, 5, 2, 4, 1, 17, "LWLWW"), ("Forest", 11, 4, 4, 3, 0, 16, "DLWDD"),
        ("Fulham", 11, 4, 3, 4, -2, 15, "LDLWW"), ("Everton", 11, 4, 2, 5, -4, 14, "WLDLL"),
        ("Leeds", 11, 3, 4, 4, -3, 13, "DDLWL")], 1))


def test_epl_standings_reparse_notice_fires_by_fourth_repeat():
    calls = _main_calls()
    hist = _history(calls)
    first, repeats, notice_at, stop_at = None, 0, None, None
    for n, e in enumerate(calls, 1):
        if not _is_standings(e):
            continue
        if first is None:
            first = n
            continue
        repeats += 1
        if notice_at is None and G._stall_notice(hist[:n], None):
            notice_at = repeats
        if stop_at is None and G._stalled(hist[:n], G.GoalConfig().stall_window):
            stop_at = repeats
            break
    assert first is not None
    assert notice_at is not None and notice_at <= 4, (notice_at, repeats)
    assert stop_at is not None and stop_at <= 5, stop_at


def test_near_only_repeat_is_caught_without_an_exact_match():
    # d0:29 and d0:32 print the same standings rows with different diagnostic lines, so their outcome keys differ.
    calls = _main_calls()
    hist = _history(calls)
    pos = {e["step"]: n for n, e in enumerate(calls)}
    prefix = hist[:pos["act:d0:32"] + 1]
    assert G._progress_key(hist[pos["act:d0:29"]]) != G._progress_key(hist[pos["act:d0:32"]])
    assert len(G._exact_run(prefix)) == 1
    assert len(G._near_run(prefix)) == 2
    note = G._stall_notice(prefix, "fetch the points table")
    assert note is not None
    assert "nearly the same" in note and "fetch the points table" in note


def test_a_new_file_is_progress_not_a_repeat():
    cells = [_cell(f"act:d0:{i}", TABLE, outputs=[{"path": f"out/part{i}.csv", "bytes": 10, "sha256": f"x{i}"}])
             for i in range(4)]
    hist = _history(cells)
    assert G._near_run(hist) == []
    assert G._stall_notice(hist, None) is None


def test_the_same_content_under_a_new_name_is_not_progress():
    """EPL 56465803 replay: the standings were saved as _full.csv and then _clean.csv with the same sha256."""
    cells = [_cell(f"act:d0:{i}", TABLE, outputs=[{"path": f"out/part{i}.csv", "bytes": 10, "sha256": "same"}])
             for i in range(4)]
    assert len(G._near_run(_history(cells))) == 4


def test_a_new_table_shape_is_progress():
    small = "Standings shape: (11, 9)\n" + TABLE.split("\n", 1)[1]
    grown = "Standings shape: (20, 9)\n" + TABLE.split("\n", 1)[1]
    hist = _history([{**_cell("act:d0:1", small)}, {**_cell("act:d0:2", small)}, {**_cell("act:d0:3", grown)}])
    assert len(G._near_run(hist)) == 1
    assert G._stall_notice(hist, None) is None


def test_a_new_carried_file_is_progress():
    hist = _history([_cell("act:d0:1", TABLE), _cell("act:d0:2", TABLE),
                     _cell("act:d0:3", TABLE, code="open('in/carried/ledger.json')")])
    assert G._near_run(hist) == []


def test_run_ids_and_file_listings_do_not_make_a_difference():
    a = _cell("act:d0:1", TABLE + "\nrun 0123456789abcdef0123456789abcdef\nFiles in in/carried:\n  pl_standings_2026_27.csv (377 bytes)\n")
    b = _cell("act:d0:2", TABLE + "\nrun fedcba9876543210fedcba9876543210\nFiles in in/carried:\n  pl_standings_2026_27.csv\n")
    hist = _history([a, b])
    assert len(G._near_run(hist)) == 2


def test_a_different_result_breaks_the_run():
    other = "Points path: 2026-27 forecast\n" + "\n".join(f"day {i} value {i * 3} {i * 7} {i * 11}" for i in range(12))
    hist = _history([_cell("act:d0:1", TABLE), _cell("act:d0:2", other), _cell("act:d0:3", TABLE)])
    assert len(G._near_run(hist)) == 1


def test_a_pointer_keeps_the_profile_of_the_call_it_points_to():
    first = _cell("act:d0:1", TABLE)
    hist = _history([first, _cell("act:d0:2", TABLE)])
    hist.append({"kind": "tool", "tool": "sandbox_python", "args": {"code": "open('in/carried/pl_standings_2026_27.csv')"},
                 "repeat_of": 1, "outcome": G._progress_key(hist[0]), "step": "act:d0:3"})
    assert len(G._near_run(hist)) == 3
    assert G._stalled(hist, 3)


def test_a_notice_names_the_repeat_and_the_next_step():
    hist = _history([_cell("act:d0:1", TABLE), _cell("act:d0:2", TABLE)])
    note = G._stall_notice(hist, "download the standings from the page")
    assert note is not None
    assert "download the standings from the page" in note
    assert "in/carried/pl_standings_2026_27.csv" in note
