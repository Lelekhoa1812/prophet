"""Round 6m, goal audit item 1: a sandbox cell is the same call when its code and the carried bytes it can read are the same.

Live stored missions (data/prophet.sqlite, read 2026-11-23): 4,917 sandbox cells, 553 of them the same code as an earlier cell.
The raw arguments differ between many of those pairs, because auto-carry mounts every newest data file of the mission and a
re-run of an upstream cell writes a new run folder with the same bytes. A comment or a layout change also made a new call.
Replayed on the stored cells, identifying a cell by its code and the bytes of the carried files it can read finds 527 repeats
(348 by raw arguments). Stall detection also compared stderr, which names the run folder, so identical outcomes looked new.

Synthetic cells and results only. Nothing here runs the sandbox.
"""
from __future__ import annotations

from prophet_agent.runtime import goal as G


def _obs(code, carry, run, outputs=(), stderr="", ok=True):
    """A stored sandbox observation: the summary the planner reads, and the full result kept beside it (as _act keeps it)."""
    summary = {"ok": ok, "run_id": run, "stdout": "", "stderr": stderr,
               "outputs": [f"{p} (1 bytes)" for p, _ in outputs]}
    full = {"ok": ok, "run_id": run, "stdout": "", "stderr": stderr, "returncode": 0 if ok else 1, "timed_out": False,
            "outputs": [{"path": p, "sha256": d, "bytes": 1} for p, d in outputs]}
    return {"kind": "tool", "tool": "sandbox_python", "args": {"code": code, "carry": carry}, "result": summary, "full": full}


def _args(code, carry=()):
    return {"code": code, "carry": list(carry)}


def test_a_cell_that_reads_no_carried_file_is_a_repeat_when_the_mission_wrote_another_file():
    # Auto-carry mounted forecast.json into the second call. The code never reads it, so the result cannot differ.
    history = [_obs("print(1)", [], "r1"), _obs("write()", [], "r2", [("out/forecast.json", "d2")])]
    args = _args("print(1)", [{"run_id": "r2", "file": "out/forecast.json"}])
    assert G._repeat_of("sandbox_python", args, history) == 1


def test_a_cell_that_reads_a_carried_file_is_a_repeat_when_another_run_wrote_the_same_bytes():
    read = "import pandas as pd\ndf = pd.read_json('in/carried/scan.json')\nprint(len(df))"
    history = [
        _obs("write()", [], "r1", [("out/scan.json", "d1")]),
        _obs(read, [{"run_id": "r1", "file": "out/scan.json"}], "r2"),
        _obs("rerun()", [], "r3", [("out/scan.json", "d1")]),       # an upstream re-run: new folder, the same bytes
    ]
    args = _args(read, [{"run_id": "r3", "file": "out/scan.json"}])
    assert G._repeat_of("sandbox_python", args, history) == 2


def test_a_cell_that_reads_a_carried_file_is_run_again_when_its_bytes_changed():
    read = "import pandas as pd\ndf = pd.read_json('in/carried/scan.json')\nprint(len(df))"
    history = [
        _obs("write()", [], "r1", [("out/scan.json", "d1")]),
        _obs(read, [{"run_id": "r1", "file": "out/scan.json"}], "r2"),
        _obs("rerun()", [], "r3", [("out/scan.json", "d2")]),       # the file really changed: new evidence
    ]
    args = _args(read, [{"run_id": "r3", "file": "out/scan.json"}])
    assert G._repeat_of("sandbox_python", args, history) is None


def test_a_comment_or_layout_change_does_not_make_a_new_cell():
    history = [_obs("x = 1\nprint(x)", [], "r1")]
    args = _args("# second look, see run 0123456789abcdef0123456789abcdef\n\nx = 1   # same\nprint( x )\n")
    assert G._repeat_of("sandbox_python", args, history) == 1


def test_a_cell_that_lists_the_carried_folder_depends_on_every_carried_file():
    # os.listdir sees each mounted file, so an unrelated carried file can change the result: it is not a repeat.
    code = "import os\nprint(sorted(os.listdir('in/carried')))"
    history = [_obs(code, [], "r1")]
    args = _args(code, [{"run_id": "r2", "file": "out/extra.json"}])
    assert G._repeat_of("sandbox_python", args, history) is None


def test_a_different_cell_is_never_a_repeat():
    history = [_obs("print(1)", [], "r1")]
    assert G._repeat_of("sandbox_python", _args("print(2)"), history) is None


def test_without_recorded_digests_a_carried_file_is_identified_by_its_run():
    # Older histories and results too large to keep carry no digest: the file is the same only if its run is.
    read = "pd.read_json('in/carried/scan.json')"
    old = _obs("write()", [], "r1")
    old.pop("full")
    history = [old, _obs(read, [{"run_id": "r1", "file": "out/scan.json"}], "r2")]
    assert G._repeat_of("sandbox_python", _args(read, [{"run_id": "r4", "file": "out/scan.json"}]), history) is None
    assert G._repeat_of("sandbox_python", _args(read, [{"run_id": "r1", "file": "out/scan.json"}]), history) == 2


def test_stall_sees_the_same_outcome_when_only_the_run_folder_in_stderr_differs():
    # The traceback names data/sandbox/<run folder>/program.py, so the same failing cell looked new on every run.
    def traceback(run):
        return f'File "/work/data/sandbox/{run}/program.py", line 19, in <module>\nKeyError: "value"'
    history = [_obs("print(df['value'])", [], f"{i}" * 32, stderr=traceback(f"{i}" * 32), ok=False) for i in (1, 2, 3)]
    assert G._stalled(history, 3) is True


def test_stall_sees_the_same_outcome_when_the_cell_differs_only_in_a_comment():
    history = [_obs(f"x = 1\nprint(x)  # look {i}", [], f"{i}" * 32) for i in (1, 2, 3)]
    assert G._stalled(history, 3) is True
