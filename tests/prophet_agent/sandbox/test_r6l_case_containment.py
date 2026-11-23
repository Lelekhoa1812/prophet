"""Round 6l, audit item 2: containment checks must ignore case on a case-insensitive file system (macOS, Windows).

Probe (before the fix, on this Mac's APFS volume): reading the project's CLAUDE.md through the home folder typed as
/Users/LIAMLE was allowed, and writing into in/ through the name IN was allowed. The guard compared paths case-sensitively.
The tests run only where the file system ignores case; elsewhere the variant path does not exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from prophet_agent.sandbox.runner import run_python

PROJECT = Path(__file__).resolve().parents[3]
HOME = str(Path.home())

darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="case-folding matters only where case is ignored")


@darwin_only
def test_reading_a_project_file_through_an_upper_case_home_path_is_refused():
    project_file = PROJECT / "CLAUDE.md"
    if not str(project_file).startswith(HOME + "/"):
        pytest.skip("the project is not under the home folder on this machine")
    variant = str(project_file).replace(HOME, HOME.upper(), 1)
    r = run_python(f"print(open({variant!r}).read()[:5])")
    assert not r.ok
    assert "reading this location is blocked" in r.stderr


@darwin_only
def test_overwriting_an_input_through_an_upper_case_in_folder_is_refused(tmp_path):
    src = tmp_path / "data.csv"
    src.write_text("orig")
    code = ("import os\n"
            "open(os.path.join(os.getcwd(), 'IN', 'data.csv'), 'w').write('CHANGED')\n")
    r = run_python(code, inputs={"data.csv": src})
    assert not r.ok
    assert "write outside the run folder is blocked" in r.stderr
    assert run_python("print(open('in/data.csv').read())", inputs={"data.csv": src}).stdout.strip() == "orig"


@darwin_only
def test_normal_reads_and_writes_in_the_run_folder_still_work(tmp_path):
    src = tmp_path / "data.csv"
    src.write_text("orig")
    r = run_python("print(open('in/data.csv').read())\nopen('out/x.txt', 'w').write('ok')\n", inputs={"data.csv": src})
    assert r.ok and r.stdout.strip() == "orig"
    assert [o.path for o in r.outputs] == ["out/x.txt"]
