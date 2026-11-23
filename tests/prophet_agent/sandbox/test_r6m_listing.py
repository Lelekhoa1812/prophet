"""Round 6m, item 3 (audit-hook bypass): listing the project or home folder is a read.

Probe (before the fix): os.listdir() and os.scandir() of the project folder returned its file names (research_notes,
PRODUCT.md, API.md, and the .env beside them). The contents were blocked, but the names were not, and the names say what
keys and notes exist. The fix applies the read rule to the two listing calls. Listing the run folder keeps working.
"""
from __future__ import annotations

from pathlib import Path

from prophet_agent.config import ROOT
from prophet_agent.sandbox.runner import run_python


def test_the_project_folder_cannot_be_listed_with_listdir():
    r = run_python(f"import os\ntry:\n    print('NAMES', os.listdir({str(ROOT)!r}))\n"
                   f"except PermissionError:\n    print('REFUSED')\n")
    assert r.ok, r.stderr
    assert "NAMES" not in r.stdout and "REFUSED" in r.stdout


def test_the_project_folder_cannot_be_listed_with_scandir():
    r = run_python(f"import os\ntry:\n    print('NAMES', [e.name for e in os.scandir({str(ROOT)!r})])\n"
                   f"except PermissionError:\n    print('REFUSED')\n")
    assert r.ok, r.stderr
    assert "NAMES" not in r.stdout and "REFUSED" in r.stdout


def test_the_home_folder_cannot_be_walked():
    # os.walk ignores errors by default, so a refused listing shows as an empty walk rather than an exception.
    home = str(Path.home())
    r = run_python(f"import os\nprint('WALKED', len(list(os.walk({home!r}))))\n")
    assert r.ok, r.stderr
    assert "WALKED 0" in r.stdout


def test_the_run_folder_can_still_be_listed():
    r = run_python("import os\nopen('out/a.txt', 'w').close()\n"
                   "print(sorted(os.listdir('out')), sorted(e.name for e in os.scandir('out')))\n")
    assert r.ok, r.stderr
    assert "['a.txt'] ['a.txt']" in r.stdout
