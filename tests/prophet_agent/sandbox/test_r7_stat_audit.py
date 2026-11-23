"""Round 7, item 1 (audit gap): os.stat and os.lstat raise no audit event, so a probe of a path outside the run folder was
never seen by the hook. Probe (before the fix): os.stat of a project file returned its size, and os.path.exists returned
True. The fix applies the read rule that open() uses: a path the sandbox may not read raises PermissionError from stat,
and os.path.exists reports False for it (the same answer as a missing file, so the probe learns nothing).
"""
from __future__ import annotations

import os

import pytest

from prophet_agent.config import ROOT
from prophet_agent.sandbox.runner import run_python

PROJECT_FILE = ROOT / "pyproject.toml"


@pytest.fixture(autouse=True)
def _project_file_exists():
    assert PROJECT_FILE.is_file(), "the read-blocked probe target must exist for the test to mean anything"


@pytest.mark.parametrize("call", ["os.stat", "os.lstat", "posix.stat", "posix.lstat"])
def test_stat_of_a_project_file_is_refused(call):
    code = (f"import os, posix\ntry:\n    {call}({str(PROJECT_FILE)!r})\n    print('STATED')\n"
            f"except PermissionError:\n    print('REFUSED')\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "STATED" not in r.stdout and "REFUSED" in r.stdout


def test_exists_on_a_project_file_reports_false_like_a_missing_file():
    r = run_python(f"import os\nprint('EXISTS', os.path.exists({str(PROJECT_FILE)!r}))\n")
    assert r.ok, r.stderr
    assert "EXISTS False" in r.stdout


def test_stat_through_a_directory_descriptor_is_checked(tmp_path):
    # tmp/link points at the project folder. A relative stat through a descriptor on tmp resolves to the project file.
    (tmp_path / "link").symlink_to(ROOT, target_is_directory=True)
    code = (f"import os\nfd = os.open({str(tmp_path)!r}, os.O_RDONLY)\n"
            f"try:\n    os.stat('link/pyproject.toml', dir_fd=fd)\n    print('STATED')\n"
            f"except PermissionError:\n    print('REFUSED')\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "STATED" not in r.stdout and "REFUSED" in r.stdout


def test_stat_outside_the_run_folder_follows_the_read_rule(tmp_path):
    # Outside the run folder, the project and the home folder, a file may be read, so it may be stat'ed too.
    target = tmp_path / "outside.txt"
    target.write_text("abc")
    r = run_python(f"import os\nprint('SIZE', os.stat({str(target)!r}).st_size)\n")
    assert r.ok, r.stderr
    assert "SIZE 3" in r.stdout


def test_stat_and_exists_inside_the_run_folder_work():
    r = run_python("import os\nopen('out/a.txt', 'w').write('abc')\n"
                   "print('SIZE', os.stat('out/a.txt').st_size, os.path.exists('out/a.txt'), os.path.isfile('out/a.txt'))\n")
    assert r.ok, r.stderr
    assert "SIZE 3 True True" in r.stdout


def test_stat_of_the_run_folder_itself_is_not_refused_by_the_input_rule():
    # in/ is readable, so stat of an input file must work (the program may read it, not write it).
    r = run_python("import os\nprint('IN', os.path.exists('in'))\n")
    assert r.ok, r.stderr
    assert "IN True" in r.stdout
