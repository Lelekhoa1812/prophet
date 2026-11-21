"""Sandbox guardrail tests (GOAL G10). These check the guardrails, which are not a security boundary (see runner.py)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from prophet_agent.sandbox.runner import memory_limit_enforced, run_python, verify_manifest


def test_runs_code_and_captures_stdout():
    r = run_python("print(2 + 3)")
    assert r.ok and r.stdout.strip() == "5"


def test_timeout_kills_the_process_group():
    r = run_python("while True:\n    pass", timeout_s=1)
    assert r.timed_out and not r.ok


def test_secrets_in_the_parent_environment_do_not_reach_the_child(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY_1", "nvapi-SHOULDNOTLEAK0000000000")
    r = run_python("import os\nprint(sorted(os.environ))")
    assert "NVIDIA_NIM_API_KEY_1" not in r.stdout
    assert "nvapi-SHOULDNOTLEAK" not in r.stdout


def test_write_outside_run_folder_is_refused(tmp_path):
    target = tmp_path / "escape.txt"
    r = run_python(f"open({str(target)!r}, 'w').write('x')")
    assert not r.ok
    assert "write outside the run folder is blocked" in r.stderr
    assert not target.exists()


def test_write_inside_out_folder_is_allowed_and_listed_with_sha256():
    code = "open('out/result.csv', 'w').write('a,b\\n1,2\\n')"
    r = run_python(code)
    assert r.ok, r.stderr
    assert [o.path for o in r.outputs] == ["out/result.csv"]
    assert len(r.outputs[0].sha256) == 64


def test_manifest_verifies_and_detects_tampering():
    r = run_python("open('out/a.txt', 'w').write('hello')")
    manifest = Path(r.workdir) / "manifest.json"
    assert verify_manifest(manifest) == []
    (Path(r.workdir) / "out" / "a.txt").write_text("changed")
    assert verify_manifest(manifest) == ["changed: out/a.txt"]


def test_subprocess_creation_is_blocked():
    r = run_python("import subprocess\nsubprocess.run(['/bin/echo', 'hi'])")
    assert not r.ok
    assert "starting processes is blocked" in r.stderr


def test_network_is_blocked():
    code = "import socket\ns = socket.socket()\ns.connect(('93.184.216.34', 80))"
    r = run_python(code, timeout_s=5)
    assert not r.ok
    assert "network access is blocked" in r.stderr


def test_input_files_must_stay_inside_the_folder():
    with pytest.raises(ValueError):
        run_python("print(1)", files={"../escape.txt": b"x"})


def test_input_file_is_readable_inside_the_run():
    r = run_python("print(open('data/in.csv').read().strip())", files={"data/in.csv": b"x,y\n1,2"})
    assert r.ok and r.stdout.strip() == "x,y\n1,2"


def test_memory_cap_is_enforced_where_the_os_supports_it():
    if not memory_limit_enforced():
        pytest.skip("RLIMIT_AS is not enforced on this OS; the result reports memory_limit_enforced=false")
    r = run_python("x = bytearray(900 * 1024 * 1024)\nprint(len(x))", mem_mb=256, timeout_s=10)
    assert not r.ok


def test_result_reports_limits_honestly():
    r = run_python("print('ok')")
    assert r.limits["memory_limit_enforced"] == memory_limit_enforced()
    assert r.to_dict()["ok"] is True


def test_deleting_a_host_file_is_refused(tmp_path):
    victim = tmp_path / "keep.txt"
    victim.write_text("keep")
    r = run_python(f"import os\nos.remove({str(victim)!r})")
    assert not r.ok
    assert "changing files outside the run folder is blocked" in r.stderr
    assert victim.read_text() == "keep"


def test_renaming_or_replacing_a_host_file_is_refused(tmp_path):
    victim = tmp_path / "keep.txt"
    victim.write_text("keep")
    r = run_python(f"import os\nos.replace('out', {str(victim)!r})")
    assert not r.ok and victim.read_text() == "keep"
    r = run_python(f"import os\nos.rename({str(victim)!r}, 'out/moved.txt')")
    assert not r.ok and victim.exists()


def test_reading_the_project_secrets_file_is_refused():
    secret = Path(__file__).resolve().parents[3] / "CLAUDE.md"   # a project file outside the run folder
    r = run_python(f"print(open({str(secret)!r}).read()[:5])")
    assert not r.ok
    assert "reading this location is blocked" in r.stderr


def test_reading_outside_the_project_and_home_is_still_allowed():
    r = run_python("print(open('/etc/hosts').read()[:1] != '')")
    assert r.ok and r.stdout.strip() == "True"


def test_forking_is_refused():
    r = run_python("import os\nos.fork()", timeout_s=5)
    assert not r.ok
    assert "starting processes is blocked" in r.stderr
