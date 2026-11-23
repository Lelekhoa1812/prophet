"""Round 6l, audit item 1: the manifest must not hash files outside the run folder through a symlink.

Probe (before the fix): os.symlink(<secret file outside the run>, 'out/k.txt') was listed in the manifest with the outside
file's sha256 and size; out/ replaced by a symlink to an outside folder listed that folder's files. Writing through such a
symlink was already refused by the audit hook (the hook resolves the real path), and that stays covered here too.
Test files live in pytest's temp folder; nothing touches the real project.
"""
from __future__ import annotations

from prophet_agent.sandbox.runner import run_python


def test_manifest_does_not_hash_a_file_outside_the_run_through_a_symlink(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET")
    code = f"import os\nos.symlink({str(secret)!r}, 'out/k.txt')\nprint('linked')\n"
    r = run_python(code)
    assert r.ok, r.stderr
    assert all(o.path != "out/k.txt" for o in r.outputs)
    assert r.outputs == []


def test_manifest_does_not_list_a_folder_outside_the_run_when_out_is_a_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    code = f"import os\nos.rmdir('out')\nos.symlink({str(outside)!r}, 'out')\nprint('ok')\n"
    r = run_python(code)
    assert r.ok, r.stderr
    assert r.outputs == []


def test_a_real_output_file_is_still_listed_next_to_a_symlink(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET")
    code = (f"import os\nos.symlink({str(secret)!r}, 'out/k.txt')\n"
            "open('out/result.csv', 'w').write('a,b\\n1,2\\n')\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert [o.path for o in r.outputs] == ["out/result.csv"]


def test_writing_through_a_symlink_to_a_file_outside_is_refused(tmp_path):
    victim = tmp_path / "keep.txt"
    victim.write_text("keep")
    code = f"import os\nos.symlink({str(victim)!r}, 'out/l')\nopen('out/l', 'w').write('X')\n"
    r = run_python(code)
    assert not r.ok and "write outside the run folder is blocked" in r.stderr
    assert victim.read_text() == "keep"
