"""Round 6m, item 2 (in/ databases): an input database is read-only, as the rest of in/ is.

Probe (before the fix): ``sqlite3.connect('in/data.sqlite')`` opened the copy read-write, so an insert into it succeeded.
SQLite opens the file itself, so the open() hook never saw the write mode. The copy is the run's own (inputs are copied, so
the source file is never reached), but in/ is documented as read-only to the program, and a WAL database also created
-wal and -shm side files inside in/. The fix: the helper opens an input database as a read-only URI with immutable=1,
which creates no side files; a URI that names in/ without mode=ro is refused by the audit hook.
"""
from __future__ import annotations

import hashlib
import sqlite3

from prophet_agent.sandbox.runner import run_python


def _db(path, sql: list[str]):
    con = sqlite3.connect(path)
    for stmt in sql:
        con.execute(stmt)
    con.commit()
    con.close()
    return path


def test_an_input_database_cannot_be_written(tmp_path):
    src = _db(tmp_path / "data.sqlite", ["create table s(v)", "insert into s values ('ORIGINAL')"])
    before = hashlib.sha256(src.read_bytes()).hexdigest()
    code = ("import sqlite3\n"
            "c = sqlite3.connect('in/data.sqlite')\n"
            "try:\n    c.execute(\"insert into s values ('TAMPERED')\"); c.commit()\n    print('WROTE')\n"
            "except sqlite3.Error as e:\n    print('REFUSED', e)\n"
            "print(c.execute('select v from s').fetchall())\n")
    r = run_python(code, inputs={"data.sqlite": src})
    assert r.ok, r.stderr
    assert "WROTE" not in r.stdout and "REFUSED" in r.stdout
    assert "TAMPERED" not in r.stdout
    assert hashlib.sha256(src.read_bytes()).hexdigest() == before


def test_an_input_database_can_be_read(tmp_path):
    src = _db(tmp_path / "rows.sqlite", ["create table s(v)", "insert into s values (41)", "insert into s values (42)"])
    r = run_python("import sqlite3\nprint(sqlite3.connect('in/rows.sqlite').execute('select sum(v) from s').fetchall())\n",
                   inputs={"rows.sqlite": src})
    assert r.ok, r.stderr
    assert "[(83,)]" in r.stdout


def test_reading_a_wal_input_database_creates_no_side_files(tmp_path):
    # A WAL database makes SQLite create -wal and -shm files next to it when it is opened read-write.
    src = tmp_path / "wal.sqlite"
    con = sqlite3.connect(src)
    con.execute("pragma journal_mode=wal")
    con.execute("create table s(v)")
    con.execute("insert into s values (9)")
    con.commit()
    con.close()
    for side in ("-wal", "-shm"):
        (tmp_path / f"wal.sqlite{side}").unlink(missing_ok=True)
    code = ("import os, sqlite3\n"
            "print(sqlite3.connect('in/wal.sqlite').execute('select v from s').fetchall())\n"
            "print(sorted(n for n in os.listdir('in') if n.startswith('wal.sqlite')))\n")
    r = run_python(code, inputs={"wal.sqlite": src})
    assert r.ok, r.stderr
    assert "[(9,)]" in r.stdout
    assert r.stdout.splitlines()[-1] == "['wal.sqlite']"


def test_a_read_write_uri_naming_the_input_folder_is_refused():
    r = run_python("import sqlite3\nsqlite3.connect('file:in/x.sqlite?mode=rw', uri=True)\nprint('OPENED')\n")
    assert not r.ok and "OPENED" not in r.stdout
    assert "database URIs are blocked" in r.stderr


def test_a_read_only_uri_naming_the_input_folder_is_allowed(tmp_path):
    src = _db(tmp_path / "ro.sqlite", ["create table s(v)", "insert into s values (3)"])
    r = run_python("import sqlite3\nprint(sqlite3.connect('file:in/ro.sqlite?mode=ro', uri=True).execute('select v from s').fetchall())\n",
                   inputs={"ro.sqlite": src})
    assert r.ok, r.stderr
    assert "[(3,)]" in r.stdout


def test_a_plain_path_to_an_input_database_passed_with_uri_true_is_refused(tmp_path):
    src = _db(tmp_path / "p.sqlite", ["create table s(v)"])
    r = run_python("import sqlite3\nc = sqlite3.connect('in/p.sqlite', 5.0, 0, 'DEFERRED', True, sqlite3.Connection, 128, True)\n"
                   "c.execute(\"insert into s values (1)\"); c.commit()\nprint('WROTE')\n",
                   inputs={"p.sqlite": src})
    assert "WROTE" not in r.stdout
