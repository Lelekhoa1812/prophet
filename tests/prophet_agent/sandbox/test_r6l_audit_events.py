"""Round 6l, audit item 3: SQLite and name lookups are guarded like files and sockets.

Probe (before the fix): sqlite3.connect() created a database file outside the run folder and read a database outside it;
socket.getaddrinfo() and socket.gethostbyname() resolved names. The fix checks the database path against the run folder
and blocks the lookups. Legitimate use inside the run folder must keep working.
"""
from __future__ import annotations

import sqlite3

from prophet_agent.sandbox.runner import run_python


def test_a_database_created_outside_the_run_folder_is_refused(tmp_path):
    db = tmp_path / "new.db"
    code = f"import sqlite3\nc = sqlite3.connect({str(db)!r})\nc.execute('create table t(x)')\nc.commit()\n"
    r = run_python(code)
    assert not r.ok
    assert "opening a database outside the run folder is blocked" in r.stderr
    assert not db.exists()


def test_a_database_outside_the_run_folder_cannot_be_read(tmp_path):
    db = tmp_path / "data.db"
    con = sqlite3.connect(db)
    con.execute("create table s(v)")
    con.execute("insert into s values ('DBSECRET')")
    con.commit()
    con.close()
    r = run_python(f"import sqlite3\nprint(sqlite3.connect({str(db)!r}).execute('select v from s').fetchall())")
    assert not r.ok and "DBSECRET" not in r.stdout


def test_a_database_inside_the_run_folder_works_and_memory_databases_too():
    code = ("import sqlite3\n"
            "c = sqlite3.connect('out/work.db'); c.execute('create table t(x)'); c.execute('insert into t values (7)')\n"
            "c.commit(); c.close()\n"
            "print(sqlite3.connect('out/work.db').execute('select x from t').fetchall())\n"
            "print(sqlite3.connect(':memory:').execute('select 1').fetchall())\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "[(7,)]" in r.stdout and "[(1,)]" in r.stdout


def test_database_uris_are_refused():
    r = run_python("import sqlite3\nsqlite3.connect('file:/etc/hosts?mode=ro', uri=True)\n")
    assert not r.ok and "database URIs are blocked" in r.stderr


def test_name_lookups_are_refused():
    for call in ("socket.getaddrinfo('example.com', 80)", "socket.gethostbyname('example.com')"):
        r = run_python(f"import socket\n{call}\n", timeout_s=10)
        assert not r.ok, call
        assert "network access is blocked" in r.stderr, call


def test_sending_a_datagram_to_an_address_is_refused():
    code = ("import socket\ns = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
            "s.sendmsg([b'x'], [], 0, ('93.184.216.34', 9))\n")
    r = run_python(code, timeout_s=10)
    assert not r.ok and "network access is blocked" in r.stderr
