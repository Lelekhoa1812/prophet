"""Round 6m, item 2 (escape): SQL-level ATTACH and VACUUM INTO bypass the database check of sqlite3.connect.

Probe (before the fix): a connection to an in-memory database ran ``ATTACH '<path outside>' AS x`` and created and wrote
a database outside the run folder; ``VACUUM INTO '<path outside>'`` wrote a file outside; ``ATTACH`` of a database under
the project data folder read its tables. SQLite opens these files itself, so neither the open() hook nor the
sqlite3.connect check sees them. Loading SQLite extensions was also not refused, and a connection opened without the
sandbox helper (``_sqlite3.connect``) skipped every check. The fix: each connection is made through the helper, which
denies ATTACH, DETACH and VACUUM INTO and refuses the extension loader; the audit hook refuses connections that do not
come from the helper.
"""
from __future__ import annotations

import sqlite3
import uuid

from prophet_agent.config import DATA
from prophet_agent.sandbox.runner import run_python


def _project_database(marker: str):
    path = DATA / f"r6m_{uuid.uuid4().hex}.sqlite"
    con = sqlite3.connect(path)
    con.execute("create table s(v)")
    con.execute("insert into s values (?)", (marker,))
    con.commit()
    con.close()
    return path


def test_attach_cannot_create_a_database_outside_the_run_folder(tmp_path):
    target = tmp_path / "attached.db"
    code = (f"import sqlite3\nc = sqlite3.connect(':memory:')\n"
            f"try:\n    c.execute(\"ATTACH '{target}' AS x\")\n    c.execute('create table x.t(a)')\n    c.commit()\n"
            f"    print('ATTACHED')\nexcept sqlite3.Error as e:\n    print('REFUSED', e)\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "ATTACHED" not in r.stdout and not target.exists()


def test_vacuum_into_cannot_write_a_file_outside_the_run_folder(tmp_path):
    target = tmp_path / "vacuumed.db"
    code = (f"import sqlite3\nc = sqlite3.connect(':memory:')\nc.execute('create table t(a)')\n"
            f"try:\n    c.execute(\"VACUUM INTO '{target}'\")\n    print('VACUUMED')\n"
            f"except sqlite3.Error as e:\n    print('REFUSED', e)\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "VACUUMED" not in r.stdout and not target.exists()


def test_attach_cannot_read_a_database_outside_the_run_folder():
    db = _project_database("PROJECTDB-MARKER")
    try:
        code = (f"import sqlite3\nc = sqlite3.connect(':memory:')\n"
                f"try:\n    c.execute(\"ATTACH '{db}' AS x\")\n"
                f"    print(c.execute('select v from x.s').fetchall())\n"
                f"except sqlite3.Error as e:\n    print('REFUSED', e)\n")
        r = run_python(code)
        assert r.ok, r.stderr
        assert "PROJECTDB-MARKER" not in r.stdout
    finally:
        db.unlink(missing_ok=True)


def test_a_connection_opened_without_the_helper_is_refused():
    # _sqlite3 is the C module underneath sqlite3; its connect() is not the helper, so it cannot attach or write.
    code = ("import _sqlite3\n"
            "c = _sqlite3.connect(':memory:')\n"
            "print('RAW_OPEN')\n"
            "c.execute(\"ATTACH '/private/tmp/r6m_raw_attach.db' AS x\")\n")
    r = run_python(code)
    assert "RAW_OPEN" not in r.stdout
    assert not r.ok and "open databases with sqlite3.connect" in r.stderr


def test_a_subclass_of_the_connection_type_is_refused():
    code = ("import sqlite3\n"
            "class Mine(sqlite3.Connection):\n    pass\n"
            "c = Mine(':memory:')\n"
            "print('SUBCLASS_OPEN')\n")
    r = run_python(code)
    assert "SUBCLASS_OPEN" not in r.stdout and not r.ok


def test_loading_sqlite_extensions_is_refused():
    code = ("import sqlite3\n"
            "c = sqlite3.connect(':memory:')\n"
            "c.enable_load_extension(True)\n"
            "print('ENABLED')\n")
    r = run_python(code)
    assert "ENABLED" not in r.stdout and not r.ok


def test_the_program_cannot_switch_the_authorizer_off():
    code = ("import sqlite3\n"
            "c = sqlite3.connect(':memory:')\n"
            "try:\n    c.set_authorizer(None)\n    print('SWITCHED_OFF')\n"
            "except PermissionError:\n    print('KEPT')\n"
            "try:\n    c.execute(\"ATTACH ':memory:' AS x\")\n    print('ATTACHED')\n"
            "except sqlite3.Error:\n    print('STILL_DENIED')\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "SWITCHED_OFF" not in r.stdout and "ATTACHED" not in r.stdout
    assert "KEPT" in r.stdout and "STILL_DENIED" in r.stdout


def test_a_database_inside_the_run_folder_still_works_with_the_helper():
    code = ("import sqlite3\n"
            "c = sqlite3.connect('out/ok.db'); c.execute('create table t(x)'); c.execute('insert into t values (5)')\n"
            "c.commit(); c.close()\n"
            "print(sqlite3.connect('out/ok.db').execute('select x from t').fetchall())\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "[(5,)]" in r.stdout
