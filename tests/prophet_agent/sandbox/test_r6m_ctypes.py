"""Round 6m, item 3 (audit-hook bypass): ctypes calls C functions directly, past every audit check.

Probe (before the fix): ``ctypes.CDLL(None).system(b"...")`` ran a shell command, and ``.socket(2, 1, 0)`` created a
native socket. The audit hook sees the lookup of the function name (``ctypes.dlsym``) and the load of a named library
(``ctypes.dlopen``), but nothing in the hook refused them. The fix: a function may be looked up by name, and a library may
be loaded by name, only through the process's own symbols, which ctypes uses on import. The scientific libraries that the
sandbox imports (numpy, pandas, scipy, statsmodels) load their own native code without ctypes, so they are unaffected.
This is a guardrail: ctypes can still read and write raw memory, so it is not a security boundary.
"""
from __future__ import annotations

import sys

from prophet_agent.sandbox.runner import run_python


def test_ctypes_cannot_run_a_shell_command_through_libc(tmp_path):
    marker = tmp_path / "shell_ran"
    code = (f"import ctypes\nlibc = ctypes.CDLL(None)\n"
            f"try:\n    libc.system(b'touch {marker}')\n    print('CALLED')\n"
            f"except (PermissionError, AttributeError) as e:\n    print('REFUSED', type(e).__name__)\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "CALLED" not in r.stdout and not marker.exists()


def test_ctypes_cannot_open_a_native_socket_through_libc():
    code = ("import ctypes\nlibc = ctypes.CDLL(None)\n"
            "try:\n    print('FD', libc.socket(2, 1, 0))\n"
            "except (PermissionError, AttributeError) as e:\n    print('REFUSED', type(e).__name__)\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "FD" not in r.stdout and "REFUSED" in r.stdout


def test_a_named_native_library_cannot_be_loaded():
    name = "/usr/lib/libz.dylib" if sys.platform == "darwin" else "libz.so.1"
    code = (f"import ctypes\ntry:\n    ctypes.CDLL({name!r})\n    print('LOADED')\n"
            f"except PermissionError as e:\n    print('REFUSED', e)\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "LOADED" not in r.stdout and "REFUSED" in r.stdout


def test_ctypes_values_still_work_inside_the_sandbox():
    r = run_python("import ctypes\nprint(ctypes.c_int(7).value, ctypes.sizeof(ctypes.c_double))\n")
    assert r.ok, r.stderr
    assert r.stdout.strip() == "7 8"


def test_the_scientific_libraries_still_import_and_compute():
    code = ("import numpy, pandas, scipy.stats\n"
            "from statsmodels.tsa.stattools import acf\n"
            "print(round(float(acf(numpy.arange(20.0), nlags=2)[1]), 3))\n")
    r = run_python(code, timeout_s=60)
    assert r.ok, r.stderr
    assert r.stdout.strip() == "0.85"
