"""Run analytics Python in a child process with guardrails. This is a guardrail, not isolation.

Motivation vs Logic
Motivation: the agent writes Python to clean data, compute statistics and draw charts. That code must not read the
user's secrets, reach the network, write outside its own folder, or run forever. A child process with a scrubbed
environment, resource limits and a hard timeout covers the common mistakes and the obvious misuse.
Logic, layer by layer:
  1. ``python -I``: no PYTHON* variables, no user site-packages, no script directory on sys.path.
  2. Scrubbed environment: only PATH, a fresh HOME and TMPDIR inside the run folder. No API keys reach the child.
  3. POSIX rlimits: CPU seconds, file size, open files, core dumps off, and address space where the OS enforces it.
     macOS does not enforce RLIMIT_AS, so the memory cap is reported as ``memory_limit_enforced: false`` there.
  4. An audit hook installed before user code: writes outside the run folder, subprocess creation, and network
     sockets raise. Audit hooks are bypassable from C extensions (for example via ctypes), which is why this is a
     guardrail and not a security boundary. The real boundary would be a container or a VM.
  5. Wall-clock timeout; the whole process group is killed.
Outputs: the program writes files into ``out/``. After it exits, every file there is listed with its sha256 and size
in a manifest, so charts and documents can be checked later.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import shutil
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from prophet_agent.config import DATA, ROOT
from prophet_agent.nim.errors import redact

SANDBOX_ROOT = DATA / "sandbox"
OUT_DIR = "out"
IN_DIR = "in"
LIB_SOURCE = Path(__file__).with_name("prophet_lib.py")
# Every prophet_*.py next to this file is importable in the sandbox (prophet_lib, prophet_stats, ...).
LIB_DIR = Path(__file__).parent
MAX_STREAM_CHARS = 20_000
# Raw output kept per stream before redaction: the first and the last RAW_HALF_BYTES bytes. The middle is dropped, so a
# program that prints gigabytes cannot exhaust the parent's memory (round 6l, audit item 5).
RAW_HALF_BYTES = MAX_STREAM_CHARS * 2
_CUT_WINDOW = 512

_BOOT = r"""
import os, platform, resource, runpy, sys, unicodedata, urllib.parse
WD = os.path.realpath(sys.argv[1])
TARGET = sys.argv[2]
ROOT = os.path.realpath(sys.argv[3]) if sys.argv[3] else ""
REAL_HOME = os.path.realpath(sys.argv[4]) if sys.argv[4] else ""
_WRITE = ("w", "a", "x", "+")
WD_IN = os.path.join(WD, "in")
_PREFIXES = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix) if p}
# Mutating calls that change the file system. Each entry lists which argument positions are paths that must stay inside.
_MUTATE = {"os.remove": (0,), "os.rmdir": (0,), "os.mkdir": (0,), "os.chmod": (0,), "os.truncate": (0,),
           "os.utime": (0,), "os.chown": (0,), "os.rename": (0, 1), "os.replace": (0, 1), "os.link": (0, 1),
           "os.symlink": (1,), "os.chflags": (0,), "os.setxattr": (0,), "os.removexattr": (0,)}
def _fold(p):
    # macOS and Windows file systems ignore case, and realpath keeps the case the caller typed. Without folding, /Users/LIAMLE
    # or in/ written as IN/ got past the prefix checks (round 6l, audit item 2: probed, the project file was read and in/ was
    # overwritten). Folding is applied only where the file system ignores case; the file is opened under its real name.
    if sys.platform in ("darwin", "win32"):
        return unicodedata.normalize("NFD", p).casefold()
    return p
def _under(p, base):
    p, base = _fold(p), _fold(base)
    return p == base or p.startswith(base + os.sep)
def _inside(p):
    rp = os.path.realpath(p)
    # The run folder is writable except in/, which holds the person's files and is read-only to the program.
    return _under(rp, WD) and not _under(rp, WD_IN)
def _readable(p):
    # Reads are allowed in the run folder, in the Python installation, and anywhere outside the project and the home
    # folder. The project holds the API keys (.env) and the notebooks; the home folder holds credentials.
    rp = os.path.realpath(p)
    if _under(rp, WD) or any(_under(rp, s) for s in _PREFIXES):
        return True
    return not any(base and _under(rp, base) for base in (ROOT, REAL_HOME))
def _check_database_uri(uri):
    # A database URI is a path plus query options. The path must be in the run folder, and a database in in/ may be
    # opened only read-only (mode=ro). vfs= would choose another file layer, so it is refused.
    parts = urllib.parse.urlsplit(uri)
    query = urllib.parse.parse_qs(parts.query, keep_blank_values=True)
    if parts.scheme != "file" or parts.netloc or "vfs" in query:
        raise PermissionError("database URIs are blocked outside the run folder")
    rp = os.path.realpath(urllib.parse.unquote(parts.path))
    if not _under(rp, WD):
        raise PermissionError("database URIs are blocked outside the run folder")
    if _under(rp, WD_IN) and query.get("mode") != ["ro"]:
        raise PermissionError("database URIs are blocked: an input database opens only with mode=ro")
def _hook(event, args):
    if event == "open":
        path, mode, flags = args[0], args[1], args[2]
        if isinstance(path, (str, bytes)):
            p = path.decode() if isinstance(path, bytes) else path
            writing = (isinstance(mode, str) and any(c in mode for c in _WRITE)) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
            if writing and not _inside(p):
                raise PermissionError("write outside the run folder is blocked: " + os.path.basename(p))
            if not writing and not _readable(p):
                raise PermissionError("reading this location is blocked in the sandbox: " + os.path.basename(p))
    elif event in _MUTATE:
        for i in _MUTATE[event]:
            if i < len(args) and isinstance(args[i], (str, bytes)):
                p = args[i].decode() if isinstance(args[i], bytes) else args[i]
                if not _inside(p):
                    raise PermissionError("changing files outside the run folder is blocked: " + os.path.basename(p))
    elif event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
                   "os.forkpty", "_posixsubprocess.fork_exec"):
        # os.forkpty was missing (round 6m): it forks a child in a new session, outside the process group the timeout kills.
        raise PermissionError("starting processes is blocked in the sandbox")
    elif event in ("os.listdir", "os.scandir"):
        # Listing the project or home folder shows which keys and notes exist, though their contents are blocked (round 6m).
        p = args[0] if args else None
        if isinstance(p, os.PathLike):
            p = os.fspath(p)
        if isinstance(p, bytes):
            p = os.fsdecode(p)
        if isinstance(p, str) and not _readable(p):
            raise PermissionError("reading this location is blocked in the sandbox: " + os.path.basename(p))
    elif event in ("os.kill", "os.killpg"):
        # Signals reach any process of the same user, the parent included (round 6m). A program may signal itself and its
        # own process group only.
        target = args[0] if args else None
        mine = os.getpgrp() if event == "os.killpg" else os.getpid()
        if isinstance(target, int) and target not in (0, mine):
            raise PermissionError("signalling other processes is blocked in the sandbox")
    elif event in ("socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg", "socket.getaddrinfo",
                   "socket.gethostbyname", "socket.gethostbyname_ex", "socket.gethostbyaddr", "socket.getnameinfo"):
        # Name lookups are refused too: a lookup of "<data>.example.com" hands the data to a DNS server (round 6l, item 3).
        raise PermissionError("network access is blocked in the sandbox")
    elif event == "sqlite3.connect":
        # SQLite opens files itself, so the open() check above never sees them: a database outside the run folder could
        # be created or read. A database must be in the run folder; URIs and other locations are refused (round 6l, item 3).
        # Round 6m: the check runs only on connections made by sandbox_connect below. A raw _sqlite3.connect() or a subclass
        # of the connection type would skip the authorizer that denies ATTACH, which is where the escape was (see below).
        if sys._getframe(1).f_code is not _CONNECT_CODE:
            raise PermissionError("open databases with sqlite3.connect, which the sandbox checks")
        db = args[0] if args else ""
        if isinstance(db, os.PathLike):
            db = os.fspath(db)
        if isinstance(db, bytes):
            db = os.fsdecode(db)
        if isinstance(db, str) and db not in ("", ":memory:"):
            if db.startswith("file:"):
                _check_database_uri(db)
            elif not _under(os.path.realpath(db), WD):
                raise PermissionError("opening a database outside the run folder is blocked: " + os.path.basename(db))
            elif _under(os.path.realpath(db), WD_IN):
                # in/ is read-only to the program. A plain path opens read-write, so only the read-only URI that
                # sandbox_connect builds for in/ may reach SQLite there (round 6m).
                raise PermissionError("input databases are read-only: open them with sqlite3.connect(path) only")
    elif event == "sqlite3.enable_load_extension":
        # An SQLite extension is native code: loading one would be the same as calling C directly.
        if len(args) > 1 and args[1]:
            raise PermissionError("loading SQLite extensions is blocked in the sandbox")
    elif event in ("ctypes.dlsym", "ctypes.dlsym/handle"):
        # ctypes calls any C function it can name, and the calls themselves are not audited (round 6m: ctypes.CDLL(None)
        # .system() ran a shell command). Names are looked up only in the process's own symbols, which ctypes needs on import.
        raise PermissionError("looking up native functions by name is blocked in the sandbox")
    elif event == "ctypes.dlopen":
        # None is the process's own symbols (CDLL(None), and ctypes.pythonapi on import). Any named library is refused.
        if args and args[0] is not None:
            raise PermissionError("loading native libraries by name is blocked in the sandbox")
sys.addaudithook(_hook)
# A relative path given with dir_fd is resolved against the folder the descriptor opens, not the working directory. The hook
# above checks the relative name, so os.remove("f.txt", dir_fd=fd) with fd on a folder outside the run deleted that file
# (probe: the file outside was removed). Each such call now resolves its real folder and applies the same check first.
def _fd_dir(fd):
    try:
        if sys.platform == "darwin":
            import fcntl
            return os.fsdecode(fcntl.fcntl(fd, 50, b"\0" * 1024).rstrip(b"\0"))     # F_GETPATH
        return os.readlink("/proc/self/fd/%d" % fd)
    except (OSError, ValueError, TypeError):
        return None
def _dir_fd_guard(fn, positions, read_only=None):
    def guarded(*a, **k):
        ro = bool(read_only and read_only(a, k))
        for key, pos in positions.items():
            fd = k.get(key)
            if fd is None or pos >= len(a):
                continue
            path = a[pos]
            if isinstance(path, bytes):
                path = os.fsdecode(path)
            if not isinstance(path, str) or os.path.isabs(path):
                continue
            base = _fd_dir(fd)
            if base is None:
                raise PermissionError("cannot tell which folder this descriptor opens in the sandbox")
            joined = os.path.join(base, path)
            if not (_readable(joined) if ro else _inside(joined)):
                raise PermissionError("changing files outside the run folder is blocked: " + os.path.basename(path))
        return fn(*a, **k)
    guarded.__name__ = fn.__name__
    return guarded
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
def _open_is_read_only(a, k):
    flags = a[1] if len(a) > 1 else k.get("flags", 0)
    return isinstance(flags, int) and not (flags & _WRITE_FLAGS)
for _name, _positions, _ro in (
        ("open", {"dir_fd": 0}, _open_is_read_only), ("remove", {"dir_fd": 0}, None), ("unlink", {"dir_fd": 0}, None),
        ("rmdir", {"dir_fd": 0}, None), ("mkdir", {"dir_fd": 0}, None), ("chmod", {"dir_fd": 0}, None),
        ("chown", {"dir_fd": 0}, None), ("utime", {"dir_fd": 0}, None), ("rename", {"src_dir_fd": 0, "dst_dir_fd": 1}, None),
        ("replace", {"src_dir_fd": 0, "dst_dir_fd": 1}, None), ("link", {"src_dir_fd": 0, "dst_dir_fd": 1}, None),
        ("symlink", {"dir_fd": 1}, None)):
    if hasattr(os, _name):
        setattr(os, _name, _dir_fd_guard(getattr(os, _name), _positions, _ro))
del _name, _positions, _ro
# A FIFO or device node is created without any audit event (round 6m), so the path itself is checked here. The same function
# is reachable as os.mkfifo and posix.mkfifo, so both names are wrapped.
def _path_guard(fn):
    def guarded(*a, **k):
        if k.get("dir_fd") is None and a:
            p = a[0]
            if isinstance(p, os.PathLike):
                p = os.fspath(p)
            if isinstance(p, bytes):
                p = os.fsdecode(p)
            if isinstance(p, str) and not _inside(p):
                raise PermissionError("changing files outside the run folder is blocked: " + os.path.basename(p))
        return fn(*a, **k)
    guarded.__name__ = fn.__name__
    return guarded
import posix as _posix
for _name in ("mkfifo", "mknod"):
    _guarded = _path_guard(_dir_fd_guard(getattr(os, _name), {"dir_fd": 0}))
    setattr(os, _name, _guarded)
    setattr(_posix, _name, _guarded)
del _name, _guarded
import builtins, io
_OPEN = builtins.open
CARRIED = os.path.join(WD, "in", "carried")
def _open(file, mode="r", *a, **k):
    # A read of out/<name> or in/carried/out/<name> that does not exist falls back to in/carried/<name>: earlier outputs are
    # mounted there, and planners write the path they wrote to (live: about 15 failed cells in one mission). Reads only.
    if isinstance(file, str) and not any(c in mode for c in "wax+") and not os.path.exists(file):
        alt = os.path.join(CARRIED, os.path.basename(file))
        if os.path.exists(alt):
            file = alt
    return _OPEN(file, mode, *a, **k)
builtins.open = _open
io.open = _open
import sqlite3, sqlite3.dbapi2
# Round 6m: SQL statements can open files without any audit event. ATTACH and VACUUM INTO create or read a database
# anywhere (probe: a database outside the run folder was created, and a project database was read). So every connection
# is made here, with an authorizer that denies ATTACH and DETACH (it also refuses VACUUM INTO) and an attachment limit of 0.
# A subclass whose set_authorizer and setlimit refuse, so the program cannot switch those off by calling them.
class _GuardedConnection(sqlite3.Connection):
    def set_authorizer(self, *a, **k):
        raise PermissionError("changing the SQLite authorizer is blocked in the sandbox")
    def setlimit(self, *a, **k):
        raise PermissionError("changing SQLite limits is blocked in the sandbox")
def _deny_attach(action, a1, a2, db, source):
    if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK
_SQL_CONNECT = sqlite3.connect
def sandbox_connect(database, *args, **kwargs):
    # Positional index 4 of the optional arguments is factory (timeout, detect_types, isolation_level, check_same_thread).
    if "factory" not in kwargs and len(args) < 5:
        kwargs["factory"] = _GuardedConnection
    # An input database is opened read-only, immutable so that no -wal or -shm file is created beside it in in/ (round 6m).
    # The uri flag is the 8th optional argument; a call that passes it positionally keeps its own flag and is checked by the hook.
    db = os.fspath(database) if isinstance(database, os.PathLike) else database
    if isinstance(db, bytes):
        db = os.fsdecode(db)
    if isinstance(db, str) and db not in ("", ":memory:") and not db.startswith("file:") \
            and len(args) < 7 and not kwargs.get("uri") and _under(os.path.realpath(db), WD_IN):
        database = "file:" + urllib.parse.quote(os.path.realpath(db)) + "?mode=ro&immutable=1"
        kwargs["uri"] = True
    conn = _SQL_CONNECT(database, *args, **kwargs)
    sqlite3.Connection.set_authorizer(conn, _deny_attach)
    sqlite3.Connection.setlimit(conn, sqlite3.SQLITE_LIMIT_ATTACHED, 0)
    return conn
_CONNECT_CODE = sandbox_connect.__code__
sqlite3.connect = sqlite3.dbapi2.connect = sandbox_connect
sys.path.insert(0, os.path.join(WD, "lib"))   # prophet_lib: loaders and profiling helpers
# Round 6m: the limits are set here, in the child, before the program's code runs (they used to be set by preexec_fn, which
# is unsafe in a multi-threaded parent). Both soft and hard values are set, so the program cannot raise them again.
TIMEOUT_S, MEM_MB, FILE_MB = (int(v) for v in sys.argv[5:8])
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S + 2, TIMEOUT_S + 2))
_FBYTES = FILE_MB * 1024 * 1024
resource.setrlimit(resource.RLIMIT_FSIZE, (_FBYTES, _FBYTES))
resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
if platform.system() == "Linux":        # macOS does not enforce RLIMIT_AS, so the memory cap is reported as not enforced there
    _MBYTES = MEM_MB * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (_MBYTES, _MBYTES))
sys.argv = [TARGET]
runpy.run_path(TARGET, run_name="__main__")
"""


@dataclass
class OutputFile:
    path: str
    sha256: str
    bytes: int


@dataclass
class SandboxResult:
    run_id: str
    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str
    outputs: list[OutputFile] = field(default_factory=list)
    workdir: str = ""
    memory_limit_enforced: bool = False
    limits: dict = field(default_factory=dict)
    output_truncated: bool = False     # true when output between the kept head and tail was dropped or clipped
    over_disk_limit: bool = False      # true when the run wrote more than its disk quota and was stopped

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.over_disk_limit

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def _clip(text: str) -> str:
    """Head and tail. A traceback and a summary table print last, so keeping only the head drops the error the planner needs
    (audit: a 31,581-character stderr lost its KeyError line). The whole text is redacted first: redact() cuts its result at
    ``limit``, so a limit smaller than the text would drop the real end of long output (round 6l, audit item 4)."""
    text = redact(text, limit=len(text) + 1)
    if len(text) <= MAX_STREAM_CHARS:
        return text
    half = MAX_STREAM_CHARS // 2
    return text[:half] + f"\n...[{len(text) - 2 * half} characters cut from the middle]...\n" + text[-half:]


def _really_inside(path: Path, root: Path) -> bool:
    """True when the real location of ``path`` (symlinks followed) is ``root`` or lies under it."""
    real = os.path.realpath(path)
    return real == str(root) or real.startswith(str(root) + os.sep)


def _cut_head(text: str) -> str:
    """Drop the end of a kept head back to a whitespace, so a key that the middle cut through is not shown in part. API keys
    contain no whitespace, so whatever follows the last space in the window belongs to the dropped middle."""
    window = text[-_CUT_WINDOW:]
    i = max(window.rfind(c) for c in " \t\r\n")
    base = len(text) - len(window)
    return text[: base + i + 1] if i >= 0 else text[:base]


def _cut_tail(text: str) -> str:
    """Drop the start of a kept tail up to its first whitespace: the rest of a key that began in the dropped middle."""
    window = text[:_CUT_WINDOW]
    found = [i for i in (window.find(c) for c in " \t\r\n") if i >= 0]
    return text[min(found) + 1:] if found else text[len(window):]


class _Capture:
    """Keeps the first and the last RAW_HALF_BYTES of one output stream and counts the bytes dropped between them."""

    def __init__(self) -> None:
        self.head = bytearray()
        self.tail = bytearray()
        self.dropped = 0

    def feed(self, chunk: bytes) -> None:
        room = RAW_HALF_BYTES - len(self.head)
        if room > 0:
            self.head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self.tail += chunk
            extra = len(self.tail) - RAW_HALF_BYTES
            if extra > 0:
                self.dropped += extra
                del self.tail[:extra]

    def text(self) -> str:
        head = bytes(self.head).decode("utf-8", "replace")
        tail = bytes(self.tail).decode("utf-8", "replace")
        if not self.dropped:
            return head + tail
        return (_cut_head(head) + f"\n...[{self.dropped} bytes dropped from the middle of the output]...\n"
                + _cut_tail(tail))


def _pump(stream, capture: _Capture) -> None:
    """Read a child's pipe to its end, so the child never blocks on a full pipe, while keeping only what the capture holds."""
    fd = stream.fileno()
    while True:
        try:
            chunk = os.read(fd, 1 << 16)
        except OSError:
            break
        if not chunk:
            break
        capture.feed(chunk)


def _writable_bytes(wd: Path) -> int:
    """Bytes the program can write in the run folder: everything except in/, which holds the copied inputs. Links are not
    followed, so a link to a large file elsewhere costs nothing."""
    total = 0
    stack = [wd]
    while stack:
        folder = stack.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if folder == wd and entry.name == IN_DIR:
                            continue
                        stack.append(Path(entry.path))
                    else:
                        total += entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
    return total


def _watch_disk(proc: subprocess.Popen, wd: Path, limit_bytes: int, stop: threading.Event, flagged: list) -> None:
    """Samples the run folder until the process ends. Past the quota, the whole process group is killed (round 6m). The
    overshoot is at most what a sample interval can write."""
    while not stop.wait(0.05):
        if _writable_bytes(wd) > limit_bytes:
            flagged.append(True)
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _scrubbed_env(home: Path) -> dict[str, str]:
    # Nothing from the parent environment is inherited except PATH. API keys never reach the child.
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(home),
        "TMPDIR": str(home),
        "LANG": "C.UTF-8",
        "MPLCONFIGDIR": str(home / ".mpl"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def memory_limit_enforced() -> bool:
    return platform.system() == "Linux"


SHORT_MODULES = {"ps": "prophet_stats", "sc": "prophet_scan"}


def _result_from_manifest(manifest: Path) -> SandboxResult:
    """The stored result of a finished run, read back from its manifest (round 4, A5-10)."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    fields = {f.name for f in SandboxResult.__dataclass_fields__.values()}
    outputs = [OutputFile(**o) for o in data.get("outputs", [])]
    kwargs = {k: v for k, v in data.items() if k in fields and k != "outputs"}
    return SandboxResult(outputs=outputs, **kwargs)


DISK_MB = 512   # the most a run may write in its folder, in total (round 6m): RLIMIT_FSIZE caps each file, not the sum


def run_python(code: str, *, timeout_s: int = 20, mem_mb: int = 1024, file_mb: int = 64, disk_mb: int = DISK_MB,
               files: dict[str, bytes] | None = None, inputs: dict[str, Path] | None = None,
               run_id: str | None = None) -> SandboxResult:
    """Run ``code`` in a fresh run folder. ``files`` are written into the folder first (for example input data).

    ``inputs`` maps a file name to a path on disk; each is copied to ``in/<name>``, which the program can read but not
    change. ``prophet_lib`` and the other ``prophet_*`` modules are importable from ``lib/``.

    ``run_id`` (round 4, A5-10) is given by an idempotent caller: its key seeds the id, so a re-run finds the folder. A folder
    with a manifest is a finished run and its stored result is returned without running the code again. A folder without a
    manifest is an interrupted run and is cleared before the code runs once under the same id.
    """
    if run_id is not None:
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("run_id must be 32 lowercase hex characters")
        folder = (SANDBOX_ROOT / run_id).resolve()
        if (folder / "manifest.json").is_file():
            return _result_from_manifest(folder / "manifest.json")
        if folder.exists():
            shutil.rmtree(folder)
    else:
        run_id = uuid.uuid4().hex
    wd = (SANDBOX_ROOT / run_id).resolve()
    home = wd / "home"
    (wd / OUT_DIR).mkdir(parents=True)
    (wd / IN_DIR).mkdir()
    (wd / "lib").mkdir()
    home.mkdir()
    for module in sorted(LIB_DIR.glob("prophet_*.py")):
        shutil.copy2(module, wd / "lib" / module.name)
    # The documented short names as modules too, so `from ps import Ledger` works like `import prophet_stats as ps`
    # (live: two cells failed on it). `pl` is left out: it would shadow polars in code that means polars.
    for short, full in SHORT_MODULES.items():
        (wd / "lib" / f"{short}.py").write_text(f"import {full} as _m\n"
                                                "globals().update({k: v for k, v in vars(_m).items() if not k.startswith('__')})\n")
    for name, src in (inputs or {}).items():
        parts = name.split("/")
        if name.startswith(".") or len(parts) > 2 or (len(parts) == 2 and parts[0] not in ("derived", "carried")) or not parts[-1]:
            raise ValueError("input names are plain file names, or derived/<name> or carried/<name> for earlier results")
        (wd / IN_DIR / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, wd / IN_DIR / name)
    for name, data in (files or {}).items():
        target = (wd / name).resolve()
        if not str(target).startswith(str(wd) + os.sep):
            raise ValueError("input file names must stay inside the run folder")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    script = wd / "program.py"
    script.write_text(code, encoding="utf-8")

    limits = {"timeout_s": timeout_s, "mem_mb": mem_mb, "file_mb": file_mb, "disk_mb": disk_mb,
              "memory_limit_enforced": memory_limit_enforced()}
    cmd = [sys.executable, "-I", "-c", _BOOT, str(wd), str(script), str(ROOT), str(Path.home()),
           str(timeout_s), str(mem_mb), str(file_mb)]
    # start_new_session makes the child a session and process-group leader, so the timeout can kill its whole group. The
    # limits are applied by _BOOT in the child (no preexec_fn: it is unsafe in a multi-threaded parent).
    proc = subprocess.Popen(cmd, cwd=str(wd), env=_scrubbed_env(home), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, start_new_session=True)
    out_cap, err_cap = _Capture(), _Capture()
    readers = [threading.Thread(target=_pump, args=(proc.stdout, out_cap), daemon=True),
               threading.Thread(target=_pump, args=(proc.stderr, err_cap), daemon=True)]
    for reader in readers:
        reader.start()
    disk_limit = disk_mb * 1024 * 1024
    over_disk: list = []
    stop_watch = threading.Event()
    watcher = threading.Thread(target=_watch_disk, args=(proc, wd, disk_limit, stop_watch, over_disk), daemon=True)
    watcher.start()
    timed_out = False
    try:
        returncode = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        returncode = proc.wait()
    stop_watch.set()
    watcher.join(timeout=10)
    if not over_disk and _writable_bytes(wd) > disk_limit:   # a run that ended between two samples is still checked
        over_disk.append(True)
    for reader in readers:
        reader.join(timeout=10)

    outputs: list[OutputFile] = []
    out_root = wd / OUT_DIR
    for p in sorted(out_root.rglob("*")):
        # A symlink in out/ (or out/ itself, once replaced by a symlink) can point at a file outside the run. The program
        # cannot write through it, but the manifest would hash and size that file, so only files that really lie inside
        # the run folder are listed (round 6l, audit item 1).
        if p.is_file() and _really_inside(p, wd):
            outputs.append(OutputFile(path=str(p.relative_to(wd)), sha256=_sha256(p), bytes=p.stat().st_size))
    out_text, err_text = out_cap.text(), err_cap.text()
    result = SandboxResult(
        run_id=run_id, returncode=returncode, timed_out=timed_out,
        stdout=_clip(out_text), stderr=_clip(err_text),
        outputs=outputs, workdir=str(wd), memory_limit_enforced=memory_limit_enforced(), limits=limits,
        output_truncated=bool(out_cap.dropped or err_cap.dropped or len(out_text) > MAX_STREAM_CHARS
                              or len(err_text) > MAX_STREAM_CHARS),
        over_disk_limit=bool(over_disk),
    )
    (wd / "manifest.json").write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return result


def verify_manifest(manifest_path: str | Path) -> list[str]:
    """Re-hash every listed output. Returns problems; an empty list means the outputs are unchanged."""
    manifest_path = Path(manifest_path)
    wd = manifest_path.parent
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    for item in data.get("outputs", []):
        p = (wd / item["path"]).resolve()
        if not p.is_file():
            problems.append(f"missing: {item['path']}")
        elif _sha256(p) != item["sha256"]:
            problems.append(f"changed: {item['path']}")
    return problems
