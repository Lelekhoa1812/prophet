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
import resource
import signal
import subprocess
import shutil
import sys
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

_BOOT = r"""
import os, runpy, sys, unicodedata
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
           "os.symlink": (1,)}
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
                   "_posixsubprocess.fork_exec"):
        raise PermissionError("starting processes is blocked in the sandbox")
    elif event in ("socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg", "socket.getaddrinfo",
                   "socket.gethostbyname", "socket.gethostbyname_ex", "socket.gethostbyaddr", "socket.getnameinfo"):
        # Name lookups are refused too: a lookup of "<data>.example.com" hands the data to a DNS server (round 6l, item 3).
        raise PermissionError("network access is blocked in the sandbox")
    elif event == "sqlite3.connect":
        # SQLite opens files itself, so the open() check above never sees them: a database outside the run folder could
        # be created or read. A database must be in the run folder; URIs and other locations are refused (round 6l, item 3).
        db = args[0] if args else ""
        if isinstance(db, os.PathLike):
            db = os.fspath(db)
        if isinstance(db, bytes):
            db = os.fsdecode(db)
        if isinstance(db, str) and db not in ("", ":memory:"):
            if db.startswith("file:"):
                raise PermissionError("database URIs are blocked in the sandbox")
            if not _under(os.path.realpath(db), WD):
                raise PermissionError("opening a database outside the run folder is blocked: " + os.path.basename(db))
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
sys.path.insert(0, os.path.join(WD, "lib"))   # prophet_lib: loaders and profiling helpers
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

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def _clip(text: str) -> str:
    """Head and tail. A traceback and a summary table print last, so keeping only the head drops the error the planner needs
    (audit: a 31,581-character stderr lost its KeyError line)."""
    text = redact(text, limit=MAX_STREAM_CHARS * 2)
    if len(text) <= MAX_STREAM_CHARS:
        return text
    half = MAX_STREAM_CHARS // 2
    return text[:half] + f"\n...[{len(text) - 2 * half} characters cut from the middle]...\n" + text[-half:]


def _really_inside(path: Path, root: Path) -> bool:
    """True when the real location of ``path`` (symlinks followed) is ``root`` or lies under it."""
    real = os.path.realpath(path)
    return real == str(root) or real.startswith(str(root) + os.sep)


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


def _limits_preexec(timeout_s: int, mem_mb: int, file_mb: int):
    def apply() -> None:
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (timeout_s + 2, timeout_s + 2))
        fbytes = file_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fbytes, fbytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        if platform.system() == "Linux":
            mbytes = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mbytes, mbytes))
    return apply


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


def run_python(code: str, *, timeout_s: int = 20, mem_mb: int = 1024, file_mb: int = 64,
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

    limits = {"timeout_s": timeout_s, "mem_mb": mem_mb, "file_mb": file_mb,
              "memory_limit_enforced": memory_limit_enforced()}
    cmd = [sys.executable, "-I", "-c", _BOOT, str(wd), str(script), str(ROOT), str(Path.home())]
    proc = subprocess.Popen(cmd, cwd=str(wd), env=_scrubbed_env(home), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, preexec_fn=_limits_preexec(timeout_s, mem_mb, file_mb))
    timed_out = False
    try:
        out_b, err_b = proc.communicate(timeout=timeout_s)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out_b, err_b = proc.communicate()
        returncode = proc.returncode

    outputs: list[OutputFile] = []
    out_root = wd / OUT_DIR
    for p in sorted(out_root.rglob("*")):
        # A symlink in out/ (or out/ itself, once replaced by a symlink) can point at a file outside the run. The program
        # cannot write through it, but the manifest would hash and size that file, so only files that really lie inside
        # the run folder are listed (round 6l, audit item 1).
        if p.is_file() and _really_inside(p, wd):
            outputs.append(OutputFile(path=str(p.relative_to(wd)), sha256=_sha256(p), bytes=p.stat().st_size))
    result = SandboxResult(
        run_id=run_id, returncode=returncode, timed_out=timed_out,
        stdout=_clip(out_b.decode("utf-8", "replace")), stderr=_clip(err_b.decode("utf-8", "replace")),
        outputs=outputs, workdir=str(wd), memory_limit_enforced=memory_limit_enforced(), limits=limits,
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
