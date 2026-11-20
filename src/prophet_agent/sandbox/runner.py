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
import resource
import signal
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from prophet_agent.config import DATA
from prophet_agent.nim.errors import redact

SANDBOX_ROOT = DATA / "sandbox"
OUT_DIR = "out"
MAX_STREAM_CHARS = 20_000

_BOOT = r"""
import os, runpy, sys
WD = os.path.realpath(sys.argv[1])
TARGET = sys.argv[2]
_WRITE = ("w", "a", "x", "+")
def _inside(p):
    rp = os.path.realpath(p)
    return rp == WD or rp.startswith(WD + os.sep)
def _hook(event, args):
    if event == "open":
        path, mode, flags = args[0], args[1], args[2]
        if isinstance(path, (str, bytes)):
            p = path.decode() if isinstance(path, bytes) else path
            writing = (isinstance(mode, str) and any(c in mode for c in _WRITE)) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
            if writing and not _inside(p):
                raise PermissionError("write outside the run folder is blocked: " + os.path.basename(p))
    elif event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn"):
        raise PermissionError("starting processes is blocked in the sandbox")
    elif event in ("socket.connect", "socket.bind", "socket.sendto"):
        raise PermissionError("network access is blocked in the sandbox")
sys.addaudithook(_hook)
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
    text = redact(text, limit=MAX_STREAM_CHARS * 2)
    return text if len(text) <= MAX_STREAM_CHARS else text[:MAX_STREAM_CHARS] + "\n...[truncated]"


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


def run_python(code: str, *, timeout_s: int = 20, mem_mb: int = 1024, file_mb: int = 64,
               files: dict[str, bytes] | None = None) -> SandboxResult:
    """Run ``code`` in a fresh run folder. ``files`` are written into the folder first (for example input data)."""
    run_id = uuid.uuid4().hex
    wd = (SANDBOX_ROOT / run_id).resolve()
    home = wd / "home"
    (wd / OUT_DIR).mkdir(parents=True)
    home.mkdir()
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
    cmd = [sys.executable, "-I", "-c", _BOOT, str(wd), str(script)]
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
        if p.is_file():
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
