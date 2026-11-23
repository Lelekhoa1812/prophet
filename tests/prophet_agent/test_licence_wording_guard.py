"""Locked decision (owner, 2026-11-22): licence and research-use wording lives only in Markdown documents.

It never appears in UI components, agent prompts, skills, contracts, API responses, tools or code strings, because it
creates noise and confusion for users. Anyone who clones the project reads the README and NOTICE.md themselves. This
guard fails if the wording comes back anywhere outside the documentation. Tests are excluded: they must name the phrases
to check for them.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCANNED = ("src", "web/src", "console", "scripts", ".claude")
SCANNED_FILES = ("pyproject.toml",)
SKIP_DIRS = {"node_modules", "__pycache__", "dist", ".venv", "data", "build",
             "worktrees"}   # .claude/worktrees holds temporary copies of the whole repo for sub-agents, not app content
FORBIDDEN = re.compile(
    r"research[ -]use|research[ -]only|non-?commercial|commercial (?:decision|use|deliverable|service)|paid services"
    r"|licen[cs]e (?:terms|gate|status|wording|notice|copy|note|disclaimer|switch)|disclaimer wording"
    # Forms the first version missed: dataset licence names and terms, reuse rules, and the "licence:" label.
    r"|\bCC[ -]BY\b|creative commons|terms of use|reuse policy|third-party restrictions|\blicen[cs]e:"
    r"|not verified: (?:read|check)|before any external use",
    re.IGNORECASE,
)

# Dataset licence wording now lives in docs/prophet/DATASET_LICENCES.md (owner decision, 2026-11-23). scripts/fetch_datasets.py
# names each dataset by a key and reads the wording from that table, so no exemption is needed. This set stays empty; add a
# line here only as a dated, owner-approved exception.
KNOWN_CODE_LICENCE_LINES: frozenset = frozenset()


def _files():
    for name in SCANNED:
        base = ROOT / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and not (set(path.relative_to(ROOT).parts) & SKIP_DIRS):
                yield path
    for name in SCANNED_FILES:
        yield ROOT / name


def _hits():
    """(relative path, line number, stripped line) for every scanned line that matches FORBIDDEN."""
    for path in _files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN.search(line):
                yield str(path.relative_to(ROOT)), number, line.strip()


def test_licence_and_research_use_wording_appears_only_in_markdown_docs_not_in_the_app():
    hits = list(_hits())
    unknown = [f"{rel}:{number}: {line[:160]}" for rel, number, line in hits if (rel, line) not in KNOWN_CODE_LICENCE_LINES]
    assert not unknown, "licence wording must live only in Markdown docs:\n" + "\n".join(unknown)
    seen = {(rel, line) for rel, _, line in hits}
    stale = sorted(KNOWN_CODE_LICENCE_LINES - seen)
    assert not stale, "these baseline lines no longer match; remove them from KNOWN_CODE_LICENCE_LINES:\n" + "\n".join(
        f"{rel}: {line[:160]}" for rel, line in stale)


def test_the_guard_itself_catches_each_known_form():
    for sample in ("research use only", "research-only outputs", "TimesFM 3 outputs are research use only", "non-commercial weights",
                   "Not for commercial decisions", "licence terms: see", "paid services", "disclaimer wording",
                   "CC BY 4.0 (UCI Machine Learning Repository)", "CC-BY licensed", "FRED terms of use apply",
                   "Some series carry third-party restrictions", "ECB reuse policy: free reuse", "creative commons attribution",
                   "Licence: CC BY", "Not verified: read the source repository", "use it before any external use"):
        assert FORBIDDEN.search(sample), sample
