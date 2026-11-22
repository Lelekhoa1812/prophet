"""Mentions: a skill, a tool or a file the person pointed at in their message, kept as a link and loaded into the run.

Motivation vs Logic
Motivation: typing "/" to pick a playbook or tool, or "@" to point at a file, used to paste plain words into the box.
The agent saw words, not the thing: nothing loaded the playbook, nothing made sure the file was attached, and the sent
message showed no sign that a pick had been made. Claude Code, Cursor and the others keep the message text readable and
carry the pick as structure that the harness resolves (reverse-engineered notes: docs/prophet/mentions_rea.md).
Logic: a pick is stored inside the message as a markdown link with a ``prophet:`` scheme,
``[/Choose model by backtest](prophet:skill/choose-model-by-backtest)``, ``[/Forecast](prophet:tool/forecast)`` or
``[@sales.csv](prophet:file/<id>)``. The label already carries its sigil, so removing the link syntax leaves the words the
person saw. The server reads the links itself and checks each against what exists, so a stale or forged id is just
text. A skill link loads the playbook into the planner's context, a tool link states that the person asked for it, and a
file link both attaches the file and names it as the subject of the question.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

KINDS = ("skill", "tool", "file")
LINK = re.compile(r"\[([^\[\]\n]{1,120})\]\(prophet:(skill|tool|file)/([A-Za-z0-9_.:\-]{1,80})\)")
MAX_SKILLS = 3             # playbooks loaded into one run; more would crowd out the question
SKILL_CHARS = 6000         # per playbook, cut at a line end
MAX_TOOLS = 6
MAX_FILES = 10


@dataclass(frozen=True)
class Mention:
    kind: str
    id: str
    label: str


def parse(text: str) -> list[Mention]:
    """Every mention in order, once per (kind, id)."""
    seen: set[tuple[str, str]] = set()
    out: list[Mention] = []
    for m in LINK.finditer(text or ""):
        key = (m.group(2), m.group(3))
        if key not in seen:
            seen.add(key)
            out.append(Mention(m.group(2), m.group(3), m.group(1)))
    return out


def plain(text: str) -> str:
    """The message as the person read it: each link replaced by its label."""
    return LINK.sub(lambda m: m.group(1), text or "")


def link(kind: str, id_: str, label: str) -> str:
    """The stored form of one pick. Brackets and line breaks cannot sit in a label, so they are dropped."""
    if kind not in KINDS:
        raise ValueError(f"unknown mention kind {kind!r}")
    clean = re.sub(r"[\[\]\n\r]+", " ", label).strip()[:120]
    return f"[{clean}](prophet:{kind}/{id_})"


def humanise(name: str) -> str:
    """'choose-model-by-backtest' -> 'Choose model by backtest'."""
    words = re.sub(r"[-_]+", " ", name).strip()
    return words[:1].upper() + words[1:]


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit]
    return head[: head.rfind("\n")] + "\n...[playbook cut]" if "\n" in head else head


def context_block(text: str, *, skill_body: Callable[[str], tuple[str, str] | None],
                  tool_info: Callable[[str], tuple[str, str] | None],
                  file_name: Callable[[str], str | None]) -> str:
    """What the planner is told about the picks, or "" when there are none that exist.

    Each resolver returns None for an id that does not exist (the pick then stays plain text): ``skill_body`` gives
    (title, body), ``tool_info`` gives (title, description) and ``file_name`` gives the file's name."""
    skills: list[str] = []
    tools: list[str] = []
    files: list[str] = []
    for m in parse(text):
        if m.kind == "skill" and len(skills) < MAX_SKILLS:
            got = skill_body(m.id)
            if got:
                title, body = got
                skills.append(f"The person invoked the playbook \"{title}\" (written {m.label} in their message). "
                              f"Follow it for this task:\n{_cut(body, SKILL_CHARS)}")
        elif m.kind == "tool" and len(tools) < MAX_TOOLS:
            got = tool_info(m.id)
            if got:
                title, desc = got
                tools.append(f"- {title} (tool {m.id}): {desc[:300]}")
        elif m.kind == "file" and len(files) < MAX_FILES:
            name = file_name(m.id)
            if name:
                files.append(f"- {name}")
    parts: list[str] = []
    if skills:
        parts.append("\n\n".join(skills))
    if tools:
        parts.append("The person asked for these tools by name. Use each one for this task, or say in the answer why the "
                     "data made it unsuitable; do not skip it silently:\n" + "\n".join(tools))
    if files:
        parts.append("The person pointed at these files with @ in their message. They are the subject of the question; "
                     "read them first:\n" + "\n".join(files))
    return "\n\n".join(parts)


def file_ids(text: str) -> list[str]:
    return [m.id for m in parse(text) if m.kind == "file"][:MAX_FILES]


def merge_ids(existing: Iterable[str], extra: Iterable[str]) -> list[str]:
    out = list(existing)
    for i in extra:
        if i not in out:
            out.append(i)
    return out
