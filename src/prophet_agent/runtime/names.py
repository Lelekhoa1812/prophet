"""Short names (1 to 4 words) for sub-agents, research lanes and code runs, so a person can tell them apart in the console.

Motivation vs Logic
Motivation: the owner asked (2026-11-27) for each sub-agent and each piece of generated code to carry a name of one to four
words saying its purpose; the console listed them as "Sub-agent 3" and by their first comment line, which read alike.
Logic: the planner is asked for a ``name``. A name it gives is cleaned (letters, digits, spaces, hyphens, ampersands; at most
four words and 40 characters). With no usable name, one is derived: for a brief, its first clause without filler words; for
code, its first comment, else the main call it makes. A derived name is never empty.
"""
from __future__ import annotations

import re

MAX_WORDS = 4
MAX_CHARS = 40
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9&'+./-]*")
FILLER = frozenset("""a an the and or of to in on for with from by at as is are be was were this that these those it its
what which who whom how when where why now then so please also into about over under between than any each every all
there their them they we you your our i me my do does did done can could should would will shall may might must""".split())
LEAD_VERBS = frozenset("research find look search check get fetch identify establish determine gather collect list compile "
                       "work out figure write produce make give tell show".split())


def clean(name: object) -> str:
    """A given name cut to four words and 40 characters, or "" when nothing usable is left."""
    words = _WORD.findall(str(name or ""))[:MAX_WORDS]
    while len(words) > 1 and words[-1].lower() in FILLER:       # "RBA policy path and" ends on its last real word
        words = words[:-1]
    text = " ".join(words)
    while len(text) > MAX_CHARS and len(words) > 1:
        words = words[:-1]
        text = " ".join(words)
    return text[:MAX_CHARS].strip(" -./")


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def for_task(task: object, given: object = None, fallback: str = "Sub-agent") -> str:
    """The name of a sub-agent or lane: the planner's, cleaned, else up to four content words of its brief's first clause."""
    if (name := clean(given)):
        return _cap(name)
    first = re.split(r"[.;:\n(]|,\s|\s[-–—]\s", str(task or "").strip(), maxsplit=1)[0]
    words = [w for w in _WORD.findall(first) if w.lower() not in FILLER]
    if len(words) > 1 and words[0].lower() in LEAD_VERBS:
        words = words[1:]
    return _cap(clean(" ".join(words))) or fallback


_COMMENT = re.compile(r"^\s*#+\s*(.+?)\s*$", re.M)
# The main call of a cell when it has no comment, in the order they usually matter.
_CALLS = (("save_series", "Save series"), ("save_panel", "Save panel"), ("backtest", "Backtest models"),
          ("calibrate", "Calibrate intervals"), ("disaggregate", "Disaggregate to monthly"), ("scan", "Scan the data"),
          ("seasonal", "Check seasonality"), ("plot", "Plot the series"), ("savefig", "Save a chart"),
          ("to_csv", "Write a table"), ("read_table", "Load the data"), ("read_csv", "Load the data"),
          ("read_excel", "Load the data"), ("describe", "Describe the data"))


def for_code(code: object, given: object = None) -> str:
    """The name of a code run: the planner's, cleaned, else its first comment, else its main call, else "Python analysis"."""
    if (name := clean(given)):
        return _cap(name)
    text = str(code or "")
    for m in _COMMENT.finditer(text):
        line = m.group(1)
        if line.startswith("!") or line.lower().startswith(("-*-", "type:", "noqa", "pylint")):
            continue
        content = [w for w in _WORD.findall(line) if w.lower() not in FILLER]
        whole = " ".join(_WORD.findall(line)[:MAX_WORDS])[:MAX_CHARS].strip()      # "# plot it" stays "Plot it"
        if (name := clean(" ".join(content)) if len(content) > 1 else whole):
            return _cap(name)
    low = text.lower()
    for needle, label in _CALLS:
        if needle in low:
            return label
    return "Python analysis"
