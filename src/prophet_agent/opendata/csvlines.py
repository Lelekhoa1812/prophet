"""Which lines of a delimited text are comments, and which line starting with '#' is its header. One rule for every reader.

Motivation vs Logic
Motivation: a league table writes its position column as "#", so its header line is "#,Club,P,W,D,L,GD,Pts". Dropping every
line that starts with '#' removed that header, and the first club's row became the header. Some sources do put real
comments before a table (NOAA, and comment blocks in other government files), and those must still be dropped.
Logic: the table width is the most common field count of the lines that do not start with '#'. A '#' line is the header
only when it has that width (and at least two fields), it comes before any table row, and the next data line is not a
row of text names (a second header). Every other '#' line is a comment. A line that does not start with '#' is never a
comment. The whitespace reader in store.py keeps its own rule, because the NOAA header is itself a '#' comment line.
"""
from __future__ import annotations

import csv
from collections import Counter


def _fields(line: str) -> list[str]:
    try:
        return next(csv.reader([line]))
    except (csv.Error, StopIteration):
        return [line]


def _numeric(cell: str) -> bool:
    try:
        float(cell.strip().replace(",", "").replace("%", ""))
    except ValueError:
        return False
    return True


def _is_text_row(fields: list[str]) -> bool:
    cells = [cell for cell in fields if cell.strip()]
    return bool(cells) and not any(_numeric(cell) for cell in cells)


def comment_flags(lines: list[str]) -> list[bool]:
    """True for each line that is a comment. Blank lines are never flagged. A '#' header line is not flagged."""
    flags = [line.startswith("#") for line in lines]
    widths = [len(_fields(line)) for line, flag in zip(lines, flags) if line.strip() and not flag]
    data_widths = Counter(width for width in widths if width >= 2)     # one-cell titles do not vote on the width
    if not data_widths:
        return flags                                   # no table rows at all: every '#' line is a comment
    width = data_widths.most_common(1)[0][0]
    if width < 2:
        return flags
    following = [index for index, line in enumerate(lines) if line.strip() and not flags[index]]
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        fields = _fields(line)
        if not flags[index]:
            if len(fields) >= 2:
                return flags                           # a table row comes before any '#' line: no header among them
            continue                                   # a one-cell title or a comment, as before
        if len(fields) != width:
            continue                                   # a comment of another width
        after = [i for i in following if i > index]
        if after and _is_text_row(_fields(lines[after[0]])):
            return flags                               # the next row is a second header, so this line is a comment
        flags[index] = False
        return flags
    return flags


def table_lines(text: str) -> list[str]:
    """The lines of a delimited text with comments removed; the header line, if it starts with '#', is kept."""
    lines = text.splitlines()
    flags = comment_flags(lines)
    return [line for line, flag in zip(lines, flags) if not flag]
