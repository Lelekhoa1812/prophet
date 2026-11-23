"""R6l, audit item 3: a long note must not be cut from report.pdf without saying so.

The PDF holds six lines of note above the chart. Before the fix the seventh line onwards was dropped with no sign on the page.
Now a cut note ends with a line that says so, and report.docx carries the full note in both packs.
"""
from __future__ import annotations

import base64
import re
import zlib
from pathlib import Path

import pytest

from prophet_agent.artifacts import build_forecast_pack, build_panel_pack

CUT_MARKER = b"Note cut to fit the page"
LONG_NOTE = " ".join(f"Sentence {i:02d} explains one synthetic assumption in full." for i in range(1, 31))
SHORT_NOTE = "Synthetic data, one short line."


def _decode_stream(raw: bytes) -> bytes:
    """Undo ASCII85 and Flate, as reportlab writes them, where present."""
    raw = raw.strip()
    if raw.endswith(b"~>"):
        raw = base64.a85decode(raw[:-2], adobe=False)
    try:
        return zlib.decompress(raw)
    except zlib.error:
        return raw


def _pdf_text(path: Path) -> bytes:
    """The decoded streams of the PDF, where the text operators are."""
    data = path.read_bytes()
    parts = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        try:
            parts.append(_decode_stream(m.group(1)))
        except (ValueError, zlib.error):
            continue
    return b"\n".join(parts)


def _out(result, name: str) -> Path:
    return Path(result.workdir) / "out" / name


def _single(note: str):
    history = [10 + (i % 7) * 0.5 for i in range(60)]
    point = [12.0 + 0.1 * i for i in range(12)]
    res = build_forecast_pack(history=history, point=point, q10=[p - 2 for p in point], q90=[p + 2 for p in point],
                              title="Long note (synthetic)", note=note)
    assert res.ok, res.stderr
    return res


def _panel(note: str):
    series = []
    for name in ("alpha", "beta"):
        history = [20 + (i % 5) for i in range(60)]
        point = [22.0 + 0.2 * i for i in range(8)]
        series.append({"name": name, "history": history, "point": point,
                       "q10": [p - 3 for p in point], "q90": [p + 3 for p in point]})
    res = build_panel_pack(series=series, title="Long note panel (synthetic)", note=note)
    assert res.ok, res.stderr
    return res


def test_a_cut_single_pack_note_says_so_on_the_pdf_and_keeps_the_full_note_in_the_docx():
    res = _single(LONG_NOTE)
    assert CUT_MARKER in _pdf_text(_out(res, "report.pdf"))
    from docx import Document
    texts = [p.text for p in Document(_out(res, "report.docx")).paragraphs]
    assert LONG_NOTE in texts


def test_a_cut_panel_pack_note_says_so_on_the_pdf_and_keeps_the_full_note_in_the_docx():
    res = _panel(LONG_NOTE)
    assert CUT_MARKER in _pdf_text(_out(res, "report.pdf"))
    from docx import Document
    texts = [p.text for p in Document(_out(res, "report.docx")).paragraphs]
    assert LONG_NOTE in texts


def test_a_short_note_is_printed_whole_without_the_cut_line():
    res = _single(SHORT_NOTE)
    pdf = _pdf_text(_out(res, "report.pdf"))
    assert b"Synthetic data, one short line." in pdf
    assert CUT_MARKER not in pdf
