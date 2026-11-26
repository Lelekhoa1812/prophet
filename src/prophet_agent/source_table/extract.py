"""Candidates from HTML a fetch already holds. This module does not contact a host.

Motivation vs Logic
Motivation: when the spreadsheet download returns a page, the grid on that page is the next candidate.
Logic: tables come from the HTML parser. File addresses come only from links the page contained.
"""
from __future__ import annotations

from prophet_agent.research.fetch import data_file_links
from prophet_agent.research.tables import html_tables


def extract_html(html: str, *, final_url: str = "") -> dict:
    """Table candidates and file links. ``final_url`` is stored on each candidate. Nothing is downloaded."""
    parsed = html_tables(html or "", base_url=final_url or "")
    links = data_file_links(html or "", final_url or "") if html else []
    return {"candidates": parsed["candidates"], "skipped_layout": parsed["skipped_layout"], "file_links": links}
