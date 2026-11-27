"""The Internet Archive's copy of a page whose live host answered with a bot check or a CAPTCHA.

Motivation vs Logic
Motivation: in the VN Unemployment notebook (2026-11-27) the national statistics office answered with a CAPTCHA, and the
mission lost its primary source. A challenge is never solved or worked around here. A public archive of the same page is a
different, legitimate source: the Wayback Machine keeps dated copies of many official pages.
Logic: the availability API names the closest snapshot of an address. The copy is read through its raw form (the "id_"
modifier, the page as archived without the archive's toolbar). The result says it is an archived copy and gives the
snapshot's date, because figures on it are as of that date, not today.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any
from urllib.parse import quote

from prophet_agent.opendata import fetch

AVAILABLE = "https://archive.org/wayback/available?url="
TIMESTAMP = re.compile(r"\d{14}")


def snapshot(url: str) -> dict[str, Any] | None:
    """The closest archived copy of url: its raw address, timestamp and date. None when there is none or the lookup fails."""
    try:
        fetched = fetch.download(AVAILABLE + quote(url, safe=":/"), max_bytes=200_000, timeout_s=20)
        data = json.loads(fetched["bytes"].decode("utf-8"))
    except (fetch.DownloadError, ValueError, OSError):
        return None
    closest = ((data or {}).get("archived_snapshots") or {}).get("closest") if isinstance(data, dict) else None
    if not isinstance(closest, dict) or not closest.get("available") or str(closest.get("status", "200")) != "200":
        return None
    stamp = str(closest.get("timestamp", ""))
    if not TIMESTAMP.fullmatch(stamp):
        return None
    when = datetime.strptime(stamp, "%Y%m%d%H%M%S")
    return {"url": f"https://web.archive.org/web/{stamp}id_/{url}", "timestamp": stamp, "date": when.date().isoformat(),
            "original": url}
