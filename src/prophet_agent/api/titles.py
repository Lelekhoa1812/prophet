"""Short names for quick forecasts and notebook questions, shown in the sidebar.

A name is at most four words. A deterministic fallback (the first words of the question) is written at once, so the
sidebar never waits on a model. A model then proposes a better name in the background; its answer is used only when it
passes the same limits, so a malformed reply never reaches the screen.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Callable

log = logging.getLogger(__name__)

MAX_WORDS = 4
MAX_CHARS = 60
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'&%.-]*")
_FILLER = {"a", "an", "the", "please", "can", "could", "would", "you", "i", "me", "my", "we", "us", "our", "to", "for",
           "of", "on", "in", "about", "and", "or", "is", "are", "be", "it", "this", "that", "with", "from"}


def fallback_title(text: str) -> str:
    """The first meaningful words of the question. Used until (or unless) a model names it."""
    words = _WORD.findall(text or "")
    kept = [w for w in words if w.lower() not in _FILLER] or words
    title = " ".join(kept[:MAX_WORDS])
    return title[:MAX_CHARS] or "Untitled"


def clean_title(raw: str) -> str | None:
    """Accept a model reply only when it is a short, plain title. Anything else returns None."""
    text = (raw or "").strip().strip("\"'`*#").strip().rstrip(".")
    text = re.sub(r"^(title|name)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    if not text or "\n" in text:
        return None
    if len(text.split()) > MAX_WORDS or len(text) > MAX_CHARS:
        return None
    return text


def propose_title(text: str, ask_model: Callable[[str], str | None]) -> str | None:
    """Ask the model for a title. Returns None when the model is unavailable or its reply does not pass the limits."""
    try:
        reply = ask_model(
            "Name this analysis request in at most four words. Reply with the name only: no quotes, no punctuation, "
            "no explanation.\n\nRequest: " + text[:600]
        )
    except Exception:  # a model outage must never break creating a forecast
        log.info("title model unavailable; keeping the fallback title", exc_info=True)
        return None
    return clean_title(reply or "")


def nim_ask_model(text: str) -> str | None:
    """Default model call. Uses the planner's provider and its enabled models (so only ids that provider knows are sent),
    through the shared client, with no retries of its own."""
    from prophet_agent.nim.transport import shared_client
    from prophet_agent.settings import Settings

    provider, models = Settings().planner_choice()
    reply = shared_client(provider).complete(
        list(models),
        [{"role": "user", "content": text}],
        max_tokens=24,
        budget_s=20.0,
    )
    return reply.text


def name_in_background(text: str, apply: Callable[[str], None], ask_model: Callable[[str], str | None] = nim_ask_model) -> threading.Thread:
    """Run the model naming on a daemon thread and hand a valid title to ``apply``. The fallback is already saved."""

    def run() -> None:
        title = propose_title(text, ask_model)
        if not title:
            return
        try:
            apply(title)
        except Exception:  # the database may already be closed (shutdown, tests); the fallback name stays
            log.info("could not save the model title; keeping the fallback", exc_info=True)

    thread = threading.Thread(target=run, name="title-namer", daemon=True)
    thread.start()
    return thread
