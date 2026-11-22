"""Boundary spacing when a free-text reply is continued after the token allowance cut it (audit 2026-11-22, item 11).

No network. A scripted httpx.MockTransport answers each call; the first reply stops on finish_reason "length" and the
continuation finishes the rest. Run with: PYTHONPATH=src .venv/bin/python -m pytest tests/prophet_agent/nim -q
"""
from __future__ import annotations

import json

import httpx

from prophet_agent.nim import NimClient, clean_text

KEYS = ("nvapi-AAAA1111",)
M1 = "nvidia/nemotron-3-ultra-550b-a55b"
MSGS = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Go on."}]


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def sse(texts: list[str], finish: str) -> httpx.Response:
    chunks = [{"choices": [{"delta": {"content": t}, "finish_reason": None}]} for t in texts]
    chunks.append({"choices": [{"delta": {}, "finish_reason": finish}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 3}})
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def reply_text(*turns: tuple[list[str], str]) -> str:
    """Runs one call whose stream answers each turn in order (first cut, then continuation) and returns the text."""
    pending = iter(turns)

    def respond(request: httpx.Request) -> httpx.Response:
        texts, finish = next(pending)
        return sse(texts, finish)

    client = NimClient(list(KEYS), transport=httpx.MockTransport(lambda r: respond(r)), clock=FakeClock(),
                       sleep=FakeClock().sleep)
    return client.complete([M1], MSGS).text


# -- (a) a continuation cut at a word boundary keeps the space ------------------------------------------------------

def test_continuation_cut_at_a_word_boundary_keeps_the_space():
    text = reply_text((["The forecast rises to 412 by "], "length"), (["June."], "stop"))
    assert text == "The forecast rises to 412 by June."


def test_continuation_keeps_one_space_when_the_model_also_starts_with_one():
    assert reply_text((["The forecast rises to 412 by"], "length"), ([" June."], "stop")) == "The forecast rises to 412 by June."
    assert reply_text((["Hello "], "length"), ([" world"], "stop")) == "Hello world"


def test_continuation_cut_at_a_line_break_keeps_the_line_break():
    assert reply_text((["Items:\n"], "length"), (["- one"], "stop")) == "Items:\n- one"


# -- (b) a continuation cut mid-word still joins with no space ------------------------------------------------------

def test_continuation_cut_mid_word_still_joins_with_no_space():
    assert reply_text((["The forecast rise"], "length"), (["s to 412 by June."], "stop")) == "The forecast rises to 412 by June."
    assert reply_text((["Hel"], "length"), (["lo there"], "stop")) == "Hello there"


# -- (c) a whole reply with leading and trailing whitespace is still trimmed ----------------------------------------

def test_whole_reply_leading_and_trailing_whitespace_is_trimmed():
    assert clean_text("  \n Hi there  \n") == "Hi there"
    assert reply_text((["  \n Hi there  \n"], "stop")) == "Hi there"


def test_whole_reply_padding_is_trimmed_when_it_is_continued():
    assert reply_text((["  Hi there "], "length"), (["again.  \n"], "stop")) == "Hi there again."


# -- (d) a think block is still removed -----------------------------------------------------------------------------

def test_think_block_is_removed_from_a_whole_reply():
    assert clean_text("<think>plan it out</think>\nAnswer") == "Answer"
    assert reply_text((["<think>plan it", " out</think>", "Answer"], "stop")) == "Answer"


def test_think_block_split_across_a_cut_is_removed_and_the_space_kept():
    assert reply_text((["<think>plan it"], "length"), ([" out</think>Answer here"], "stop")) == "Answer here"


# -- (e) the json fence unwrap still works --------------------------------------------------------------------------

def test_json_fence_is_unwrapped_from_a_whole_reply():
    assert clean_text('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert reply_text((["```json\n", '{"a": 1}', "\n```"], "stop")) == '{"a": 1}'


def test_json_fence_is_unwrapped_when_the_reply_is_continued_across_a_space():
    assert reply_text((['```json\n{"a": '], "length"), (["1}\n```"], "stop")) == '{"a": 1}'
