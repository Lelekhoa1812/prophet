"""An answered question stays in the mission record with the answer the person gave, so the chat can show the decision
after its card has gone. Open questions are still listed separately."""
from __future__ import annotations

import pytest

from prophet_agent.runtime.store import Store

PAYLOAD = {"questions": [{"question": "Ignore the 2025-06-03 spike?", "header": "Spike",
                          "multiSelect": False, "options": [{"label": "Ignore it", "description": ""}]}]}


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "answered.sqlite")
    yield s
    s.close()


def _ask(store, mid, key):
    # A question always belongs to a tool call the runtime started, so the test starts one too.
    store.tool_start(mid, 0, "ask_user", {}, key)
    return store.open_question(mid, key, PAYLOAD)


def test_answered_question_is_kept_with_its_answer_and_leaves_the_open_list(store):
    mid = store.create_mission("forecast sales", {})
    answered = _ask(store, mid, "tool-a")
    still_open = _ask(store, mid, "tool-b")

    store.answer_question(answered, {"Spike": "Ignore it"})

    kept = store.answered_questions(mid)
    assert [q["id"] for q in kept] == [answered]
    assert kept[0]["answers"] == {"Spike": "Ignore it"}
    assert kept[0]["payload"] == PAYLOAD
    assert kept[0]["answered"] is not None
    assert [q["id"] for q in store.open_questions(mid)] == [still_open]


def test_no_answered_questions_gives_an_empty_list(store):
    mid = store.create_mission("forecast sales", {})
    _ask(store, mid, "tool-a")
    assert store.answered_questions(mid) == []
