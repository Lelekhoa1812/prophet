"""The model already streams upstream. The text the person reads is shown as it arrives.

Planner calls stay buffered. A free-text answer is released only while the lead is writing it. The draft is not a
step: a replay does not read it, and storing the answer clears it.
"""
from __future__ import annotations

from types import SimpleNamespace

from prophet_agent.api.service import MissionService
from prophet_agent.nim.transport import preview_text
from prophet_agent.runtime.goal import NimBrain
from prophet_agent.runtime.store import Store
from prophet_agent.tools import default_tools


def test_preview_hides_an_open_think_block_and_a_partial_tag():
    assert preview_text("Hello <think>secret") == "Hello "
    assert preview_text("Hello <thi") == "Hello "
    assert preview_text("<think>secret</think>The total is 12.") == "The total is 12."
    assert preview_text("The total is 12.") == "The total is 12."


class _Stream:
    chain = ("m",)

    def __init__(self, pieces, text):
        self.pieces = pieces
        self.text = text
        self.calls = []

    def stream(self, models, messages, **kwargs):
        self.calls.append(("stream", kwargs))
        for piece in self.pieces:
            yield "delta", piece
        yield "done", SimpleNamespace(text=self.text)

    def complete(self, models, messages, **kwargs):
        self.calls.append(("complete", kwargs))
        return SimpleNamespace(text=self.text)


def test_a_revealed_write_passes_arriving_text_then_the_finished_text():
    client = _Stream(["The demand ", "rises in June."], "The demand rises in June.")
    seen: list[str] = []
    brain = NimBrain(client, ["m"])
    brain.on_text = seen.append
    with brain.reveal():
        assert brain.write([]) == "The demand rises in June."
    assert seen[0].startswith("The demand")
    assert seen[-1] == "The demand rises in June."
    assert client.calls[0][0] == "stream"


def test_a_write_outside_reveal_waits_for_the_whole_reply():
    client = _Stream(["The demand "], "The demand rises in June.")
    brain = NimBrain(client, ["m"])
    brain.on_text = lambda text: None
    assert brain.write([]) == "The demand rises in June."
    assert client.calls == [("complete", client.calls[0][1])]


def test_the_draft_is_on_the_snapshot_and_the_stored_answer_clears_it(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    mid = service.store.create_mission("forecast demand", {})
    service.store.set_draft(mid, "The demand rises")
    snap = service.snapshot(mid)
    assert snap["draft"] == "The demand rises"
    service.store.set_answer(mid, {"status": "met", "answer": "The demand rises in June."})
    assert service.store.mission(mid)["draft_text"] is None
    assert service.snapshot(mid)["draft"] is None


def test_an_older_database_gains_the_draft_column(tmp_path):
    path = tmp_path / "old.sqlite"
    store = Store(path)
    mid = store.create_mission("forecast demand", {})
    store.close()
    import sqlite3
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE missions DROP COLUMN draft_text")
    conn.close()
    again = Store(path)
    again.set_draft(mid, "still writing")
    assert again.mission(mid)["draft_text"] == "still writing"
    again.close()
