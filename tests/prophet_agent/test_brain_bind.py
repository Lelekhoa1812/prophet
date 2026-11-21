"""NimBrain hands the mission's checkpoint and wait note to the transport only when they were bound, so a cancel is felt
inside a provider wait and a scripted client that does not know those arguments still works."""
from __future__ import annotations

import json
from types import SimpleNamespace

from prophet_agent.runtime.goal import NimBrain


class Client:
    chain = ("m",)

    def __init__(self):
        self.kwargs = None

    def complete(self, models, messages, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(text=json.dumps({"ok": True}))


def test_unbound_brain_passes_no_extra_arguments():
    c = Client()
    NimBrain(c).decide([], {}, "plan")
    assert "checkpoint" not in c.kwargs and "on_wait" not in c.kwargs


def test_bound_brain_passes_checkpoint_and_wait_note_to_every_call():
    c = Client()
    brain = NimBrain(c)
    stop, note = (lambda: None), (lambda info: None)
    brain.bind(stop, note)
    brain.decide([], {}, "plan")
    assert c.kwargs["checkpoint"] is stop and c.kwargs["on_wait"] is note
    brain.write([])
    assert c.kwargs["checkpoint"] is stop
