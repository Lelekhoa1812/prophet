"""Round 5: after a model-service checkpoint, the next automatic part waits before it starts.

Live (measured, 2026-11-23): a network outage ("Could not resolve host", NIM ConnectError) stopped two missions at checkpoints;
both automatic parts restarted at once, failed again within seconds, and the missions sat at a checkpoint after the network
came back until a person pressed Continue.
"""
from __future__ import annotations

from prophet_agent.api.service import MODEL_SERVICE_WAIT_MAX_S, MissionService
from prophet_agent.runtime.goal import GoalConfig


def _service(tmp_path):
    svc = MissionService(tmp_path / "w.sqlite", lambda: None, {}, GoalConfig(max_iterations=1))
    slept, now = [], [0.0]

    def sleep(s):
        slept.append(s)
        now[0] += s
    svc._sleep, svc._clock = sleep, lambda: now[0]
    return svc, slept


def _checkpoint(svc, which):
    mid = svc.store.create_mission("forecast", {})
    svc.store.set_status(mid, "checkpoint")
    svc.store.append_event(mid, "mission.checkpoint", {"which": which})
    return mid


def test_a_model_service_checkpoint_waits_and_doubles_up_to_the_cap(tmp_path):
    svc, slept = _service(tmp_path)
    mid = _checkpoint(svc, "model_service")
    assert svc._wait_for_model_service(mid, 1) is True              # second automatic part: 60 s doubled
    assert sum(slept) == 120.0 and max(slept) == 5.0
    waits = [e["payload"]["seconds"] for e in svc.store.events(mid) if e["kind"] == "mission.auto_continue_waiting"]
    assert waits == [120.0]
    svc._sleep = lambda s: svc.store.set_status(mid, "running")     # the person pressed Continue while it waited
    assert svc._wait_for_model_service(mid, 3) is False
    assert [e["payload"]["seconds"] for e in svc.store.events(mid)
            if e["kind"] == "mission.auto_continue_waiting"][-1] == MODEL_SERVICE_WAIT_MAX_S


def test_other_checkpoints_do_not_wait(tmp_path):
    svc, slept = _service(tmp_path)
    mid = _checkpoint(svc, "max_seconds")
    assert svc._wait_for_model_service(mid, 0) is True and slept == []
