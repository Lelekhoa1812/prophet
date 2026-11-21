"""Settings: bounded limits, masked keys in an owner-only .env, and data export/clear that keep settings and keys.

Every test uses a temporary data root and .env, and restores the process environment, so the real data/ and keys are
never touched.
"""
from __future__ import annotations

import os
import stat

import pytest

from prophet_agent.runtime.goal import GoalConfig
from prophet_agent.runtime.store import Store
from prophet_agent.settings import BOUNDS, Settings, mask

KEY_A = "nvapi-AAAAAAAA1111"
KEY_B = "nvapi-BBBBBBBB2222"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for name in ["NVIDIA_NIM_API_KEYS"] + [f"NVIDIA_NIM_API_KEY_{i}" for i in range(1, 10)]:
        monkeypatch.setenv(name, "")                 # recorded, so teardown restores the real environment
        monkeypatch.delenv(name)
    dotenv = tmp_path / ".env"
    dotenv.write_text("# keep me\nOTHER=1\nNVIDIA_NIM_API_KEY_1=nvapi-OLDOLDOLD0000\n")
    settings = Settings(data_root=tmp_path / "data", dotenv=dotenv)
    yield settings, dotenv, tmp_path


def test_defaults_are_returned_until_a_value_is_saved(env):
    settings, _, _ = env
    assert settings.values()["agent"]["part_steps"] == 200
    settings.update({"agent": {"part_steps": 50}})
    assert settings.values()["agent"]["part_steps"] == 50
    assert settings.goal_config(GoalConfig()).max_iterations == 50


def test_out_of_range_or_unknown_settings_are_refused_not_clamped(env):
    settings, _, _ = env
    low, high = BOUNDS[("agent", "part_steps")]
    with pytest.raises(ValueError, match="whole number"):
        settings.update({"agent": {"part_steps": high + 1}})
    with pytest.raises(ValueError, match="whole number"):
        settings.update({"agent": {"part_steps": True}})          # a bool is not a count
    with pytest.raises(ValueError, match="unknown setting"):
        settings.update({"agent": {"made_up": 3}})
    with pytest.raises(ValueError, match="unknown settings section"):
        settings.update({"sandbox": {"timeout": 3}})
    assert settings.values()["agent"]["part_steps"] == 200, "a refused update changes nothing"


def test_a_refused_patch_writes_nothing_even_when_part_of_it_is_valid(env):
    settings, _, _ = env
    with pytest.raises(ValueError):
        settings.update({"agent": {"part_steps": 40, "sub_steps": 0}})
    assert settings.values()["agent"]["part_steps"] == 200


def test_default_budget_follows_the_saved_allowance(env):
    settings, _, _ = env
    settings.update({"agent": {"tool_calls_per_part": 7, "part_seconds": 120}})
    assert settings.default_budget() == {"max_tool_calls": 7, "max_seconds": 120}


def test_keys_are_written_to_dotenv_owner_only_and_never_shown_whole(env):
    settings, dotenv, _ = env
    settings.set_keys([KEY_A, KEY_B])
    text = dotenv.read_text()
    assert "# keep me" in text and "OTHER=1" in text, "unrelated lines are kept"
    assert "OLDOLD" not in text, "the old indexed key is removed, so it cannot come back"
    assert f"NVIDIA_NIM_API_KEYS={KEY_A},{KEY_B}" in text
    assert stat.S_IMODE(os.stat(dotenv).st_mode) == 0o600
    masked = settings.masked_keys()
    assert [m["masked"] for m in masked] == [mask(KEY_A), mask(KEY_B)]
    assert all(KEY_A not in m["masked"] for m in masked)


def test_a_masked_entry_keeps_its_stored_key_so_echoing_the_form_cannot_erase_it(env):
    settings, _, _ = env
    settings.set_keys([KEY_A, KEY_B])
    settings.set_keys([mask(KEY_A), KEY_B])
    assert settings.key_slots() == [KEY_A, KEY_B]


def test_a_masked_entry_matching_no_key_and_a_malformed_key_are_refused(env):
    settings, _, _ = env
    settings.set_keys([KEY_A])
    with pytest.raises(ValueError, match="does not match"):
        settings.set_keys(["nvapi-ZZZZ...9999"])
    with pytest.raises(ValueError, match="starts with nvapi-"):
        settings.set_keys(["not-a-key"])
    assert settings.key_slots() == [KEY_A], "a refused key list leaves the stored keys alone"


def test_export_carries_settings_and_ledger_but_no_key_material(env, tmp_path):
    settings, _, _ = env
    settings.set_keys([KEY_A])
    (tmp_path / "data").mkdir(exist_ok=True)
    store = Store(tmp_path / "data" / "prophet.sqlite")
    store.create_mission("forecast the demand", {"max_tool_calls": 5})

    class _Datasets:
        def list(self, limit=50):
            return []

    data = settings.export(store, _Datasets())
    store.close()
    assert data["missions"][0]["goal"] == "forecast the demand"
    assert KEY_A not in str(data)


def test_clear_needs_the_exact_phrase_and_refuses_while_a_mission_runs(env, tmp_path):
    settings, _, _ = env
    (tmp_path / "data").mkdir(exist_ok=True)
    store = Store(tmp_path / "data" / "prophet.sqlite")
    mid = store.create_mission("g", {})
    with pytest.raises(ValueError, match="type"):
        settings.clear(store, ("created", "running", "waiting_user"), "yes")
    store.set_status(mid, "running")
    with pytest.raises(RuntimeError, match="still running"):
        settings.clear(store, ("created", "running", "waiting_user"), "delete all data")
    store.set_status(mid, "completed")
    removed = settings.clear(store, ("created", "running", "waiting_user"), "delete all data")
    assert removed["missions"] == 1
    assert store.conn.execute("SELECT COUNT(*) FROM missions").fetchone()[0] == 0
    store.close()


def test_clear_keeps_settings_and_keys_and_removes_uploads(env, tmp_path):
    settings, dotenv, _ = env
    settings.set_keys([KEY_A])
    settings.update({"agent": {"part_steps": 80}})
    data = settings.data_root
    (data / "datasets").mkdir(parents=True)
    (data / "datasets" / "sales.csv").write_text("a,b\n1,2\n")
    store = Store(data / "prophet.sqlite")
    settings.clear(store, ("created", "running", "waiting_user"), "delete all data")
    store.close()
    assert not (data / "datasets").exists()
    assert (data / "settings.json").exists(), "settings survive a clear"
    assert settings.values()["agent"]["part_steps"] == 80
    assert "NVIDIA_NIM_API_KEYS" in dotenv.read_text(), "keys survive a clear"
    assert (data / "prophet.sqlite").exists(), "the database file is kept; its rows are emptied"


def test_only_the_bounded_numbers_are_settings():
    assert set(BOUNDS) == {("agent", "part_steps"), ("agent", "tool_calls_per_part"), ("agent", "part_seconds"),
                           ("agent", "sub_steps"), ("context", "view_chars"), ("context", "keep_recent"),
                           ("context", "memory_chars")}
