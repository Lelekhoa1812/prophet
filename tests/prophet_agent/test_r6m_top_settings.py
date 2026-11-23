"""Round 6m: key storage in Settings. Every test uses a temporary .env and data folder; the repo .env is never read or written."""
from __future__ import annotations

import pytest

from prophet_agent.settings import Settings, mask

GEMINI_NAMES = ["GOOGLE_API_KEY", "GEMINI_API_KEY"] + [f"{p}_{i}" for p in ("GOOGLE_API_KEY", "GEMINI_API_KEY") for i in range(1, 10)]
NIM_NAMES = ["NVIDIA_NIM_API_KEYS"] + [f"NVIDIA_NIM_API_KEY_{i}" for i in range(1, 10)]


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for name in GEMINI_NAMES + NIM_NAMES:
        monkeypatch.setenv(name, "")            # recorded, so teardown restores the real environment
        monkeypatch.delenv(name)
    dotenv = tmp_path / ".env"
    dotenv.write_text("# keep me\n")
    return Settings(data_root=tmp_path / "data", dotenv=dotenv)


def test_a_mask_shows_at_most_a_quarter_of_a_short_key():
    for key in ["abcdefgh", "abcdefghijklmnop", "abcdefghijklmnopqrstuv"]:
        shown = sum(len(part) for part in mask(key).split("...") if part)
        assert shown <= len(key) // 4, (key, mask(key))


def test_a_full_length_nim_key_keeps_the_familiar_mask():
    key = "nvapi-" + "A" * 58 + "WXYZ"
    assert mask(key) == "nvapi-...WXYZ"


def test_two_stored_keys_that_mask_alike_both_survive_an_echo_of_their_masks(settings):
    keys = ["AIzaSyAAAAAAAAAA-1111", "AIzaSyBBBBBBBBBB-1111"]
    settings.set_keys(keys, provider="gemini")
    masks = [row["masked"] for row in settings.masked_keys("gemini")]
    assert masks[0] == masks[1], "the premise: these two keys look the same once masked"
    settings.set_keys(masks, provider="gemini")
    assert settings.key_slots(None, "gemini") == keys, "echoing the form must not erase the second key"
