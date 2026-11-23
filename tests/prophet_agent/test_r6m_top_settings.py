"""Round 6m: key storage in Settings. Every test uses a temporary .env and data folder; the repo .env is never read or written."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from prophet_agent import config
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


def test_keys_are_read_from_the_settings_own_dotenv_not_the_repo_file(settings, tmp_path, monkeypatch):
    decoy = tmp_path / "repo.env"
    decoy.write_text("NVIDIA_NIM_API_KEYS=nvapi-DECOYDECOY1234\n")
    monkeypatch.setattr(config, "DOTENV", decoy)
    assert settings.key_slots() == [], "a key in another file must not be used by this Settings"
    settings.set_keys(["nvapi-OWNKEYOWNKEY1234"])
    assert settings.key_slots() == ["nvapi-OWNKEYOWNKEY1234"]


def test_the_env_file_is_replaced_from_a_temp_file_in_data_and_flushed_first(settings, monkeypatch):
    events: list[tuple[str, str]] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append(("fsync", ""))
        return real_fsync(fd)

    def replace(src, dst, *args, **kwargs):
        events.append(("replace", str(src)))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    settings.set_keys(["nvapi-ATOMICKEY12345678"])
    kinds = [kind for kind, _ in events]
    assert "fsync" in kinds and kinds.index("fsync") < kinds.index("replace"), "the key bytes reach disk before the swap"
    temp = Path(events[kinds.index("replace")][1])
    assert temp.parent == settings.data_root, "a temp file that a crash can leave behind must sit in the git-ignored data folder"
    assert temp.parent != settings.dotenv.parent


def test_a_failed_swap_leaves_the_old_env_intact_and_no_temp_file_behind(settings, monkeypatch):
    settings.set_keys(["nvapi-ORIGINALKEY1234"])
    before = settings.dotenv.read_text()

    def refuse(src, dst, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(OSError):
        settings.set_keys(["nvapi-REPLACEMENT5678"])
    assert settings.dotenv.read_text() == before
    assert settings.key_slots() == ["nvapi-ORIGINALKEY1234"]
    stray = [p.name for p in settings.dotenv.parent.iterdir() if p.name.startswith(".env")]
    if settings.data_root.exists():
        stray += [p.name for p in settings.data_root.iterdir() if p.name.startswith(".")]
    assert stray == [".env"], "no temp file may keep a key outside the owner-only .env"
    assert stat.S_IMODE(os.stat(settings.dotenv).st_mode) == 0o600
