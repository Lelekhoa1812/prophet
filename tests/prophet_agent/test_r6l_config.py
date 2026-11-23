"""config.py: paths, environment overrides, directory creation and the .env parser.

Each test loads a fresh copy of the module, because ROOT, DATA and the other paths are fixed when the module is first
imported. The real prophet_agent.config is left alone.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

CONFIG_PATH = Path(__file__).resolve().parents[2] / "src" / "prophet_agent" / "config.py"
REPO_ROOT = CONFIG_PATH.parents[2]
OVERRIDES = ("PROPHET_HOME", "PROPHET_DB", "PROPHET_TIMESFM_CHECKPOINT")


def _fresh_config(monkeypatch, **env):
    for name in OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    spec = importlib.util.spec_from_file_location("prophet_config_under_test", CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_without_overrides_the_state_folder_sits_under_the_repo_root(monkeypatch):
    cfg = _fresh_config(monkeypatch)
    assert cfg.ROOT == REPO_ROOT
    assert cfg.DATA == REPO_ROOT / "data"
    assert cfg.MISSIONS_DIR == REPO_ROOT / "data" / "missions"
    assert cfg.DB_PATH == REPO_ROOT / "data" / "prophet.sqlite"
    assert cfg.DOTENV == REPO_ROOT / ".env"
    assert cfg.TIMESFM_CHECKPOINT == "google/timesfm-3.0-pytorch"


def test_prophet_home_moves_every_default_path_and_prophet_db_moves_only_the_database(monkeypatch, tmp_path):
    cfg = _fresh_config(monkeypatch, PROPHET_HOME=tmp_path)
    assert cfg.ROOT == tmp_path
    assert cfg.DATA == tmp_path / "data"
    assert cfg.MISSIONS_DIR == tmp_path / "data" / "missions"
    assert cfg.DOTENV == tmp_path / ".env"
    assert cfg.DB_PATH == tmp_path / "data" / "prophet.sqlite"

    custom = _fresh_config(monkeypatch, PROPHET_HOME=tmp_path, PROPHET_DB=tmp_path / "elsewhere.db")
    assert custom.DB_PATH == tmp_path / "elsewhere.db"
    assert custom.DATA == tmp_path / "data"


def test_the_timesfm_checkpoint_can_be_replaced_from_the_environment(monkeypatch):
    cfg = _fresh_config(monkeypatch, PROPHET_TIMESFM_CHECKPOINT="local/checkpoint-folder")
    assert cfg.TIMESFM_CHECKPOINT == "local/checkpoint-folder"


def test_ensure_dirs_creates_the_data_and_mission_folders_and_is_safe_to_repeat(monkeypatch, tmp_path):
    cfg = _fresh_config(monkeypatch, PROPHET_HOME=tmp_path)
    cfg.ensure_dirs()
    cfg.ensure_dirs()
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "data" / "missions").is_dir()


def test_load_dotenv_values_parses_keys_values_quotes_and_skips_noise(monkeypatch, tmp_path):
    cfg = _fresh_config(monkeypatch, PROPHET_HOME=tmp_path)
    env_file = tmp_path / "sample.env"
    env_file.write_text(
        "\n".join([
            "# a comment line",
            "",
            "PLAIN_VALUE=abc123",
            "export EXPORTED = \"quoted value\"",
            "SINGLE='single quoted'",
            "EMPTY=",
            "UNBALANCED=\"no closing quote",
            "NO_EQUALS_SIGN",
            "URL=https://host.example/path?a=b",
            "   INDENTED = value   ",
        ]) + "\n"
    )
    assert cfg.load_dotenv_values(env_file) == {
        "PLAIN_VALUE": "abc123",
        "EXPORTED": "quoted value",
        "SINGLE": "single quoted",
        "EMPTY": "",
        "UNBALANCED": "\"no closing quote",
        "URL": "https://host.example/path?a=b",
        "INDENTED": "value",
    }


def test_load_dotenv_values_returns_empty_for_a_missing_file_or_a_directory(monkeypatch, tmp_path):
    cfg = _fresh_config(monkeypatch, PROPHET_HOME=tmp_path)
    assert cfg.load_dotenv_values(tmp_path / "does-not-exist.env") == {}
    assert cfg.load_dotenv_values(tmp_path) == {}


def test_load_dotenv_values_reads_the_repo_dotenv_by_default_and_never_exports_to_the_environment(monkeypatch, tmp_path):
    cfg = _fresh_config(monkeypatch, PROPHET_HOME=tmp_path)
    (tmp_path / ".env").write_text("R6L_CONFIG_TEST_KEY=not-a-real-key\n")
    monkeypatch.delenv("R6L_CONFIG_TEST_KEY", raising=False)

    assert cfg.load_dotenv_values() == {"R6L_CONFIG_TEST_KEY": "not-a-real-key"}
    assert "R6L_CONFIG_TEST_KEY" not in os.environ
