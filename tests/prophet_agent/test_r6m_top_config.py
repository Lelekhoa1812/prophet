"""Round 6m: the .env reader in config.py. Offline: every file is a temporary one; the repo .env is never read."""
from __future__ import annotations

from prophet_agent.config import load_dotenv_values


def parse(tmp_path, text: str) -> dict[str, str]:
    path = tmp_path / "sample.env"
    path.write_text(text)
    return load_dotenv_values(path)


def test_a_comment_after_an_unquoted_key_is_not_part_of_the_key(tmp_path):
    # A key pasted with a trailing note used to keep the note, so the key was sent as "nvapi-... # my key" and refused.
    values = parse(tmp_path, "NVIDIA_NIM_API_KEYS=nvapi-abcdefgh1234  # my key\n")
    assert values == {"NVIDIA_NIM_API_KEYS": "nvapi-abcdefgh1234"}


def test_a_hash_inside_an_unquoted_value_is_kept_when_no_space_comes_before_it(tmp_path):
    assert parse(tmp_path, "TOKEN=abc#def\n") == {"TOKEN": "abc#def"}


def test_a_hash_inside_quotes_is_kept_and_a_comment_after_closing_quotes_is_dropped(tmp_path):
    values = parse(tmp_path, 'QUOTED="abc # not a comment"\nAFTER="abc" # note\n')
    assert values == {"QUOTED": "abc # not a comment", "AFTER": "abc"}


def test_an_unbalanced_quote_is_kept_as_typed(tmp_path):
    assert parse(tmp_path, 'UNBALANCED="no closing quote\n') == {"UNBALANCED": '"no closing quote'}
