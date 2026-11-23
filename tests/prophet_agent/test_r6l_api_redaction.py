"""Redaction of key-shaped text beyond the NVIDIA nvapi- form: Bearer tokens, key-like name=value pairs, and the other
providers' key formats. The nvapi- behaviour itself is pinned too, so widening never changes it. Keys in these tests are
invented values, never real credentials.
"""
from __future__ import annotations

import pytest

from prophet_agent.api.bundle import _key_shaped, _scrub
from prophet_agent.nim.errors import NimError, redact, scrub_keys


@pytest.mark.parametrize("text, secret", [
    ("Authorization: Bearer abcDEF123456789xyz rejected", "abcDEF123456789xyz"),
    ("GET https://host.example/v1/models?api_key=abcdef123456789 failed", "abcdef123456789"),
    ("GET https://host.example/v1/models?key=zyxwvu987654321 failed", "zyxwvu987654321"),
    ("https://host.example/v1?x=1&token=ghp_abcdef1234567890 failed", "ghp_abcdef1234567890"),
    ("form body password=hunter2hunter2 was echoed", "hunter2hunter2"),
    ('{"password": "hunter2"}', "hunter2"),
    ("upstream echoed sk-ant-api03-ABCDEFGH12345678", "sk-ant-api03-ABCDEFGH12345678"),
    ("openai said sk-proj-abcdef1234567890 is invalid", "sk-proj-abcdef1234567890"),
    ("google AIzaSyB0123456789abcdefghijABCDEFGHIJKL rejected", "AIzaSyB0123456789abcdefghijABCDEFGHIJKL"),
    ("xai key xai-abcdefghijklmnop1234 rejected", "xai-abcdefghijklmnop1234"),
])
def test_other_key_shapes_are_redacted(text, secret):
    out = redact(text)
    assert secret not in out
    assert "[key]" in out


def test_the_nvidia_form_is_still_replaced_as_before():
    assert redact("auth failed for nvapi-SECRETVALUE123") == "auth failed for [key]"
    err = NimError("auth failed for nvapi-SECRETVALUE123")
    assert "SECRETVALUE" not in str(err) and "[key]" in str(err)


def test_the_length_cap_still_applies():
    assert len(redact("x" * 1000, limit=300)) == 300


def test_ordinary_text_and_code_are_left_alone():
    text = "max_tokens=100 total_tokens=5 sorted(rows, key=len) token count 42"
    assert redact(text) == text


def test_code_scrub_keeps_name_value_pairs_that_are_ordinary_code():
    code = "rows.sort(key=lambda r: r[0]); password=None"
    assert scrub_keys(code) == code


def test_the_bundle_scrub_removes_the_other_key_shapes_and_keeps_its_own_label():
    raw = "Bearer abcDEF123456789xyz and sk-proj-abcdef1234567890 and nvapi-ABCDEFGH1234".encode()
    out = _scrub(raw).decode()
    assert "abcDEF123456789xyz" not in out and "sk-proj-abcdef" not in out and "ABCDEFGH1234" not in out
    assert out.count("[key removed]") == 3


def test_the_bundle_refusal_check_sees_other_key_shapes_but_not_plain_code():
    assert _key_shaped("token: sk-proj-abcdef1234567890")
    assert _key_shaped("Authorization: Bearer abcDEF123456789xyz")
    assert not _key_shaped("rows.sort(key=len)\nprint(total)")
