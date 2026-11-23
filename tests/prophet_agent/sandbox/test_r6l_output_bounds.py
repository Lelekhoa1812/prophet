"""Round 6l, audit items 4 and 5: captured output is bounded in memory, redacted, and keeps the real end of the output.

Probes (before the fix): a child printing 150 MB raised the parent's peak memory by about 526 MB (item 5: the whole stream was
held). Output of 45,000 characters kept the first 40,000 and lost the real end, so a traceback printed last was missing
(found while checking item 4: redaction ran on the whole text before the slice, so a key past the slice was not leaked, but
the end of the output was). The tests use outputs of a few megabytes at most.
"""
from __future__ import annotations

from prophet_agent.sandbox.runner import MAX_STREAM_CHARS, RAW_HALF_BYTES, _Capture, _clip, run_python


def test_the_capture_holds_at_most_the_head_and_tail_however_much_is_fed():
    cap = _Capture()
    chunk = b"y" * 65536
    for _ in range(200):                      # about 13 MB
        cap.feed(chunk)
    assert len(cap.head) == RAW_HALF_BYTES and len(cap.tail) == RAW_HALF_BYTES
    assert cap.dropped == 200 * 65536 - 2 * RAW_HALF_BYTES


def test_a_long_output_keeps_its_real_end_and_says_it_was_cut():
    r = run_python("print('x' * 45000 + ' TAILMARK')")
    assert r.ok, r.stderr
    assert "TAILMARK" in r.stdout
    assert "characters cut from the middle" in r.stdout
    assert r.output_truncated is True


def test_a_multi_megabyte_output_is_bounded_and_reports_truncation():
    r = run_python("import sys\nsys.stdout.write('z' * 3_000_000 + ' ENDMARK\\n')", file_mb=64)
    assert r.ok, r.stderr
    assert "ENDMARK" in r.stdout
    assert len(r.stdout) <= MAX_STREAM_CHARS + 200
    assert r.output_truncated is True


def test_a_short_output_is_not_reported_as_truncated():
    r = run_python("print('short')")
    assert r.stdout.strip() == "short" and r.output_truncated is False


def test_a_complete_key_in_the_kept_text_is_redacted():
    r = run_python("print('before nvapi-" + "C" * 40 + " after')")
    assert "[key]" in r.stdout and "CCCC" not in r.stdout


def test_a_key_straddling_the_cut_into_the_middle_is_not_shown_in_part():
    # A space sits just before the key, so the key starts inside the head's last window and runs into the dropped middle.
    total = 2 * RAW_HALF_BYTES + 1000
    start = RAW_HALF_BYTES - 10
    text = "a" * (start - 1) + " " + "nvapi-" + "B" * 60 + " " + "a" * (total - start - 67)
    r = run_python(f"print({text!r})")
    assert r.ok, r.stderr
    assert "nvapi" not in r.stdout and "BBBB" not in r.stdout


def test_a_key_straddling_the_cut_into_the_kept_tail_is_not_shown_in_part():
    total = 2 * RAW_HALF_BYTES + 500
    start = (total - RAW_HALF_BYTES) - 10
    text = "a" * start + "nvapi-" + "B" * 60 + " " + "a" * (total - start - 67)
    r = run_python(f"print({text!r})")
    assert r.ok, r.stderr
    assert "nvapi" not in r.stdout and "BBBB" not in r.stdout


def test_clip_redacts_the_whole_text_before_keeping_head_and_tail():
    text = "q" * (MAX_STREAM_CHARS * 3) + " nvapi-" + "D" * 40 + " tail"
    out = _clip(text)
    assert "DDDD" not in out and "tail" in out
