"""Round 6m, item 1: redaction corrupts analysis output that is not a credential.

The sandbox prints its output through prophet_agent.nim.errors.redact, and the rule lives there, outside the sandbox
package. The rule masks every name=value pair whose name is credential-like, whatever the value is. Reproduced through
sandbox output (run_python), the false positives are:

  token=customer_segment_a      an identifier after a generic name
  token=2026-11-23              a date
  token=12345678                a number
  {"token": "forecast"}         a JSON field named token, with no length floor
  token=<40-character hash>     a hash that a user may need to see
  Bearer securities issued      the word "Bearer" in finance text, followed by an 8+ letter word
  use sk-learn-utilities        the sk- vendor key pattern on a hyphenated identifier

The decided rule, for the owner of nim/errors.py to apply (not applied here, because that file is outside this package):
  1. A bare "token" name masks only inside a URL query (after ? or &), as "key" already does. Free text, JSON and code
     keep it. api_key, access_token, auth_token, secret and password keep masking in every form: a false positive costs a
     masked word, a missed password costs far more.
  2. A Bearer value masks only when it holds a digit or is 20 characters or longer.
  3. An sk- value masks only when it holds a digit and is 20 characters or longer (the vendor key shape).

The guard tests below pass now and must keep passing: real credentials in these shapes stay masked. The false-positive
tests pass under the corrected rule in prophet_agent/nim/errors.py.
"""
from __future__ import annotations

import pytest

from prophet_agent.sandbox.runner import run_python

FALSE_POSITIVES = [
    ("token=customer_segment_a", "token=customer_segment_a"),
    ("token=2026-11-23", "token=2026-11-23"),
    ("token=12345678", "token=12345678"),
    ('{"token": "forecast", "n": 3}', '{"token": "forecast", "n": 3}'),
    ("token=f3a9c1d2e4b5a6978012cdef34567890abcdef12", "token=f3a9c1d2e4b5a6978012cdef34567890abcdef12"),
    ("Bearer securities issued", "Bearer securities issued"),
    ("use sk-learn-utilities here", "use sk-learn-utilities here"),
]

@pytest.mark.parametrize("text, expected", FALSE_POSITIVES, ids=[t for t, _ in FALSE_POSITIVES])
def test_analysis_output_that_is_not_a_credential_is_printed_as_written(text, expected):
    r = run_python(f"print({text!r})\n")
    assert r.ok, r.stderr
    assert r.stdout.strip() == expected


@pytest.mark.parametrize("text, secret", [
    ("api_key=abcdef123456", "abcdef123456"),
    ("password=correcthorse", "correcthorse"),
    ("https://host.example/v1?token=ghp_abcdef1234567890", "ghp_abcdef1234567890"),
    ("Authorization: Bearer abcDEF123456789xyz", "abcDEF123456789xyz"),
    ("the key was sk-proj-abcdef1234567890", "sk-proj-abcdef1234567890"),
])
def test_real_credentials_in_sandbox_output_stay_masked(text, secret):
    r = run_python(f"print({text!r})\n")
    assert r.ok, r.stderr
    assert secret not in r.stdout and "[key]" in r.stdout
