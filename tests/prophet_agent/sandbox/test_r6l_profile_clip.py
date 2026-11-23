"""Round 6l, audit item 8: a profile too large for the clipped stdout is still read, and a missing profile fails explicitly.

Probe (before the fix): a profile printed to stdout above 20,000 characters was clipped to a head and a tail, and json.loads
then failed with "Expecting property name enclosed in double quotes", which did not say the output had been cut. The profile
is now written to out/profile.json and read from there.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from prophet_agent.sandbox.runner import MAX_STREAM_CHARS
from prophet_agent.tools import data_tools


def _wide_csv(path, columns=60):
    # Long text cells: the profile keeps 40 columns with a few samples each, so the samples must be long for the profile
    # to pass the stdout clip (about 24,000 characters here).
    names = [f"measurement_{i:04d}_" + "descriptive_" * 50 for i in range(columns)]
    rows = [",".join(f"r{r}c{c}-" + "x" * 200 for c in range(columns)) for r in range(6)]
    path.write_text(",".join(names) + "\n" + "\n".join(rows) + "\n")
    return path


def test_a_profile_longer_than_the_clipped_stdout_is_read_in_full(tmp_path):
    src = _wide_csv(tmp_path / "wide.csv")
    result = data_tools.make_inspect({"wide.csv": src})({"file": "wide.csv"}, "k")
    assert result["profile"]["rows"] == 6
    assert len(result["profile"]["columns"]) == 40
    assert result["profile"]["columns"][0]["name"].startswith("measurement_0000_")


def test_the_profile_text_would_have_been_clipped_on_stdout(tmp_path):
    # Guard for the test above: it only proves something if the profile really exceeds the stdout clip.
    import json
    src = _wide_csv(tmp_path / "wide.csv")
    result = data_tools.make_inspect({"wide.csv": src})({"file": "wide.csv"}, "k")
    assert len(json.dumps(result["profile"])) > MAX_STREAM_CHARS


def test_a_missing_profile_file_is_reported_explicitly(tmp_path, monkeypatch):
    fake = SimpleNamespace(ok=True, stderr="", run_id="x", workdir=str(tmp_path))
    monkeypatch.setattr(data_tools, "run_python", lambda *a, **k: fake)
    run = data_tools.make_inspect({"f.csv": tmp_path / "f.csv"})
    with pytest.raises(RuntimeError, match="profile was not written"):
        run({"file": "f.csv"}, "k")
