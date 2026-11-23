"""Round 6m: malformed inputs are refused with a message the model can act on, not a bare TypeError or a silent value.

Probes (before the fixes): a list or dict as the file name of inspect_data raised "unhashable type: 'list'"; a dict as
source.name raised the same; a series with nested lists was accepted as a two-dimensional array; a driver window given as
the integer 2026 was accepted as the date 1970-01-01.
"""
from __future__ import annotations

import pytest

from prophet_agent.tools import data_tools


def test_inspect_data_refuses_a_file_name_that_is_not_text(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("date,value\n2026-01-01,1\n")
    run = data_tools.make_inspect({"a.csv": path})
    for bad in (["a.csv"], {"name": "a.csv"}, 5):
        with pytest.raises(ValueError, match="file must be one of"):
            run({"file": bad}, "k")
