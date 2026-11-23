"""Every annotation in the research tools module resolves to a real name.

Defect (docs/prophet/GOAL.md): research_tools.py annotated a helper with ``Page`` without importing it. The module imports
with ``from __future__ import annotations``, so nothing failed until something asked for the annotations' values, such as
typing.get_type_hints, a schema builder or a type checker.
"""
from __future__ import annotations

import inspect
import typing

from prophet_agent.tools import research_tools


def test_fetch_with_backfill_annotations_resolve():
    hints = typing.get_type_hints(research_tools._fetch_with_backfill)
    assert set(hints) == {"ranked", "fetch_top", "return"}


def test_every_function_in_the_module_has_resolvable_annotations():
    for name, obj in vars(research_tools).items():
        if inspect.isfunction(obj) and obj.__module__ == research_tools.__name__:
            typing.get_type_hints(obj)   # raises NameError on an undefined annotation name
