"""Pair results that explain themselves when they are read the wrong way.

Motivation vs Logic
Motivation: many helpers return ``(value, report)``. A model that expects a dict writes ``result["changepoints"]`` and gets
"tuple indices must be integers", then repeats the same call with a different print (live: three identical crashes).
Logic: ``Pair`` is a tuple, so unpacking, indexing by position and equality work as before. Indexing by a text key is
answered from the report (or from the value when that is a dict), and when the key is in neither the error says what the
function returns and which keys exist. ``wrap_pairs`` wraps every public function of a module whose return annotation is a
tuple, so no helper has to be edited one by one.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any

import numpy as np


class ExplainedArray(np.ndarray):
    """An array that answers a text-key read with an explanation of what it is, instead of numpy's "only integers, slices ... are
    valid indices" (live: one mission hit that line 30 times, reading the flags array of ``hampel`` as a dict)."""

    _who = "this value"
    _keys: list = []

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            kind = "a boolean array of flags, one per input point" if self.dtype == bool else f"a numeric array of {self.size} values"
            raise TypeError(f"{self._who} returned {kind} as its first item, not a dict, so [{key!r}] does not exist. Use it as an "
                            f"array (flags.sum(), flags.nonzero()[0]). The counts and settings are in the report, the second item: "
                            f"`value, report = ...`; report keys: {self._keys}.")
        return super().__getitem__(key)


class Pair(tuple):
    """``(value, report)``: read it with ``value, report = f(...)``; ``pair["key"]`` reads the report as a convenience."""

    _who = "this function"

    @property
    def value(self) -> Any:
        return self[0]

    @property
    def report(self) -> Any:
        return self[1] if len(self) > 1 else None

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, str):
            return tuple.__getitem__(self, key)
        for part in (self[1:2] or ()) + (self[0:1] or ()):
            if isinstance(part, dict) and key in part:
                return part[key]
        keys = sorted(self[1]) if len(self) > 1 and isinstance(self[1], dict) else []
        raise TypeError(f"{self._who} returns a pair (value, report), not a dict: unpack it with `value, report = ...`. "
                        f"The report has the keys {keys}.")


def wrap_pairs(namespace: dict) -> None:
    """Wrap the public functions in ``namespace`` that are annotated to return a tuple, so they return a ``Pair``."""
    for name, fn in list(namespace.items()):
        if name.startswith("_") or not inspect.isfunction(fn) or fn.__module__ != namespace.get("__name__"):
            continue
        ret = fn.__annotations__.get("return")
        if not (isinstance(ret, str) and ret.replace(" ", "").startswith("tuple[")):
            continue

        def make(f, label):
            kind = type(label, (Pair,), {"_who": f"{label}()"})

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                out = f(*args, **kwargs)
                if not (isinstance(out, tuple) and type(out) is tuple):
                    return out
                if out and isinstance(out[0], np.ndarray) and out[0].ndim >= 1 and not isinstance(out[0], ExplainedArray):
                    arr = out[0].view(ExplainedArray)
                    arr._who = f"{label}()"
                    arr._keys = sorted(out[1]) if len(out) > 1 and isinstance(out[1], dict) else []
                    out = (arr, *out[1:])
                return kind(out)
            return wrapper

        namespace[name] = make(fn, name)
