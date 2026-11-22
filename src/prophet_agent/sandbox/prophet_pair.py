"""Results and arguments that explain themselves when they are used the wrong way.

Motivation vs Logic
Motivation: many helpers return ``(value, report)``. A model that expects a dict writes ``result["changepoints"]`` and gets
"tuple indices must be integers", then repeats the same call with a different print (live: three identical crashes).
Models also guess keyword names (``n_sigma=``, ``min_segment=``) and call dict-style methods on arrays (``flags.get``).
Logic: ``Pair`` is a tuple, so unpacking, indexing by position and equality work as before. Indexing by a text key is
answered from the report (or from the value when that is a dict), and when the key is in neither the error says what the
function returns and which keys exist. ``ReportDict`` is a dict whose missing-key error lists the keys that exist, with a
few documented aliases. ``accepts_aliases`` maps guessed keyword names to the real ones and raises a TypeError that
shows the real signature for any other unknown keyword. ``wrap_pairs`` applies the explanation to every public function of
a module whose return annotation is a tuple, so no helper has to be edited one by one.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

import numpy as np


class ReportDict(dict):
    """A report or summary dict. A missing key raises KeyError that names the keys that exist.

    Documented aliases: ``missing_steps`` reads ``filled_periods`` (the number of gap periods filled) and ``blockers`` reads
    ``blocked`` (True when the scan found a blocker); a change point's ``mean_before``, ``mean_after``, ``jump`` and ``at``
    read ``before_mean``, ``after_mean``, ``shift`` and ``index``; ``n_outliers`` and ``n_flagged`` read ``count``. An alias
    only answers when its target key is present. When the pair's value is a flag array, ``outliers``, ``flags``, ``mask``
    and ``outlier_mask`` read it (live: 5 failed cells wrote ``report['outliers']`` after ``ps.hampel``), and
    ``outlier_indices``, ``indices`` and ``positions`` read the flagged positions (live: 6 more cells).
    """

    ALIASES = {"missing_steps": "filled_periods", "blockers": "blocked", "mean_before": "before_mean",
               "mean_after": "after_mean", "jump": "shift", "at": "index", "n_outliers": "count", "n_flagged": "count",
               "outlier_count": "count", "num_outliers": "count", "outlier_share": "share", "pct": "share",
               "date": "label"}
    FLAG_KEYS = ("outliers", "flags", "mask", "outlier_mask", "is_outlier")
    INDEX_KEYS = ("outlier_indices", "indices", "outlier_index", "positions", "outlier_positions", "index")
    _flags: Any = None

    def __missing__(self, key: Any) -> Any:
        target = self.ALIASES.get(key) if isinstance(key, str) else None
        if target is not None and target in self:
            return self[target]
        if key in self.FLAG_KEYS and self._flags is not None:
            return self._flags
        if key in self.INDEX_KEYS and self._flags is not None:
            return np.flatnonzero(self._flags)
        raise KeyError(f"no key {key!r} in this report; its keys are {sorted(self)}")

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class ExplainedArray(np.ndarray):
    """An array that answers a text-key read with an explanation of what it is, instead of numpy's "only integers, slices ... are
    valid indices" (live: one mission hit that line 30 times, reading the flags array of ``hampel`` as a dict)."""

    _who = "this value"
    _keys: list = []

    def _explain(self, how: str) -> None:
        kind = "a boolean array of flags, one per input point" if self.dtype == bool else f"a numeric array of {self.size} values"
        raise TypeError(f"{self._who} returned {kind} as its first item, not a dict, so {how} does not exist. Use it as an "
                        f"array (flags.sum(), flags.nonzero()[0]). The counts and settings are in the report, the second item: "
                        f"`value, report = ...`; report keys: {self._keys}.")

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            self._explain(f"[{key!r}]")
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        """Dict-style read, refused with the same explanation as a text-key read (``.get`` is not an array method)."""
        self._explain(f".get({key!r})")


class FlagArray(ExplainedArray):
    """A boolean flag array whose truth value is "any flag set", so ``if report['outliers']:`` reads as intended instead of
    numpy's "truth value of an array is ambiguous" (live retail: 12 failed cells on that line)."""

    def __bool__(self) -> bool:
        return bool(np.asarray(self).any())


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
        if len(self) > 1 and isinstance(self[1], ReportDict):
            try:
                return self[1][key]                      # answers the documented aliases
            except KeyError:
                pass
        keys = sorted(self[1]) if len(self) > 1 and isinstance(self[1], dict) else []
        raise TypeError(f"{self._who} returns a pair (value, report), not a dict: unpack it with `value, report = ...`. "
                        f"The report has the keys {keys}.")


def _plain_parameters(sig: inspect.Signature) -> str:
    """Parameter names and defaults without annotations: ``x, window=7, causal=True``."""
    parts = []
    for p in sig.parameters.values():
        prefix = {inspect.Parameter.VAR_POSITIONAL: "*", inspect.Parameter.VAR_KEYWORD: "**"}.get(p.kind, "")
        parts.append(prefix + p.name if p.default is inspect.Parameter.empty else f"{p.name}={p.default!r}")
    return ", ".join(parts)


def accepts_aliases(aliases: dict[str, str | None] | None = None) -> Callable:
    """Decorator: map guessed keyword names to the real ones, and refuse any other unknown keyword with the real signature.

    ``aliases`` maps a guessed name to the real parameter name. A value of None means the name is accepted and has no effect
    (for example ``robust=True`` on a function that is always robust). Passing both a guessed name and its real name is a
    TypeError. The message for an unknown keyword shows ``name(parameters)`` so one traceback is enough to fix the call.
    """
    alias_map = dict(aliases or {})

    def decorate(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        takes_any_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for guessed, real in alias_map.items():
                if guessed not in kwargs:
                    continue
                value = kwargs.pop(guessed)
                if real is None:
                    continue
                if real in kwargs:
                    raise TypeError(f"{fn.__name__}() got both {guessed!r} and {real!r}; pass only {real!r}")
                kwargs[real] = value
            unknown = [k for k in kwargs if k not in sig.parameters]
            if unknown and not takes_any_keyword:
                raise TypeError(f"{fn.__name__}() got unexpected keyword argument(s) {unknown}; its parameters are "
                                f"{fn.__name__}({_plain_parameters(sig)})")
            return fn(*args, **kwargs)
        return wrapper
    return decorate


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
                    arr = out[0].view(FlagArray if out[0].dtype == bool else ExplainedArray)
                    arr._who = f"{label}()"
                    arr._keys = sorted(out[1]) if len(out) > 1 and isinstance(out[1], dict) else []
                    out = (arr, *out[1:])
                if len(out) > 1 and type(out[1]) is dict:
                    rep = ReportDict(out[1])
                    if isinstance(out[0], np.ndarray) and out[0].dtype == bool:
                        rep._flags = out[0]
                    out = (out[0], rep, *out[2:])
                if out and isinstance(out[0], list) and out[0] and all(type(d) is dict for d in out[0]):
                    # a list of records (change points, segments): each answers the same aliases and explains a missing key
                    out = ([ReportDict(d) for d in out[0]], *out[1:])
                return kind(out)
            return wrapper
        namespace[name] = make(fn, name)
