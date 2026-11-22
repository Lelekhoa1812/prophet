"""What the agent may change about a TimesFM 3 forecast, and the checks that keep each change honest.

Motivation vs Logic
Motivation: TimesFM 3 has no fine-tuning hook, so everything that adapts it to a series is a choice made at call time:
how much history it reads, which points it should ignore (a keying error, a stock-out, a one-off event), which known
drivers it may condition on (a promotion calendar, public holidays, price), and two numeric devices (z-normalisation and
symmetric averaging). The agent needs to try these and keep only what a backtest shows helps; the earlier calendar
covariate gave no gain on the synthetic twin, so a change must never be assumed to help.
Logic: ``Control`` holds one validated set of choices. ``build_control`` turns tool arguments into it and refuses
anything the model cannot honour with a message that says what to fix (a covariate of the wrong length is the usual one).
``Control.at_origin`` cuts every array to what was knowable at a past origin, so the same control can be backtested
without leaking the future. A covariate is "past" when it is only known up to now (price paid, weather) and "future"
when it is known ahead (a planned promotion, a holiday); the model conditions on both and does not forecast them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

MAX_COVARIATES = 8
MAX_IGNORE_RANGES = 200
PADDINGS = ("none", "edge")
CONTROL_ARGS = {"context_points", "ignore_ranges", "past_covariates", "future_covariates", "znorm", "symmetric_averaging",
                "covariate_padding"}

CONTROL_HELP = {
    "context_points": "optional integer, use only the most recent N points of history (default: all, up to the model's cap)",
    "ignore_ranges": "optional list of [start, end) index pairs into the history (0 = first point) that the model should "
                     "treat as missing: keying errors, stock-outs, one-off events. The values there are not used.",
    "past_covariates": "optional, drivers known only up to now: {name: list as long as the history} or a list of names "
                       "stored under 'covariates' in the source file",
    "future_covariates": "optional, drivers known ahead (planned promotions, holidays): {name: list as long as the "
                         "history PLUS the horizon} or a list of names stored under 'covariates' in the source file",
    "znorm": "optional boolean, standardise the series and covariates before the model sees them (default false)",
    "symmetric_averaging": "optional boolean, also forecast the negated series and average, which can reduce bias "
                           "(runs the model twice; default false)",
    "covariate_padding": "optional, 'none' (default) or 'edge': with edge, the future covariates repeat their last value past "
                         "the horizon up to the model's output patch, so the model sees a driver for every step it decodes. "
                         "Needs past_covariates or future_covariates.",
}


@dataclass(frozen=True)
class Control:
    context_points: int | None = None
    ignore: tuple[tuple[int, int], ...] = ()
    past: dict[str, np.ndarray] = field(default_factory=dict)
    future: dict[str, np.ndarray] = field(default_factory=dict)
    znorm: bool = False
    symmetric: bool = False
    padding_mode: str = "none"      # A6-10: "edge" extends future covariates past the horizon (see timesfm.edge_pad_future)

    @property
    def is_default(self) -> bool:
        return not (self.context_points or self.ignore or self.past or self.future or self.znorm or self.symmetric)

    def describe(self) -> dict[str, Any]:
        """What was asked for, without the arrays, for the result's recorded configuration."""
        d: dict[str, Any] = {}
        if self.context_points:
            d["context_points"] = self.context_points
        if self.ignore:
            d["ignored_ranges"] = [list(r) for r in self.ignore]
            d["ignored_points"] = sum(e - s for s, e in self.ignore)
        if self.past:
            d["past_covariates"] = sorted(self.past)
        if self.future:
            d["future_covariates"] = sorted(self.future)
        if self.znorm:
            d["znorm"] = True
        if self.symmetric:
            d["symmetric_averaging"] = True
        if self.padding_mode != "none":
            d["covariate_padding"] = self.padding_mode
        return d

    def ignore_mask(self, n: int) -> np.ndarray:
        m = np.zeros(n, dtype=bool)
        for s, e in self.ignore:
            m[max(0, s):min(n, e)] = True
        return m

    def at_origin(self, origin: int, horizon: int) -> "Control":
        """The same choices as they would have been made with only the first ``origin`` points known."""
        return Control(
            context_points=self.context_points,
            ignore=tuple((s, min(e, origin)) for s, e in self.ignore if s < origin),
            past={k: v[:origin] for k, v in self.past.items()},
            future={k: v[:origin + horizon] for k, v in self.future.items()},
            znorm=self.znorm, symmetric=self.symmetric, padding_mode=self.padding_mode)


def _bool(args: dict, key: str) -> bool:
    v = args.get(key, False)
    if not isinstance(v, bool):
        raise ValueError(f"{key} must be true or false")
    return v


def _ranges(raw: Any, n: int) -> tuple[tuple[int, int], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or len(raw) > MAX_IGNORE_RANGES:
        raise ValueError(f"ignore_ranges must be a list of up to {MAX_IGNORE_RANGES} [start, end) pairs")
    out = []
    for r in raw:
        if (not isinstance(r, (list, tuple)) or len(r) != 2 or not all(isinstance(i, int) and not isinstance(i, bool) for i in r)):
            raise ValueError(f"ignore_ranges entry {r!r} must be [start, end) with two whole numbers")
        s, e = int(r[0]), int(r[1])
        if not 0 <= s < e <= n:
            raise ValueError(f"ignore_ranges entry {[s, e]} must satisfy 0 <= start < end <= {n} (the history has {n} points)")
        out.append((s, e))
    merged: list[tuple[int, int]] = []
    for s, e in sorted(out):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return tuple(merged)


def _covariates(raw: Any, pool: dict[str, Any], want: int, kind: str) -> dict[str, np.ndarray]:
    if raw is None:
        return {}
    if isinstance(raw, list):
        if not all(isinstance(k, str) for k in raw):
            raise ValueError(f"{kind}_covariates as a list must hold names (strings); give {{name: values}} to pass numbers")
        missing = [k for k in raw if k not in pool]
        if missing:
            raise ValueError(f"{kind}_covariates {missing} not found under 'covariates' in the source file; "
                             f"available: {sorted(pool)[:20] or 'none (save them with covariates= in save_series)'}")
        raw = {k: pool[k] for k in raw}
    if not isinstance(raw, dict):
        raise ValueError(f"{kind}_covariates must be {{name: list of numbers}} or a list of names")
    if len(raw) > MAX_COVARIATES:
        raise ValueError(f"{len(raw)} {kind} covariates; the limit is {MAX_COVARIATES}")
    out = {}
    for name, vals in raw.items():
        try:
            a = np.asarray(vals, dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError(f"{kind} covariate {name!r} contains values that are not numbers")
        if a.ndim != 1 or not np.isfinite(a).all():
            raise ValueError(f"{kind} covariate {name!r} must be a flat list of finite numbers (fill gaps first)")
        if len(a) != want:
            extra = " (the history PLUS the horizon: append the values for the periods being forecast)" if kind == "future" else ""
            raise ValueError(f"{kind} covariate {name!r} has {len(a)} values; it needs exactly {want}{extra}")
        out[str(name)] = a
    return out


def build_control(args: dict, n: int, horizon: int, pool: dict[str, Any] | None = None) -> Control:
    """Validate the control arguments for a history of ``n`` points and a forecast of ``horizon`` periods."""
    pool = pool or {}
    ctx = args.get("context_points")
    if ctx is not None and (not isinstance(ctx, int) or isinstance(ctx, bool) or ctx < 8):
        raise ValueError("context_points must be a whole number of at least 8")
    ignore = _ranges(args.get("ignore_ranges"), n)
    left = n - sum(e - s for s, e in ignore)
    if ignore and left < 8:
        raise ValueError(f"ignoring those ranges leaves {left} usable points; at least 8 are needed")
    padding = args.get("covariate_padding", "none")
    if padding not in PADDINGS:
        raise ValueError(f"covariate_padding must be one of {list(PADDINGS)}, got {padding!r}")
    past = _covariates(args.get("past_covariates"), pool, n, "past")
    future = _covariates(args.get("future_covariates"), pool, n + horizon, "future")
    if padding != "none" and not (past or future):
        raise ValueError("covariate_padding needs past_covariates or future_covariates; with none it has nothing to pad")
    return Control(
        context_points=ctx, ignore=ignore, past=past, future=future,
        znorm=_bool(args, "znorm"), symmetric=_bool(args, "symmetric_averaging"), padding_mode=padding)
