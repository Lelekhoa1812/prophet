"""Model-agnostic forecast adapter with per-series routing.

Motivation vs Logic
Motivation: the platform will serve many clients with different demand shapes, and one model (TimesFM 3, whose weights
carry a non-commercial licence) must not be a single point of failure. Series with long steady history, short history
and lumpy (intermittent) demand also favour different methods.
Logic: every model implements one interface and returns nine quantiles per horizon step, the same `Forecast` the rest
of the system already consumes. `Router` classifies each series (history length, intermittency, volume) and sends it to
the model chosen for its class. `fit_routing` picks that model by walk-forward pinball loss on the tenant's own history,
and keeps the default unless a challenger wins by a clear margin, so routing stays stable and explainable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from . import baselines, metrics
from .forecast import Forecast, TimesFMEngine

SHORT_HISTORY_WEEKS = 26
INTERMITTENT_ZERO_SHARE = 0.30


class ForecastModel(Protocol):
    name: str
    licence: str
    min_history: int

    def predict(self, contexts: list[np.ndarray], horizon: int, season: int = 52) -> Forecast: ...


@dataclass
class TimesFM:
    name: str = "timesfm"
    licence: str = "TimesFM 3 weights: non-commercial until cleared"
    min_history: int = 8

    def predict(self, contexts, horizon, season=52):
        return TimesFMEngine.get().predict(contexts, horizon)


@dataclass
class Baseline:
    """Wraps the honest baselines (ma4, ma8, seasonal_naive, seasonal_year)."""
    name: str
    licence: str = "own code, no restriction"
    min_history: int = 4

    def predict(self, contexts, horizon, season=52):
        return baselines.run(self.name, [np.asarray(c, dtype=float) for c in contexts], horizon, season)


@dataclass
class TSB:
    """Teunter-Syntetos-Babai for lumpy demand: chance of any demand times the size of a typical order.

    Quantiles come from a deterministic Monte-Carlo of (occurs ? size : 0) summed over each step, so a series that sells
    one big lot every few weeks gets a median near zero and a long right tail instead of a smooth average.
    """
    name: str = "tsb"
    licence: str = "own code, no restriction"
    min_history: int = 8
    alpha: float = 0.15
    n: int = 400
    seed: int = 5

    def predict(self, contexts, horizon, season=52):
        rng = np.random.default_rng(self.seed)
        u = rng.random((self.n, horizon))
        pick = rng.random((self.n, horizon))
        qs = np.empty((len(contexts), horizon, 9))
        for i, c in enumerate(contexts):
            s = np.nan_to_num(np.asarray(c, dtype=float))
            p = 0.5
            for v in s:
                p = p + self.alpha * ((1.0 if v > 0 else 0.0) - p)
            sizes = s[s > 0][-52:]
            if len(sizes) == 0:
                qs[i] = 0.0
                continue
            draw = sizes[(pick * len(sizes)).astype(int) % len(sizes)]
            samples = np.where(u < p, draw, 0.0)
            qs[i] = np.quantile(samples, metrics.QS, axis=0).T
        q = np.sort(qs, axis=-1)
        return Forecast(q[:, :, 4].copy(), q)


@dataclass
class Ensemble:
    """Average of members' quantiles (quantile averaging keeps each member's tails in proportion)."""
    members: list
    name: str = "ensemble"
    licence: str = ""
    min_history: int = 8

    def __post_init__(self):
        self.licence = "; ".join(sorted({m.licence for m in self.members}))
        self.min_history = max(m.min_history for m in self.members)

    def predict(self, contexts, horizon, season=52):
        fcs = [m.predict(contexts, horizon, season) for m in self.members]
        q = np.sort(np.mean([f.q for f in fcs], axis=0), axis=-1)
        return Forecast(np.mean([f.point for f in fcs], axis=0), q)


def series_class(context: np.ndarray, head_cut: float | None = None) -> str:
    """short | intermittent | steady, from the series itself (weekly data)."""
    s = np.nan_to_num(np.asarray(context, dtype=float))
    if len(s) < SHORT_HISTORY_WEEKS:
        return "short"
    recent = s[-52:]
    if (recent == 0).mean() >= INTERMITTENT_ZERO_SHARE:
        return "intermittent"
    return "steady"


def default_registry() -> dict:
    reg = {m.name: m for m in [TimesFM(), TSB(), Baseline("ma8"), Baseline("ma4"), Baseline("seasonal_year")]}
    reg["ensemble"] = Ensemble([reg["timesfm"], reg["ma8"]])
    return reg


@dataclass
class Router:
    """Sends each series to the model chosen for its class. `route` maps class -> model name; `default` fills gaps."""
    registry: dict = field(default_factory=default_registry)
    default: str = "timesfm"
    route: dict = field(default_factory=dict)
    allow_non_commercial: bool = True
    licence_fallback: str = "tsb"   # best free model on the twin: 14% worse pinball on steady series, 2% on lumpy

    def model_for(self, cls: str):
        name = self.route.get(cls, self.default)
        m = self.registry[name]
        if not self.allow_non_commercial and "non-commercial" in m.licence:
            return self.registry[self.licence_fallback]      # never serve a non-commercial model in a production tenant
        return m

    def classify(self, contexts) -> list[str]:
        return [series_class(c) for c in contexts]

    def predict(self, contexts: list[np.ndarray], horizon: int, season: int = 52, **_) -> Forecast:
        classes = self.classify(contexts)
        n = len(contexts)
        pt = np.zeros((n, horizon)); q = np.zeros((n, horizon, 9))
        for cls in sorted(set(classes)):
            idx = [i for i, c in enumerate(classes) if c == cls]
            m = self.model_for(cls)
            ok = [i for i in idx if len(contexts[i]) >= m.min_history]
            for i in set(idx) - set(ok):             # too short for the chosen model: fall back to a moving average
                f = self.registry["ma4"].predict([contexts[i]], horizon, season)
                pt[i], q[i] = f.point[0], f.q[0]
            if ok:
                f = m.predict([contexts[i] for i in ok], horizon, season)
                for j, i in enumerate(ok):
                    pt[i], q[i] = f.point[j], f.q[j]
        return Forecast(pt, q)

    def describe(self) -> dict:
        return {"default": self.default, "route": dict(self.route), "allow_non_commercial": self.allow_non_commercial,
                "models": {k: {"licence": v.licence, "min_history": v.min_history} for k, v in self.registry.items()}}


def fit_routing(mat: np.ndarray, models: list, origins: list[int], horizon: int = 4, default: str = "timesfm",
                margin: float = 0.02, season: int = 52, min_series: int = 8) -> dict:
    """Walk-forward pinball loss per series class and model. A challenger replaces the default only if it is at least
    `margin` (relative) better and the class has at least `min_series` series-origins to judge on.

    mat is (series, periods) weekly demand. Returns {"route": {...}, "table": {class: {model: pinball}}, "counts": {...}}.
    """
    acc = {m.name: {} for m in models}
    counts: dict[str, int] = {}
    for o in origins:
        hist = mat[:, :o]
        active = np.where(hist[:, -26:].sum(axis=1) > 0)[0]
        if len(active) == 0:
            continue
        ctxs = [hist[i] for i in active]
        cls = np.array([series_class(c) for c in ctxs])
        actual = mat[active, o:o + horizon]
        for c in set(cls):
            counts[c] = counts.get(c, 0) + int((cls == c).sum())
        for m in models:
            ok = np.array([len(c) >= m.min_history for c in ctxs])
            f = m.predict([c for c, k in zip(ctxs, ok) if k], horizon, season)
            sel = np.where(ok)[0]
            for c in set(cls):
                rows = [j for j, i in enumerate(sel) if cls[i] == c]
                if not rows:
                    continue
                a = actual[sel[rows]]
                acc[m.name].setdefault(c, []).append((metrics.pinball(a, f.q[rows]), a.mean(), len(rows)))
    table: dict[str, dict] = {}
    for name, per in acc.items():
        for c, rows in per.items():
            w = np.array([r[1] * r[2] for r in rows]); v = np.array([r[0] for r in rows])
            table.setdefault(c, {})[name] = float((v * w).sum() / w.sum()) if w.sum() > 0 else float("nan")
    route = {}
    for c, scores in table.items():
        base = scores.get(default)
        best = min(scores, key=lambda k: scores[k] if scores[k] == scores[k] else 9e9)
        route[c] = best if (base is not None and scores[best] <= base * (1 - margin) and counts.get(c, 0) >= min_series) else default
    return {"route": route, "table": table, "counts": counts}


_ROUTER: Router | None = None


def get_forecaster() -> Router:
    """The process-wide forecaster. Default routes every series to TimesFM, so behaviour is unchanged until a tenant
    fits and installs a routing table with `set_router`."""
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = Router()
    return _ROUTER


def set_router(r: Router | None) -> None:
    global _ROUTER
    _ROUTER = r
