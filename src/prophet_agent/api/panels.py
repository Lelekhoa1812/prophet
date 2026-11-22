"""Per-entity results of a panel backtest, paged for the console (A7-09).

Motivation vs Logic
Motivation: a panel of many series (stations, SKUs, clients) has one MASE per entity and model. The planner sees only the
first rows, and the console had no view at all, so a person could not find the entity that broke.
Logic: each stored ``panel_backtest`` result (the ledger keeps every entity) becomes a summary, and its entities are
returned a page at a time, sorted by name, best model or any model's MASE. The best model is the lowest finite MASE.
"""
from __future__ import annotations

import math
from typing import Any

SORTS = ("entity", "best")          # plus "mase:<model>" for any model in the run
MAX_PAGE = 200


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _best(mase: dict[str, Any]) -> str | None:
    finite = {m: float(v) for m, v in mase.items() if _finite(v)}
    return min(finite, key=finite.get) if finite else None


# The stored JSON has its keys sorted, so the models are put back in the order a person reads them.
MODEL_ORDER = ("naive", "seasonal_naive", "global_gbm", "timesfm", "blend")


def _models(result: dict) -> list[str]:
    names: set[str] = set()
    for models in (result.get("per_entity") or {}).values():
        names.update(models)
    return sorted(names, key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER), m))


def panel_summary(step: int, result: dict) -> dict:
    per = result.get("per_entity") or {}
    return {"step": step, "entities": len(per), "models": _models(result), "horizon": result.get("horizon"),
            "origins": result.get("origins"), "season": result.get("season"), "overall": result.get("overall") or {},
            "wins": result.get("wins") or {}, "best_overall": result.get("best_overall"),
            "coverage_80": result.get("coverage_80") or {}, "reading": result.get("reading"), "basis": result.get("basis")}


def entity_page(result: dict, page: int = 1, page_size: int = 25, sort: str = "entity", order: str = "asc") -> dict:
    """One page of entities. Raises ValueError for a bad page, size, sort or order (the API answers 422)."""
    if page < 1:
        raise ValueError("page must be 1 or more")
    if not 1 <= page_size <= MAX_PAGE:
        raise ValueError(f"page_size must be from 1 to {MAX_PAGE}")
    if order not in ("asc", "desc"):
        raise ValueError("order must be asc or desc")
    models = _models(result)
    per = result.get("per_entity") or {}
    rows = [{"entity": name, "best_model": _best(mase), "mase": {m: mase.get(m) for m in models}} for name, mase in per.items()]
    if sort == "entity":
        key = lambda r: r["entity"]
    elif sort == "best":
        key = lambda r: (r["best_model"] or "", r["entity"])
    elif sort.startswith("mase:") and sort[5:] in models:
        model = sort[5:]
        key = lambda r: (not _finite(r["mase"].get(model)), float(r["mase"][model]) if _finite(r["mase"].get(model)) else 0.0, r["entity"])
    else:
        raise ValueError(f"sort must be entity, best, or mase:<model> for one of {models}")
    rows.sort(key=key, reverse=(order == "desc"))
    if sort.startswith("mase:") or sort == "best":
        # Missing values stay last whichever way the order runs.
        rows.sort(key=lambda r: r["best_model"] is None if sort == "best" else not _finite(r["mase"].get(sort[5:])))
    total = len(rows)
    pages = max(1, math.ceil(total / page_size))
    start = (page - 1) * page_size
    return {"total": total, "page": page, "page_size": page_size, "pages": pages, "sort": sort, "order": order,
            "models": models, "rows": rows[start:start + page_size]}
