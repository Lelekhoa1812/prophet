"""Chart series read from a web page's own markup: Highcharts, Plotly, Vega-Lite, Chart.js, ECharts and inline SVG.

Motivation vs Logic
Motivation: a chart on a page usually carries its numbers. Reading them is exact, while digitising the picture is an
estimate. The forecast needs the exact series when the page has one, and must be told which series came from the page's
data and which came from geometry.
Logic: the page is parsed with BeautifulSoup. Inline scripts are searched for the library calls (Highcharts.chart,
Plotly.newPlot, new Chart, setOption, vegaEmbed) and their option objects are read by a small literal parser. The parser
handles the JavaScript object syntax pages use (unquoted keys, single quotes, trailing commas, comments) and never runs
JavaScript: a function, a call or an unknown expression becomes UNPARSED, and a series that contains one is skipped rather
than guessed. Variables defined in the same script are resolved once. SVG charts are read from their own geometry: numeric
tick labels on each axis give a linear pixel-to-value map (least squares over two or more ticks, checked for residuals),
and polyline, path and circle marks are inverted through it. Every series records its method, its exactness ("embedded"
for data in the page, "svg-geometry" for values inverted from SVG pixels) and where it was found.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from bs4 import BeautifulSoup

EMBEDDED = "embedded"
SVG_GEOMETRY = "svg-geometry"
MAX_SERIES = 40
MAX_POINTS = 20_000
_MS_DAY = 86_400_000


class _Unparsed:
    """Marks a value the literal reader could not take: a function, a call, an unknown expression."""

    def __repr__(self) -> str:
        return "UNPARSED"


UNPARSED = _Unparsed()


@dataclass(frozen=True)
class Ref:
    """A bare identifier used as a value, resolved later from a variable in the same script."""
    name: str


class _Bad(Exception):
    """The text at this position is not a literal the reader supports."""


_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_IDENT = re.compile(r"[A-Za-z_$][\w$]*")
_WS = " \t\r\n\f\v ﻿"


def _skip_string(t: str, i: int) -> int:
    quote = t[i]
    i += 1
    while i < len(t):
        c = t[i]
        if c == "\\":
            i += 2
            continue
        if c == quote:
            return i + 1
        i += 1
    return len(t)


def _skip_expression(t: str, i: int) -> int:
    """Index of the first comma or closing bracket at depth zero, from i. Strings and comments are passed over, and so are
    balanced brackets, so a function body with commas inside it is skipped whole."""
    depth, n = 0, len(t)
    while i < n:
        c = t[i]
        if c in "\"'`":
            i = _skip_string(t, i)
            continue
        if t.startswith("//", i):
            j = t.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if t.startswith("/*", i):
            j = t.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                return i
            depth -= 1
        elif c == "," and depth == 0:
            return i
        i += 1
    return n


class LiteralReader:
    """Reads JavaScript literals (objects, arrays, strings, numbers, true/false/null) from a script.

    Motivation vs Logic
    Motivation: chart options are written as JavaScript, not JSON. A strict JSON reader would miss most real pages, and
    eval would run page code. This reader accepts only the literal subset, so the page cannot run anything here.
    Logic: each value is read at its position. A value that is not a literal is reported as UNPARSED and skipped to the next
    delimiter, so the rest of the object still reads. A bare identifier becomes a Ref. Date.UTC(...) and new Date("...")
    are read because pages use them for datetime axes.
    """

    def __init__(self, text: str):
        self.t = text
        self.n = len(text)

    def ws(self, i: int) -> int:
        t, n = self.t, self.n
        while i < n:
            c = t[i]
            if c in _WS:
                i += 1
            elif t.startswith("//", i):
                j = t.find("\n", i)
                i = n if j < 0 else j + 1
            elif t.startswith("/*", i):
                j = t.find("*/", i + 2)
                i = n if j < 0 else j + 2
            else:
                break
        return i

    def value(self, i: int) -> tuple[Any, int]:
        i = self.ws(i)
        if i >= self.n:
            raise _Bad("end of text")
        t, c = self.t, self.t[i]
        if c == "{":
            return self._object(i)
        if c == "[":
            return self._array(i)
        if c in "\"'`":
            return self._string(i)
        if c.isdigit() or (c in "+-." and i + 1 < self.n and (t[i + 1].isdigit() or t[i + 1] == ".")):
            m = _NUMBER.match(t, i)
            if not m:
                raise _Bad("number")
            text = m.group(0)
            return (float(text) if any(ch in text for ch in ".eE") else int(text)), m.end()
        m = _IDENT.match(t, i)
        if not m:
            raise _Bad("token")
        word, j = m.group(0), m.end()
        if word in ("true", "false"):
            return word == "true", j
        if word in ("null", "undefined", "NaN", "Infinity"):
            return None, j
        if word == "new":
            k = self.ws(j)
            m2 = _IDENT.match(t, k)
            if m2 and m2.group(0) == "Date" and self.ws(m2.end()) < self.n and t[self.ws(m2.end())] == "(":
                args, end = self.args(self.ws(m2.end()))
                if len(args) == 1 and isinstance(args[0], str):
                    return args[0], end
                if len(args) == 1 and isinstance(args[0], (int, float)):
                    return int(args[0]), end
            raise _Bad("new")
        if word == "Date" and t.startswith(".UTC(", j):
            args, end = self.args(j + 4)
            if args and all(isinstance(a, (int, float)) and not isinstance(a, bool) for a in args):
                return _date_utc_ms(args), end
            raise _Bad("Date.UTC")
        k = self.ws(j)
        if word == "function" or t.startswith("=>", k) or (k < self.n and t[k] in ".([`"):
            raise _Bad("expression")
        return Ref(word), j

    def args(self, i: int) -> tuple[list, int]:
        """Arguments of a call whose '(' is at i. Non-literal arguments are UNPARSED, so the positions stay aligned."""
        t, out = self.t, []
        i += 1
        while True:
            i = self.ws(i)
            if i >= self.n:
                raise _Bad("unclosed call")
            if t[i] == ")":
                return out, i + 1
            if t[i] in ",]}":
                i += 1
                continue
            start = i
            try:
                val, k = self.value(i)
                k2 = self.ws(k)
                if k2 >= self.n or t[k2] not in ",)":
                    raise _Bad("after argument")
            except _Bad:
                val, k2 = UNPARSED, _skip_expression(t, start)
                if k2 <= start:
                    k2 = start + 1
            out.append(val)
            i = k2

    def _string(self, i: int) -> tuple[str, int]:
        t, quote = self.t, self.t[i]
        j = i + 1
        chunks: list[str] = []
        escapes = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}
        while j < self.n:
            c = t[j]
            if c == quote:
                return "".join(chunks), j + 1
            if quote == "`" and t.startswith("${", j):
                raise _Bad("template")
            if c == "\\" and j + 1 < self.n:
                e = t[j + 1]
                if e == "u" and j + 5 < self.n and re.fullmatch(r"[0-9a-fA-F]{4}", t[j + 2:j + 6]):
                    chunks.append(chr(int(t[j + 2:j + 6], 16)))
                    j += 6
                    continue
                if e == "x" and j + 3 < self.n and re.fullmatch(r"[0-9a-fA-F]{2}", t[j + 2:j + 4]):
                    chunks.append(chr(int(t[j + 2:j + 4], 16)))
                    j += 4
                    continue
                if e == "\n":
                    j += 2
                    continue
                chunks.append(escapes.get(e, e))
                j += 2
                continue
            chunks.append(c)
            j += 1
        raise _Bad("unterminated string")

    def _object(self, i: int) -> tuple[dict, int]:
        t, out = self.t, {}
        i += 1
        while True:
            i = self.ws(i)
            if i >= self.n:
                raise _Bad("unclosed object")
            if t[i] == "}":
                return out, i + 1
            if t[i] == ",":
                i += 1
                continue
            if t[i] in "\"'":
                key, j = self._string(i)
            elif t[i].isdigit():
                m = _NUMBER.match(t, i)
                key, j = m.group(0), m.end()
            else:
                m = _IDENT.match(t, i)
                if not m:                       # a computed key, a spread or junk: skip the entry
                    k = _skip_expression(t, i)
                    i = k + 1 if k < self.n and t[k] == "," else (k if k > i else i + 1)
                    continue
                key, j = m.group(0), m.end()
            j = self.ws(j)
            if j < self.n and t[j] == "(":      # method shorthand: name(args) { body }
                k = _skip_expression(t, j)
                out[str(key)] = UNPARSED
                i = k
                continue
            if j >= self.n or t[j] != ":":
                out[str(key)] = UNPARSED
                i = _skip_expression(t, j)
                continue
            start = self.ws(j + 1)
            try:
                val, k = self.value(start)
                k2 = self.ws(k)
                if k2 >= self.n or t[k2] not in ",}":
                    raise _Bad("after property")
            except _Bad:
                val, k2 = UNPARSED, _skip_expression(t, start)
                if k2 <= start:
                    k2 = start + 1
            out[str(key)] = val
            i = k2

    def _array(self, i: int) -> tuple[list, int]:
        t, out = self.t, []
        i += 1
        while True:
            i = self.ws(i)
            if i >= self.n:
                raise _Bad("unclosed array")
            if t[i] == "]":
                return out, i + 1
            if t[i] == ",":
                i += 1
                continue
            if t[i] in ")}":
                i += 1
                continue
            start = i
            try:
                val, k = self.value(i)
                k2 = self.ws(k)
                if k2 >= self.n or t[k2] not in ",]":
                    raise _Bad("after item")
            except _Bad:
                val, k2 = UNPARSED, _skip_expression(t, start)
                if k2 <= start:
                    k2 = start + 1
            out.append(val)
            i = k2


def _date_utc_ms(parts: list) -> int:
    """Date.UTC(year, monthIndex, day, hours, minutes, seconds, ms) in milliseconds since the epoch."""
    fields = [int(p) for p in parts[:7]] + [0] * (7 - len(parts[:7]))
    year, month, day, hour, minute, second, ms = fields
    years, months = year + month // 12, month % 12
    base = datetime(years, months + 1, 1, tzinfo=timezone.utc)
    delta = base - datetime(1970, 1, 1, tzinfo=timezone.utc)
    total = delta.days * _MS_DAY + delta.seconds * 1000 + delta.microseconds // 1000
    return total + (day - 1) * _MS_DAY + hour * 3_600_000 + minute * 60_000 + second * 1000 + ms


def _resolve(node: Any, table: dict[str, Any], depth: int = 0) -> Any:
    """Replaces Refs with the literal their variable holds in the same script. An unknown name becomes UNPARSED."""
    if depth > 20:
        return UNPARSED
    if isinstance(node, Ref):
        if node.name not in table:
            return UNPARSED
        return _resolve(table[node.name], table, depth + 1)
    if isinstance(node, dict):
        return {k: _resolve(v, table, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(v, table, depth + 1) for v in node]
    return node


_ASSIGN = re.compile(r"(?<![\w$.])(?:(?:var|let|const)\s+)?([A-Za-z_$][\w$]*)\s*=(?!=)(?=\s*[\[{\"'\d+-])")


def _variable_table(script: str) -> dict[str, Any]:
    """Literal values of the variables assigned in one script. The first assignment of a name wins."""
    table: dict[str, Any] = {}
    reader = LiteralReader(script)
    for m in _ASSIGN.finditer(script):
        name = m.group(1)
        if name in table:
            continue
        try:
            value, _ = reader.value(m.end())
        except _Bad:
            continue
        table[name] = value
    return table


def _iso(ms: Any) -> Any:
    """A datetime axis value as the page states it: a date when it falls on midnight UTC, else a UTC timestamp."""
    if not isinstance(ms, (int, float)) or isinstance(ms, bool):
        return ms
    when = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    if int(ms) % _MS_DAY == 0:
        return when.date().isoformat()
    return when.isoformat(timespec="seconds").replace("+00:00", "Z")


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not (isinstance(v, float) and math.isnan(v))


def _make(method: str, name: str, xs: list, ys: list, x_kind: str, source: dict) -> dict | None:
    """One series, or None when it is not a series of values (no points, unread items, non-numeric y)."""
    if not ys or len(xs) != len(ys) or len(ys) > MAX_POINTS:
        return None
    if any(x is UNPARSED for x in xs) or any(y is UNPARSED for y in ys):
        return None
    if not all(y is None or _is_num(y) for y in ys):
        return None
    return {"name": str(name), "x": xs, "y": ys, "method": method, "exactness": EMBEDDED, "x_kind": x_kind,
            "n": len(ys), "source": source}


@dataclass
class Extraction:
    series: list[dict] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _sets(ex: Extraction, series: dict | None, reason: str) -> None:
    if series is None:
        ex.skipped.append(reason)
    else:
        ex.series.append(series)


def _first_dict(args: list) -> dict | None:
    for a in args:
        if isinstance(a, dict):
            return a
    return None


def _axis(node: Any) -> dict:
    if isinstance(node, list):
        node = node[0] if node else {}
    return node if isinstance(node, dict) else {}


def _highcharts(opts: dict, source: dict, ex: Extraction) -> None:
    """Highcharts: series[].data as numbers, [x, y] pairs, {x, y} or {name, y} (pie). Categories and datetime x."""
    xaxis = _axis(opts.get("xAxis"))
    categories = xaxis.get("categories") if isinstance(xaxis.get("categories"), list) else None
    datetime_x = xaxis.get("type") == "datetime"
    chart_type = _axis(opts.get("chart")).get("type")
    series_list = opts.get("series")
    if not isinstance(series_list, list):
        ex.skipped.append("highcharts: no series list")
        return
    for i, s in enumerate(series_list):
        name = (s.get("name") if isinstance(s, dict) else None) or f"series {i + 1}"
        data = s.get("data") if isinstance(s, dict) else None
        if not isinstance(data, list):
            ex.skipped.append(f"highcharts {name!r}: no data array")
            continue
        xs, ys, explicit_x = [], [], False
        for j, item in enumerate(data):
            x, y = j, UNPARSED
            if isinstance(item, dict):
                if "y" in item:
                    y = item["y"]
                    x = item["x"] if "x" in item else item.get("name", j)
                    explicit_x = True
            elif isinstance(item, list) and len(item) >= 2:
                x, y = item[0], item[1]
                explicit_x = True
            elif _is_num(item) or item is None:
                y = item
                if categories is not None and j < len(categories):
                    x = categories[j]
            if categories is not None and _is_num(x) and not datetime_x and float(x) == int(x) \
                    and 0 <= int(x) < len(categories):
                x = categories[int(x)]     # numeric x on a categorical axis is a position: the category at it
            xs.append(_iso(x) if datetime_x and explicit_x else x)
            ys.append(y)
        if datetime_x:
            x_kind = "datetime"
        elif any(isinstance(x, str) for x in xs):
            x_kind = "category"
        elif explicit_x:
            x_kind = "number"
        else:
            x_kind = "index"
        _sets(ex, _make("highcharts", name, xs, ys, x_kind,
                        {**source, "series_index": i}), f"highcharts {name!r}: unreadable data")


def _plotly(traces: list, source: dict, ex: Extraction) -> None:
    """Plotly traces: x and y arrays, or labels and values for a pie."""
    for i, tr in enumerate(traces):
        if not isinstance(tr, dict):
            ex.skipped.append(f"plotly trace {i + 1}: not an object")
            continue
        name = tr.get("name") or f"trace {i + 1}"
        if tr.get("type") == "pie":
            xs, ys = tr.get("labels"), tr.get("values")
        else:
            xs, ys = tr.get("x"), tr.get("y")
        if not isinstance(ys, list):
            ex.skipped.append(f"plotly {name!r}: no y array")
            continue
        if not isinstance(xs, list):
            xs = list(range(len(ys)))
        x_kind = "category" if any(isinstance(v, str) for v in xs) else "number"
        if x_kind == "number" and xs and all(isinstance(v, str) for v in xs[:1]):
            x_kind = "category"
        _sets(ex, _make("plotly", str(name), list(xs), list(ys), x_kind, {**source, "trace_index": i}),
              f"plotly {name!r}: unreadable data")


def _vega_lite(spec: dict, source: dict, ex: Extraction) -> None:
    """Vega-Lite with inline data.values: one series per colour group (or one overall) from encoding x and y fields.
    Aggregated or binned encodings are not the page's values, so they are reported and skipped."""
    layers = spec.get("layer") if isinstance(spec.get("layer"), list) else [spec]
    data = spec.get("data") if isinstance(spec.get("data"), dict) else None
    values = (data or {}).get("values")
    if not isinstance(values, list):
        ex.skipped.append("vega-lite: no inline data.values")
        return
    base = layers[0] if layers and isinstance(layers[0], dict) else {}
    enc = base.get("encoding") or spec.get("encoding")
    if not isinstance(enc, dict):
        ex.skipped.append("vega-lite: no encoding")
        return
    x_enc, y_enc = enc.get("x") or {}, enc.get("y") or {}
    color = (enc.get("color") or {}).get("field")
    if not isinstance(x_enc, dict) or not isinstance(y_enc, dict) or not x_enc.get("field") or not y_enc.get("field"):
        ex.skipped.append("vega-lite: encoding has no x and y field")
        return
    if x_enc.get("aggregate") or y_enc.get("aggregate") or x_enc.get("bin") or y_enc.get("bin"):
        ex.skipped.append("vega-lite: aggregated or binned encoding is not an embedded value")
        return
    xf, yf = x_enc["field"], y_enc["field"]
    groups: dict[str, tuple[list, list]] = {}
    for row in values:
        if not isinstance(row, dict):
            continue
        key = str(row.get(color)) if color else (spec.get("title") if isinstance(spec.get("title"), str) else "series 1")
        gx, gy = groups.setdefault(key, ([], []))
        gx.append(row.get(xf))
        gy.append(row.get(yf))
    if not groups:
        ex.skipped.append("vega-lite: no rows")
        return
    x_kind = "category" if any(isinstance(v, str) for v in next(iter(groups.values()))[0]) else "number"
    for key, (gx, gy) in groups.items():
        _sets(ex, _make("vega", key, gx, gy, x_kind, {**source, "fields": [xf, yf]}),
              f"vega-lite {key!r}: unreadable rows")


def _vega_full(spec: dict, source: dict, ex: Extraction) -> None:
    """Vega (not Vega-Lite): the first data set with values, and the x and y fields of the first mark's encode block."""
    data_sets = spec.get("data") if isinstance(spec.get("data"), list) else []
    marks = spec.get("marks") if isinstance(spec.get("marks"), list) else []
    values = next((d.get("values") for d in data_sets if isinstance(d, dict) and isinstance(d.get("values"), list)), None)
    if values is None or not marks or not isinstance(marks[0], dict):
        ex.skipped.append("vega: no inline values or mark encoding")
        return
    encode = marks[0].get("encode") or {}
    block = encode.get("update") or encode.get("enter") or {}
    xf = (block.get("x") or {}).get("field") if isinstance(block.get("x"), dict) else None
    yf = (block.get("y") or {}).get("field") if isinstance(block.get("y"), dict) else None
    if not xf or not yf:
        ex.skipped.append("vega: mark encode has no x and y field")
        return
    xs = [r.get(xf) for r in values if isinstance(r, dict)]
    ys = [r.get(yf) for r in values if isinstance(r, dict)]
    _sets(ex, _make("vega", "series 1", xs, ys, "number", {**source, "fields": [xf, yf]}), "vega: unreadable rows")


def _chartjs(config: dict, source: dict, ex: Extraction) -> None:
    """Chart.js: data.labels for the category axis and data.datasets[].data as numbers or {x, y} points."""
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    labels = data.get("labels") if isinstance(data.get("labels"), list) else None
    datasets = data.get("datasets") if isinstance(data.get("datasets"), list) else []
    if not datasets:
        ex.skipped.append("chart.js: no datasets")
        return
    for i, ds in enumerate(datasets):
        if not isinstance(ds, dict) or not isinstance(ds.get("data"), list):
            ex.skipped.append(f"chart.js dataset {i + 1}: no data array")
            continue
        name = ds.get("label") or f"dataset {i + 1}"
        xs, ys, kind = [], [], "index"
        for j, item in enumerate(ds["data"]):
            if isinstance(item, dict) and "y" in item:
                xs.append(item.get("x"))
                ys.append(item.get("y"))
                kind = "number"
            else:
                xs.append(labels[j] if labels is not None and j < len(labels) else j)
                ys.append(item)
                kind = "category" if labels is not None else "index"
        _sets(ex, _make("chartjs", name, xs, ys, kind, {**source, "series_index": i}),
              f"chart.js {name!r}: unreadable data")


def _echarts(opts: dict, source: dict, ex: Extraction) -> None:
    """ECharts setOption: xAxis.data for categories, series[].data as numbers, [x, y] pairs or {value} items."""
    xaxis = _axis(opts.get("xAxis"))
    cats = xaxis.get("data") if isinstance(xaxis.get("data"), list) else None
    time_x = xaxis.get("type") == "time"
    series_list = opts.get("series")
    if not isinstance(series_list, list):
        ex.skipped.append("echarts: no series list")
        return
    for i, s in enumerate(series_list):
        if not isinstance(s, dict) or not isinstance(s.get("data"), list):
            ex.skipped.append(f"echarts series {i + 1}: no data array")
            continue
        name = s.get("name") or f"series {i + 1}"
        xs, ys, kind = [], [], "index"
        for j, item in enumerate(s["data"]):
            if isinstance(item, dict) and "value" in item:
                item = item["value"]
            if isinstance(item, list) and len(item) >= 2:
                xs.append(_iso(item[0]) if time_x else item[0])
                ys.append(item[1])
                kind = "datetime" if time_x else "number"
            elif _is_num(item) or item is None or item is UNPARSED:
                ys.append(item)
                if cats is not None and j < len(cats):
                    xs.append(cats[j])
                    kind = "category"
                else:
                    xs.append(j)
            else:
                xs.append(j)
                ys.append(UNPARSED)
        _sets(ex, _make("echarts", name, xs, ys, kind, {**source, "series_index": i}),
              f"echarts {name!r}: unreadable data")


_CALLS = (
    ("highcharts", re.compile(r"Highcharts\.(?:stock|map|gantt)?[Cc]hart\s*\(")),
    ("plotly", re.compile(r"Plotly\.(?:newPlot|react|plot)\s*\(")),
    ("chartjs", re.compile(r"new\s+Chart\s*\(")),
    ("echarts", re.compile(r"\.setOption\s*\(")),
    ("vega", re.compile(r"(?:vegaEmbed|vega\.embed|vl\.embed)\s*\(")),
)


def _script_series(script: str, index: int, ex: Extraction) -> None:
    """Library calls found in one inline script. Arguments are read as literals, variables from the same script resolved."""
    table = None
    for method, pattern in _CALLS:
        for m in pattern.finditer(script):
            if table is None:
                table = _variable_table(script)
            source = {"kind": "script", "index": index, "call": m.group(0).rstrip("( ").strip(), "offset": m.start()}
            try:
                args, _ = LiteralReader(script).args(m.end() - 1)
            except _Bad:
                ex.skipped.append(f"{method} call at script {index}: arguments not readable")
                continue
            args = [_resolve(a, table) for a in args]
            if method == "highcharts":
                opts = next((a for a in args if isinstance(a, dict) and isinstance(a.get("series"), list)), None)
                if opts is None:
                    ex.skipped.append("highcharts: no options object with series")
                    continue
                _highcharts(opts, source, ex)
            elif method == "plotly":
                traces = args[1] if len(args) > 1 else None
                if not isinstance(traces, list):
                    ex.skipped.append("plotly: data is not a literal list of traces")
                    continue
                _plotly(traces, source, ex)
            elif method == "chartjs":
                config = _first_dict(args)
                if config is None or not isinstance(config.get("data"), dict):
                    ex.skipped.append("chart.js: config is not a literal object")
                    continue
                _chartjs(config, source, ex)
            elif method == "echarts":
                opts = _first_dict(args)
                if opts is None:
                    ex.skipped.append("echarts: option is not a literal object")
                    continue
                _echarts(opts, source, ex)
            elif method == "vega":
                spec = next((a for a in args if isinstance(a, dict)), None)
                if spec is None:
                    ex.skipped.append("vega: spec is not a literal object")
                    continue
                if "marks" in spec:
                    _vega_full(spec, source, ex)
                else:
                    _vega_lite(spec, source, ex)


def _json_spec_series(text: str, index: int, ex: Extraction) -> None:
    """A Vega-Lite spec held in a JSON script tag."""
    try:
        spec = json.loads(text)
    except ValueError:
        return
    if isinstance(spec, dict) and ("encoding" in spec or "layer" in spec) and "mark" in json.dumps(spec)[:4000]:
        _vega_lite(spec, {"kind": "script", "index": index, "call": "application/json", "offset": 0}, ex)


# -- SVG geometry --------------------------------------------------------------------------------------------------------

_SVG_CHART_CLASS = re.compile(r"highcharts|plotly|echarts|vega|apexcharts|chartjs|chart\.js", re.I)
_NUMERIC_TEXT = re.compile(r"^[+\-−]?[$€£]?\d[\d,]*(?:\.\d+)?%?$")
_TRANSLATE = re.compile(r"translate\(\s*([+-]?[\d.]+(?:e[+-]?\d+)?)(?:[\s,]+([+-]?[\d.]+(?:e[+-]?\d+)?))?\s*\)")
_PATH_TOKEN = re.compile(r"[MmLlHhVvZz]|[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
TICK_TOL_PX = 3.0
MAX_AXIS_RESIDUAL = 0.005          # residual of a tick fit, as a share of the tick value span


def _num_text(text: str) -> float | None:
    """A tick label read as a number: thousands separators, a unicode minus, currency and percent signs are accepted."""
    s = " ".join(text.split()).replace("−", "-")
    if not _NUMERIC_TEXT.match(s):
        return None
    pct = s.endswith("%")
    s = s.rstrip("%").replace(",", "")
    s = re.sub(r"^([+-]?)[$€£]", r"\1", s)
    try:
        return float(s)
    except ValueError:
        return None


def _float_attr(el, name: str, default: float | None = None) -> float | None:
    raw = el.get(name)
    if raw is None:
        return default
    m = re.match(r"\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", str(raw))
    return float(m.group(1)) if m else default


def _offset(el) -> tuple[float, float]:
    """Sum of translate() on the element and its ancestors. Other transforms are not applied (see the module notes)."""
    dx = dy = 0.0
    node = el
    while node is not None and getattr(node, "name", None) not in (None, "[document]"):
        tr = node.get("transform") if hasattr(node, "get") else None
        if tr:
            for m in _TRANSLATE.finditer(str(tr)):
                dx += float(m.group(1))
                dy += float(m.group(2)) if m.group(2) else 0.0
        node = node.parent
    return dx, dy


def _fit_axis(pairs: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Least-squares value = a * pixel + b from (pixel, value) ticks. None when the ticks do not lie on a line."""
    if len(pairs) < 2:
        return None
    pix = [p for p, _ in pairs]
    val = [v for _, v in pairs]
    if max(pix) - min(pix) < 1e-9:
        return None
    n = len(pairs)
    mp, mv = sum(pix) / n, sum(val) / n
    sxx = sum((p - mp) ** 2 for p in pix)
    sxy = sum((p - mp) * (v - mv) for p, v in pairs)
    a = sxy / sxx
    b = mv - a * mp
    span = (max(val) - min(val)) or 1.0
    residual = max(abs(a * p + b - v) for p, v in pairs)
    if residual > MAX_AXIS_RESIDUAL * span + 1e-9:
        return None
    return a, b


def _cluster(items: list[tuple[float, float, str]], axis: int) -> list[list[tuple[float, float, str]]]:
    """Groups labels that share one coordinate (axis 0: same x, a y-axis; axis 1: same y, an x-axis) within TICK_TOL_PX."""
    groups: list[list] = []
    for item in sorted(items, key=lambda it: it[axis]):
        for g in groups:
            if abs(g[0][axis] - item[axis]) <= TICK_TOL_PX:
                g.append(item)
                break
        else:
            groups.append([item])
    return groups


def _svg_axes(labels: list[tuple[float, float, str]]) -> tuple[dict | None, dict | None, list]:
    """(y_map, x_map, categories). A map is (a, b) for value = a * pixel + b. Categories are non-numeric x labels, in order."""
    numeric = [(x, y, _num_text(t)) for x, y, t in labels if _num_text(t) is not None]
    texts = [(x, y, t) for x, y, t in labels if _num_text(t) is None and t.strip()]
    y_map = x_map = None
    # Y ticks share one x (right-aligned labels) and differ in y; X ticks share one y and differ in x.
    column = [g for g in _cluster(numeric, 0) if len({round(it[1], 1) for it in g}) >= 2]
    if column:
        y_map = _fit_axis([(it[1], it[2]) for it in max(column, key=len)])
    row = [g for g in _cluster(numeric, 1) if len({round(it[0], 1) for it in g}) >= 2]
    if row:
        x_map = _fit_axis([(it[0], it[2]) for it in max(row, key=len)])
    cats: list = []
    text_row = max(_cluster(texts, 1), key=len, default=[])
    if len(text_row) >= 2 and x_map is None:
        cats = [t for _, _, t in sorted(text_row, key=lambda it: it[0])]
    return y_map, x_map, cats


def _path_points(d: str) -> list[tuple[float, float]] | None:
    """Vertices of a path made of M, L, H, V and Z (either case). None when it has curves or arcs, which are not data."""
    tokens = _PATH_TOKEN.findall(d or "")
    points: list[tuple[float, float]] = []
    cx = cy = 0.0
    cmd = None
    i = 0

    def take(count: int) -> list[float] | None:
        nonlocal i
        if i + count > len(tokens):
            return None
        vals = []
        for tok in tokens[i:i + count]:
            if tok.isalpha():
                return None
            vals.append(float(tok))
        i += count
        return vals

    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]
            i += 1
            if cmd in "Zz":
                continue
        if cmd is None:
            return None
        if cmd in "Mm":
            v = take(2)
            if v is None:
                return None
            cx, cy = (cx + v[0], cy + v[1]) if cmd == "m" else (v[0], v[1])
            points.append((cx, cy))
            cmd = "l" if cmd == "m" else "L"
        elif cmd in "Ll":
            v = take(2)
            if v is None:
                return None
            cx, cy = (cx + v[0], cy + v[1]) if cmd == "l" else (v[0], v[1])
            points.append((cx, cy))
        elif cmd in "Hh":
            v = take(1)
            if v is None:
                return None
            cx = cx + v[0] if cmd == "h" else v[0]
            points.append((cx, cy))
        elif cmd in "Vv":
            v = take(1)
            if v is None:
                return None
            cy = cy + v[0] if cmd == "v" else v[0]
            points.append((cx, cy))
        else:
            return None
    return points


def _element_name(el) -> str | None:
    """A name for a mark: its own title, aria-label or data-series, else the nearest labelled group."""
    node = el
    while node is not None and getattr(node, "name", None) not in (None, "[document]"):
        title = node.find("title", recursive=False)
        if title is not None and title.get_text(strip=True):
            return " ".join(title.get_text(" ", strip=True).split())
        for attr in ("aria-label", "data-series", "data-name"):
            if node.get(attr):
                return str(node.get(attr)).strip()
        node = node.parent
    return None


def _svg_series(svg, index: int, ex: Extraction) -> None:
    labels: list[tuple[float, float, str]] = []
    for el in svg.find_all("text"):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        dx, dy = _offset(el)
        x = _float_attr(el, "x")
        y = _float_attr(el, "y")
        if x is None or y is None:
            continue
        labels.append((x + dx, y + dy, text))
    y_map, x_map, cats = _svg_axes(labels)
    if y_map is None and x_map is None and not cats:
        ex.skipped.append(f"svg {index}: no readable axis ticks")
        return
    source = {"kind": "svg", "index": index}
    marks: list[tuple[str, list[tuple[float, float]]]] = []
    circles: dict[tuple, list[tuple[float, float]]] = {}
    for el in svg.find_all(["polyline", "path"]):
        dx, dy = _offset(el)
        if el.name == "polyline":
            nums = [float(v) for v in re.findall(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", el.get("points", ""))]
            pts = list(zip(nums[0::2], nums[1::2]))
        else:
            pts = _path_points(el.get("d", ""))
            if pts is None:
                continue
        if len(pts) < 3 or len({round(p[0], 3) for p in pts}) < 2 or len({round(p[1], 3) for p in pts}) < 2:
            continue                        # gridlines, axis lines and single segments are not series
        marks.append((_element_name(el) or f"series {len(marks) + 1}", [(px + dx, py + dy) for px, py in pts]))
    for el in svg.find_all("circle"):
        dx, dy = _offset(el)
        cx, cy = _float_attr(el, "cx"), _float_attr(el, "cy")
        if cx is None or cy is None:
            continue
        circles.setdefault((el.get("fill"), el.get("r")), []).append((cx + dx, cy + dy))
    for key, pts in circles.items():
        if len(pts) >= 3:
            marks.append((f"markers {key[0] or ''}".strip(), sorted(pts)))
    for name, pts in marks:
        if y_map is None:
            ex.skipped.append(f"svg {index} {name!r}: no y-axis scale")
            continue
        ya, yb = y_map
        if cats:
            xs = []
            for px, _ in pts:
                nearest = min(range(len(cats)), key=lambda k: abs(_cat_px(labels, cats[k]) - px))
                xs.append(cats[nearest])
            x_kind = "category"
        elif x_map is not None:
            xa, xb = x_map
            xs = [round(xa * px + xb, 9) for px, _ in pts]
            x_kind = "number"
        else:
            xs = list(range(len(pts)))
            x_kind = "index"
        ys = [round(ya * py + yb, 9) for _, py in pts]
        series = {"name": name, "x": xs, "y": ys, "method": "svg", "exactness": SVG_GEOMETRY, "x_kind": x_kind,
                  "n": len(ys), "source": {**source, "x_scale": "linear" if x_map else None}}
        ex.series.append(series)


def _cat_px(labels: list[tuple[float, float, str]], text: str) -> float:
    for x, _, t in labels:
        if t == text:
            return x
    return 0.0


def extract_page_series(html: str) -> Extraction:
    """Every chart series readable from the page: embedded library data first, then inline SVG geometry.

    Series are returned in page order. ``skipped`` names what was found but not read, so nothing is silently dropped.
    """
    ex = Extraction()
    soup = BeautifulSoup(html or "", "html.parser")
    for i, script in enumerate(soup.find_all("script"), start=1):
        if script.get("src"):
            continue
        kind = (script.get("type") or "").lower()
        text = script.get_text()
        if not text.strip():
            continue
        if kind in ("application/json", "application/vnd.vegalite+json", "application/vnd.vega+json"):
            _json_spec_series(text, i, ex)
            continue
        if kind.startswith("application/ld+json") or kind.startswith("text/template"):
            continue
        _script_series(text, i, ex)
    for i, svg in enumerate(soup.find_all("svg"), start=1):
        ancestors = " ".join(str(p.get("class", "")) + " " + str(p.get("id", "")) for p in svg.parents if hasattr(p, "get"))
        if _SVG_CHART_CLASS.search(ancestors):
            continue                        # a library's own rendering: its data is read from the script above
        _svg_series(svg, i, ex)
    if len(ex.series) > MAX_SERIES:
        ex.skipped.append(f"{len(ex.series) - MAX_SERIES} series beyond the {MAX_SERIES} limit were not kept")
        ex.series = ex.series[:MAX_SERIES]
    return ex


def filter_series(series: list[dict], name: str | None) -> list[dict]:
    """Series whose name contains the given text (case-insensitive). No name keeps all."""
    if not name or not name.strip():
        return list(series)
    needle = name.strip().lower()
    return [s for s in series if needle in str(s.get("name", "")).lower()]
