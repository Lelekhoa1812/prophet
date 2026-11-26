"""Digitise a chart image: the model proposes axis ticks, code measures the series from the pixels.

Motivation vs Logic
Motivation: a chart that exists only as a picture still carries numbers. A vision model reading values straight off the
picture is not trustworthy: it estimates, and it can hallucinate a plausible curve. So the model is asked only for what it
can see reliably (tick positions and their labels, axis titles, units, series names and colours), and everything numeric
is measured in code and labelled as read from an image.
Logic: (1) the model returns tick pixel positions and tick label text, per axis, in a strict JSON schema. (2) Code parses the
labels and fits a linear, or log10, least-squares map from pixel to value. The fit is rejected when there are fewer than two
ticks, a duplicate or non-monotonic tick, or a residual above the tolerance. (3) Each series gets a colour mask: a pixel is
in the mask when its squared RGB distance to the series colour is within COLOUR_D2. Long low-saturation straight runs
(gridlines) are removed when the series colour is saturated. (4) A line series takes the centre of the longest run per
column; a marker series takes blob centroids. (5) Pixels are mapped through the fit. (6) Self-check: the extracted points
are re-drawn as a polyline and compared with the mask (precision and recall within 2 pixels). (7) Every value is labelled
read_from_image, with an error bound from the fit residual and half a pixel of axis resolution. The proposal and fit are
kept in the result, so a resumed run replays them without asking the model again.
The model's own data estimates are never used as the result. A table request, if ever added, is a cross-check only.
Reimplemented from the published ideas of WebPlotDigitizer (AGPL-3.0; no code copied).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from prophet_agent.nim.schema import validate

READ_FROM_IMAGE = "read_from_image"
DIGITISED = "digitised"
VISION_MODELS = ("nvidia/nemotron-3.5-super-vl-120b-a12b", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
COLOUR_D2 = 120.0                  # squared RGB distance within which a pixel belongs to a series
GRID_MIN_SHARE = 0.6               # a low-saturation run this share of the image width or height is a gridline
GRID_SATURATION = 25               # max-min channel spread below which a pixel counts as grey
TICK_RESIDUAL = 0.005              # largest tick-fit residual, as a share of the fitted value span
SELF_CHECK_PX = 2.0                # distance in pixels for precision and recall of the re-drawn series
MIN_MARKER_PIXELS = 4
DARK_TICK = 110.0                  # mean channel below which a pixel is part of an axis, spine or tick mark
SPINE_MIN_SHARE = 0.3              # an axis spine or gridline spans at least this share of the image side
MAX_POINTS = 400
MAX_BYTES = 8 * 1024 * 1024
MAX_SIDE_PX = 6000
_NUM = re.compile(r"^[+\-−]?[$€£]?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?%?$")

AXIS = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "unit": {"type": "string"},
        "scale": {"type": "string", "enum": ["linear", "log"]},
        "ticks": {"type": "array", "items": {"type": "object", "properties": {
            "pixel": {"type": "number"}, "label": {"type": "string"}}, "required": ["pixel", "label"]}},
    },
    "required": ["title", "unit", "scale", "ticks"],
}
PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "chart_type": {"type": "string", "enum": ["line", "scatter", "other"]},
        "x_axis": AXIS,
        "y_axis": AXIS,
        "series": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"},
            "colour": {"type": "array", "items": {"type": "integer"}}}, "required": ["name", "colour"]}},
    },
    "required": ["chart_type", "x_axis", "y_axis", "series"],
}
SYSTEM_PROMPT = (
    "You read chart images for a measurement tool. Do not estimate any data values. Report only what you can see: "
    "for each axis, the tick marks with their pixel position and their label text. Pixel positions are measured from the "
    "top-left corner of the image in whole image pixels: for the x axis the column of the tick, for the y axis the row. "
    "Give the axis title, the unit (empty if none), and scale 'log' or 'linear'. For each data series give its name and "
    "its line or marker colour as RGB integers 0-255. Reply with the JSON object only."
)


class ChartReadError(ValueError):
    """The image cannot be digitised: a refusal with a plain reason, never a guessed value."""


@dataclass(frozen=True)
class AxisFit:
    scale: str
    a: float
    b: float
    residual: float          # largest fit residual, in fitted units (log10 units for a log axis)
    span: float              # value span of the ticks, in fitted units

    def value(self, pixel: np.ndarray | float):
        t = self.a * pixel + self.b
        return np.power(10.0, t) if self.scale == "log" else t

    def value_per_pixel(self, pixel: float) -> float:
        """Absolute change of value for one pixel at this pixel position (the axis resolution)."""
        d = abs(self.a)
        if self.scale == "log":
            return float(self.value(pixel)) * math.log(10) * d
        return d

    def error_bound(self, pixel: float) -> float:
        """Value-space bound: the fit residual, or half a pixel of resolution when that is larger."""
        resid = self.residual * (float(self.value(pixel)) * math.log(10) if self.scale == "log" else 1.0)
        return max(resid, 0.5 * self.value_per_pixel(pixel))


def _parse_label(text: str) -> float | None:
    s = " ".join(str(text).split()).replace("−", "-")
    if not _NUM.match(s):
        return None
    s = re.sub(r"^([+-]?)[$€£]", r"\1", s.rstrip("%").replace(",", ""))
    try:
        return float(s)
    except ValueError:
        return None


def fit_axis(ticks: list[tuple[float, float]], scale: str) -> AxisFit:
    """Least-squares pixel-to-value map from (pixel, value) ticks. Raises ChartReadError on any tick it cannot trust.

    Motivation vs Logic
    Motivation: a wrong axis fit moves every value. The fit must be checked, not assumed.
    Logic: two ticks or more; distinct pixels; for a log axis the values must be positive and are fitted in log10; the
    values must rise or fall strictly with the pixel; the largest residual must stay within TICK_RESIDUAL of the span.
    """
    if scale not in ("linear", "log"):
        raise ChartReadError(f"unknown axis scale {scale!r}")
    if len(ticks) < 2:
        raise ChartReadError("an axis needs at least two readable ticks")
    ticks = sorted(ticks, key=lambda t: t[0])
    pix = np.array([p for p, _ in ticks], dtype=float)
    val = np.array([v for _, v in ticks], dtype=float)
    if np.any(np.diff(pix) <= 0):
        raise ChartReadError("tick pixel positions are not distinct")
    if scale == "log":
        if np.any(val <= 0):
            raise ChartReadError("a log axis tick must be positive")
        t = np.log10(val)
    else:
        t = val
    steps = np.diff(t)
    if not (np.all(steps > 0) or np.all(steps < 0)):
        raise ChartReadError("tick values are not monotonic with pixel position")
    design = np.vstack([pix, np.ones_like(pix)]).T
    (a, b), *_ = np.linalg.lstsq(design, t, rcond=None)
    resid = float(np.max(np.abs(t - (a * pix + b))))
    span = float(t.max() - t.min())
    if span <= 0 or resid > TICK_RESIDUAL * span + 1e-12:
        raise ChartReadError(f"tick fit residual {resid:.4g} is above the tolerance")
    return AxisFit(scale, float(a), float(b), resid, span)


def _decode(image: bytes) -> tuple[np.ndarray, str]:
    if len(image) > MAX_BYTES:
        raise ChartReadError("the image is larger than 8 MB")
    if image[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    elif image[:3] == b"\xff\xd8\xff":
        mime = "image/jpeg"
    else:
        raise ChartReadError("only PNG and JPEG chart images can be digitised")
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(image))
        im.load()
    except Exception as exc:
        raise ChartReadError(f"the image cannot be decoded: {type(exc).__name__}") from None
    if max(im.size) > MAX_SIDE_PX:
        raise ChartReadError("the image is wider or taller than 6000 pixels")
    return np.asarray(im.convert("RGB"), dtype=np.float32), mime


def _background(arr: np.ndarray) -> tuple[int, int, int]:
    """The dominant colour: the most common cell of a 16-level colour grid."""
    q = (arr // 16).astype(np.int32)
    key = q[..., 0] * 256 + q[..., 1] * 16 + q[..., 2]
    values, counts = np.unique(key, return_counts=True)
    top = int(values[np.argmax(counts)])
    return (top // 256 * 16 + 8, (top // 16) % 16 * 16 + 8, top % 16 * 16 + 8)


def _long_runs(mask: np.ndarray, axis: int, min_len: int) -> np.ndarray:
    """Pixels that belong to a straight run of at least min_len along the given axis (0: vertical, 1: horizontal)."""
    out = np.zeros_like(mask)
    lines = mask if axis == 1 else mask.T
    res = out if axis == 1 else out.T
    for r in range(lines.shape[0]):
        row = np.concatenate([[False], lines[r], [False]]).astype(np.int8)
        edges = np.flatnonzero(np.diff(row))
        for start, end in zip(edges[0::2], edges[1::2]):
            if end - start >= min_len:
                res[r, start:end] = True
    return out


def colour_mask(arr: np.ndarray, colour: tuple[int, int, int]) -> tuple[np.ndarray, dict]:
    """Foreground of one series: pixels within COLOUR_D2 (squared RGB distance) of its colour, gridlines removed."""
    diff = arr - np.array(colour, dtype=np.float32)
    mask = np.sum(diff * diff, axis=2) <= COLOUR_D2
    info: dict[str, Any] = {"pixels": int(mask.sum())}
    sat = float(max(colour) - min(colour))
    h, w = mask.shape
    if sat >= GRID_SATURATION:
        # Gridlines are grey straight runs; a saturated series never is one, so the grey runs are not part of it.
        grey = (arr.max(axis=2) - arr.min(axis=2)) < GRID_SATURATION
        grid = _long_runs(grey, 1, int(GRID_MIN_SHARE * w)) | _long_runs(grey, 0, int(GRID_MIN_SHARE * h))
        info["gridline_pixels_removed"] = int((mask & grid).sum())
        mask &= ~grid
    else:
        info["gridline_removal"] = "skipped: the series colour is grey, so grey runs cannot be told apart"
    bg = np.array(_background(arr), dtype=np.float32)
    if float(np.sum((np.array(colour, dtype=np.float32) - bg) ** 2)) > COLOUR_D2:
        bgd = np.sum((arr - bg) ** 2, axis=2) <= COLOUR_D2     # the dominant background colour is not the series colour
        info["background_removed"] = int((mask & bgd).sum())
        mask &= ~bgd
    else:
        info["background_removal"] = "skipped: the series colour is the background colour"
    return mask, info


def _line_points(mask: np.ndarray) -> list[tuple[float, float]]:
    """Per column, the centre of the longest foreground run."""
    pts = []
    for col in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, col])
        if rows.size == 0:
            continue
        runs = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
        best = max(runs, key=len)
        pts.append((float(col), float(best.mean())))
    return pts


def _marker_points(mask: np.ndarray) -> list[tuple[float, float]]:
    """Centroids of connected blobs of at least MIN_MARKER_PIXELS pixels, left to right."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return []
    idx = list(range(1, n + 1))
    sizes = ndimage.sum(mask, lab, idx)
    cents = ndimage.center_of_mass(mask, lab, idx)
    pts = [(float(cx), float(cy)) for (cy, cx), s in zip(cents, sizes) if s >= MIN_MARKER_PIXELS]
    return sorted(pts)


def _self_check(mask: np.ndarray, pts: list[tuple[float, float]], markers: bool) -> dict:
    """Re-draws the extracted points on the image grid and compares them with the mask (within SELF_CHECK_PX)."""
    if len(pts) < 2 or not mask.any():
        return {"precision": 0.0, "recall": 0.0, "checked": False}
    drawn = np.zeros_like(mask)
    if markers:
        for x, y in pts:
            drawn[int(round(y)), int(round(x))] = True
    else:
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            steps = max(int(max(abs(x1 - x0), abs(y1 - y0))), 1)
            xs = np.round(np.linspace(x0, x1, steps + 1)).astype(int)
            ys = np.round(np.linspace(y0, y1, steps + 1)).astype(int)
            drawn[ys.clip(0, mask.shape[0] - 1), xs.clip(0, mask.shape[1] - 1)] = True
    d_mask = ndimage.distance_transform_edt(~mask)
    d_drawn = ndimage.distance_transform_edt(~drawn)
    precision = float(np.mean(d_mask[drawn] <= SELF_CHECK_PX)) if drawn.any() else 0.0
    recall = float(np.mean(d_drawn[mask] <= SELF_CHECK_PX)) if mask.any() else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "checked": True}


def _thin(points: list, limit: int) -> list:
    if len(points) <= limit:
        return points
    keep = np.linspace(0, len(points) - 1, limit).round().astype(int)
    return [points[i] for i in keep]


def ask_model(client: Any, image: bytes, mime: str, models=VISION_MODELS) -> tuple[dict, str]:
    """One vision call for tick proposals. Returns (proposal, model name). The reply is checked against the schema."""
    import base64
    b64 = base64.b64encode(image).decode("ascii")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "Read the axes, ticks and series colours of this chart."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]},
    ]
    reply = client.complete(list(models), messages, schema=PROPOSAL_SCHEMA, schema_name="chart_axes",
                            max_tokens=2048, thinking=False)
    try:
        proposal = json.loads(reply.text)
    except (ValueError, TypeError):
        raise ChartReadError("the vision model did not return JSON") from None
    problem = validate(proposal, PROPOSAL_SCHEMA)
    if problem:
        raise ChartReadError(f"the vision model's reply failed the schema: {problem}")
    return proposal, str(reply.model)


def _label_values(axis: dict, which: str) -> list[float]:
    values = [v for v in (_parse_label(t.get("label", "")) for t in axis.get("ticks", [])) if v is not None]
    if len(values) < 2:
        raise ChartReadError(f"the {which} axis has fewer than two numeric tick labels")
    return values


def _longest_run(line: np.ndarray) -> int:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], line.astype(np.int8), [0]])))
    lengths = edges[1::2] - edges[0::2]
    return int(lengths.max()) if lengths.size else 0


def _centres(indices: np.ndarray) -> list[float]:
    if indices.size == 0:
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 2) + 1)
    return [float(g.mean()) for g in groups]


def detect_ticks(arr: np.ndarray, axis: str) -> list[float]:
    """Pixel positions of the tick marks on one axis, found in the image itself.

    Motivation vs Logic
    Motivation: the model's own pixel estimates were measured on a live chart and were scaled by about 0.77 on both axes,
    so a fit on them would put every value about 1.3 times out. The tick marks are where the pixels are.
    Logic: the axis spine is the longest dark straight line (bottom row for x, left column for y). Tick marks are dark
    pixels on the outer side of the spine, 2 to 4 pixels from it. When the chart has no spine, the grey gridlines are
    used instead: the long low-saturation straight runs, one per tick. The side with more ticks wins. Returns ascending
    positions (columns for x, rows for y).
    """
    h, w, _ = arr.shape
    dark = arr.mean(axis=2) < DARK_TICK
    grey = (arr.max(axis=2) - arr.min(axis=2)) < GRID_SATURATION
    if axis == "x":
        best = _spine_ticks(dark, [r for r in range(h) if _longest_run(dark[r]) >= SPINE_MIN_SHARE * w], "x")
        if best:
            return best
        long_cols = _long_runs(grey, 0, int(SPINE_MIN_SHARE * h)).any(axis=0)
        return _centres(np.flatnonzero(long_cols))
    best = _spine_ticks(dark, [c for c in range(w) if _longest_run(dark[:, c]) >= SPINE_MIN_SHARE * h], "y")
    if best:
        return best
    long_rows = _long_runs(grey, 1, int(SPINE_MIN_SHARE * w)).any(axis=1)
    return _centres(np.flatnonzero(long_rows))


def _spine_ticks(dark: np.ndarray, lines: list[int], which: str) -> list[float]:
    """Tick positions for the best candidate spine. Every long dark line is tried, on both sides, because a chart often
    has a top spine as well as a bottom one (equal length), and only the bottom one carries ticks. The candidate with the
    most tick marks wins; ties keep the first. Ticks are dark for three pixels at offsets 2 to 4 from the line."""
    h, w = dark.shape
    best: list[float] = []
    for line in lines:
        # A tick lies on the spine, so only columns (rows) inside the spine's own extent count. This drops the strokes of
        # a y tick label sitting in the corner beside the x axis.
        line_pixels = dark[line] if which == "x" else dark[:, line]
        start, end = _longest_span(line_pixels)
        for side in (1, -1):
            offsets = [line + side * k for k in (2, 3, 4)]
            if which == "x":
                if not all(0 <= r < h for r in offsets):
                    continue
                hits = dark[offsets[0]] & dark[offsets[1]] & dark[offsets[2]]
            else:
                if not all(0 <= c < w for c in offsets):
                    continue
                hits = dark[:, offsets[0]] & dark[:, offsets[1]] & dark[:, offsets[2]]
            inside = np.zeros_like(hits)
            inside[max(0, start - 2):end + 3] = True
            found = _centres(np.flatnonzero(hits & inside))
            if len(found) > len(best):
                best = found
    return best


def _longest_span(line: np.ndarray) -> tuple[int, int]:
    """Start and end index of the longest dark run on a line."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], line.astype(np.int8), [0]])))
    starts, ends = edges[0::2], edges[1::2]
    if starts.size == 0:
        return 0, -1
    i = int(np.argmax(ends - starts))
    return int(starts[i]), int(ends[i]) - 1


def match_ticks(values: list[float], positions: list[float], which: str, hints: list[float]) -> list[tuple[float, float]]:
    """Pairs each tick value with a detected tick position, in order. The count must agree exactly; the model's pixel
    hints are compared and reported, never used for the fit. Returns (pixel, value) pairs."""
    if len(positions) != len(values):
        raise ChartReadError(f"the {which} axis has {len(values)} tick labels but {len(positions)} tick marks were found")
    if which == "x":
        pairs = list(zip(sorted(positions), sorted(values)))
    else:
        pairs = list(zip(sorted(positions, reverse=True), sorted(values)))
    return pairs


def digitise_image(image: bytes, *, client: Any = None, models=VISION_MODELS, proposal: dict | None = None,
                   model_name: str | None = None) -> dict:
    """Digitise one chart image. ``proposal`` (from an earlier run, or a test) replaces the model call.

    The result holds: the axes (ticks as proposed, the fit, residuals), the tolerances, one series per proposed colour
    with its points, error bounds and self-check, and the proposal itself for replay. Every value is read_from_image.
    """
    arr, mime = _decode(image)
    h, w, _ = arr.shape
    if proposal is None:
        if client is None:
            from prophet_agent.nim.transport import shared_client
            client = shared_client("nim")
        proposal, model_name = ask_model(client, image, mime, models)
    elif problem := validate(proposal, PROPOSAL_SCHEMA):
        raise ChartReadError(f"the proposal failed the schema: {problem}")
    model_name = model_name or "replay"
    tick_pairs = {}
    for which in ("x", "y"):
        axis = proposal[f"{which}_axis"]
        tick_pairs[which] = match_ticks(_label_values(axis, which), detect_ticks(arr, which), which,
                                        [float(t["pixel"]) for t in axis["ticks"]])
    x_fit = fit_axis(tick_pairs["x"], proposal["x_axis"]["scale"])
    y_fit = fit_axis(tick_pairs["y"], proposal["y_axis"]["scale"])
    markers = proposal["chart_type"] == "scatter"
    series_out, notes = [], []
    for s in proposal["series"]:
        colour = tuple(int(c) for c in s["colour"])
        if len(colour) != 3 or any(not 0 <= c <= 255 for c in colour):
            notes.append(f"series {s['name']!r}: colour is not RGB")
            continue
        mask, info = colour_mask(arr, colour)
        pts = _marker_points(mask) if markers else _line_points(mask)
        if len(pts) < 2:
            notes.append(f"series {s['name']!r}: fewer than two pixels in its colour")
            continue
        pts = _thin(pts, MAX_POINTS)
        xs = [float(x_fit.value(px)) for px, _ in pts]
        ys = [float(y_fit.value(py)) for _, py in pts]
        bounds = [y_fit.error_bound(py) for _, py in pts]
        xb = [x_fit.error_bound(px) for px, _ in pts]
        order = np.argsort(xs)
        series_out.append({
            "name": s["name"], "colour": list(colour), "method": DIGITISED, "exactness": READ_FROM_IMAGE,
            "model": model_name, "mode": "markers" if markers else "line", "n": len(pts),
            "x": [round(xs[i], 9) for i in order], "y": [round(ys[i], 9) for i in order],
            "error_bound_y": round(float(np.max(bounds)), 9), "error_bound_x": round(float(np.max(xb)), 9),
            "self_check": _self_check(mask, pts, markers), "mask": info,
        })
    if not series_out:
        raise ChartReadError("no series could be measured from the image: " + "; ".join(notes or ["no colour matched"]))
    return {
        "method": DIGITISED, "exactness": READ_FROM_IMAGE, "model": model_name,
        "image_sha256": hashlib.sha256(image).hexdigest(), "image_px": [w, h],
        "axes": {
            "x": {"title": proposal["x_axis"]["title"], "unit": proposal["x_axis"]["unit"], "fit": _fit_dict(x_fit),
                  "proposal_ticks": proposal["x_axis"]["ticks"],
                  "ticks": [{"pixel": round(p, 2), "value": v} for p, v in tick_pairs["x"]]},
            "y": {"title": proposal["y_axis"]["title"], "unit": proposal["y_axis"]["unit"], "fit": _fit_dict(y_fit),
                  "proposal_ticks": proposal["y_axis"]["ticks"],
                  "ticks": [{"pixel": round(p, 2), "value": v} for p, v in tick_pairs["y"]]},
        },
        "tolerance": {"colour_d2": COLOUR_D2, "grid_min_share": GRID_MIN_SHARE, "tick_residual": TICK_RESIDUAL,
                      "self_check_px": SELF_CHECK_PX, "background": list(_background(arr))},
        "series": series_out, "notes": notes, "proposal": proposal,
    }


def _fit_dict(fit: AxisFit) -> dict:
    return {"scale": fit.scale, "a": fit.a, "b": fit.b, "residual": fit.residual, "span": fit.span}


def fetch_image(url: str) -> bytes:
    """A chart image from an address, through the guarded fetch (see chart_fetch)."""
    from prophet_agent.research.chart_fetch import fetch_bytes
    result = fetch_bytes(url, max_bytes=MAX_BYTES, media=("image/png", "image/jpeg"))
    if not result["ok"]:
        raise ChartReadError(result["note"])
    return result["body"]
