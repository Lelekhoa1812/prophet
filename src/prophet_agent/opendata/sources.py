"""The registry of no-key open-data sources, and the URL each one is fetched from.

Motivation vs Logic
Motivation: the platform should download public series by itself, with the exact URL, the parameters and the licence key
recorded, and without any way for a caller to aim a download at another host or to slip extra text into a query.
Logic: each source is a fixed URL template and an ordered table of parameters. Every value is checked against its rule
before anything is assembled, unknown or missing names are refused, and the URL is built from the checked, canonical
values only. No source takes a key or a token, so no secret can reach a URL. The licence is named by key only; the
wording lives in docs/prophet/DATASET_LICENCES.md.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from urllib.parse import quote, urlencode

VARIABLE = re.compile(r"[a-z_0-9]{1,40}")
TIMEZONE = re.compile(r"[A-Za-z_]+(?:/[A-Za-z_+-]+)*")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
FLOW = re.compile(r"[A-Z0-9_]{1,30}")
DATA_KEY = re.compile(r"[A-Za-z0-9_.+]{1,120}")
PLACEHOLDER = re.compile(r"\{(\w+)\}")
MAX_SPAN_YEARS = 20


@dataclass(frozen=True)
class Param:
    kind: str          # the type shown by describe(): number, integer, variables, timezone, date, choice, code
    required: bool
    description: str
    check: Callable[[Any], str]   # returns the canonical text for the value, or raises ValueError


@dataclass(frozen=True)
class Source:
    key: str
    title: str
    url_template: str          # {name} placeholders are filled from path parameters
    fixed_query: tuple         # ((name, value), ...) always sent, before the parameters
    params: dict               # name -> Param, in the order they appear in the URL
    licence_key: str           # must name a row in docs/prophet/DATASET_LICENCES.md
    cross_check: Callable[[dict], None] = lambda canonical: None

    @property
    def path_params(self) -> tuple:
        return tuple(PLACEHOLDER.findall(self.url_template))


def _number(low: float, high: float) -> Callable[[Any], str]:
    def check(value: Any) -> str:
        if isinstance(value, bool):
            raise ValueError("must be a number")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError("must be a number") from None
        if not math.isfinite(number):
            raise ValueError("must be a finite number")
        if not low <= number <= high:
            raise ValueError(f"must be between {low:g} and {high:g}")
        return repr(number)
    return check


def _integer(low: int, high: int) -> Callable[[Any], str]:
    def check(value: Any) -> str:
        if isinstance(value, bool):
            raise ValueError("must be a whole number")
        if isinstance(value, int):
            number = value
        elif isinstance(value, str) and re.fullmatch(r"\d{1,4}", value.strip()):
            number = int(value)
        else:
            raise ValueError("must be a whole number")
        if not low <= number <= high:
            raise ValueError(f"must be between {low} and {high}")
        return str(number)
    return check


def _variables(value: Any) -> str:
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)) and all(isinstance(part, str) for part in value):
        names = [part.strip() for part in value]
    else:
        raise ValueError("must be a comma-separated string or a list of variable names")
    if not names or not all(VARIABLE.fullmatch(name) for name in names):
        raise ValueError("each variable name must be 1 to 40 characters from a to z, 0 to 9 and underscore")
    return ",".join(names)


def _timezone(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("must be an IANA timezone name such as Europe/London, or auto")
    if value == "auto" or TIMEZONE.fullmatch(value):
        return value
    raise ValueError("must be an IANA timezone name such as Europe/London, or auto")


def _date(value: Any) -> str:
    if isinstance(value, date) and not isinstance(value, datetime):
        text = value.isoformat()
    elif isinstance(value, str) and ISO_DATE.fullmatch(value):
        text = value
    else:
        raise ValueError("must be a date in YYYY-MM-DD form")
    try:
        date.fromisoformat(text)
    except ValueError:
        raise ValueError("is not a real calendar date") from None
    return text


def _choice(*options: str) -> Callable[[Any], str]:
    def check(value: Any) -> str:
        if isinstance(value, str) and value in options:
            return value
        raise ValueError(f"must be one of: {', '.join(options)}")
    return check


def _code(pattern: re.Pattern, what: str) -> Callable[[Any], str]:
    def check(value: Any) -> str:
        if not isinstance(value, str) or not pattern.fullmatch(value):
            raise ValueError(f"is not a valid {what}")
        if set(value) <= {"."}:   # '.' and '..' are path segments, not codes
            raise ValueError(f"is not a valid {what}")
        return value
    return check


def _years_after(start: date, years: int) -> date:
    try:
        return start.replace(year=start.year + years)
    except ValueError:            # 29 February in a non-leap year: the last day of February is the limit
        return start.replace(year=start.year + years, day=28)


def _check_span(canonical: dict) -> None:
    start = date.fromisoformat(canonical["start_date"])
    end = date.fromisoformat(canonical["end_date"])
    if end < start:
        raise ValueError("end_date is before start_date")
    if end > _years_after(start, MAX_SPAN_YEARS):
        raise ValueError(f"the span from start_date to end_date is more than {MAX_SPAN_YEARS} years")


_OPEN_METEO_COMMON = {
    "latitude": Param("number", True, "Latitude in degrees, -90 to 90.", _number(-90, 90)),
    "longitude": Param("number", True, "Longitude in degrees, -180 to 180.", _number(-180, 180)),
    "daily": Param("variables", False, "Comma-separated daily variable names, for example temperature_2m_max.",
                   _variables),
    "hourly": Param("variables", False, "Comma-separated hourly variable names, for example temperature_2m.",
                    _variables),
    "timezone": Param("timezone", False, "IANA timezone name, or auto for the location's timezone.", _timezone),
    "format": Param("choice", False, "Response format: csv or json.", _choice("csv", "json")),
}

SOURCES: dict[str, Source] = {
    "open_meteo_forecast": Source(
        key="open_meteo_forecast",
        title="Open-Meteo weather forecast for a point (up to 16 days)",
        url_template="https://api.open-meteo.com/v1/forecast",
        fixed_query=(),
        params={
            **_OPEN_METEO_COMMON,
            "forecast_days": Param("integer", False, "Number of forecast days, 1 to 16.", _integer(1, 16)),
        },
        licence_key="open_meteo",
    ),
    "open_meteo_archive": Source(
        key="open_meteo_archive",
        title="Open-Meteo historical weather for a point (span up to 20 years)",
        url_template="https://archive-api.open-meteo.com/v1/archive",
        fixed_query=(),
        params={
            **_OPEN_METEO_COMMON,
            "start_date": Param("date", True, "First day, YYYY-MM-DD.", _date),
            "end_date": Param("date", True, "Last day, YYYY-MM-DD. Not before start_date.", _date),
        },
        licence_key="open_meteo",
        cross_check=_check_span,
    ),
    "ecb_sdmx": Source(
        key="ecb_sdmx",
        title="European Central Bank statistical data, one SDMX flow and key, as CSV",
        url_template="https://data-api.ecb.europa.eu/service/data/{flow}/{key}",
        fixed_query=(("format", "csvdata"),),
        params={
            "flow": Param("code", True, "SDMX dataflow id, for example EXR.", _code(FLOW, "dataflow id")),
            "key": Param("code", True, "SDMX series key, for example D.USD.EUR.SP00.A.",
                         _code(DATA_KEY, "series key")),
        },
        licence_key="ecb",
    ),
}


def get(source_key: str) -> Source:
    if not isinstance(source_key, str) or source_key not in SOURCES:
        raise ValueError(f"unknown source key {source_key!r}; known keys: {', '.join(SOURCES)}")
    return SOURCES[source_key]


def describe() -> list[dict]:
    return [
        {
            "key": source.key,
            "title": source.title,
            "params": {
                name: {"type": param.kind, "required": param.required, "description": param.description}
                for name, param in source.params.items()
            },
        }
        for source in SOURCES.values()
    ]


def normalise(source_key: str, params: dict) -> dict:
    """The canonical text of each parameter that was given (None counts as not given). Raises ValueError with the
    parameter's name when a value is invalid, a name is unknown, or a required name is missing."""
    source = get(source_key)
    if not isinstance(params, dict):
        raise ValueError("params must be a mapping of parameter names to values")
    unknown = sorted(str(name) for name in params if name not in source.params)
    if unknown:
        raise ValueError(f"unknown parameter {unknown[0]!r} for source {source_key!r}; "
                         f"allowed: {', '.join(source.params)}")
    given = {name: value for name, value in params.items() if value is not None}
    canonical: dict = {}
    for name, param in source.params.items():
        if name not in given:
            if param.required:
                raise ValueError(f"missing required parameter {name!r} for source {source_key!r}")
            continue
        try:
            canonical[name] = param.check(given[name])
        except ValueError as exc:
            raise ValueError(f"parameter {name!r} {exc}") from None
    source.cross_check(canonical)
    return canonical


def build_url(source_key: str, params: dict) -> str:
    """The URL for one download. Contains only the fixed query and the checked parameter values, never a secret."""
    source = get(source_key)
    canonical = normalise(source_key, params)
    base = source.url_template.format(**{name: canonical[name] for name in source.path_params})
    query = list(source.fixed_query) + [
        (name, canonical[name]) for name in source.params if name in canonical and name not in source.path_params
    ]
    if not query:
        return base
    return f"{base}?{urlencode(query, safe=',/', quote_via=quote)}"
