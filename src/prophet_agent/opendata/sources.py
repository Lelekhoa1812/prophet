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

import difflib
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
# SDMX REST names the period filters startPeriod and endPeriod. Measured 2026-11-26: ABS answered start_period with HTTP 403,
# BIS and OECD ignored it and sent the full history, and Eurostat ignored it. All four honour startPeriod.
SDMX_PERIODS = (("start_period", "startPeriod"), ("end_period", "endPeriod"))
PERIOD = re.compile(r"\d{4}(?:-\d{2}(?:-\d{2})?|-Q[1-4])?")
YEAR_SPAN = re.compile(r"\d{4}:\d{4}")
BOE_DATE = re.compile(r"\d{2}/[A-Z][a-z]{2}/\d{4}")
BOE_CODE = re.compile(r"[A-Z0-9]{2,16}")
GFZ_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
SDMX_FLOW = re.compile(r"[A-Za-z0-9._,@-]{1,80}")
VERSION = re.compile(r"[0-9]+\.[0-9]+")
LOWER_CODE = re.compile(r"[A-Za-z0-9_]{1,40}")
DIM_SEL = re.compile(r"[A-Z0-9(),]{1,80}")
ILO_FLOW = re.compile(r"DF_[A-Z0-9_]{3,60}")
DBN_PROVIDER = re.compile(r"[A-Za-z0-9_-]{1,40}")
DBN_CODE = re.compile(r"[A-Za-z0-9_.:@+-]{1,160}")
CUBE = re.compile(r"[a-z0-9]{1,20}")
SERIES_ID = re.compile(r"[A-Za-z0-9._-]{1,80}")
RBA_TABLE = re.compile(r"[a-j]\d{1,2}(?:\.\d{1,2})?")
RBA_SERIES = re.compile(r"[A-Z][A-Z0-9]{2,19}")
FRED_ID = re.compile(r"[A-Z0-9_]{1,40}")


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
    body_kind: str = "csv"     # csv, json, text, zip, xlsx, dbnomics or rba. All but json and text are stored as csv
    local_params: tuple = ()   # checked and stored, never placed on the URL
    defaults: tuple = ()       # ((name, value), ...) filled in when the caller omits an optional parameter
    query_names: tuple = ()    # ((name, name on the URL), ...) where the server's name differs from the parameter's

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


def _month(value: Any) -> str:
    """A calendar month as YYYYMM (AEMO's file names). YYYY-MM is accepted and written without the dash."""
    text = str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) else ""
    text = text.replace("-", "")
    if not re.fullmatch(r"\d{6}", text) or not 1998 <= int(text[:4]) <= 2100 or not 1 <= int(text[4:]) <= 12:
        raise ValueError("must be a month as YYYYMM, for example 202610")
    return text


def _years_after(start: date, years: int) -> date:
    try:
        return start.replace(year=start.year + years)
    except ValueError:            # 29 February in a non-leap year: the last day of February is the limit
        return start.replace(year=start.year + years, day=28)


def _period(value: Any) -> str:
    if not isinstance(value, str) or not PERIOD.fullmatch(value):
        raise ValueError("must be YYYY, YYYY-MM, YYYY-MM-DD, or YYYY-Q1 to YYYY-Q4")
    if len(value) == 4:
        return value
    if value[4:6] == "-Q":
        return value
    try:
        if len(value) == 7:
            date.fromisoformat(value + "-01")
        else:
            date.fromisoformat(value)
    except ValueError:
        raise ValueError("is not a real calendar period") from None
    return value


def _year_span(value: Any) -> str:
    if not isinstance(value, str) or not YEAR_SPAN.fullmatch(value):
        raise ValueError("must be a year span YYYY:YYYY")
    start, end = (int(part) for part in value.split(":"))
    if not 1900 <= start <= end <= 2100:
        raise ValueError("must run from 1900 to 2100, with the end year not before the start year")
    return value


def _wb_country(value: Any) -> str:
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z]{2,3}", value):
        return value.upper()
    raise ValueError("must be a 2 or 3 letter country code")


def _wb_indicator(value: Any) -> str:
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9.]{2,40}", value) and ".." not in value:
        return value.upper()
    raise ValueError("is not a valid indicator code")


def _boe_date(value: Any) -> str:
    if value == "now":
        return "now"
    if not isinstance(value, str) or not BOE_DATE.fullmatch(value):
        raise ValueError("must be DD/Mon/YYYY or now")
    try:
        datetime.strptime(value, "%d/%b/%Y")
    except ValueError:
        raise ValueError("is not a real calendar date") from None
    return value


def _boe_codes(value: Any) -> str:
    if isinstance(value, str):
        codes = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)) and all(isinstance(part, str) for part in value):
        codes = [part.strip() for part in value]
    else:
        raise ValueError("must be a comma-separated string or a list of series codes")
    if not codes or len(codes) > 40 or not all(BOE_CODE.fullmatch(code) for code in codes):
        raise ValueError("must be 1 to 40 series codes of 2 to 16 letters and digits")
    return ",".join(codes)


def _gfz_time(value: Any) -> str:
    if not isinstance(value, str) or not GFZ_TIME.fullmatch(value):
        raise ValueError("must be a UTC time YYYY-MM-DDThh:mm:ssZ")
    try:
        date.fromisoformat(value[:10])
    except ValueError:
        raise ValueError("is not a real calendar date") from None
    hour, minute, second = (int(part) for part in value[11:19].split(":"))
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise ValueError("must be a real UTC time")
    return value


def _check_periods(canonical: dict) -> None:
    start = canonical.get("start_period")
    end = canonical.get("end_period")
    if start and end and len(start) == len(end) and end < start:
        raise ValueError("end_period is before start_period")


def _check_gfz(canonical: dict) -> None:
    if canonical["end"] < canonical["start"]:
        raise ValueError("end is before start")


def _check_boe(canonical: dict) -> None:
    if canonical["Dateto"] == "now":
        return
    start = datetime.strptime(canonical["Datefrom"], "%d/%b/%Y")
    end = datetime.strptime(canonical["Dateto"], "%d/%b/%Y")
    if end < start:
        raise ValueError("Dateto is before Datefrom")


def _check_fred(canonical: dict) -> None:
    if "cosd" in canonical and "coed" in canonical and canonical["coed"] < canonical["cosd"]:
        raise ValueError("coed is before cosd")


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
    "bis_sdmx": Source(
        key="bis_sdmx",
        title="BIS statistics, one dataflow, version and key, as CSV",
        url_template="https://stats.bis.org/api/v2/data/dataflow/BIS/{flow}/{version}/{key}",
        fixed_query=(("format", "csv"),),
        params={
            "flow": Param("code", True, "Dataflow id, for example WS_CBPOL or WS_XRU.", _code(FLOW, "dataflow id")),
            "version": Param("code", True, "Dataflow version, for example 1.0.", _code(VERSION, "version")),
            "key": Param("code", True, "Series key, for example M.US or D.AU..", _code(DATA_KEY, "series key")),
            "start_period": Param("code", False, "First period: YYYY, YYYY-MM, YYYY-MM-DD, or YYYY-Q1.", _period),
            "end_period": Param("code", False, "Last period, same form as start_period.", _period),
        },
        licence_key="bis",
        cross_check=_check_periods,
        query_names=SDMX_PERIODS,
    ),
    "oecd_sdmx": Source(
        key="oecd_sdmx",
        title="OECD statistics, one dataflow and key, as CSV",
        url_template="https://sdmx.oecd.org/public/rest/data/{dataflow}/{key}",
        fixed_query=(("format", "csvfile"),),
        params={
            "dataflow": Param("code", True, "Agency and dataflow, for example OECD.SDD.STES,DSD_STES@DF_CLI.",
                              _code(SDMX_FLOW, "dataflow id")),
            "key": Param("code", True, "Dimension key. Dots separate dimensions and + joins codes.",
                         _code(DATA_KEY, "series key")),
            "start_period": Param("code", False, "First period: YYYY, YYYY-MM, YYYY-MM-DD, or YYYY-Q1.", _period),
            "end_period": Param("code", False, "Last period, same form as start_period.", _period),
        },
        licence_key="oecd",
        cross_check=_check_periods,
        query_names=SDMX_PERIODS,
    ),
    "eurostat_sdmx": Source(
        key="eurostat_sdmx",
        title="Eurostat statistics, one dataset and key, as CSV",
        url_template="https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{dataset}/{key}",
        fixed_query=(("format", "SDMX-CSV"),),
        params={
            "dataset": Param("code", True, "Dataset code, for example prc_hicp_manr or env_wat_use.",
                             _code(LOWER_CODE, "dataset code")),
            "key": Param("code", True, "Dimension key, for example M.RCH_A.CP00.EA.", _code(DATA_KEY, "series key")),
            "start_period": Param("code", False, "First period: YYYY, YYYY-MM, YYYY-MM-DD, or YYYY-Q1.", _period),
            "end_period": Param("code", False, "Last period, same form as start_period.", _period),
        },
        licence_key="eurostat",
        cross_check=_check_periods,
        query_names=SDMX_PERIODS,
    ),
    "abs_sdmx": Source(
        key="abs_sdmx",
        title="ABS statistics, one dataflow and key, as CSV",
        url_template="https://data.api.abs.gov.au/rest/data/{flow}/{key}",
        fixed_query=(("format", "csv"),),
        params={
            "flow": Param("code", True, "Agency, dataflow and version, for example ABS,CPI,1.1.0.",
                          _code(SDMX_FLOW, "dataflow id")),
            "key": Param("code", True, "Dimension key, for example 1.10001.10.50.Q.", _code(DATA_KEY, "series key")),
            "start_period": Param("code", False, "First period: YYYY, YYYY-MM, YYYY-MM-DD, or YYYY-Q1.", _period),
            "end_period": Param("code", False, "Last period, same form as start_period.", _period),
        },
        licence_key="abs",
        cross_check=_check_periods,
        query_names=SDMX_PERIODS,
    ),
    "boe_database": Source(
        key="boe_database",
        title="Bank of England Database, up to 40 series, as CSV",
        url_template="https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp",
        fixed_query=(("csv.x", "yes"), ("UsingCodes", "Y"), ("CSVF", "TN")),
        params={
            "Datefrom": Param("date", True, "First day, DD/Mon/YYYY.", _boe_date),
            "Dateto": Param("date", True, "Last day, DD/Mon/YYYY, or now.", _boe_date),
            "SeriesCodes": Param("code", True, "One to 40 series codes, for example LPMAUZI.", _boe_codes),
        },
        licence_key="boe",
        cross_check=_check_boe,
    ),
    "boc_overnight": Source(
        key="boc_overnight",
        title="Bank of Canada target for the overnight rate, series V39079, as CSV",
        url_template="https://www.bankofcanada.ca/valet/observations/V39079/csv",
        fixed_query=(),
        params={
            "recent": Param("integer", False, "How many latest observations to keep, from 1 to 100. Omit for the full series.",
                            _integer(1, 100)),
        },
        licence_key="boc",
    ),
    "rba_cash_rate": Source(
        key="rba_cash_rate",
        title=("Reserve Bank of Australia statistics table as CSV: date and one column per series ID (default F1.1, the "
               "monthly cash rate target FIRMMCRT; F1 is daily, F13 international policy rates, G1 consumer prices)"),
        url_template="https://www.rba.gov.au/statistics/tables/csv/{table}-data.csv",
        fixed_query=(),
        params={
            "table": Param("code", False, "RBA table code in lower case, for example f1.1, f1, f13, g1 or h5. Defaults to f1.1.",
                           _code(RBA_TABLE, "RBA table code")),
            "series": Param("code", False, "One series ID to keep, for example FIRMMCRT. Omit for every series in the table.",
                            _code(RBA_SERIES, "RBA series ID")),
        },
        licence_key="rba",
        body_kind="rba",
        local_params=("series",),
        defaults=(("table", "f1.1"),),
    ),
    "fhfa_hpi": Source(
        key="fhfa_hpi",
        title="FHFA monthly house price index master file, as CSV",
        url_template="https://www.fhfa.gov/hpi/download/monthly/hpi_master.csv",
        fixed_query=(),
        params={},
        licence_key="fhfa",
    ),
    "fao_food_price": Source(
        key="fao_food_price",
        title="FAO food price index, monthly, as CSV",
        url_template="https://www.fao.org/media/docs/worldfoodsituationlibraries/wfs-library/food_price_indices_data.csv",
        fixed_query=(),
        params={},
        licence_key="fao",
    ),
    "swpc_json": Source(
        key="swpc_json",
        title="NOAA SWPC JSON index file",
        url_template="https://services.swpc.noaa.gov/json/{name}.json",
        fixed_query=(),
        params={
            "name": Param("choice", True, "File name without the extension: planetary_k_index_1m or f107_cm_flux.",
                          _choice("planetary_k_index_1m", "f107_cm_flux")),
        },
        licence_key="nws",
        body_kind="json",
    ),
    "cpc_oni": Source(
        key="cpc_oni",
        title="NOAA CPC Oceanic Nino Index, as a plain-text table",
        url_template="https://cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
        fixed_query=(),
        params={},
        licence_key="nws",
        body_kind="text",
    ),
    "gfz_kp": Source(
        key="gfz_kp",
        title="GFZ geomagnetic index for one window, as JSON",
        url_template="https://kp.gfz.de/app/json/",
        fixed_query=(),
        params={
            "start": Param("date", True, "Window start, UTC, YYYY-MM-DDThh:mm:ssZ.", _gfz_time),
            "end": Param("date", True, "Window end, UTC, YYYY-MM-DDThh:mm:ssZ.", _gfz_time),
            "index": Param("choice", True, "Index name.",
                           _choice("Kp", "ap", "Ap", "Cp", "C9", "Hp30", "Hp60", "ap30", "ap60")),
            "status": Param("choice", False, "def for definitive values, or pre for preliminary.",
                            _choice("def", "pre")),
        },
        licence_key="gfz",
        cross_check=_check_gfz,
        body_kind="json",
    ),
    "hadcrut_monthly": Source(
        key="hadcrut_monthly",
        title="HadCRUT 5.2.0.0 global monthly summary series, as CSV",
        url_template=("https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.2.0.0/analysis/diagnostics/"
                      "HadCRUT.5.2.0.0.analysis.summary_series.global.monthly.csv"),
        fixed_query=(),
        params={},
        licence_key="hadcrut",
    ),
    # Panel round 3 GISTEMP (2026-11-26): the planner asked for source "gistemp" and was refused as unknown.
    "gistemp_monthly": Source(
        key="gistemp_monthly",
        title="NASA GISTEMP v4 global land-ocean monthly temperature anomaly (one row per year, Jan to Dec), as CSV",
        url_template="https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
        fixed_query=(),
        params={},
        licence_key="nasa_giss",
    ),
    # Panel round 4 (NEM, 2026-11-27): aemo.com.au pages answer with a bot check, and the NEMWEB dispatch file the planner
    # found holds unit prices, not demand. This file is reachable and holds 5-minute TOTALDEMAND per region (measured).
    "aemo_price_demand": Source(
        key="aemo_price_demand",
        # Panel rounds 9 to 11 (NEM): the month file ends at 00:00, which closes the day before, so a pandas daily mean
        # made that one reading the last "day"; round 11 also fetched two months, 58 days, too few to calibrate.
        title="AEMO National Electricity Market price and demand for one region and one month (5-minute TOTALDEMAND in "
              "MW and RRP), as CSV. SETTLEMENTDATE is the end of each interval. For a daily forecast fetch at least 6 "
              "months and build days with pl.build_series(df, 'TOTALDEMAND', 'SETTLEMENTDATE', freq='D', agg='mean'), "
              "which leaves out a partial day",
        url_template="https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{month}_{region}.csv",
        fixed_query=(),
        params={
            "month": Param("month", True, "Month as YYYYMM, for example 202610. One file per month.", _month),
            "region": Param("choice", True, "NEM region: NSW1, QLD1, VIC1, SA1 or TAS1.",
                            _choice("NSW1", "QLD1", "VIC1", "SA1", "TAS1")),
        },
        licence_key="aemo",
    ),
    "snb_cube": Source(
        key="snb_cube",
        title="Swiss National Bank data cube, as CSV",
        url_template="https://data.snb.ch/api/cube/{cube}/data/csv/{lang}",
        fixed_query=(),
        params={
            "cube": Param("code", True, "Cube id, for example devkum.", _code(CUBE, "cube id")),
            "lang": Param("choice", True, "Language of the file: de, en, fr or it.", _choice("de", "en", "fr", "it")),
            "dimSel": Param("code", True, "Dimension selection, for example D0(M1),D1(EUR1,USD1).",
                            _code(DIM_SEL, "dimension selection")),
        },
        licence_key="snb",
    ),
    "world_bank": Source(
        key="world_bank",
        title="World Bank indicator for one country, as JSON",
        url_template="https://api.worldbank.org/v2/country/{country}/indicator/{indicator}",
        fixed_query=(("format", "json"),),
        params={
            "country": Param("code", True, "Country code of 2 or 3 letters, for example US.", _wb_country),
            "indicator": Param("code", True, "Indicator code, for example FP.CPI.TOTL.ZG.", _wb_indicator),
            "date": Param("code", False, "Year span YYYY:YYYY.", _year_span),
            "per_page": Param("integer", False, "Observations per page, 1 to 2000. Defaults to 1000.",
                              _integer(1, 2000)),
        },
        licence_key="world_bank",
        body_kind="json",
        defaults=(("per_page", 1000),),
    ),
    "eia_bulk": Source(
        key="eia_bulk",
        title="EIA bulk file, one series taken out of the ZIP",
        url_template="https://www.eia.gov/opendata/bulk/{dataset}.zip",
        fixed_query=(),
        params={
            "dataset": Param("choice", True, "Bulk dataset: ELEC, NG or EBA.", _choice("ELEC", "NG", "EBA")),
            "series_id": Param("code", True, "Series id inside the file, for example ELEC.PRICE.US-ALL.A.",
                               _code(SERIES_ID, "series id")),
        },
        licence_key="eia",
        body_kind="zip",
        local_params=("series_id",),
    ),
    "world_bank_pink_sheet": Source(
        key="world_bank_pink_sheet",
        title="World Bank monthly commodity price workbook",
        url_template=("https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/"
                      "CMO-Historical-Data-Monthly.xlsx"),
        fixed_query=(),
        params={},
        licence_key="world_bank",
        body_kind="xlsx",
    ),
    # Panel CPI (2026-11-26): the planner asked download_open_data for source "fred" and was refused as unknown, then spent
    # its budget on web research. FRED's graph CSV needs no key; the series code is the only required value.
    "fred": Source(
        key="fred",
        title="FRED series (St. Louis Fed) as CSV, by series code, for example CPIAUCSL, UNRATE, SP500, DCOILWTICO",
        url_template="https://fred.stlouisfed.org/graph/fredgraph.csv",
        fixed_query=(),
        params={
            "id": Param("code", True, "One FRED series code, for example CPIAUCSL.", _code(FRED_ID, "FRED series code")),
            "cosd": Param("date", False, "First observation date, YYYY-MM-DD. Omit for the full history.", _date),
            "coed": Param("date", False, "Last observation date, YYYY-MM-DD. Omit for the latest.", _date),
        },
        licence_key="fred",
        cross_check=_check_fred,
    ),
    # VN Unemployment notebook (2026-11-27): the national office answered with a challenge page and OECD has no Vietnam
    # series, so the mission found nothing in 28 steps. ILOSTAT publishes labour statistics for every country, keyless, as
    # SDMX; Vietnam's unemployment rate is quarterly from 2011-Q1 there (measured 2026-11-27: 56 quarters to 2024-Q4).
    "ilostat_sdmx": Source(
        key="ilostat_sdmx",
        title=("ILOSTAT (International Labour Organization) labour statistics for any country, as CSV: unemployment, "
               "employment, labour force, wages, hours. Unemployment rate: dataflow DF_UNE_DEAP_SEX_AGE_RT with key "
               "<ISO3>.<A, Q or M>.UNE_DEAP_RT.SEX_T.AGE_YTHADULT_YGE15, for example VNM.Q.UNE_DEAP_RT.SEX_T.AGE_YTHADULT_YGE15"),
        url_template="https://sdmx.ilo.org/rest/data/ILO,{dataflow},1.0/{key}",
        fixed_query=(("format", "csv"),),
        params={
            "dataflow": Param("code", True, "ILOSTAT dataflow id, for example DF_UNE_DEAP_SEX_AGE_RT.",
                              _code(ILO_FLOW, "ILOSTAT dataflow id")),
            "key": Param("code", True, "Dimension key: country (ISO3), frequency (A, Q or M), measure, sex, age, "
                                       "for example VNM.Q.UNE_DEAP_RT.SEX_T.AGE_YTHADULT_YGE15.",
                         _code(DATA_KEY, "series key")),
            "start_period": Param("code", False, "First period: YYYY, YYYY-MM or YYYY-Q1.", _period),
            "end_period": Param("code", False, "Last period, same form as start_period.", _period),
        },
        licence_key="ilo",
        cross_check=_check_periods,
        query_names=SDMX_PERIODS,
    ),
    "dbnomics": Source(
        key="dbnomics",
        title=("DBnomics mirror of 80+ providers (IMF, ILO, OECD, Eurostat, World Bank, central banks, national statistics "
               "offices): one series by provider, dataset and series code, as CSV of period and value. Find the codes with "
               "search_open_data. The mirror can lag its provider; the result says when it was indexed"),
        url_template="https://api.db.nomics.world/v22/series/{provider}/{dataset}/{series}",
        fixed_query=(("observations", "1"),),
        params={
            "provider": Param("code", True, "Provider code, for example IMF, ILO, OECD, Eurostat, WB.",
                              _code(DBN_PROVIDER, "provider code")),
            "dataset": Param("code", True, "Dataset code, for example WEO:latest or UNE_DEAP_SEX_AGE_RT.",
                             _code(DBN_CODE, "dataset code")),
            "series": Param("code", True, "Series code, for example VNM.LUR.pcent_total_labor_force.",
                            _code(DBN_CODE, "series code")),
        },
        licence_key="dbnomics",
        body_kind="dbnomics",
    ),
}

# Names a planner reaches for that are not the registry's keys (VN Unemployment: 'oecd', then 'flow' and 'country'). Each
# resolves to the registry's own name, and the result says it did, so a near miss costs no step.
SOURCE_ALIASES = {
    "oecd": "oecd_sdmx", "ecb": "ecb_sdmx", "bis": "bis_sdmx", "eurostat": "eurostat_sdmx", "abs": "abs_sdmx",
    "ilo": "ilostat_sdmx", "ilostat": "ilostat_sdmx", "ilo_sdmx": "ilostat_sdmx", "worldbank": "world_bank",
    "wb": "world_bank", "wdi": "world_bank", "world_bank_api": "world_bank", "fred_csv": "fred", "db_nomics": "dbnomics",
}
_SDMX_PARAM_ALIASES = {"flow": "dataflow", "dataflow_id": "dataflow", "flow_ref": "dataflow", "series_key": "key",
                       "startPeriod": "start_period", "endPeriod": "end_period", "start": "start_period",
                       "end": "end_period"}
PARAM_ALIASES = {
    **{key: _SDMX_PARAM_ALIASES for key in ("oecd_sdmx", "ecb_sdmx", "bis_sdmx", "eurostat_sdmx", "abs_sdmx",
                                            "ilostat_sdmx")},
    "fred": {"series_id": "id", "series": "id", "code": "id", "start": "cosd", "end": "coed", "observation_start": "cosd",
             "observation_end": "coed"},
    "world_bank": {"country_code": "country", "iso3": "country", "indicator_id": "indicator", "code": "indicator"},
    "dbnomics": {"provider_code": "provider", "dataset_code": "dataset", "series_code": "series"},
}


def resolve(source_key: Any, params: Any) -> tuple[str, Any, list[str]]:
    """The registry's own source key and parameter names for a request that used a known alias, with a note per change.
    Anything not an alias is returned as given, so get() and normalise() still refuse it with their own message."""
    notes: list[str] = []
    key = source_key
    if isinstance(key, str) and key not in SOURCES:
        alias = SOURCE_ALIASES.get(key.strip().lower())
        if alias:
            notes.append(f"source {key!r} is {alias!r} in this registry")
            key = alias
    if isinstance(params, dict) and isinstance(key, str) and key in SOURCES:
        names = PARAM_ALIASES.get(key, {})
        allowed = SOURCES[key].params
        fixed = {}
        for name, value in params.items():
            target = names.get(name) if name not in allowed else None
            if target and target not in params:
                notes.append(f"parameter {name!r} is {target!r} for {key}")
                fixed[target] = value
            else:
                fixed[name] = value
        params = fixed
    return key, params, notes


def get(source_key: str) -> Source:
    if not isinstance(source_key, str) or source_key not in SOURCES:
        close = difflib.get_close_matches(str(source_key).lower(), list(SOURCES), n=2, cutoff=0.5)
        hint = f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
        raise ValueError(f"unknown source key {source_key!r}.{hint} Known keys: {', '.join(SOURCES)}. To find a series "
                         "whose source you do not know, call search_open_data")
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
                         f"allowed: {', '.join(source.params)} ({source.title})")
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
    for name, value in source.defaults:
        if name not in canonical:
            canonical[name] = source.params[name].check(value)
    # Open-Meteo answers JSON when no format is named, while the store reads a download without one as CSV. The default is
    # therefore written into the request and the provenance: a download always says which format it asked for.
    if "format" in source.params and "format" not in canonical:
        canonical["format"] = "csv"
    source.cross_check(canonical)
    return canonical


def build_url(source_key: str, params: dict) -> str:
    """The URL for one download. Contains only the fixed query and the checked parameter values, never a secret."""
    source = get(source_key)
    canonical = normalise(source_key, params)
    base = source.url_template.format(**{name: canonical[name] for name in source.path_params})
    on_url = dict(source.query_names)
    query = list(source.fixed_query) + [
        (on_url.get(name, name), canonical[name]) for name in source.params
        if name in canonical and name not in source.path_params and name not in source.local_params
    ]
    if not query:
        return base
    return f"{base}?{urlencode(query, safe=',/()', quote_via=quote)}"
