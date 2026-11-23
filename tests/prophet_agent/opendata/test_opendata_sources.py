"""Source registry: describe(), URL building for each source, and every validation rule. No network."""
import pytest

from prophet_agent.opendata import sources

FORECAST_OK = {"latitude": 52.52, "longitude": 13.41}
ARCHIVE_OK = {"latitude": 52, "longitude": 13, "start_date": "2024-01-01", "end_date": "2024-01-31"}


def test_describe_lists_the_three_sources_with_their_parameters():
    listed = {item["key"]: item for item in sources.describe()}
    assert set(listed) == {"open_meteo_forecast", "open_meteo_archive", "ecb_sdmx"}
    forecast = listed["open_meteo_forecast"]["params"]
    assert set(forecast) == {"latitude", "longitude", "daily", "hourly", "timezone", "format", "forecast_days"}
    assert forecast["latitude"] == {"type": "number", "required": True,
                                    "description": "Latitude in degrees, -90 to 90."}
    assert forecast["forecast_days"]["required"] is False
    archive = listed["open_meteo_archive"]["params"]
    assert archive["start_date"]["required"] and archive["end_date"]["required"]
    assert "forecast_days" not in archive
    assert set(listed["ecb_sdmx"]["params"]) == {"flow", "key"}
    assert all(item["required"] for item in listed["ecb_sdmx"]["params"].values())


def test_forecast_url_uses_registry_order_and_canonical_values():
    url = sources.build_url("open_meteo_forecast", {
        "latitude": 52.52, "longitude": 13.41, "daily": "temperature_2m_max,precipitation_sum",
        "timezone": "Europe/Berlin", "format": "csv", "forecast_days": 7})
    assert url == ("https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41"
                   "&daily=temperature_2m_max,precipitation_sum&timezone=Europe/Berlin&format=csv&forecast_days=7")


def test_forecast_url_accepts_a_list_of_variables_and_ignores_none():
    url = sources.build_url("open_meteo_forecast", {
        "latitude": 52, "longitude": 13, "hourly": ["temperature_2m", "relative_humidity_2m"], "timezone": None})
    assert url == ("https://api.open-meteo.com/v1/forecast?latitude=52.0&longitude=13.0"
                   "&hourly=temperature_2m,relative_humidity_2m")


def test_forecast_url_with_only_a_point_has_no_optional_parts():
    assert sources.build_url("open_meteo_forecast", FORECAST_OK) == \
        "https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41"


def test_archive_url_takes_both_dates_and_its_own_host():
    url = sources.build_url("open_meteo_archive", {**ARCHIVE_OK, "daily": "temperature_2m_max"})
    assert url == ("https://archive-api.open-meteo.com/v1/archive?latitude=52.0&longitude=13.0"
                   "&daily=temperature_2m_max&start_date=2024-01-01&end_date=2024-01-31")


def test_archive_span_of_exactly_twenty_years_is_allowed():
    sources.build_url("open_meteo_archive", {**ARCHIVE_OK, "start_date": "2004-02-29", "end_date": "2024-02-29"})
    sources.build_url("open_meteo_archive", {**ARCHIVE_OK, "start_date": "2004-01-01", "end_date": "2024-01-01"})


def test_ecb_url_puts_flow_and_key_in_the_path_and_fixes_the_format():
    assert sources.build_url("ecb_sdmx", {"flow": "EXR", "key": "D.USD.EUR.SP00.A"}) == \
        "https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?format=csvdata"
    assert sources.build_url("ecb_sdmx", {"flow": "EXR", "key": "D.USD+GBP.EUR_X"}) == \
        "https://data-api.ecb.europa.eu/service/data/EXR/D.USD+GBP.EUR_X?format=csvdata"


@pytest.mark.parametrize("source_key, params, message", [
    ("open_meteo_forecast", {"longitude": 13}, "missing required parameter 'latitude'"),
    ("open_meteo_forecast", {"latitude": 52}, "missing required parameter 'longitude'"),
    ("open_meteo_archive", {"latitude": 52, "longitude": 13, "start_date": "2024-01-01"},
     "missing required parameter 'end_date'"),
    ("ecb_sdmx", {"flow": "EXR"}, "missing required parameter 'key'"),
    ("open_meteo_forecast", {"latitude": 91, "longitude": 13}, "must be between -90 and 90"),
    ("open_meteo_forecast", {"latitude": -90.5, "longitude": 13}, "must be between -90 and 90"),
    ("open_meteo_forecast", {"latitude": 52, "longitude": 181}, "must be between -180 and 180"),
    ("open_meteo_forecast", {"latitude": True, "longitude": 13}, "must be a number"),
    ("open_meteo_forecast", {"latitude": "north", "longitude": 13}, "must be a number"),
    ("open_meteo_forecast", {"latitude": float("nan"), "longitude": 13}, "must be a finite number"),
    ("open_meteo_forecast", {"latitude": float("inf"), "longitude": 13}, "must be a finite number"),
    ("open_meteo_forecast", {**FORECAST_OK, "forecast_days": 0}, "must be between 1 and 16"),
    ("open_meteo_forecast", {**FORECAST_OK, "forecast_days": 17}, "must be between 1 and 16"),
    ("open_meteo_forecast", {**FORECAST_OK, "forecast_days": 3.5}, "must be a whole number"),
    ("open_meteo_forecast", {**FORECAST_OK, "forecast_days": True}, "must be a whole number"),
    ("open_meteo_forecast", {**FORECAST_OK, "daily": "Temperature"}, "each variable name must be"),
    ("open_meteo_forecast", {**FORECAST_OK, "daily": "temp&format=json"}, "each variable name must be"),
    ("open_meteo_forecast", {**FORECAST_OK, "daily": "a" * 41}, "each variable name must be"),
    ("open_meteo_forecast", {**FORECAST_OK, "daily": "temp,,rain"}, "each variable name must be"),
    ("open_meteo_forecast", {**FORECAST_OK, "hourly": ["ok_name", 5]}, "must be a comma-separated string"),
    ("open_meteo_forecast", {**FORECAST_OK, "timezone": "Europe/London\n"}, "must be an IANA timezone"),
    ("open_meteo_forecast", {**FORECAST_OK, "timezone": "../etc/passwd"}, "must be an IANA timezone"),
    ("open_meteo_forecast", {**FORECAST_OK, "timezone": 5}, "must be an IANA timezone"),
    ("open_meteo_forecast", {**FORECAST_OK, "format": "xml"}, "must be one of: csv, json"),
    ("open_meteo_archive", {**ARCHIVE_OK, "start_date": "2024-1-1"}, "must be a date in YYYY-MM-DD form"),
    ("open_meteo_archive", {**ARCHIVE_OK, "start_date": "2023-02-29"}, "is not a real calendar date"),
    ("open_meteo_archive", {**ARCHIVE_OK, "end_date": "2023-12-31"}, "end_date is before start_date"),
    ("open_meteo_archive", {**ARCHIVE_OK, "end_date": "2024-01-02", "start_date": "2004-01-01"},
     "more than 20 years"),
    ("ecb_sdmx", {"flow": "exr", "key": "D"}, "is not a valid dataflow id"),
    ("ecb_sdmx", {"flow": "EXR/../X", "key": "D"}, "is not a valid dataflow id"),
    ("ecb_sdmx", {"flow": "EXR", "key": ".."}, "is not a valid series key"),
    ("ecb_sdmx", {"flow": "EXR", "key": ""}, "is not a valid series key"),
    ("ecb_sdmx", {"flow": "EXR", "key": "D?format=xml"}, "is not a valid series key"),
    ("ecb_sdmx", {"flow": "EXR", "key": "D/X"}, "is not a valid series key"),
])
def test_every_validation_rule_refuses_with_the_parameter_named(source_key, params, message):
    with pytest.raises(ValueError, match=message):
        sources.build_url(source_key, params)


def test_unknown_parameter_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown parameter 'api_key' for source 'open_meteo_forecast'"):
        sources.build_url("open_meteo_forecast", {**FORECAST_OK, "api_key": "abc"})


def test_unknown_source_key_is_refused():
    with pytest.raises(ValueError, match="unknown source key 'nope'"):
        sources.build_url("nope", {})
    with pytest.raises(ValueError, match="unknown source key"):
        sources.build_url(None, {})


def test_params_must_be_a_mapping():
    with pytest.raises(ValueError, match="mapping"):
        sources.build_url("open_meteo_forecast", ["latitude", 52])


def test_every_source_names_a_licence_key():
    assert {source.key: source.licence_key for source in sources.SOURCES.values()} == {
        "open_meteo_forecast": "open_meteo", "open_meteo_archive": "open_meteo", "ecb_sdmx": "ecb"}
