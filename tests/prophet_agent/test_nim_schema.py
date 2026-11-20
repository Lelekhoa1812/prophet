"""Schema helpers: strip grammar keywords only, parse the server's 'Unimplemented keys' list, validate locally."""
from prophet_agent.nim.schema import is_format_error, strip_schema, unimplemented_keys, validate


def test_keywords_are_stripped_at_every_depth():
    schema = {"$schema": "x", "title": "Top", "type": "array", "uniqueItems": True,
              "items": {"$id": "y", "type": "string", "title": "Item"}}
    assert strip_schema(schema) == {"type": "array", "items": {"type": "string"}}


def test_a_property_named_like_a_keyword_is_kept():
    schema = {"type": "object", "title": "Doc", "properties": {"title": {"type": "string", "title": "T"},
                                                              "uniqueItems": {"type": "boolean"}},
              "required": ["title"]}
    out = strip_schema(schema)
    assert out["properties"] == {"title": {"type": "string"}, "uniqueItems": {"type": "boolean"}}
    assert "title" not in out
    assert out["required"] == ["title"]


def test_definitions_maps_keep_their_names():
    schema = {"$defs": {"title": {"type": "string", "title": "x"}}, "type": "object"}
    assert strip_schema(schema) == {"$defs": {"title": {"type": "string"}}, "type": "object"}


def test_unimplemented_keys_parses_escaped_json_bodies():
    body = '{"detail": "Unimplemented keys: [\\"minItems\\", \\"uniqueItems\\"]"}'
    assert unimplemented_keys(body) == {"minItems", "uniqueItems"}


def test_unimplemented_keys_parses_plain_bodies_and_returns_empty_otherwise():
    assert unimplemented_keys("Unimplemented keys: ['minItems']") == {"minItems"}
    assert unimplemented_keys("something else entirely") == set()
    assert unimplemented_keys("") == set()


def test_format_error_detection():
    assert is_format_error("Grammar error: bad schema")
    assert is_format_error("response_format is not supported")
    assert not is_format_error("model not found")


def test_validate_reports_the_first_error_with_its_path_and_none_when_valid():
    schema = {"type": "object", "properties": {"items": {"type": "array", "minItems": 1}}, "required": ["items"]}
    assert validate({"items": ["a"]}, schema) is None
    reason = validate({"items": []}, schema)
    assert reason is not None and reason.startswith("items:")


def test_validate_flags_an_invalid_schema_rather_than_raising():
    assert validate({}, {"type": "not-a-type"}).startswith("invalid schema")
