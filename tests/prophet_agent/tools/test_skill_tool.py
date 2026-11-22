"""Analyst skills: every playbook parses, stays within the size and code rules, and names only helpers that exist."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from prophet_agent.sandbox import prophet_stats as ps
from prophet_agent.skills import SKILLS_DIR, list_skills, read_skill
from prophet_agent.tools.skill_tool import ANALYST_SKILL_SPEC, _run, skill_tool
from prophet_agent.tools.timesfm_tool import TIMESFM_TOOL_SPEC

EXPECTED = {
    "first-contact-profile", "missing-data-and-gaps", "outliers-and-level-shifts", "zeros-censoring-and-intermittent",
    "seasonality-and-calendar", "smoothing-without-leakage", "choose-model-by-backtest", "timesfm-control-and-flat-forecasts",
    "drivers-and-covariates", "intervals-and-trust-statement", "deliverables-and-handover",
    "knn-and-analogue-imputation", "kalman-and-gradient-features", "bayesian-shrinkage-hierarchies",
    "forecast-combination-and-conformal", "changepoints-and-regimes",
    "web-research-playbook", "market-and-macro-research",
}
PS_NAMES = {"hampel", "stl_outliers", "impute", "smooth", "changepoints", "derivative_features", "censoring_check",
            "seasonal_strength", "stationarity", "backtest_treatment", "isolation_outliers", "Ledger",
            "knn_impute", "kalman_smooth", "savgol_smooth", "rolling_slope_features", "shrink_to_group",
            "combine_forecasts", "conformal_interval", "fuse_report", "causal_canary"}
PL_NAMES = {"load", "profile", "build_series", "prepare_series", "save_series", "dominant_period", "find_season",
            "detect_anomalies", "event_covariate", "holiday_covariate", "list_holidays", "chart", "query", "size_report",
            "read_table", "to_number", "to_datetime"}
HELPER = re.compile(r"\b(ps|pl)\.([A-Za-z_]\w*)")


def _files() -> list[Path]:
    return sorted(SKILLS_DIR.glob("*.md"))


def _front_matter(path: Path) -> tuple[str, str, str, str]:
    """(name, summary, blank separator, body) from the raw file; fails the test if the layout is wrong."""
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[0].startswith("name: "), f"{path.name}: first line must be 'name: <slug>'"
    assert lines[1].startswith("summary: "), f"{path.name}: second line must be 'summary: <one line>'"
    assert lines[2] == "", f"{path.name}: a blank line must follow the front matter"
    return lines[0][len("name: "):].strip(), lines[1][len("summary: "):].strip(), lines[2], "\n".join(lines[3:]).strip()


def _body(path: Path) -> str:
    return _front_matter(path)[3]


def test_expected_slugs_are_all_present():
    assert {p.stem for p in _files()} == EXPECTED


def test_every_file_parses_its_front_matter():
    for path in _files():
        name, summary, _, body = _front_matter(path)
        assert name and summary and body, path.name


def test_names_are_unique_and_match_filenames():
    names = [_front_matter(p)[0] for p in _files()]
    assert len(names) == len(set(names))
    for path in _files():
        assert _front_matter(path)[0] == path.stem


def test_summaries_are_one_line_and_under_110_characters():
    for path in _files():
        summary = _front_matter(path)[1]
        assert 0 < len(summary) < 110, (path.name, len(summary))


def test_body_word_counts_are_within_bounds():
    for path in _files():
        words = len(_body(path).split())
        assert 250 <= words <= 600, (path.name, words)


def test_code_blocks_are_at_most_12_lines():
    for path in _files():
        blocks = re.findall(r"```[a-z]*\n(.*?)```", _body(path), flags=re.S)
        assert blocks, f"{path.name} has no code block"
        for block in blocks:
            assert len(block.rstrip("\n").split("\n")) <= 12, path.name


def test_every_body_ends_with_a_do_not_line():
    for path in _files():
        last = [line for line in _body(path).split("\n") if line.strip()][-1]
        assert last.startswith("Do not"), path.name


def test_every_helper_named_in_a_skill_is_in_the_allowed_lists():
    for path in _files():
        for module, fn in HELPER.findall(_body(path)):
            allowed = PS_NAMES if module == "ps" else PL_NAMES
            assert fn in allowed, f"{path.name} names {module}.{fn}, which is not in the allowed list"


NEW_SKILLS = {
    "knn-and-analogue-imputation", "kalman-and-gradient-features", "bayesian-shrinkage-hierarchies",
    "forecast-combination-and-conformal", "changepoints-and-regimes",
}


def test_every_ps_callable_named_in_a_skill_exists_and_is_callable():
    for path in _files():
        for fn in re.findall(r"\bps\.([A-Za-z_]\w*)", _body(path)):
            assert hasattr(ps, fn) and callable(getattr(ps, fn)), (path.name, fn)


def test_each_new_skill_names_at_least_one_callable():
    for path in _files():
        if path.stem in NEW_SKILLS:
            assert re.findall(r"\bps\.[A-Za-z_]\w*", _body(path)), path.name


def test_list_skills_returns_names_and_summaries():
    listed = list_skills()
    assert {item["name"] for item in listed} == EXPECTED
    assert all(set(item) == {"name", "summary"} for item in listed)


def test_read_skill_returns_the_body_without_front_matter():
    text = read_skill("first-contact-profile")
    assert not text.startswith("name:") and "summary:" not in text.split("\n")[0]
    assert text.startswith("# First contact")


def test_read_skill_unknown_name_lists_the_valid_names():
    with pytest.raises(KeyError) as err:
        read_skill("no-such-skill")
    assert "first-contact-profile" in str(err.value) and "deliverables-and-handover" in str(err.value)


def test_tool_without_a_name_lists_every_skill():
    out = _run({}, "k")
    assert {item["name"] for item in out["skills"]} == EXPECTED


def test_tool_with_a_name_returns_name_summary_and_text():
    out = _run({"name": "outliers-and-level-shifts"}, "k")
    assert out["name"] == "outliers-and-level-shifts"
    assert out["summary"] == next(i["summary"] for i in list_skills() if i["name"] == out["name"])
    assert out["text"] == read_skill("outliers-and-level-shifts")


def test_tool_unknown_name_raises_value_error_with_valid_names():
    with pytest.raises(ValueError, match="valid names") as err:
        _run({"name": "nope"}, "k")
    assert "choose-model-by-backtest" in str(err.value)


def test_tool_refuses_unknown_arguments():
    with pytest.raises(ValueError, match="unknown arguments"):
        _run({"name": "drivers-and-covariates", "path": "x"}, "k")


def test_spec_has_the_same_shape_as_the_timesfm_spec():
    assert set(ANALYST_SKILL_SPEC) == set(TIMESFM_TOOL_SPEC)
    assert ANALYST_SKILL_SPEC["name"] == "analyst_skill"
    assert ANALYST_SKILL_SPEC["title"] == "Analyst skills"
    assert ANALYST_SKILL_SPEC["activity"] == "Reading an analysis playbook"
    assert set(ANALYST_SKILL_SPEC["args"]) == {"name"}


def test_tool_is_named_and_idempotent():
    tool = skill_tool()
    assert tool.name == "analyst_skill"
    assert tool.idempotent is True
