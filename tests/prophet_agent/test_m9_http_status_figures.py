"""Panel NGSTORAGE (2026-11-26): HTTP status codes in an answer's account of failed downloads are not forecast figures."""
from prophet_agent.runtime import grounding as g


def test_status_codes_after_a_fetch_verb_or_in_a_list_are_not_figures():
    assert g.figures("The history files return 403/404 and were skipped.") == set() or not g.figures(
        "The history files return 403/404 and were skipped.")
    assert not g.figures("Downloads failed with 403 or 404.")
    assert not g.figures("The CSV link returned a 404.")


def test_ordinary_counts_after_a_verb_are_still_figures():
    found = {str(f) for f in g.figures("The page returned 250 points and 403 rows.")}
    assert "250" in found and "403" in found
