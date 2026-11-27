"""Figures from page text: the model proposes rows, code keeps only what the page states (VN Unemployment, 2026-11-27)."""
import json

import pytest

from prophet_agent.research import text_figures as tf

PAGE = ("Labour force report. The unemployment rate of working-age persons in the third quarter of 2025 was 2.22%, "
        "down 0.02 percentage points from the previous quarter. In the second quarter of 2025 the unemployment rate "
        "was 2.24%. The government expects the unemployment rate to stay below 3% in 2026. "
        "A later table shows the unemployment rate in Q2 2025 at 2.27%.")


def numbered():
    return tf.sentences(PAGE, "unemployment rate")


@pytest.mark.parametrize("text, label, start", [
    ("2025", "2025", "2025-01-01"), ("2025-Q3", "2025-Q3", "2025-07-01"), ("Q3 2025", "2025-Q3", "2025-07-01"),
    ("third quarter of 2025", "2025-Q3", "2025-07-01"), ("March 2025", "2025-03", "2025-03-01"),
    ("2025-03", "2025-03", "2025-03-01"), ("2025-03-31", "2025-03-31", "2025-03-31")])
def test_periods_parse_to_one_dated_period(text, label, start):
    assert tf.parse_period(text) == (label, start)


@pytest.mark.parametrize("text", ["last quarter", "2025-13", "Q5 2025", "2024-2025", ""])
def test_relative_or_impossible_periods_are_refused(text):
    assert tf.parse_period(text) is None


def _row(sentence, quote, period, value, unit="percent"):
    return {"sentence": sentence, "quote": quote, "period": period, "value": value, "unit": unit}


def test_a_row_is_kept_only_when_its_quote_and_number_are_in_the_sentence():
    s = numbered()
    q3 = next(i for i, x in enumerate(s) if "third quarter" in x)
    proposal = {"rows": [
        _row(q3, "the third quarter of 2025 was 2.22%", "2025-Q3", 2.22),
        _row(q3, "the third quarter of 2025 was 2.2%", "2025-Q3", 2.2),          # not word for word
        _row(q3, "the third quarter of 2025 was 2.22%", "2025-Q3", 2.3),          # value not in the quote
        _row(q3, "the third quarter of 2025 was 2.22%", "last quarter", 2.22),    # not a dated period
        _row(99, "x", "2025", 1.0),
    ]}
    kept, rejected = tf.verify(proposal, s)
    assert [(r["period"], r["value"]) for r in kept] == [("2025-Q3", 2.22)]
    reasons = " ".join(r["reason"] for r in rejected)
    assert "word for word" in reasons and "not a number written" in reasons and "dated period" in reasons
    assert "no such sentence" in reasons


def test_two_values_for_one_period_are_a_conflict_and_neither_is_kept():
    s = numbered()
    q2a = next(i for i, x in enumerate(s) if "second quarter" in x)
    q2b = next(i for i, x in enumerate(s) if "Q2 2025" in x)
    kept, rejected = tf.verify({"rows": [_row(q2a, "the unemployment rate was 2.24%", "2025-Q2", 2.24),
                                         _row(q2b, "the unemployment rate in Q2 2025 at 2.27%", "2025-Q2", 2.27)]}, s)
    assert kept == []
    assert all("conflict" in r["reason"] for r in rejected)


def test_thousands_separators_and_the_csv_shape():
    assert tf._numbers("employed 51,234.5 thousand persons") == [51234.5]
    body = tf.to_csv([{"period": "2025-Q3", "date": "2025-07-01", "value": 2.22, "unit": "percent", "entity": "Viet Nam",
                       "quote": "was 2.22%"}]).decode()
    assert body.splitlines()[0] == "period,date,value,unit,entity,quote"


def test_the_tool_stores_verified_rows_with_the_page_as_source(tmp_path):
    from prophet_agent.datasets import DatasetStore
    from prophet_agent.research.fetch import Page
    from prophet_agent.tools.opendata_tools import LiveMounts, OpenDataContext, extract_text_figures_tool

    class Client:
        chain = ["m"]

        def complete(self, models, messages, **kw):
            lines = messages[-1]["content"].split("\n")
            idx = next(int(l[1:l.index("]")]) for l in lines if "third quarter" in l)
            text = json.dumps({"rows": [_row(idx, "the third quarter of 2025 was 2.22%", "2025-Q3", 2.22)]})
            return type("R", (), {"text": text, "model": "m"})()

    page = Page(url="https://example.org/lfs", final_url="https://example.org/lfs", text=PAGE, ok=True,
                sha256="ab", fetched_at="2026-11-27T00:00:00Z", media="text/html")
    store = DatasetStore(tmp_path / "datasets")
    ids = []
    tool = extract_text_figures_tool(OpenDataContext(store=store, attach=ids.append), LiveMounts({}),
                                     fetch_fn=lambda url: page, client=Client())
    out = tool.fn({"url": "https://example.org/lfs", "measure": "unemployment rate"}, "k")
    assert out["row_count"] == 1 and out["first_rows"][0][:3] == ["2025-Q3", "2025-07-01", "2.22"]
    assert out["sandbox_file"].startswith("in/") and ids == [out["dataset_id"]]
    meta = json.loads((store.path(out["dataset_id"]).parent / "meta.json").read_text())
    assert meta["provenance"]["url"] == "https://example.org/lfs"
    assert meta["provenance"]["extraction"]["method"] == "text_figures"
