"""passages: splitting, BM25-lite scoring with a number bonus, best passages and quote checking. Pure functions."""
import pytest

from prophet_agent.research.fetch import Page
from prophet_agent.research.passages import (Passage, verify_figures, best_passages, quote_in_page, score_passages,
                                             split_passages)

PAGE = (
    "Background on the oil market and what drives it over the long term.\n\n"
    "Brent crude is forecast to average 78.5 dollars per barrel in 2027, according to the agency's report.\n\n"
    "Analysts discussed shipping routes, refinery maintenance and currency moves at length this quarter.\n\n"
    "Inventories rose by 3.2 million barrels in the latest week, weighing on prices across the region."
)


def test_short_paragraphs_stay_whole():
    passages = split_passages(PAGE, size=700, overlap=100)
    assert len(passages) == 1 and passages[0].text == PAGE.strip()


def test_packing_respects_size_and_paragraph_boundaries():
    passages = split_passages(PAGE, size=170, overlap=0)
    assert len(passages) >= 3
    for p in passages:
        assert len(p.text) <= 170 + 5
        assert PAGE[p.start:p.start + len(p.text)] == p.text
    assert any("Brent crude" in p.text for p in passages)


def test_oversized_paragraph_is_cut_into_overlapping_windows_covering_all_text():
    long = "alpha beta gamma delta " * 200
    passages = split_passages(long, size=300, overlap=60)
    assert len(passages) > 5
    assert passages[0].start == 0
    assert passages[-1].start + len(passages[-1].text) >= len(long.rstrip()) - 5
    starts = [p.start for p in passages]
    assert starts == sorted(starts)


def test_split_rejects_bad_sizes():
    with pytest.raises(ValueError):
        split_passages("x", size=10)
    with pytest.raises(ValueError):
        split_passages("x", size=100, overlap=100)


def test_passage_with_query_terms_and_numbers_scores_first():
    passages = [Passage("General remarks about markets and sentiment.", 0),
                Passage("Brent crude is forecast to average 78.5 dollars in 2027.", 50),
                Passage("Shipping routes and insurance costs were discussed.", 120)]
    scores = score_passages(passages, ["Brent crude forecast 2027"])
    assert scores.index(max(scores)) == 1
    assert scores[0] == 0 and scores[2] == 0


def test_numbers_get_a_bonus_over_otherwise_equal_passages():
    with_number = Passage("oil price rose to 82 dollars", 0)
    without = Passage("oil price rose to some dollars", 40)
    s_num, s_plain = score_passages([with_number, without], ["oil price dollars"])
    assert s_num > s_plain


def test_best_passages_returns_scored_dicts_in_rank_order():
    page = Page(url="https://x.example", final_url="https://x.example", text=PAGE)
    best = best_passages(page, ["inventories barrels latest week"], k=2, size=150, overlap=0)
    assert best and set(best[0]) == {"text", "score", "start"}
    assert best[0]["text"].startswith("Inventories rose")
    assert best[0]["score"] > 0


def test_best_passages_returns_nothing_when_no_query_matches():
    assert best_passages(PAGE, ["quantum chromodynamics lattice"], k=3) == []
    assert best_passages("", ["oil"], k=3) == []


def test_best_passages_respects_k_and_skips_overlapping_spans():
    text = "oil oil oil price. " * 120
    best = best_passages(text, ["oil price"], k=2, size=200, overlap=50)
    assert len(best) <= 2
    spans = sorted((b["start"], b["start"] + len(b["text"])) for b in best)
    for (a1, b1), (a2, _) in zip(spans, spans[1:]):
        assert b1 <= a2


def test_quote_check_ignores_whitespace_and_typographic_quotes():
    text = "The agency said “prices will ease”\nin   2027."
    assert quote_in_page('The agency said "prices will ease" in 2027.', text)


def test_quote_check_catches_a_changed_figure():
    assert not quote_in_page("average 78.6 dollars", "forecast to average 78.5 dollars")


def test_empty_quote_is_never_verified():
    assert quote_in_page("   ", "anything") is False


def test_imf_example_flags_the_figure_that_is_not_in_the_evidence():
    sources = [{"id": "S1", "passages": [{"text": "JPY 159.47 per USD"}]}]
    assert verify_figures("IMF reports EUR/JPY 162.34", sources) == ["162.34"]


def test_a_figure_found_in_a_passage_is_not_flagged_whatever_the_separators():
    sources = [{"passages": [{"text": "It was 1234.5 dollars, or 78.50 a barrel."}]}]
    assert verify_figures("1,234.50 dollars and 78.5 a barrel", sources) == []


def test_years_and_small_whole_numbers_are_not_checked():
    assert verify_figures("In 2026 there were 3 meetings and 9 cuts", []) == []


def test_each_unverified_figure_is_listed_once_in_order():
    sources = [{"text": "the level was 4.25 per cent"}]
    assert verify_figures("Rates 5.10, then 4.25, then 5.10 and 6.75", sources) == ["5.10", "6.75"]


def test_sources_without_passages_or_text_verify_nothing():
    assert verify_figures("it is 12.5", [{"url": "https://x.gov", "passages": []}, None]) == ["12.5"]
    assert verify_figures("", []) == []
