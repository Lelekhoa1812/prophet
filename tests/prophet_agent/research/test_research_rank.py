"""rank: credibility tiers, dedupe, per-domain cap, recency and ordering. Pure functions, no network."""
from prophet_agent.research.rank import credibility, parse_date, rank_hits
from prophet_agent.research.search import Hit

NOW = 1_795_000_000.0   # a fixed clock so recency checks do not drift
DAY = 86400


def hit(url, rank=1, date=None, title="t"):
    return Hit(title=title, url=url, date=date, rank=rank, backend="bing")


def test_primary_sources_are_tier_one():
    for url in ["https://www.federalreserve.gov/news", "https://ecb.europa.eu/press", "https://www.rba.gov.au/x",
                "https://data.worldbank.org/indicator", "https://www.eia.gov/petroleum", "https://arxiv.org/abs/1"]:
        assert credibility(url)[0] == 1, url


def test_government_and_university_suffixes_are_tier_one():
    assert credibility("https://www.agency.gov.au/x")[0] == 1
    assert credibility("https://www.mit.edu/x")[0] == 1


def test_established_outlets_are_tier_two():
    assert credibility("https://www.reuters.com/markets/oil")[0] == 2
    assert credibility("https://www.bbc.co.uk/news/business")[0] == 2


def test_low_quality_patterns_are_tier_four():
    assert credibility("https://someone.blogspot.com/2026/oil")[0] == 4
    assert credibility("https://medium.com/@x/oil")[0] == 4
    assert credibility("https://reviews.example.com/best-oil-stocks")[0] == 4
    assert credibility("https://example.com/top-10-oil-picks")[0] == 4
    assert credibility("https://example.com/deals?aff=123")[0] == 4


def test_everything_else_is_tier_three_with_a_score():
    assert credibility("https://small-consultancy.example/insights") == (3, 55)


def test_dedupes_by_normalised_url():
    ranked = rank_hits([hit("https://www.reuters.com/a?utm_source=x", rank=1),
                        hit("https://www.reuters.com/a#top", rank=2)], now=NOW)
    assert len(ranked) == 1


def test_caps_each_domain_at_two_hits():
    hits = [hit(f"https://www.reuters.com/story-{i}", rank=i + 1) for i in range(5)]
    hits.append(hit("https://www.ft.com/x", rank=9))
    ranked = rank_hits(hits, now=NOW)
    domains = [h.url for h in ranked]
    assert sum("reuters.com" in u for u in domains) == 2
    assert any("ft.com" in u for u in domains)


def test_orders_by_tier_then_recency_then_search_rank():
    hits = [
        hit("https://random.example/x", rank=1, date="2026-11-20"),
        hit("https://www.reuters.com/old", rank=2, date="2026-10-01"),
        hit("https://www.reuters.com/new", rank=3, date="2026-11-18"),
        hit("https://www.federalreserve.gov/rate", rank=4),
    ]
    order = [h.url for h in rank_hits(hits, now=NOW)]
    assert order == ["https://www.federalreserve.gov/rate", "https://www.reuters.com/new",
                     "https://www.reuters.com/old", "https://random.example/x"]


def test_undated_hits_come_after_dated_ones_within_a_tier():
    hits = [hit("https://a.example/undated", rank=1), hit("https://b.example/dated", rank=2, date="2026-11-01")]
    assert [h.url for h in rank_hits(hits, now=NOW)][0] == "https://b.example/dated"


def test_recency_filter_drops_old_dated_hits_and_keeps_undated():
    hits = [hit("https://a.example/old", date="2020-01-01"), hit("https://b.example/undated"),
            hit("https://c.example/recent", date="2026-11-20")]
    kept = {h.url for h in rank_hits(hits, recency_days=7, now=NOW)}
    assert kept == {"https://b.example/undated", "https://c.example/recent"}


def test_returned_hits_carry_tier_and_score():
    ranked = rank_hits([hit("https://www.bis.org/x")], now=NOW)
    assert (ranked[0].tier, ranked[0].score) == (1, 100)


def test_parse_date_reads_iso_and_english_forms():
    assert parse_date("2026-11-20T10:00:00Z").year == 2026
    assert parse_date("Mar 3, 2026").month == 3
    assert parse_date("3 March 2026").day == 3
    assert parse_date("sometime last week") is None
    assert parse_date(None) is None
