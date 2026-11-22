"""rank: credibility tiers, dedupe, per-domain cap, relevance, recency and ordering. Pure functions, no network."""
import pytest

from prophet_agent.research.rank import credibility, is_primary, parse_date, rank_hits, rank_report
from prophet_agent.research.search import Hit

NOW = 1_795_000_000.0   # a fixed clock (about 24 November 2026) so recency checks do not drift
DAY = 86400


def hit(url, rank=1, date=None, title="t", snippet=""):
    return Hit(title=title, url=url, date=date, rank=rank, backend="bing", snippet=snippet)


@pytest.mark.parametrize("url", [
    "https://www.federalreserve.gov/news", "https://www.ecb.europa.eu/press", "https://ec.europa.eu/eurostat/web/x",
    "https://www.boj.or.jp/en/mopo", "https://www.bankofengland.co.uk/x", "https://www.bankofcanada.ca/x",
    "https://www.rba.gov.au/x", "https://www.snb.ch/en", "https://www.bundesbank.de/x", "https://www.bok.or.kr/x",
    "https://www.eia.gov/petroleum", "https://www.opec.org/x", "https://www.iea.org/x", "https://www.imf.org/x",
    "https://www.bis.org/x", "https://data.worldbank.org/indicator", "https://www.oecd.org/x", "https://www.census.gov/x",
    "https://www.bls.gov/cpi", "https://www.bea.gov/news", "https://www.stat.go.jp/english", "https://www.e-stat.go.jp/x",
    "https://www.mof.go.jp/x", "https://www.ons.gov.uk/x", "https://www.abs.gov.au/x", "https://www.stats.govt.nz/x",
    "https://www.statcan.gc.ca/x", "https://www.stats.gov.cn/x", "https://www.pboc.gov.cn/x", "https://www.insee.gouv.fr/x",
    "https://fred.stlouisfed.org/series/DCOILWTICO",
])
def test_official_publishers_are_tier_one(url):
    assert credibility(url)[0] == 1, url
    assert is_primary(url)


@pytest.mark.parametrize("url", [
    "https://arxiv.org/abs/2601.00001", "https://www.mit.edu/x", "https://www.ox.ac.uk/research", "https://www.lse.ac.uk/x",
])
def test_academic_and_preprint_pages_are_tier_two_and_not_primary(url):
    assert credibility(url)[0] == 2, url
    assert not is_primary(url)


@pytest.mark.parametrize("url", [
    "https://www.reuters.com/markets/oil", "https://www.bloomberg.com/news/x", "https://www.ft.com/content/x",
    "https://markets.ft.com/data/x", "https://www.wsj.com/x", "https://apnews.com/x", "https://www.cnbc.com/x",
    "https://www.nikkei.com/x", "https://www.economist.com/x", "https://www.bbc.co.uk/news/business",
])
def test_wires_and_financial_press_are_tier_two(url):
    assert credibility(url)[0] == 2, url


def test_low_quality_patterns_are_tier_four():
    assert credibility("https://someone.blogspot.com/2026/oil")[0] == 4
    assert credibility("https://medium.com/@x/oil")[0] == 4
    assert credibility("https://reviews.example.com/best-oil-stocks")[0] == 4
    assert credibility("https://example.com/top-10-oil-picks")[0] == 4
    assert credibility("https://example.com/deals?aff=123")[0] == 4


def test_everything_else_is_tier_three_with_a_score():
    assert credibility("https://small-consultancy.example/insights") == (3, 55)


def test_admiralmarkets_and_similar_aggregators_stay_tier_three():
    assert credibility("https://admiralmarkets.com/education/oil")[0] == 3


def test_dedupes_by_normalised_url():
    ranked = rank_hits([hit("https://www.reuters.com/a?utm_source=x", rank=1),
                        hit("https://www.reuters.com/a#top", rank=2)], now=NOW)
    assert len(ranked) == 1


def test_near_identical_titles_are_dropped_and_counted():
    hits = [hit("https://www.reuters.com/a", rank=1, title="Oil prices rise after OPEC cut output"),
            hit("https://www.ft.com/b", rank=2, title="Oil prices rise after OPEC cut output."),
            hit("https://www.bloomberg.com/c", rank=3, title="Gold steady ahead of payrolls data")]
    ranked, duplicates = rank_report(hits, now=NOW)
    assert len(ranked) == 2 and duplicates == 1


def test_short_titles_are_not_treated_as_duplicates():
    ranked, duplicates = rank_report([hit("https://a.example/x", title="News"),
                                      hit("https://b.example/y", title="News")], now=NOW)
    assert len(ranked) == 2 and duplicates == 0


def test_caps_each_domain_at_two_hits():
    topics = ["Copper imports", "Wheat harvest", "Tin mining", "Cobalt prices", "Nickel output"]
    hits = [hit(f"https://www.reuters.com/story-{i}", rank=i + 1, title=topics[i]) for i in range(5)]
    hits.append(hit("https://www.ft.com/x", rank=9))
    ranked = rank_hits(hits, now=NOW)
    domains = [h.url for h in ranked]
    assert sum("reuters.com" in u for u in domains) == 2
    assert any("ft.com" in u for u in domains)


def test_news_recency_outweighs_tier():
    hits = [hit("https://www.federalreserve.gov/2021/old-index", rank=1, date="2021-03-01",
                title="Oil market outlook"),
            hit("https://www.cnbc.com/2026/11/23/oil-prices", rank=2, date="2026-11-23",
                title="Oil prices move on OPEC output")]
    order = [h.url for h in rank_hits(hits, now=NOW, queries=["oil prices OPEC"], kind="news")]
    assert order[0] == "https://www.cnbc.com/2026/11/23/oil-prices"


def test_text_kind_keeps_tier_ahead_of_recency():
    hits = [hit("https://random.example/x", rank=1, date="2026-11-20"),
            hit("https://www.federalreserve.gov/rate", rank=2)]
    assert rank_hits(hits, now=NOW)[0].url == "https://www.federalreserve.gov/rate"


def test_orders_by_combined_score_for_text():
    hits = [
        hit("https://random.example/x", rank=1, date="2026-11-20"),
        hit("https://www.reuters.com/old", rank=2, date="2026-10-01"),
        hit("https://www.reuters.com/new", rank=3, date="2026-11-18"),
        hit("https://www.federalreserve.gov/rate", rank=4),
    ]
    order = [h.url for h in rank_hits(hits, now=NOW)]
    assert order == ["https://www.reuters.com/new", "https://www.reuters.com/old",
                     "https://www.federalreserve.gov/rate", "https://random.example/x"]


def test_undated_hits_come_after_dated_ones_within_a_tier():
    hits = [hit("https://a.example/undated", rank=1), hit("https://b.example/dated", rank=2, date="2026-11-01")]
    assert [h.url for h in rank_hits(hits, now=NOW)][0] == "https://b.example/dated"


def test_undated_hit_ranks_below_an_old_dated_hit_of_the_same_tier():
    hits = [hit("https://www.bis.org/undated", rank=1, title="Quarterly statistics"),
            hit("https://www.bis.org/dated", rank=2, date="2019-06-01", title="Annual review")]
    assert rank_hits(hits, now=NOW)[0].url == "https://www.bis.org/dated"


def test_relevance_lifts_a_matching_hit_within_a_tier():
    hits = [hit("https://a.example/x", rank=1, title="Weather report", snippet="rain in the north"),
            hit("https://b.example/y", rank=2, title="Brent crude forecast", snippet="Brent 2027 average")]
    assert rank_hits(hits, now=NOW, queries=["Brent crude 2027"])[0].url == "https://b.example/y"


def test_recency_filter_drops_old_dated_hits_and_keeps_undated():
    hits = [hit("https://a.example/old", date="2020-01-01"), hit("https://b.example/undated"),
            hit("https://c.example/recent", date="2026-11-20")]
    kept = {h.url for h in rank_hits(hits, recency_days=7, now=NOW)}
    assert kept == {"https://b.example/undated", "https://c.example/recent"}


def test_returned_hits_carry_tier_and_score():
    ranked = rank_hits([hit("https://www.bis.org/x")], now=NOW)
    assert (ranked[0].tier, ranked[0].score) == (1, 100)


def test_future_dated_hit_counts_as_fresh_not_as_an_error():
    hits = [hit("https://a.example/future", date="2026-12-01"), hit("https://b.example/old", date="2019-01-01")]
    assert rank_hits(hits, now=NOW, kind="news")[0].url == "https://a.example/future"


def test_parse_date_reads_iso_and_english_forms():
    assert parse_date("2026-11-20T10:00:00Z").year == 2026
    assert parse_date("Mar 3, 2026").month == 3
    assert parse_date("3 March 2026").day == 3
    assert parse_date("sometime last week") is None
    assert parse_date(None) is None
