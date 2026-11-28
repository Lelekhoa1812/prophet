"""Query variants, backend cooldown and retry, registry labelling and ranking for data discovery. ddgs is replaced; offline."""
import threading

import pytest

import ddgs
from ddgs.exceptions import DDGSException
from prophet_agent.research import search
from prophet_agent.research.queryplan import MAX_VARIANTS, is_data_seeking, plan_query
from prophet_agent.research.rank import data_bearing, interleave, rank_hits, registry_hits
from prophet_agent.research.search import Hit, discover, reset_backend_health, web_search

QUERY = "NVDA historical stock price CSV download"


class FakeClock:
    def __init__(self):
        self.now = 1000.0


class FakeDDGS:
    """Stands in for ddgs.DDGS. ``handler(backend, query)`` returns rows or an exception."""
    handler = None
    calls: list = []
    lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        pass

    def _answer(self, query, **kwargs):
        backend = kwargs["backend"]
        with FakeDDGS.lock:
            FakeDDGS.calls.append((backend, query))
        outcome = FakeDDGS.handler(backend, query)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def text(self, query, **kwargs):
        return self._answer(query, **kwargs)

    def news(self, query, **kwargs):
        return self._answer(query, **kwargs)


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(search, "_clock", lambda: fake.now)
    return fake


@pytest.fixture
def fake(monkeypatch, tmp_path, clock):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("PROPHET_SEARCH_BACKENDS", raising=False)
    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    FakeDDGS.calls = []
    FakeDDGS.handler = lambda backend, query: []
    reset_backend_health()
    yield FakeDDGS
    reset_backend_health()


ROWS = [
    {"title": "NVDA stock price history CSV", "href": "https://stooq.com/q/d/l/?s=nvda.us&i=d", "body": "Daily."},
    {"title": "NVDA historical prices", "href": "https://www.nasdaq.com/market-activity/stocks/nvda/historical",
     "body": "Table."},
]


# ---- query variants -------------------------------------------------------------------------------------------------

def test_plan_keeps_the_query_first_and_adds_a_short_csv_form():
    plan = plan_query(QUERY)
    assert plan.variants[0] == QUERY
    assert "NVDA stock price" in plan.variants
    assert "NVDA stock price csv download" in plan.variants
    assert plan.data_seeking is True and plan.entity == "NVDA"


def test_plan_site_targets_domain_hosts_for_a_table_query():
    plan = plan_query("Premier League 2026-27 table")
    assert "site:wikipedia.org Premier League 2026-27 table" in plan.variants
    assert "Premier League 2026-27 table historical data" in plan.variants


def test_plan_gives_only_two_variants_for_a_non_data_query():
    assert plan_query("what is nvidia").variants == ("what is nvidia", "nvidia")
    assert is_data_seeking("what is nvidia") is False


def test_plan_is_deterministic_bounded_and_free_of_duplicates():
    first = plan_query(QUERY).variants
    assert first == plan_query(QUERY).variants
    assert len(first) <= MAX_VARIANTS
    keys = [" ".join(v.lower().split()) for v in first]
    assert len(keys) == len(set(keys))


def test_plan_refuses_an_empty_query():
    with pytest.raises(ValueError):
        plan_query("   ")


# ---- backend health: cooldown, retry, recorded answers -------------------------------------------------------------

def test_three_bad_answers_cool_a_backend_and_it_is_skipped(fake):
    fake.handler = lambda backend, query: DDGSException("blocked") if backend == "bing" else ROWS
    for i in range(3):
        web_search(f"q{i}", backends=["bing", "google"], backoff_s=0, use_cache=False)
    result = web_search("q4", backends=["bing", "google"], backoff_s=0, use_cache=False)
    assert result.tried == ["google"]
    assert any(d.startswith("bing: cooling down") for d in result.diagnostics)
    assert search.backend_health()["bing"]["cooling_s"] > 0


def test_cooldown_ends_after_the_window(fake, clock):
    fake.handler = lambda backend, query: DDGSException("blocked") if backend == "bing" else ROWS
    for i in range(3):
        web_search(f"q{i}", backends=["bing", "google"], backoff_s=0, use_cache=False)
    clock.now += search.COOLDOWN_S + 1
    result = web_search("later", backends=["bing", "google"], backoff_s=0, use_cache=False)
    assert result.tried == ["bing", "google"]


def test_a_good_answer_clears_the_streak(fake):
    answers = {"n": 0}

    def handler(backend, query):
        if backend != "bing":
            return ROWS
        answers["n"] += 1
        return [] if answers["n"] in (1, 2, 4, 5) else ROWS

    fake.handler = handler
    for i in range(6):
        web_search(f"q{i}", backends=["bing", "google"], backoff_s=0, use_cache=False)
    assert search.backend_health()["bing"]["cooling_s"] == 0


def test_all_backends_cooling_still_probes_one(fake):
    fake.handler = lambda backend, query: DDGSException("down")
    for i in range(3):
        web_search(f"a{i}", backends=["bing", "google"], backoff_s=0, use_cache=False)
    fake.calls.clear()
    result = web_search("again", backends=["bing", "google"], backoff_s=0, use_cache=False)
    assert len(result.tried) == 1
    assert any("all backends cooling" in d for d in result.diagnostics)
    assert len(fake.calls) == 1


def test_transient_error_is_retried_once(fake):
    outcomes = iter([TimeoutError("slow"), ROWS])

    def handler(backend, query):
        return next(outcomes) if backend == "bing" else []

    fake.handler = handler
    result = web_search("q", backends=["bing"], backoff_s=0, use_cache=False)
    assert result.backend == "bing" and len(result) == 2
    assert [b for b, _ in fake.calls] == ["bing", "bing"]


def test_non_transient_error_is_not_retried(fake):
    fake.handler = lambda backend, query: RuntimeError("bad request")
    result = web_search("q", backends=["bing"], backoff_s=0, use_cache=False)
    assert result == []
    assert [b for b, _ in fake.calls] == ["bing"]
    assert search.backend_health()["bing"]["failed"] == 1


def test_result_records_which_backends_were_tried_and_answered(fake):
    fake.handler = lambda backend, query: [] if backend == "bing" else ROWS
    result = web_search("q", backends=["bing", "google"], backoff_s=0, use_cache=False)
    assert result.tried == ["bing", "google"]
    assert result.answered == ["bing", "google"]
    assert result.backend == "google"


def test_health_counts_ok_empty_and_failed(fake):
    fake.handler = lambda backend, query: {"bing": DDGSException("x"), "google": [], "yahoo": ROWS}[backend]
    web_search("q", backends=["bing", "google", "yahoo"], backoff_s=0, use_cache=False)
    health = search.backend_health()
    assert health["bing"]["failed"] == 1
    assert health["google"]["empty"] == 1
    assert health["yahoo"]["ok"] == 1
    assert "mojeek" not in health


# ---- discovery: merge, registry labelling, ranking ---------------------------------------------------------------

def test_discover_appends_labelled_registry_entries_after_search_hits(fake):
    fake.handler = lambda backend, query: ROWS if "NVDA" in query else []
    result = discover(QUERY, backoff_s=0, use_cache=False)
    registry = registry_hits(QUERY)
    assert registry, "the equities family should have registry pages"
    search_hits = [h for h in result if h.source != "registry"]
    assert len(search_hits) == 2
    assert [h.url for h in result[:2]] == [h.url for h in search_hits]
    assert all(h.source == "registry" and h.backend == "registry" for h in result[2:])
    assert len(result) == 2 + len(registry)
    assert result.variants[0] == QUERY


def test_registry_urls_never_appear_as_search_hits(fake):
    registry_url = registry_hits(QUERY)[0].url
    fake.handler = lambda backend, query: [{"title": "SEC newsroom", "href": registry_url + "/", "body": ""}] + ROWS
    result = discover(QUERY, backoff_s=0, use_cache=False)
    search_urls = [h.url for h in result if h.source != "registry"]
    assert registry_url not in search_urls


def test_no_registry_entries_for_a_query_without_a_family(fake):
    fake.handler = lambda backend, query: ROWS
    result = discover("what is nvidia", backoff_s=0, use_cache=False)
    assert all(h.source != "registry" for h in result)


def test_discover_interleaves_variants_so_each_contributes_early(fake):
    variants = plan_query("Premier League 2026-27 table").variants

    def handler(backend, query):
        i = variants.index(query)
        return [{"title": f"{query} result {j} standings table", "href": f"https://src{i}.example.org/p{j}",
                 "body": ""} for j in range(3)]

    fake.handler = handler
    result = discover("Premier League 2026-27 table", backoff_s=0, use_cache=False)
    first = {h.url.split("/")[2] for h in result[:len(variants)]}
    assert first == {f"src{i}.example.org" for i in range(len(variants))}


def test_discover_dedupes_the_same_page_across_variants(fake):
    fake.handler = lambda backend, query: ROWS
    result = discover(QUERY, backoff_s=0, use_cache=False)
    search_urls = [h.url for h in result if h.source != "registry"]
    assert len(search_urls) == len(set(search_urls)) == 2


def test_discover_refuses_bad_arguments():
    with pytest.raises(ValueError):
        discover(QUERY, max_results=0)
    with pytest.raises(ValueError):
        discover("  ")


def test_data_seeking_prefers_a_data_file_over_an_article_of_equal_standing():
    article = Hit(title="Nvidia shares rise on chip demand NVDA price history",
                  url="https://news.example.com/nvda-price-history", rank=1)
    data = Hit(title="NVDA price history download csv file", url="https://data.example.com/nvda/prices.csv", rank=2)
    plain = rank_hits([article, data], queries=["NVDA price history"], now=0.0)
    assert plain[0].url == article.url
    seeking = rank_hits([article, data], queries=["NVDA price history"], now=0.0, data_seeking=True)
    assert seeking[0].url == data.url


def test_data_bearing_scores_files_then_paths_then_tables():
    assert data_bearing("https://x.example.com/a/prices.csv") == 1.0
    assert data_bearing("https://x.example.com/api/v1/series") == 0.6
    assert data_bearing("https://en.wikipedia.org/wiki/Premier_League") == 0.5
    assert data_bearing("https://news.example.com/story") == 0.0


def test_interleave_is_round_robin_without_repeats():
    a = [Hit(title="a", url="https://a.example.com/1", rank=1), Hit(title="a2", url="https://a.example.com/2", rank=2)]
    b = [Hit(title="b", url="https://a.example.com/1/", rank=1), Hit(title="b2", url="https://b.example.com/2", rank=2)]
    out = [h.url for h in interleave([a, b])]
    assert out == ["https://a.example.com/1", "https://a.example.com/2", "https://b.example.com/2"]


def test_registry_hits_are_labelled_registry_and_tiered():
    hits = registry_hits(QUERY)
    assert hits and all(h.source == "registry" and h.backend == "registry" for h in hits)
    assert all(h.tier == 1 for h in hits)
    assert registry_hits("what is nvidia") == []


# ---- pacing across research lanes (round 20261128-125459) ----------------------------------------------------------

def test_a_send_error_streak_cools_for_ninety_seconds_not_ten_minutes(fake, clock):
    fake.handler = lambda backend, query: (DDGSException("ConnectError: error sending request for url")
                                           if backend == "bing" else ROWS)
    for i in range(3):
        web_search(f"s{i}", backends=["bing", "google"], backoff_s=0, use_cache=False)
    assert 0 < search.backend_health()["bing"]["cooling_s"] <= search.TRANSIENT_COOLDOWN_S
    clock.now += search.TRANSIENT_COOLDOWN_S + 1
    assert web_search("back", backends=["bing", "google"], backoff_s=0, use_cache=False).tried == ["bing", "google"]


def test_a_backend_whose_slot_is_too_far_off_is_skipped_not_called(fake, monkeypatch):
    from prophet_agent.research.netguard import HostThrottle

    pace = HostThrottle(100.0, sleep=lambda s: None)
    monkeypatch.setattr(search, "_PACE", pace)
    monkeypatch.setattr(search, "PACE_S", 100.0)
    fake.handler = lambda backend, query: ROWS
    pace.wait("bing")                                   # another lane took bing's slot; the next is 100 s away
    result = web_search("paced", backends=["bing", "google"], backoff_s=0.01, use_cache=False)
    assert result.tried == ["google"] and len(result) == 2
    assert "bing: busy with other searches, skipped" in result.diagnostics
    assert [c[0] for c in fake.calls] == ["google"]
