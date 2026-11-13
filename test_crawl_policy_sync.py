"""
The synchronous side of crawl_policy, and the scripts and page worker built
on it: one honest name, robots.txt respected (each redirect included), a
refusal is final, and web search only through Brave's API. No network:
every fetch is injected.
"""

import asyncio
import io
import urllib.error
from pathlib import Path

import httpx
try:
    import pytest
except ImportError:
    # The editorial job image checks the pipeline with unittest and has no
    # pytest, and these tests are pytest's (fixtures, monkeypatch, parametrize).
    # Skipped there rather than failing to import, which stopped the admin
    # release from deploying its jobs (2026-11-13). Run them with pytest.
    import unittest

    raise unittest.SkipTest("pytest-style tests: run them with pytest")

from enrichment import crawl_policy
from enrichment.crawl_policy import SyncRobotsPolicy, USER_AGENT

ROOT = Path(__file__).parent
ROBOTS_TXT = "User-agent: *\nDisallow: /private\n\nUser-agent: ProjArchBot\nDisallow: /no-bots\nCrawl-delay: 7\n"


def _policy(status=200, body=ROBOTS_TXT, calls=None, clock=None):
    def fetch(url):
        if calls is not None:
            calls.append(url)
        return status, body

    return SyncRobotsPolicy(fetch=fetch, **({"clock": clock} if clock else {}))


# ── SyncRobotsPolicy ───────────────────────────────────────────────────────

def test_each_site_is_read_once_and_the_group_naming_us_is_obeyed():
    calls = []
    policy = _policy(calls=calls)
    assert policy.allows("https://u.edu/people/a")
    assert not policy.allows("https://u.edu/no-bots/b")
    assert policy.allows("https://u.edu/private/c"), "a group naming us replaces the * group (RFC 9309)"
    assert policy.crawl_delay("https://U.EDU/people/a") == 7.0
    assert calls == ["https://u.edu/robots.txt"]


def test_with_no_group_naming_us_the_rules_for_every_crawler_apply():
    policy = _policy(body="User-agent: *\nDisallow: /private\n")
    assert not policy.allows("https://u.edu/private/x")
    assert policy.allows("https://u.edu/people/x")
    assert policy.crawl_delay("https://u.edu/people/x") is None


@pytest.mark.parametrize("status,allowed", [(401, False), (403, False), (404, True), (410, True)])
def test_robots_answers_are_read_as_crawlers_read_them(status, allowed):
    assert _policy(status=status, body="").allows("https://u.edu/people/a") is allowed


def test_an_unreadable_robots_txt_disallows_until_it_is_read_again_soon():
    now = [0.0]
    answers = [(None, ""), (503, ""), (200, "User-agent: *\nDisallow:\n")]
    calls = []

    def fetch(url):
        calls.append(url)
        return answers[len(calls) - 1]

    policy = SyncRobotsPolicy(fetch=fetch, clock=lambda: now[0])
    assert not policy.allows("https://u.edu/a")
    assert not policy.allows("https://u.edu/a"), "cached while unreadable"
    now[0] += 11 * 60
    assert not policy.allows("https://u.edu/a"), "a 5xx is still no answer"
    now[0] += 11 * 60
    assert policy.allows("https://u.edu/a")
    assert len(calls) == 3


def test_only_web_urls_are_ever_allowed_and_nothing_is_fetched_for_others():
    calls = []
    policy = _policy(calls=calls)
    for url in ("file:///etc/passwd", "ftp://u.edu/a", "javascript:alert(1)", "//u.edu/a", ""):
        assert not policy.allows(url), url
        assert policy.crawl_delay(url) is None
    assert calls == []


def test_the_module_helpers_share_one_policy(monkeypatch):
    monkeypatch.setattr(crawl_policy, "SYNC_ROBOTS", _policy())
    assert crawl_policy.robots_allows("https://u.edu/people/a")
    assert not crawl_policy.robots_allows("https://u.edu/no-bots/a")
    assert crawl_policy.crawl_delay("https://u.edu/people/a") == 7.0


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_robots_txt_is_requested_under_our_name(monkeypatch):
    seen = []

    def urlopen(req, timeout=None, context=None):
        seen.append((req.full_url, req.get_header("User-agent"), context))
        return _Resp(ROBOTS_TXT.encode())

    monkeypatch.setattr(crawl_policy.urllib.request, "urlopen", urlopen)
    assert crawl_policy.fetch_robots_txt("https://u.edu/robots.txt") == (200, ROBOTS_TXT)
    assert seen == [("https://u.edu/robots.txt", USER_AGENT, None)]

    def missing(req, timeout=None, context=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(crawl_policy.urllib.request, "urlopen", missing)
    assert crawl_policy.fetch_robots_txt("https://u.edu/robots.txt") == (404, "")

    def down(req, timeout=None, context=None):
        raise OSError("connection refused")

    monkeypatch.setattr(crawl_policy.urllib.request, "urlopen", down)
    assert crawl_policy.fetch_robots_txt("https://u.edu/robots.txt") == (None, "")


# ── The page worker ────────────────────────────────────────────────────────

@pytest.fixture
def worker(monkeypatch):
    import playwright_subprocess_worker as pw

    # ROBOTS_TXT without its Crawl-delay, so tests do not wait between pages.
    monkeypatch.setattr(pw, "ROBOTS", _policy(body="User-agent: *\nDisallow: /private\n\nUser-agent: ProjArchBot\nDisallow: /no-bots\n"))
    w = pw.ScraperWorker()
    w.playwright_calls = []
    monkeypatch.setattr(w, "fetch_playwright", lambda url, timeout=30: w.playwright_calls.append(url) or ("error", None, None, None, {}))
    monkeypatch.setattr(w, "fetch_playwright_cv", lambda url, timeout=60: w.playwright_calls.append(url) or ("error", None, None, None, {}))
    return pw, w


class _Page(_Resp):
    def __init__(self, body, url, ctype="text/html; charset=utf-8"):
        super().__init__(body.encode())
        self._url = url
        self.headers = {"Content-Type": ctype}

    def geturl(self):
        return self._url


def _serve(pw, monkeypatch, answer):
    requested = []

    def open_(req, timeout=None):
        requested.append(req.full_url)
        assert req.get_header("User-agent") == USER_AGENT
        return answer(req)

    monkeypatch.setattr(pw.OPENER, "open", open_)
    return requested


PROFILE = "<html><body><main><h1>Jeff Volek</h1>" + "<p>Professor of human sciences at Ohio State.</p>" * 20 + "</main></body></html>"


def test_the_worker_sends_our_name_and_reads_an_allowed_page(worker, monkeypatch):
    pw, w = worker
    requested = _serve(pw, monkeypatch, lambda req: _Page(PROFILE, req.full_url))
    kind, html, *_ = w.fetch("https://u.edu/people/volek")
    assert kind == "html" and "Jeff Volek" in html
    assert requested == ["https://u.edu/people/volek"]


def test_a_page_robots_disallows_or_whose_site_forbids_bots_is_never_requested(worker, monkeypatch):
    pw, w = worker
    requested = _serve(pw, monkeypatch, lambda req: _Page(PROFILE, req.full_url))
    for url in ("https://u.edu/no-bots/volek", "https://scholar.google.com/citations?user=abc",
                "https://www.researchgate.net/profile/X", "https://www.ratemyprofessors.com/professor/1"):
        result = w.run(url)
        assert result["status"] == "error" and result.get("refused"), url
    assert requested == [] and w.playwright_calls == []


@pytest.mark.parametrize("status", [401, 403, 429])
def test_a_refusal_is_final_and_never_retried_in_a_browser(worker, monkeypatch, status):
    pw, w = worker

    def refuse(req):
        raise urllib.error.HTTPError(req.full_url, status, "no", {}, io.BytesIO(b"no"))

    requested = _serve(pw, monkeypatch, refuse)
    result = w.run("https://u.edu/people/volek")
    assert result.get("refused") and str(status) in result["error"]
    assert requested == ["https://u.edu/people/volek"]
    assert w.playwright_calls == []
    assert w.extract_cv("https://u.edu/people/volek-cv.pdf")["status"] == "refused"
    assert w.playwright_calls == []


def test_a_bot_check_page_is_a_refusal_not_content(worker, monkeypatch):
    pw, w = worker
    challenge = "<html><title>Just a moment...</title>" + "Checking your browser before accessing u.edu. " * 5 + "</html>"
    _serve(pw, monkeypatch, lambda req: _Page(challenge, req.full_url))
    result = w.run("https://u.edu/people/volek")
    assert result.get("refused")
    assert w.playwright_calls == []


def test_a_redirect_is_followed_only_where_robots_txt_allows(worker, monkeypatch):
    pw, _ = worker
    handler = pw._PolicyRedirectHandler()
    req = pw.urllib.request.Request("https://u.edu/people/volek", headers=pw.HEADERS)
    with pytest.raises(pw.NotFetchable):
        handler.redirect_request(req, io.BytesIO(), 302, "Found", {}, "https://u.edu/no-bots/volek")
    with pytest.raises(pw.NotFetchable):
        handler.redirect_request(req, io.BytesIO(), 302, "Found", {}, "https://scholar.google.com/citations?user=a")
    followed = handler.redirect_request(req, io.BytesIO(), 302, "Found", {}, "https://u.edu/people/jeff-volek")
    assert followed.full_url == "https://u.edu/people/jeff-volek"


def test_a_long_crawl_delay_is_kept_by_skipping_not_by_fetching_sooner(worker, monkeypatch):
    pw, w = worker
    monkeypatch.setattr(pw, "ROBOTS", _policy(body="User-agent: *\nCrawl-delay: 120\n"))
    requested = _serve(pw, monkeypatch, lambda req: _Page(PROFILE, req.full_url))
    assert w.fetch("https://u.edu/a")[0] == "html"
    kind, _, _, _, meta = w.fetch("https://u.edu/b")
    assert kind == "error" and meta["error"] == "crawl_delay_longer_than_job"
    assert requested == ["https://u.edu/a"]


# ── Search and URL collection ──────────────────────────────────────────────

def test_without_a_brave_key_no_search_runs(monkeypatch):
    import collect_osu_scholar_urls as osu

    monkeypatch.delenv("BRAVE_SEARCH_API", raising=False)
    monkeypatch.setattr(osu.httpx, "AsyncClient", lambda *a, **k: pytest.fail("no request without a key"))
    assert asyncio.run(osu.brave_search('"Jeff Volek" "Ohio State"', 10)) == []


def test_search_goes_through_brave_s_api_and_keeps_no_snippets(monkeypatch):
    import collect_osu_scholar_urls as osu

    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"web": {"results": [
            {"url": "https://u.edu/people/volek", "title": "<strong>Jeff Volek</strong> | OSU", "description": "a snippet"},
            {"title": "no url"},
        ]}})

    real_client = httpx.AsyncClient
    monkeypatch.setenv("BRAVE_SEARCH_API", "test-key")
    monkeypatch.setattr(osu.httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    results = asyncio.run(osu.brave_search('"Jeff Volek"', 50))
    assert results == [{"url": "https://u.edu/people/volek", "title": "Jeff Volek | OSU", "snippet": ""}]
    [request] = seen
    assert request.url.host == "api.search.brave.com"
    assert request.headers["X-Subscription-Token"] == "test-key"
    assert request.headers["User-Agent"] == USER_AGENT
    assert request.url.params["count"] == "20", "capped at the API's maximum"


def test_no_query_looks_for_google_scholar_researchgate_or_an_email_handle():
    import collect_osu_scholar_urls as osu

    scholar = {
        "name": {"full": "Jeff Volek", "first": "Jeff", "last": "Volek"},
        "about": {"department": "Human Sciences"},
        "_excel_email": "volek.1@osu.edu",
        "_excel_source_url": "https://ehe.osu.edu/people/volek.1",
    }
    queries = " | ".join(osu._build_queries(osu._identity_keywords(scholar), aggressive=True)).lower()
    for banned in ("scholar.google", "google scholar", "researchgate", '"volek.1"', "volek.1 "):
        assert banned not in queries, banned
    for url in ("https://scholar.google.com/citations?user=x", "https://www.researchgate.net/profile/x", "https://www.academia.edu/x"):
        assert osu._should_skip(url), url

    universities = (ROOT / "run_universities_folder_pipeline.py").read_text(encoding="utf-8")
    assert "site:scholar.google" not in universities and "site:researchgate" not in universities
    assert 'f\'"{email_local}"' not in universities


SCRIPTS = [
    "collect_osu_scholar_urls.py", "collect_legendary_scholar_urls.py", "run_universities_folder_pipeline.py",
    "playwright_subprocess_worker.py", "unified_pipeline.py", "daily_story_worker.py",
    "api/services/document_processor.py", "api/utils/url_categorizer.py", "api/utils/robots_checker.py",
]


@pytest.mark.parametrize("name", SCRIPTS)
def test_no_script_impersonates_a_browser_or_scrapes_a_search_engine(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    for marker in ("Mozilla/5.0", "Chrome/1", "AutomationControlled", "navigator, 'webdriver'", "add_init_script",
                   "storage_state", "curl_cffi", "free_proxy", "from scholarly", "import scholarly",
                   "duckduckgo_search", "from ddgs", "html.duckduckgo.com", "youtube_transcript_api", "timedtext"):
        assert marker not in text, (name, marker)
