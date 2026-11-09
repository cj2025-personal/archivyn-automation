"""
How the enrichment collectors gather public information: one honest name,
robots.txt respected, a refusal is final, nothing from sources whose terms
forbid automated access - and the three collectors that found nobody in the
2026 run (Semantic Scholar, the OSU Knowledge Bank, OSU courses) reading
their sources correctly. No network: every fetch is injected.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from enrichment import base_collector, crawl_policy, orchestrator
from enrichment.base_collector import BaseCollector, ProfessorQuery
from enrichment.crawl_policy import RobotsPolicy, is_challenge_page, parser_for
from enrichment.osu_courses import course_record, recent_terms, term_name
from enrichment.osu_expertise import item_for_scholar
from enrichment.semantic_scholar import anchored_author_id, openalex_dois

ENRICHMENT = Path(__file__).parent / "enrichment"
QUERY = ProfessorQuery(profile_id="p1", name="Jeff Volek")


def run(coro):
    return asyncio.run(coro)


# ── robots.txt ────────────────────────────────────────────────────────────

ROBOTS_TXT = "User-agent: *\nDisallow: /private\n\nUser-agent: ProjArchBot\nDisallow: /no-bots\nCrawl-delay: 7\n"


def test_robots_rules_are_read_as_crawlers_read_them():
    rp, settled = parser_for(200, ROBOTS_TXT)
    assert settled and rp.can_fetch("ProjArchBot", "https://u.edu/people/a")
    assert not rp.can_fetch("ProjArchBot", "https://u.edu/no-bots/a")
    rp, settled = parser_for(403, "")
    assert settled and not rp.can_fetch("ProjArchBot", "https://u.edu/people/a")
    rp, settled = parser_for(404, "")
    assert settled and rp.can_fetch("ProjArchBot", "https://u.edu/people/a")
    for status in (500, 503, None):
        rp, settled = parser_for(status, "")
        assert not settled and not rp.can_fetch("ProjArchBot", "https://u.edu/people/a")


def test_robots_policy_reads_each_site_once_and_honours_crawl_delay():
    calls = []

    async def fetch(url):
        calls.append(url)
        return 200, ROBOTS_TXT

    policy = RobotsPolicy(fetch=fetch)
    assert run(policy.allows("https://u.edu/people/a"))
    assert not run(policy.allows("https://u.edu/no-bots/b"))
    assert run(policy.crawl_delay("https://u.edu/people/a")) == 7.0
    assert calls == ["https://u.edu/robots.txt"]


def test_a_bot_check_page_is_recognised():
    assert is_challenge_page("<title>Just a moment...</title>")
    assert not is_challenge_page("<h1>Jeff Volek, Professor</h1>")
    # A faculty page whose contact form embeds reCAPTCHA is content, not a refusal.
    assert not is_challenge_page('<h1>Contact</h1><div class="g-recaptcha" data-sitekey="x"></div>')


# ── page fetches ──────────────────────────────────────────────────────────

class _Collector(BaseCollector):
    source_name = "test"

    async def collect(self, query):  # pragma: no cover - not used
        raise NotImplementedError


def _collector_with(handler, robots=ROBOTS_TXT, monkeypatch=None):
    async def fetch(url):
        return 200, robots

    monkeypatch.setattr(base_collector, "ROBOTS", RobotsPolicy(fetch=fetch))
    c = _Collector(rate_limit_delay=0)
    c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), headers={"User-Agent": crawl_policy.USER_AGENT})
    return c


def test_a_page_robots_disallows_is_never_requested(monkeypatch):
    requested = []
    handler = lambda r: requested.append(str(r.url)) or httpx.Response(200, text="hi")
    # A group naming us is the one we obey (RFC 9309)...
    c = _collector_with(handler, monkeypatch=monkeypatch)
    assert run(c._fetch_html("https://u.edu/no-bots/x")) is None
    # ...and with no such group, the rules for every crawler apply to us.
    c = _collector_with(handler, robots="User-agent: *\nDisallow: /private\n", monkeypatch=monkeypatch)
    assert run(c._fetch_html("https://u.edu/private/x")) is None
    assert requested == []


@pytest.mark.parametrize("status", [401, 403, 429])
def test_a_refusal_is_final(monkeypatch, status):
    requested = []

    def handler(request):
        requested.append(request)
        return httpx.Response(status, text="no")

    c = _collector_with(handler, monkeypatch=monkeypatch)
    assert run(c._fetch_any("https://u.edu/people/a")) is None
    assert len(requested) == 1


def test_a_bot_check_page_is_not_content(monkeypatch):
    c = _collector_with(lambda r: httpx.Response(200, text="Checking your browser before accessing"), monkeypatch=monkeypatch)
    assert run(c._fetch_html("https://u.edu/people/a")) is None


def test_only_our_name_is_sent_whatever_the_caller_passes(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request.headers["user-agent"])
        return httpx.Response(200, text="<p>Jeff Volek</p>", headers={"content-type": "text/html"})

    c = _collector_with(handler, monkeypatch=monkeypatch)
    page = run(c._fetch_any("https://u.edu/people/a", headers={"User-Agent": "Mozilla/5.0 Chrome/124"}))
    assert page and page["text"] == "<p>Jeff Volek</p>"
    assert seen == [crawl_policy.USER_AGENT]


def test_no_collector_impersonates_a_browser_or_keeps_evasion_code():
    for path in ENRICHMENT.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "Mozilla/5.0" not in text, path.name
        for marker in ("curl_cffi", "AutomationControlled", "cf_clearance", "free_proxy", "scholarly", "duckduckgo_search"):
            assert marker not in text, (path.name, marker)


def test_sources_whose_terms_forbid_automated_access_are_gone():
    for name in ("google_scholar", "rate_my_professor", "youtube_transcripts"):
        assert name not in orchestrator.ALL_COLLECTORS
        assert not (ENRICHMENT / f"{name}.py").exists()
    assert not (ENRICHMENT / "bypass.py").exists()


# ── Semantic Scholar: identified through the scholar's own papers ─────────

def _paper(*authors):
    return {"authors": [{"authorId": aid, "name": name} for aid, name in authors]}


def test_the_s2_author_named_on_several_of_the_scholars_papers_is_the_scholar():
    papers = [
        _paper(("111", "Jeffrey S. Volek"), ("9", "A. Other")),
        _paper(("111", "J. Volek")),
        _paper(("222", "J. Volek"), ("9", "A. Other")),
        None,  # a DOI S2 does not have
    ]
    assert anchored_author_id(QUERY, papers, dois_looked_up=4) == "111"


def test_one_paper_is_not_enough_and_a_tie_is_not_a_match():
    assert anchored_author_id(QUERY, [_paper(("111", "Jeff Volek"))], dois_looked_up=5) is None
    tie = [_paper(("111", "Jeff Volek")), _paper(("111", "Jeff Volek")), _paper(("222", "Jeff Volek")), _paper(("222", "J. Volek"))]
    assert anchored_author_id(QUERY, tie, dois_looked_up=4) is None
    assert anchored_author_id(QUERY, [_paper(("111", "Jeff Volek"))], dois_looked_up=1) == "111"


def test_dois_come_from_the_runs_openalex_works_most_cited_first(tmp_path, monkeypatch):
    profile = tmp_path / "profiles" / "p1"
    profile.mkdir(parents=True)
    works = [{"doi": "https://doi.org/10.1/A", "cited_by_count": 3}, {"doi": "10.1/b", "cited_by_count": 9}, {"doi": None}]
    (profile / "enrichment.json").write_text(json.dumps({"sources": {"openalex": {"data": {"works": works}}}}))
    monkeypatch.setenv("ENRICHMENT_OUTPUT_DIR", str(tmp_path))
    assert openalex_dois("p1") == ["10.1/b", "10.1/a"]
    assert openalex_dois("missing") == []


def test_semantic_scholar_runs_after_openalex():
    assert orchestrator.DEPENDENT_SOURCES["semantic_scholar"] == ["openalex"]


# ── OSU Knowledge Bank ────────────────────────────────────────────────────

def _kb_item(**fields):
    return {"handle": "1811/1", "metadata": {k: [{"value": v} for v in vs] for k, vs in fields.items()}}


def test_a_work_the_scholar_authored_is_kept_with_its_authors():
    rec = item_for_scholar(QUERY, _kb_item(**{"dc.title": ["Ketones"], "dc.creator": ["Volek, Jeff S."], "dc.type": ["Article"]}))
    assert rec["role"] == "author" and rec["authors"] == ["Volek, Jeff S."]


def test_an_advised_thesis_is_kept_without_the_students_name():
    rec = item_for_scholar(QUERY, _kb_item(**{"dc.title": ["Cold water"], "dc.creator": ["Arce, Lucas"], "dc.contributor.advisor": ["Volek, Jeff"]}))
    assert rec["role"] == "advisor" and "authors" not in rec
    assert "Arce" not in json.dumps(rec)


def test_an_abstract_is_kept_only_under_an_open_license():
    base = {"dc.title": ["T"], "dc.creator": ["Volek, Jeff"], "dc.description.abstract": ["We found..."]}
    assert "abstract" not in item_for_scholar(QUERY, _kb_item(**base, **{"dc.rights": ["This item may be protected by copyright"]}))
    rec = item_for_scholar(QUERY, _kb_item(**base, **{"dc.rights.uri": ["http://creativecommons.org/licenses/by/4.0/"]}))
    assert rec["abstract"] == "We found..."


def test_an_item_the_scholar_has_no_part_in_is_skipped():
    assert item_for_scholar(QUERY, _kb_item(**{"dc.title": ["T"], "dc.creator": ["Smith, Ann"]})) is None


# ── OSU courses ───────────────────────────────────────────────────────────

def test_osu_term_codes_are_four_digits():
    assert recent_terms(datetime(2026, 11, 8))[:4] == ["1268", "1264", "1262", "1258"]
    assert term_name("1268") == "Autumn 2026" and term_name("1262") == "Spring 2026"


def _course_result(instructor="Jeff Volek"):
    return {
        "course": {"subject": "KNSISM", "catalogNumber": "5500", "title": "Nutrition", "description": "About food.", "minUnits": 3},
        "sections": [{"classNumber": "123", "meetings": [{
            "monday": True, "startTime": "9:10", "buildingDescription": "PAES Building", "room": "203",
            "instructors": [{"displayName": instructor, "email": "volek.1@osu.edu", "role": "PI"}],
        }]}],
    }


def test_a_course_is_described_without_when_or_where_it_meets():
    rec = course_record(QUERY, _course_result(), "1268")
    assert rec["subject"] == "KNSISM" and rec["title"] == "Nutrition" and rec["role"] == "PI"
    flat = json.dumps(rec)
    for leaked in ("PAES", "203", "9:10", "volek.1@osu.edu", "classNumber", "monday"):
        assert leaked not in flat


def test_a_course_someone_else_teaches_is_not_kept():
    assert course_record(QUERY, _course_result("Ann Smith"), "1268") == {}


def test_an_instructor_listed_without_a_name_is_skipped_not_an_error():
    result = _course_result()
    result["sections"][0]["meetings"][0]["instructors"].insert(0, {"displayName": None, "role": "TA"})
    assert course_record(QUERY, result, "1268")["role"] == "PI"


# ── OpenAlex: abstracts only under an open license ────────────────────────

def test_openalex_keeps_an_abstract_only_for_openly_licensed_works(monkeypatch):
    from enrichment.openalex import OpenAlexCollector

    def work(license_):
        return {"title": "T", "publication_year": 2020, "cited_by_count": 1, "doi": "10.1/x",
                "abstract_inverted_index": {"We": [0], "found": [1]},
                "primary_location": {"license": license_, "source": {"display_name": "J"}}}

    async def fake_get_json(url, params=None):
        return {"results": [work("cc-by"), work("cc-by-nc"), work(None), work("publisher-specific-oa")]}

    c = OpenAlexCollector(rate_limit_delay=0)
    monkeypatch.setattr(c, "_get_json", fake_get_json)
    works = run(c._get_works("A1"))
    assert [w["abstract"] for w in works] == ["We found", "", "", ""]
    assert all(w["title"] == "T" and w["doi"] == "10.1/x" for w in works)
