"""The profile-quality diagnosis and the targeted profile repair.

Nothing here reaches the web or a model: every socket but the local test
database's is refused for the whole module, Wikipedia's API and the Brave
Search API are stubs at the collector's own request functions, pages come
from a stub scraper, and the identity judge is a stub. The tests that need a
database use only the local test MongoDB at 127.0.0.1:27018, each in a fresh
database dropped afterwards, and are skipped when it is not running. Nothing
here reads a .env file.

Run with:  python -m unittest test_legend_profile_repair
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import re
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

try:
    import legend_profile_repair as repair  # first: it refuses .env files before the pipeline modules load
    import collect_legendary_scholar_urls as collect
    SCRAPING_STACK = ""
except ImportError as exc:
    # The editorial job image installs only requirements-editorial.txt and runs
    # these tests with unittest; the repair needs the scraping stack of
    # requirements.txt (pandas, the page worker), so its tests are skipped there
    # while the diagnosis's still run.
    repair = collect = None  # type: ignore[assignment]
    SCRAPING_STACK = f"needs the scraping stack of requirements.txt ({exc})"
needs_scraping_stack = unittest.skipIf(bool(SCRAPING_STACK), SCRAPING_STACK)

import legend_identity
import legend_identity_review as identity_stage
import legend_profile_diagnosis as diagnosis
import profile_quality_refresh as quality
from enrichment.crawl_policy import USER_AGENT
from legend_evergreen import parse_ids
from readonly_mongo import ReadOnlyViolation, connect_read_only
from test_editorial_pipeline import _identified
from test_legend_evergreen import QUIET_ENV, LocalDatabase, make_worker, needs_local_mongo

NAME = "Ada Brightwater"
PID = "ada-brightwater"
WIKI = "https://en.wikipedia.org/wiki/Ada_Brightwater"
MUSEUM = "https://museum.example.org/ada-brightwater"
NASA = "https://www.nasa.gov/people/ada-brightwater"
SMITHSONIAN = "https://www.si.edu/ada-brightwater-biography"
COLLEGE = "https://history.example.edu/ada-brightwater"
NEVER_FETCHED = (
    "researchgate.net", "scholar.google", "nytimes.com", "books.google", "facebook.com",
)

_real_connect = socket.socket.connect


def _local_only(sock: socket.socket, address: Any) -> Any:
    host = address[0] if isinstance(address, tuple) else address
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise AssertionError(f"a test reached for the network: {address}")
    return _real_connect(sock, address)


def setUpModule() -> None:
    socket.socket.connect = _local_only  # type: ignore[assignment]


def tearDownModule() -> None:
    socket.socket.connect = _real_connect  # type: ignore[assignment]


# ── Fixtures ────────────────────────────────────────────────────────────────

TOPICS = ["fireflies", "glaciers", "volcanoes", "orchards", "rivers", "comets", "deserts", "coral"]
PLACES = ["Ohio", "Nevada", "Maine", "Oregon", "Texas", "Vermont", "Kansas", "Alaska"]


def page_text(page: int, sentences: int) -> str:
    """A page about the legend in words of its own, so each passage is traced
    to the page it came from."""
    topic, place = TOPICS[page % len(TOPICS)], PLACES[page % len(PLACES)]
    return " ".join(
        f"{NAME} studied {topic} near {place} in {1950 + n}, measuring how light and water changed through the seasons. "
        f"She wrote careful field notes about the {topic} of {place} and taught students to repeat each experiment {n + 2} times."
        for n in range(sentences)
    )


def stored_passage(index: int) -> str:
    """Over 420 characters, so length scores in full, as Chappelle's do."""
    return (
        f"{NAME} began laboratory study {index} in {1960 + index}, testing how young plants answer to light, to salt and to cold nights. "
        f"In {1961 + index} {NAME} shared those results with teachers, nurses and young chemists at schools and clinics across the state, "
        f"and the notes from study {index} were kept in the college library so that students could check every number for themselves "
        f"and repeat the work with simple tools they could hold in their own hands."
    )


def legend(count: int = 19, *, label: str = "needs_data_repair", score: int = 64, **extra: Any) -> Dict[str, Any]:
    """A legend like emmett-chappelle on 2026-11-13: three core sections,
    long passages, 19 of them, so 64 points and still not ready."""
    sections: Dict[str, List[Dict[str, Any]]] = {"biography": [], "background_and_work": [], "legacy": []}
    names = list(sections)
    pages = [WIKI, MUSEUM]
    for index in range(count):
        sections[names[index % 3]].append({
            "chunk_id": f"{PID}-c{index}",
            "text": stored_passage(index),
            "primary_source_url": pages[index % 2],
        })
    doc = {
        "_id": PID,
        "profile_id": PID,
        "professor_name": NAME,
        "name": {"full": NAME, "display": NAME},
        "about": {
            "short_bio": f"{NAME} was a chemist.",
            "field_of_study": "Chemistry",
            "bio_by_band": {"grade_3_5": {"status": "ready", "text": "Ada liked light.", "input_hash": "kept"}},
        },
        "display": {"visibility": "published"},
        "admin": {"review_status": "curated"},
        "rag_context": {"chunk_count": count, "sections_available": names, "section_chunks": sections},
        "daily_story_profile": {"status": label, "quality_score": score},
        **extra,
    }
    return _identified(doc)


def wikipedia_api_linking(links: List[str]) -> Any:
    """Wikipedia's API, as collect._get_json is answered, for an article
    whose external links are links."""

    def answer(url: str, params: Dict[str, Any], headers: Dict[str, str] | None = None) -> Dict[str, Any]:
        assert url == "https://en.wikipedia.org/w/api.php" and (headers or {}).get("User-Agent") == USER_AGENT
        if params.get("list") == "search":
            return {"query": {"search": [{"title": NAME}]}}
        if params.get("prop") == "info":
            return {"query": {"pages": {"1": {"fullurl": WIKI}}}}
        return {"query": {"pages": {"1": {"extlinks": [{"*": link} for link in links]}}}}

    return answer


wikipedia_api = wikipedia_api_linking([NASA, "https://books.google.com/books?id=ada", "https://www.researchgate.net/profile/Ada_Brightwater"])


SEARCH_RESULTS = [
    {"url": SMITHSONIAN, "title": f"{NAME} biography | Smithsonian", "snippet": ""},
    {"url": "https://www.nytimes.com/1999/ada-brightwater.html", "title": f"{NAME}, chemistry pioneer, obituary", "snippet": ""},
    {"url": "https://scholar.google.com/citations?user=ada", "title": f"{NAME} - Google Scholar", "snippet": ""},
    {"url": COLLEGE, "title": f"{NAME}, chemist and teacher", "snippet": ""},
    {"url": "https://www.facebook.com/ada.brightwater", "title": NAME, "snippet": ""},
]


class Searches:
    def __init__(self) -> None:
        self.queries: List[str] = []

    async def __call__(self, query: str, max_results: int) -> List[Dict[str, str]]:
        if not os.getenv("BRAVE_SEARCH_API", "").strip():
            return []  # as brave_search: without a key, no request and no results
        self.queries.append(query)
        return [dict(r) for r in SEARCH_RESULTS]


def same_pages(urls: List[str]) -> List[str]:
    """URLs as one page each (the collector drops www. and the scheme's case)."""
    return sorted(repair._url_key(u) for u in urls)


class Scraper:
    """Stands in for the page worker: a page for each URL it knows."""

    def __init__(self, sentences: int = 44) -> None:
        self.sentences = sentences
        self.asked: List[str] = []
        self.pages = [repair._url_key(u) for u in (WIKI, MUSEUM, NASA, SMITHSONIAN, COLLEGE)]

    async def extract_all(self, url: str) -> Dict[str, Any]:
        self.asked.append(url)
        if repair._url_key(url) not in self.pages:
            return {"profile_data": {}, "source_records": []}
        index = self.pages.index(repair._url_key(url))
        return {
            "profile_data": {"name": NAME},
            "source_records": [{
                "source_type": "profile_page",
                "source_url": url,
                "resolved_url": url,
                "content": page_text(index, self.sentences),
                "fetch_metadata": {"status_code": 200},
                "processing_status": "success",
                # A title with periods: section names are page titles, and
                # MongoDB must store them as keys.
                "title": f"Dr. {NAME}, page {index + 1} - Example",
            }],
        }


def judge(prompt: str, schema: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    assert schema is identity_stage.IDENTITY_SCHEMA
    page_ids = re.findall(r"^\[(P\d+)\] ", prompt, flags=re.M)
    assert page_ids, "the judge is shown the pages"
    return {
        "pages": [{"page_id": p, "verdict": legend_identity.ABOUT_LEGEND, "person_described": NAME, "reason": "about her"} for p in page_ids],
        "intended_person": {"full_name": NAME, "field": "Chemistry", "years": "1931-2001", "known_for": "light and plants"},
        "name_check": "correct",
        "usual_spelling": "",
    }


def snapshot(db: Any) -> Dict[str, List[str]]:
    return {
        name: sorted(json.dumps(doc, sort_keys=True, default=str) for doc in db[name].find({}))
        for name in sorted(db.list_collection_names())
    }


def old_evaluate(signals: Dict[str, Any], avg_unrounded: float, profile_id: str) -> Any:
    """profile_quality_refresh.evaluate_profile's arithmetic as written before
    its tables were named (commit e8d6e88), the oracle for the refactor."""
    score = 20 if signals["name_quality"] else 0
    n = signals["chunk_count"]
    score += 40 if n >= 80 else 30 if n >= 40 else 20 if n >= 20 else 10 if n >= 10 else 0
    c = signals["section_coverage"]
    score += 20 if c >= 4 else 14 if c == 3 else 8 if c == 2 else 0
    a = avg_unrounded
    score += 20 if a >= 420 else 14 if a >= 250 else 8 if a >= 140 else 0
    status = "ready"
    if score < 60 or n < 20:
        status = "needs_data_repair"
    if not profile_id:
        status = "invalid_profile"
    return score, status


# ── Pure pieces ─────────────────────────────────────────────────────────────

class ScoreTests(unittest.TestCase):
    def test_the_named_tables_score_exactly_as_the_inline_arithmetic_did(self) -> None:
        cases = [
            (["biography", "background_and_work", "legacy"], [1700] * 19, NAME),
            (["Margaret S. Collins"], [1451] * 10, "Margaret S. Collins"),
            (["W. E. B. Du Bois", "Personal Website"], [1677] * 97, "W.E.B. Du bois"),
            (["about", "biography", "legacy", "publications", "background"], [300] * 21, "John Hope Franklin"),
            (["about"], [420] * 24 + [419], "x"),  # averages 419.96: stored as 420.0, scored under 420
            ([], [], ""),
        ]
        for sections, lengths, name in cases:
            chunks: Dict[str, List[Dict[str, str]]] = {s: [] for s in sections}
            for i, length in enumerate(lengths):
                chunks[sections[i % len(sections)]].append({"text": "y" * length})
            doc = {"profile_id": "p" if name else "", "professor_name": name, "rag_context": {"section_chunks": chunks}}
            got = quality.evaluate_profile(diagnosis.NAME_RULES, doc)
            unrounded = quality._avg_chunk_len(chunks)
            self.assertEqual((got["score"], got["status"]), old_evaluate(got["signals"], unrounded, doc["profile_id"]), (sections, len(lengths)))
            self.assertEqual(sum(got["points"].values()), got["score"])
        edge = {"profile_id": "p", "rag_context": {"section_chunks": {"about": [{"text": "y" * n} for n in [420] * 24 + [419]]}}}
        got = quality.evaluate_profile(diagnosis.NAME_RULES, edge)
        self.assertEqual((got["signals"]["avg_chunk_len"], got["points"]["avg_len"]), (420.0, 14))

    def test_the_name_part_never_fails_because_the_fallback_name_passes(self) -> None:
        for name in ("unknown", "http://x.org", "African American History Museum", ""):
            got = quality.evaluate_profile(diagnosis.NAME_RULES, {"profile_id": "p", "professor_name": name, "rag_context": {}})
            self.assertEqual((got["name_normalized"], got["points"]["name"]), (diagnosis.UNREAD_NAME, 20), name)

    def test_sixty_four_points_with_nineteen_passages_is_not_ready(self) -> None:
        got = quality.evaluate_profile(diagnosis.NAME_RULES, legend())
        self.assertEqual((got["score"], got["status"]), (64, "needs_data_repair"))
        self.assertEqual(got["points"], {"name": 20, "chunks": 10, "coverage": 14, "avg_len": 20})

    def test_least_passages_is_read_off_the_tiers(self) -> None:
        self.assertEqual(diagnosis.least_passages(54), 20)   # name + 3 sections + long passages
        self.assertEqual(diagnosis.least_passages(40), 20)   # re-collected: no core section named
        self.assertEqual(diagnosis.least_passages(34), 40)
        self.assertEqual(diagnosis.least_passages(20), 80)
        self.assertIsNone(diagnosis.least_passages(0))
        self.assertEqual(diagnosis.least_passages(40, 70), 40, "a stricter gate (STORY_PROFILE_MIN_QUALITY_SCORE) needs more")
        self.assertEqual(diagnosis.least_passages(40, 75), 80)


class DiagnosisTests(unittest.TestCase):
    def test_a_near_miss_says_it_is_one_passage_short_and_what_a_re_collection_needs(self) -> None:
        row = diagnosis.diagnose(legend(), url_list=[WIKI, MUSEUM, "https://lux.example.edu/person/ada"])
        self.assertFalse(row["ready"])
        self.assertFalse(row["label_is_stale"], "the stored 64 is what a recomputation gives")
        self.assertIn("19 passages stored; ready needs at least 20", row["reasons"])
        self.assertTrue(any(r.startswith("section titles name 3 of 5 core sections; missing about, publications") for r in row["reasons"]))
        self.assertTrue(row["needs"][0].startswith("1 more passages, 20 in all"), row["needs"])
        self.assertIn("collected again (sections titled by page, coverage 0): at least 20 passages", row["needs"][1])
        self.assertTrue(row["identity"]["eligible"])
        self.assertEqual(row["writer_gates"]["reviewed_creditable_passages"], 19)
        self.assertEqual([u["gave_passages"] for u in row["url_list"]], [True, True, False])
        self.assertEqual(row["child_bio"], "ready")
        text = diagnosis.format_row({**row, "found": True})
        self.assertIn("recomputed now: needs_data_repair 64 = name 20 + passages 10 (19 stored)", text)

    def test_what_needs_a_person_is_named(self) -> None:
        doc = legend(5, label="needs_data_repair", score=20)
        doc["professor_name"] = "unknown"
        doc["name"] = {"full": "unknown"}
        # Sections the name reader does not look in, so no name is found.
        chunks = [c for cs in doc["rag_context"]["section_chunks"].values() for c in cs]
        doc["rag_context"] = {"chunk_count": 17, "section_chunks": {"Lab Notes": chunks}}
        doc["identity_review"] = {**doc["identity_review"], "name_check": "misspelled", "usual_spelling": "Ada Brightwaters"}
        row = diagnosis.diagnose(doc, url_list=[], duplicates=[{"profile_id": "f0d7838e", "passages": 72, "status": "ready", "score": 80}])
        human = " | ".join(row["needs_a_person"])
        self.assertIn("its name could not be read", human)
        self.assertEqual(row["now"]["points"]["name"], 20, "the fallback name still scores")
        self.assertIn("misspelled (usually Ada Brightwaters)", human)
        self.assertIn("no stored URL list", human)
        self.assertIn("f0d7838e (72 passages, ready 80)", human)
        self.assertIn("rag_context.chunk_count says 17 but section_chunks holds 5 passages (the score uses 17)", row["reasons"])
        self.assertTrue(row["label_is_stale"], "the stored 20 is not what the rules give now")


class IdsFileTests(unittest.TestCase):
    def test_an_ids_file_written_by_powershell_5_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.txt"
            path.write_bytes("﻿emmett-chappelle\r\nwelton-taylor\r\n# a note\r\n".encode("utf-8"))
            self.assertEqual(parse_ids("", str(path)), ["emmett-chappelle", "welton-taylor"])


@needs_scraping_stack
class UrlRuleTests(unittest.TestCase):
    def test_only_pages_the_collector_allows_and_a_story_could_credit_are_fetched(self) -> None:
        kept, dropped = repair.screen_urls([
            WIKI, NASA, "https://www.researchgate.net/profile/Ada", "https://scholar.google.com/citations?user=a",
            "https://www.academia.edu/ada", "https://www.linkedin.com/in/ada", "https://www.nytimes.com/1999/ada.html",
            "https://www.bbc.com/news/ada", "https://www.facebook.com/ada", "ftp://files.example.org/ada",
        ])
        self.assertEqual(kept, [WIKI, NASA])
        self.assertEqual(len(dropped), 8)
        self.assertIn("could not credit", next(d["reason"] for d in dropped if "nytimes" in d["url"]))

    def test_the_stored_list_comes_first_without_repeats_and_within_the_limit(self) -> None:
        merged = repair.merge_urls([WIKI, MUSEUM], ["http://en.wikipedia.org/wiki/Ada_Brightwater/", NASA, SMITHSONIAN], 3)
        self.assertEqual(merged, [WIKI, MUSEUM, NASA])


@needs_scraping_stack
class CollectorTests(unittest.TestCase):
    def _collect(self, always_search: bool) -> Any:
        import asyncio

        scholar = collect.ScholarRow(row_number=1, scholar_id=PID, full_name=NAME, field="Chemistry", slug=PID)
        searches = Searches()
        with mock.patch.object(collect, "_get_json", side_effect=wikipedia_api), mock.patch.object(collect, "brave_search", searches), \
                mock.patch.dict(os.environ, {"BRAVE_SEARCH_API": "test-key"}):
            result = asyncio.run(collect.collect_for_scholar(scholar, max_urls=50, sleep_s=0, per_query=15, always_search=always_search))
        return result, searches

    def test_a_first_collection_searches_only_when_wikipedia_gives_under_four_seeds(self) -> None:
        # Wikipedia gives the article and NASA (books.google and ResearchGate
        # are filtered out of its links): two seeds, so it searches.
        result, searches = self._collect(always_search=False)
        self.assertEqual(len(searches.queries), 19)
        kept = same_pages(result["kept_urls"])
        self.assertNotIn(repair._url_key("https://scholar.google.com/citations?user=ada"), kept)
        self.assertNotIn(repair._url_key("https://www.facebook.com/ada.brightwater"), kept)

    def test_a_repair_searches_even_with_enough_seeds(self) -> None:
        with mock.patch.object(collect, "_filter_wikipedia_extlinks", return_value=[NASA, MUSEUM, COLLEGE]):
            stopped, none = self._collect(always_search=False)
            searched, some = self._collect(always_search=True)
        self.assertEqual(none.queries, [], "four seeds: the first collection does not search")
        self.assertEqual(len(some.queries), 19)
        self.assertIn(repair._url_key(SMITHSONIAN), same_pages(searched["kept_urls"]))
        self.assertNotIn(repair._url_key(SMITHSONIAN), same_pages(stopped["kept_urls"]))


@needs_scraping_stack
class EnvFileTests(unittest.TestCase):
    def test_the_repair_refuses_env_files_for_its_process(self) -> None:
        import dotenv

        self.assertEqual(dotenv.load_dotenv.__name__, "no_env_file")
        self.assertFalse(dotenv.load_dotenv())

    def test_the_sync_builder_handed_a_database_reads_no_env_file(self) -> None:
        from sync_local_chunked_to_mongodb import LocalChunkedMongoSync
        import sync_local_chunked_to_mongodb as sync_module

        with mock.patch.object(sync_module, "load_dotenv", side_effect=AssertionError("no .env")), \
                mock.patch.dict(os.environ, {"MONGODB_URI": ""}):
            builder = LocalChunkedMongoSync(use_llm=False, database={"legend_scholars": object()})
        self.assertIsNone(builder.mongo_client)
        builder.close()


@needs_scraping_stack
class ModelPathTests(unittest.TestCase):
    def test_every_model_path_of_the_pipeline_is_refused_before_a_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            pipeline = repair.make_pipeline(Path(tmp) / "out")
            self.assertIsNone(pipeline.llm_cleaner)
            with self.assertRaisesRegex(RuntimeError, "calls no model"):
                pipeline.chunking_pipeline.process_profile("p", "text")
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"PROFILE_ROLE_FILTER_ENABLED": "1", "PROFILE_ROLE_FILTER_USE_LLM": "1", "OPENAI_API_KEY": "sk-test-not-used"}):
            with self.assertRaisesRegex(RuntimeError, "page role filter"):
                repair.make_pipeline(Path(tmp) / "out")


# ── Against the local test database ─────────────────────────────────────────

@needs_local_mongo
class DiagnosisDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.local = LocalDatabase()
        self.addCleanup(self.local.close)
        dup = legend(25, label="ready", score=74)
        dup.update(_id="f0d7838e-dup", profile_id="f0d7838e-dup")
        self.local.db["legend_scholars"].insert_many([legend(), dup])
        self.local.db["legend_scholar_daily_stories"].insert_one({"story_key": "s1", "status": "failed_validation", "scholar": {"profile_id": PID}})
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        Path(self.tmp.name, f"{PID}.txt").write_text(f"{WIKI}\n{COLLEGE}\n", encoding="utf-8")

    def test_the_diagnosis_reads_through_a_read_only_database_and_loads_no_env_file(self) -> None:
        import daily_story_worker as worker_module
        import dotenv

        before = snapshot(self.local.db)
        out = io.StringIO()
        refused = mock.Mock(side_effect=AssertionError("no .env"))
        with mock.patch.dict(os.environ, {diagnosis.URI_ENV: self.local.uri, diagnosis.DB_ENV: ""}), \
                mock.patch.object(dotenv, "load_dotenv", refused), mock.patch.object(worker_module, "load_dotenv", refused), \
                mock.patch.object(sys, "argv", ["legend_profile_diagnosis.py", "--legend-ids", f"{PID},missing-legend", "--url-lists-dir", self.tmp.name]), \
                contextlib.redirect_stdout(out):
            self.assertEqual(diagnosis.main(), 0)
        text = out.getvalue()
        self.assertIn(f"{PID}  {NAME}  NOT READY", text)
        self.assertIn("19 passages stored; ready needs at least 20", text)
        self.assertIn("another profile of the same name, f0d7838e-dup (25 passages, ready 74)", text)
        self.assertIn(f"stored URL list (2): 1 gave stored passages; 1 gave none:\n    - {COLLEGE}", text)
        self.assertIn("stories: failed_validation 1", text)
        self.assertIn("missing-legend: NOT FOUND", text)
        self.assertEqual(snapshot(self.local.db), before)

        client, db = connect_read_only(self.local.uri, label="test", db_name=self.local.name)
        self.addCleanup(client.close)
        with self.assertRaises(ReadOnlyViolation):
            db["legend_scholars"].update_one({"_id": PID}, {"$set": {"x": 1}})

    def test_without_its_address_it_does_not_start(self) -> None:
        with mock.patch.dict(os.environ, {diagnosis.URI_ENV: ""}), mock.patch.object(sys, "argv", ["x", "--legend-ids", PID]), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(diagnosis.main(), 2)


@needs_scraping_stack
@needs_local_mongo
class RepairTests(unittest.TestCase):
    """The full re-collection (--recollect)."""

    def setUp(self) -> None:
        self.env = mock.patch.dict(os.environ, {**QUIET_ENV, "BRAVE_SEARCH_API": "test-key", "OPENAI_API_KEY": "", "LEGEND_IDENTITY_MODEL": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.local = LocalDatabase()
        self.addCleanup(self.local.close)
        ready = legend(30, label="ready", score=74)
        ready.update(_id="ready-legend", profile_id="ready-legend", professor_name="Ready Legend", name={"full": "Ready Legend"})
        ready = _identified({k: v for k, v in ready.items() if k != "identity_review"})
        stale = legend(30, label="needs_data_repair", score=40)
        stale.update(_id="stale-label", profile_id="stale-label", professor_name="Stale Label", name={"full": "Stale Label"})
        stale = _identified({k: v for k, v in stale.items() if k != "identity_review"})
        self.local.db["legend_scholars"].insert_many([legend(), ready, stale])
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.lists = Path(self.tmp.name, "final")
        self.lists.mkdir()
        (self.lists / f"{PID}.txt").write_text(
            "\n".join([WIKI, MUSEUM, "https://www.researchgate.net/profile/Ada_Brightwater",
                       "https://scholar.google.com/citations?user=ada", "https://www.nytimes.com/1999/ada.html"]) + "\n",
            encoding="utf-8",
        )
        (self.lists / f"{PID}.meta.json").write_text(json.dumps({"profile_id": PID, "profile_name": NAME, "profile_url": WIKI, "field": "Chemistry", "row_number": 7}), encoding="utf-8")
        self.ids = Path(self.tmp.name, "pilot_ids.txt")
        self.ids.write_bytes(f"﻿{PID}\r\nready-legend\r\nstale-label\r\nnot-a-legend\r\n".encode("utf-8"))
        self.work = Path(self.tmp.name, "work")
        self.judge = mock.Mock(side_effect=judge)

    def _args(self, **overrides: Any) -> argparse.Namespace:
        values = {
            "legend_ids_file": str(self.ids), "legend_ids": "", "dry_run": False, "max_urls": 50, "sleep_seconds": 0,
            "per_query_results": 15, "min_quality": 60, "url_lists_dir": str(self.lists), "work_dir": str(self.work),
            "fresh": False, "model": None, "price": [], "recollect": True, "fetch": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def _run(self, scraper: Scraper, searches: Searches, **overrides: Any) -> Any:
        args = self._args(**overrides)
        worker = make_worker(self.local.uri, structured=self.judge, read_only=args.dry_run)
        self.addCleanup(worker.close)
        out = io.StringIO()
        with mock.patch.object(collect, "_get_json", side_effect=wikipedia_api), mock.patch.object(collect, "brave_search", searches), \
                contextlib.redirect_stdout(out):
            code = repair.run(args, worker=worker, scraper=scraper)
        text = out.getvalue()
        if args.dry_run:
            return code, text
        return code, json.loads(text[text.rindex('{\n  "run_id"'):])

    def _results(self, report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {r["profile_id"]: r for r in report["results"]}

    def test_a_dry_run_writes_fetches_and_asks_nothing_and_shows_its_arithmetic(self) -> None:
        before = snapshot(self.local.db)
        scraper, searches = Scraper(), Searches()
        code, text = self._run(scraper, searches, dry_run=True)
        self.assertEqual(code, 0)
        self.assertEqual(snapshot(self.local.db), before)
        self.assertFalse(self.work.exists(), "nothing is saved to disk")
        self.assertEqual((scraper.asked, searches.queries, self.judge.call_count), ([], [], 0))
        self.assertIn(f"{PID} ({NAME}): repair, needs_data_repair 64: 19 passages stored; ready needs at least 20", text)
        self.assertIn("5 in its stored list (3 left out:", text)
        self.assertIn("ready-legend (Ready Legend): ready, ready 74: nothing to fetch or write", text)
        self.assertIn("stale-label (Stale Label): relabel: stored needs_data_repair 40, its passages score ready 74: would correct the label only", text)
        self.assertIn("not-a-legend: not found", text)
        self.assertIn("Brave Search API: 19 = 19 queries", text)
        self.assertIn("Wikipedia API: 3 to 6 requests a legend x 1 legends to look up = 3 to 6", text)
        self.assertIn("Seed pages: at most 50", text)
        self.assertRegex(text, r"Identity review, ada-brightwater \[gemini-2\.5-flash\]: 1 calls x \([\d,]+ in x \$0\.30/M \+ [\d,]+\.\.8,192 out x \$2\.50/M\)")
        self.assertIn("Total: $", text)

    def test_a_repair_writes_only_the_passages_and_the_label_then_the_legend_is_writable(self) -> None:
        scraper, searches = Scraper(), Searches()
        code, report = self._run(scraper, searches)
        self.assertEqual(code, 0, report)
        results = self._results(report)
        ada = results[PID]
        self.assertEqual(ada["outcome"], "written", ada)
        self.assertEqual(same_pages(scraper.asked), same_pages([WIKI, MUSEUM, NASA, SMITHSONIAN, COLLEGE]))
        for asked in scraper.asked:
            self.assertFalse(any(bad in asked for bad in NEVER_FETCHED), asked)
        self.assertEqual(len(searches.queries), 19)
        self.assertEqual(ada["candidate"]["status"], "ready")
        self.assertGreaterEqual(ada["candidate"]["passages"], 20)
        self.assertTrue(ada["identity"]["eligible"], ada["identity"])
        self.assertEqual(ada["now"], {"quality_gate": True, "status": "ready", "score": ada["candidate"]["score"], "blocked_by": [], "writable": True})

        doc = self.local.db["legend_scholars"].find_one({"_id": PID})
        self.assertEqual(doc["daily_story_profile"]["status"], "ready")
        self.assertEqual(doc["rag_context"]["chunk_count"], ada["candidate"]["passages"])
        self.assertTrue(any(key.startswith("Dr. Ada Brightwater") for key in doc["rag_context"]["section_chunks"]))
        self.assertEqual(doc["about"]["bio_by_band"]["grade_3_5"]["text"], "Ada liked light.", "the bios are kept")
        self.assertEqual((doc["display"], doc["admin"], doc["name"]), ({"visibility": "published"}, {"review_status": "curated"}, {"full": NAME, "display": NAME}))
        self.assertEqual(legend_identity.story_eligibility(doc), (True, legend_identity.VERIFIED))
        row = self.local.db["daily_story_profile_quality"].find_one({"profile_id": PID})
        self.assertEqual((row["status"], row["quality_score"]), ("ready", ada["candidate"]["score"]))
        backup = self.local.db[repair.BACKUPS_COLLECTION].find_one({"profile_id": PID})
        self.assertEqual(backup["previous"]["rag_context"]["chunk_count"], 19)
        self.assertEqual(backup["previous"]["daily_story_profile"], {"status": "needs_data_repair", "quality_score": 64})

        self.assertEqual(results["ready-legend"]["outcome"], "already_ready")
        self.assertEqual(results["stale-label"]["outcome"], "relabeled")
        stale = self.local.db["legend_scholars"].find_one({"_id": "stale-label"})
        self.assertEqual(stale["daily_story_profile"]["status"], "ready")
        self.assertEqual(results["not-a-legend"]["outcome"], "not_found")
        self.assertEqual(self.judge.call_count, 1, "only the re-sourced legend is judged again")
        self.assertEqual(self.local.db[repair.RUNS_COLLECTION].count_documents({}), 1)

        # Idempotent: a second run finds every legend ready and does nothing.
        self.judge.reset_mock()
        again, searches_again = Scraper(), Searches()
        before = snapshot(self.local.db)
        code, report = self._run(again, searches_again)
        self.assertEqual(code, 0)
        self.assertEqual({r["outcome"] for r in report["results"]} - {"not_found"}, {"already_ready"})
        self.assertEqual((again.asked, searches_again.queries, self.judge.call_count), ([], [], 0))
        after = snapshot(self.local.db)
        after.pop(repair.RUNS_COLLECTION)
        before.pop(repair.RUNS_COLLECTION)
        self.assertEqual(after, before, "nothing but the run record is written the second time")

    def test_a_candidate_that_falls_short_is_not_written_and_a_rerun_reuses_the_fetch(self) -> None:
        before = self.local.db["legend_scholars"].find_one({"_id": PID})
        thin, searches = Scraper(sentences=6), Searches()
        code, report = self._run(thin, searches, legend_ids_file="", legend_ids=PID)
        ada = self._results(report)[PID]
        self.assertEqual(code, 0)
        self.assertEqual(ada["outcome"], "not_written")
        self.assertTrue(any("would score needs_data_repair" in s for s in ada["still_short"]), ada)
        self.assertEqual(self.local.db["legend_scholars"].find_one({"_id": PID}), before, "the stored profile is as it was")
        self.assertEqual(self.local.db[repair.BACKUPS_COLLECTION].count_documents({}), 0)
        self.assertEqual(self.judge.call_count, 0)

        again, searches_again = Scraper(sentences=6), Searches()
        code, report = self._run(again, searches_again, legend_ids_file="", legend_ids=PID)
        self.assertEqual(self._results(report)[PID]["outcome"], "not_written")
        self.assertEqual((again.asked, searches_again.queries), ([], []), "saved work is reused: no search, no fetch")

        fresh, searches_fresh = Scraper(sentences=6), Searches()
        self._run(fresh, searches_fresh, legend_ids_file="", legend_ids=PID, fresh=True)
        self.assertEqual(len(searches_fresh.queries), 19)
        self.assertEqual(len(fresh.asked), 5)

    def test_a_collection_made_without_a_search_key_is_redone_once_one_is_set(self) -> None:
        with mock.patch.dict(os.environ, {"BRAVE_SEARCH_API": ""}):
            unsearched, no_requests = Scraper(sentences=6), Searches()
            code, report = self._run(unsearched, no_requests, legend_ids_file="", legend_ids=PID)
        ada = self._results(report)[PID]
        self.assertEqual((ada["search_ran"], no_requests.queries), (False, []))
        self.assertEqual(same_pages(unsearched.asked), same_pages([WIKI, MUSEUM, NASA]), "the stored list and Wikipedia's links only")

        searched, searches = Scraper(), Searches()
        code, report = self._run(searched, searches, legend_ids_file="", legend_ids=PID)
        ada = self._results(report)[PID]
        self.assertEqual((ada["search_ran"], len(searches.queries), ada["outcome"]), (True, 19, "written"))
        self.assertEqual(len(searched.asked), 5)

    def test_a_run_stopped_after_its_fetch_resumes_without_fetching_again(self) -> None:
        scraper, searches = Scraper(), Searches()
        with mock.patch.object(repair, "write_candidate", side_effect=RuntimeError("the connection dropped")):
            code, report = self._run(scraper, searches, legend_ids_file="", legend_ids=PID)
        self.assertEqual(code, 1)
        self.assertEqual(self._results(report)[PID]["outcome"], "error")
        self.assertEqual(self.local.db["legend_scholars"].find_one({"_id": PID})["daily_story_profile"]["status"], "needs_data_repair")

        again, searches_again = Scraper(), Searches()
        code, report = self._run(again, searches_again, legend_ids_file="", legend_ids=PID)
        self.assertEqual(code, 0)
        self.assertEqual(self._results(report)[PID]["outcome"], "written")
        self.assertEqual((again.asked, searches_again.queries), ([], []))

    def test_a_dry_run_after_a_fetch_says_what_the_saved_pages_would_give(self) -> None:
        with mock.patch.object(repair, "write_candidate", side_effect=RuntimeError("stopped")):
            self._run(Scraper(), Searches(), legend_ids_file="", legend_ids=PID)
        before = snapshot(self.local.db)
        code, text = self._run(Scraper(), Searches(), legend_ids_file="", legend_ids=PID, dry_run=True)
        self.assertEqual(code, 0)
        self.assertRegex(text, r"saved work: pages fetched \S+ \(5 URLs\): would be ready \d+ with \d+ passages, \d+ creditable from 5 pages: would write")
        self.assertIn("Brave Search API: 0 = 0 queries", text)
        self.assertRegex(text, r"Identity review, ada-brightwater \[gemini-2\.5-flash\]: 1 calls x \([\d,]+ in .*input from its prompt")
        self.assertEqual(snapshot(self.local.db), before)

    def test_without_a_model_or_ids_it_does_not_start(self) -> None:
        with mock.patch.dict(os.environ, {"STORY_LLM_MODEL": "", "LEGEND_IDENTITY_MODEL": ""}), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(repair.run(self._args(), worker=mock.Mock()), 2)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(repair.run(self._args(legend_ids_file=""), worker=mock.Mock()), 2)
        with mock.patch.dict(os.environ, {repair.URI_ENV: ""}), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(repair.run(self._args()), 2)


def stored_bytes(rag_context: Dict[str, Any]) -> Dict[str, List[bytes]]:
    """Each stored section's passages as BSON, the bytes MongoDB keeps."""
    import bson

    return {section: [bson.encode(c) for c in chunks] for section, chunks in rag_context["section_chunks"].items()}


@needs_scraping_stack
class AppendMergeTests(unittest.TestCase):
    def test_stored_passages_stay_as_they_are_and_duplicates_are_skipped(self) -> None:
        stored = legend()["rag_context"]
        stored["section_text"] = {"legacy": "the stored legacy text"}
        stored["source_catalog"] = [{"source_url": WIKI}]
        stored["chunk_source_map"] = {}
        before = copy.deepcopy(stored)
        first_stored = stored["section_chunks"]["biography"][0]
        new = {
            "section_chunks": {
                "Dr. Ada Brightwater, page 3": [
                    {"chunk_id": "src_nasa:0", "text": "  " + first_stored["text"].upper() + "  "},  # a stored passage again
                    {"chunk_id": "src_nasa:1", "text": "Ada Brightwater measured the light of fireflies at night."},
                    {"chunk_id": "src_nasa:2", "text": "ada brightwater measured   the light of fireflies at night."},  # the line above again
                ],
                "legacy": [{"chunk_id": f"{PID}-c0", "text": "Ada Brightwater's notebooks are kept at the college."}],
            },
            "source_catalog": [{"source_url": WIKI}, {"source_url": NASA}],
            "chunk_source_map": {"src_nasa:1": [{"source_url": NASA}], f"{PID}-c0": [{"source_url": COLLEGE}]},
        }
        merged, added, duplicates = repair.appended_rag_context(stored, new, record={"at": "now", "urls": [NASA]})
        self.assertEqual((added, duplicates), (2, 2))
        self.assertEqual(stored, before, "the stored rag_context handed in is not changed")
        after = stored_bytes(merged)
        for section, chunks in stored_bytes(before).items():
            self.assertEqual(after[section][: len(chunks)], chunks, section)
        self.assertEqual(list(merged["section_chunks"])[:3], list(before["section_chunks"]), "stored titles keep their place")
        legacy = merged["section_chunks"]["legacy"]
        self.assertEqual(len(legacy), len(before["section_chunks"]["legacy"]) + 1)
        self.assertEqual(legacy[-1]["chunk_id"], f"{PID}-c0:appended:{19 + 1}", "a clashing id is renamed, the stored one kept")
        self.assertEqual(legacy[-1]["order"], 20)
        self.assertEqual(merged["section_chunks"]["Dr. Ada Brightwater, page 3"][0]["order"], 19)
        self.assertEqual(merged["chunk_count"], 21)
        self.assertEqual(merged["sections_available"], [*before["sections_available"], "Dr. Ada Brightwater, page 3"])
        self.assertTrue(merged["section_text"]["legacy"].startswith("the stored legacy text\n\n"))
        self.assertEqual([s["source_url"] for s in merged["source_catalog"]], [WIKI, NASA])
        self.assertEqual(set(merged["chunk_source_map"]), {"src_nasa:1", f"{PID}-c0:appended:20"})
        self.assertEqual(merged["appended"], [{"at": "now", "urls": [NASA], "passages": 2, "duplicates_skipped": 2}])


@needs_scraping_stack
@needs_local_mongo
class AppendTests(unittest.TestCase):
    """The default: fetch the stored list's pages that gave no passage (and
    Wikipedia's links not yet stored) and add their passages to the stored
    ones."""

    WIKI_LINKS = [NASA, SMITHSONIAN, "https://books.google.com/books?id=ada", "https://www.researchgate.net/profile/Ada_Brightwater"]
    FORBIDDEN = [
        "https://www.researchgate.net/profile/Ada_Brightwater", "https://scholar.google.com/citations?user=ada",
        "https://www.nytimes.com/1999/ada.html", "https://www.facebook.com/ada.brightwater", "ftp://files.example.org/ada",
    ]

    def setUp(self) -> None:
        self.env = mock.patch.dict(os.environ, {**QUIET_ENV, "BRAVE_SEARCH_API": "", "OPENAI_API_KEY": "", "LEGEND_IDENTITY_MODEL": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.local = LocalDatabase()
        self.addCleanup(self.local.close)
        self.local.db["legend_scholars"].insert_one(legend())
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.lists = Path(self.tmp.name, "final")
        self.lists.mkdir()
        (self.lists / f"{PID}.txt").write_text("\n".join([WIKI, MUSEUM, NASA, COLLEGE, *self.FORBIDDEN]) + "\n", encoding="utf-8")
        (self.lists / f"{PID}.meta.json").write_text(json.dumps({"profile_id": PID, "profile_name": NAME, "profile_url": WIKI, "field": "Chemistry"}), encoding="utf-8")
        self.work = Path(self.tmp.name, "work")
        self.judge = mock.Mock(side_effect=judge)
        self.wikipedia = mock.Mock(side_effect=wikipedia_api_linking(self.WIKI_LINKS))

    def _run(self, scraper: Scraper, *, dry_run: bool = False, fetch: bool = False, cli: List[str] = ()) -> Any:
        args = repair.build_arg_parser().parse_args([
            "--legend-ids", PID, "--url-lists-dir", str(self.lists), "--work-dir", str(self.work), *cli,
            *(["--dry-run"] if dry_run else []), *(["--fetch"] if fetch else []),
        ])
        worker = make_worker(self.local.uri, structured=self.judge, read_only=dry_run)
        self.addCleanup(worker.close)
        out = io.StringIO()
        searches = Searches()
        with mock.patch.object(collect, "_get_json", self.wikipedia), mock.patch.object(collect, "brave_search", searches), \
                contextlib.redirect_stdout(out):
            code = repair.run(args, worker=worker, scraper=scraper)
        self.assertEqual(searches.queries, [], "an append never searches")
        text = out.getvalue()
        if dry_run:
            return code, text
        return code, {r["profile_id"]: r for r in json.loads(text[text.rindex('{\n  "run_id"'):])["results"]}

    def _stored(self) -> Dict[str, Any]:
        return self.local.db["legend_scholars"].find_one({"_id": PID})

    def test_append_is_the_default(self) -> None:
        self.assertFalse(repair.build_arg_parser().parse_args(["--legend-ids-file", "ids.txt"]).recollect)

    def test_only_pages_that_gave_nothing_are_fetched_and_the_stored_passages_are_kept_byte_for_byte(self) -> None:
        before = self._stored()["rag_context"]
        scraper = Scraper()
        code, results = self._run(scraper)
        ada = results[PID]
        self.assertEqual((code, ada["outcome"], ada["mode"]), (0, "written", "append"), ada)
        # NASA and the college from the stored list; the Smithsonian from
        # Wikipedia's links. Not the pages that already gave passages, not
        # the article (stored), not a URL the rules refuse.
        self.assertEqual(same_pages(scraper.asked), same_pages([NASA, COLLEGE, SMITHSONIAN]))
        for asked in scraper.asked:
            self.assertFalse(any(bad in asked for bad in NEVER_FETCHED), asked)
        self.assertEqual(ada["candidate"]["before"], 19)
        self.assertEqual(ada["candidate"]["passages"], 19 + ada["candidate"]["added"])

        after = self._stored()["rag_context"]
        stored = stored_bytes(after)
        for section, chunks in stored_bytes(before).items():
            self.assertEqual(stored[section][: len(chunks)], chunks, f"{section}: stored passages kept as they were")
        self.assertEqual(list(after["section_chunks"])[:3], ["biography", "background_and_work", "legacy"])
        self.assertEqual(after["sections_available"][:3], ["biography", "background_and_work", "legacy"])
        scored = quality.evaluate_profile(diagnosis.NAME_RULES, self._stored())
        self.assertEqual(scored["points"]["coverage"], 14, "the stored titles keep their coverage")
        self.assertEqual((self._stored()["daily_story_profile"]["status"], scored["status"]), ("ready", "ready"))
        backup = self.local.db[repair.BACKUPS_COLLECTION].find_one({"profile_id": PID})
        self.assertEqual(stored_bytes(backup["previous"]["rag_context"]), stored_bytes(before))
        self.assertEqual(backup["policy"], repair.APPEND_POLICY_VERSION)
        self.assertTrue(ada["identity"]["eligible"])
        self.assertEqual(self.judge.call_count, 1, "the whole legend is judged again: its sources changed")
        self.assertTrue(ada["now"]["writable"])

        again = Scraper()
        code, results = self._run(again)
        self.assertEqual((results[PID]["outcome"], again.asked, self.judge.call_count), ("already_ready", [], 1))

    def test_no_write_below_the_gate(self) -> None:
        self.local.db["legend_scholars"].replace_one({"_id": PID}, legend(10, score=58))
        before = self._stored()
        code, results = self._run(Scraper(sentences=6))
        ada = results[PID]
        self.assertEqual(ada["outcome"], "not_written")
        self.assertLess(ada["candidate"]["passages"], 20)
        self.assertTrue(any("would score needs_data_repair" in s for s in ada["still_short"]), ada)
        self.assertEqual(self._stored(), before)
        self.assertEqual(self.local.db[repair.BACKUPS_COLLECTION].count_documents({}), 0)
        self.assertEqual(self.judge.call_count, 0)

    def test_a_passage_already_stored_is_skipped(self) -> None:
        self.local.db["legend_scholars"].replace_one({"_id": PID}, legend(10, score=58))
        code, results = self._run(Scraper(sentences=6))
        first = results[PID]["candidate"]
        # Store one passage the pipeline made from a fetched page, as if an
        # earlier run had kept it; the saved pages are used again.
        state = json.loads((self.work / PID / repair.STATE_FILES[repair.APPEND]).read_text(encoding="utf-8"))
        chunks = json.loads(Path(state["fetched"]["chunks_path"]).read_text(encoding="utf-8"))["sections"]
        text = next(c["text"] for cs in chunks.values() for c in cs)
        self.local.db["legend_scholars"].update_one({"_id": PID}, {"$push": {"rag_context.section_chunks.legacy": {"chunk_id": "kept", "text": text, "primary_source_url": MUSEUM}}})
        again = Scraper(sentences=6)
        code, results = self._run(again)
        second = results[PID]["candidate"]
        self.assertEqual(again.asked, [], "the saved pages are reused")
        self.assertEqual((second["added"], second["duplicates_skipped"]), (first["added"] - 1, first["duplicates_skipped"] + 1))

    def test_a_rag_context_changed_meanwhile_is_not_overwritten(self) -> None:
        db = self.local.db

        class Meddling(Scraper):
            async def extract_all(inner, url: str) -> Dict[str, Any]:
                if not inner.asked:  # another writer adds a passage while the pages are fetched
                    db["legend_scholars"].update_one({"_id": PID}, {"$push": {"rag_context.section_chunks.legacy": {"chunk_id": "theirs", "text": "x" * 500}}})
                return await Scraper.extract_all(inner, url)

        code, results = self._run(Meddling())
        self.assertEqual((code, results[PID]["outcome"]), (1, "error"))
        self.assertIn("changed since this run read it", results[PID]["error"])
        self.assertEqual(self._stored()["rag_context"]["section_chunks"]["legacy"][-1]["chunk_id"], "theirs")
        self.assertEqual(db[repair.BACKUPS_COLLECTION].count_documents({}), 0)

    def test_a_dry_run_lists_the_urls_and_with_fetch_says_whether_each_would_pass(self) -> None:
        before = snapshot(self.local.db)
        scraper = Scraper()
        code, text = self._run(scraper, dry_run=True)
        self.assertEqual((code, scraper.asked, self.wikipedia.call_count), (0, [], 0))
        self.assertFalse(self.work.exists(), "a plain dry run saves nothing")
        self.assertIn(f"{PID} ({NAME}): repair, needs_data_repair 64: 19 passages stored", text)
        self.assertIn("stored list: 9 URLs; 2 gave stored passages; 5 left out by the rules", text)
        self.assertIn(f"would fetch the 2 that gave none:\n        - {NASA}\n        - {COLLEGE}\n", text)
        self.assertIn("after appending: 19 + what these pages give; it passes only with at least 1 new passages (20 in all", text)
        self.assertIn("Brave Search API: none (an append does not search)", text)
        self.assertIn("Wikipedia API: 3 to 6 requests a legend x 1 legends to look up", text)

        code, text = self._run(scraper, dry_run=True, fetch=True)
        self.assertEqual(code, 0)
        self.assertEqual(same_pages(scraper.asked), same_pages([NASA, COLLEGE, SMITHSONIAN]))
        self.assertEqual((snapshot(self.local.db), self.judge.call_count), (before, 0), "--fetch writes nothing to the database, asks no model")
        self.assertRegex(text, r"fetched \S+: 3 URLs \(2 from the stored list, 1 from Wikipedia\):")
        self.assertRegex(text, r"after appending: 19 \+ \d+ new = \d+ passages, ready \d+, \d+ creditable from \d+ pages: WOULD PASS and be written")

        again = Scraper()
        code, results = self._run(again)
        self.assertEqual((results[PID]["outcome"], again.asked), ("written", []), "the run uses the pages the dry run fetched")


if __name__ == "__main__":
    unittest.main()
