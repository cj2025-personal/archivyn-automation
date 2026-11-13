"""Evergreen legend stories, their chosen-list runner, the grade 3-5 bios and
the selection report.

No model is called: every model answer is a stub. The tests that need a
database use only the local test MongoDB at 127.0.0.1:27018, each in a fresh
database dropped afterwards, and are skipped when it is not running (the job
image's CI has none). Nothing here reads a .env file.

Run with:  python -m unittest test_legend_evergreen
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import unittest
import uuid
from datetime import date
from typing import Any, Dict, List
from unittest import mock

import daily_story_personalize as personalize
import daily_story_retry_failed as retry_stage
import daily_story_worker as worker_module
import legend_child_bio as bios
import legend_evergreen as evergreen
import legend_selection_report as selection
import model_costs
import source_licences
from readonly_mongo import ReadOnlyCollection, ReadOnlyDatabase, ReadOnlyViolation, write_stage_in
from test_editorial_pipeline import _identified

LOCAL_MONGO = "mongodb://127.0.0.1:27018"
TODAY = "2026-11-12"


def _local_mongo_up() -> bool:
    try:
        from pymongo import MongoClient

        client = MongoClient(LOCAL_MONGO, serverSelectionTimeoutMS=800)
        client.admin.command("ping")
        client.close()
        return True
    except Exception:
        return False


LOCAL_UP = _local_mongo_up()
needs_local_mongo = unittest.skipUnless(LOCAL_UP, "needs the local test MongoDB at 127.0.0.1:27018")

QUIET_ENV = {
    "STORY_ML_ENABLED": "0",
    "STORY_ENFORCE_PROFILE_QUALITY": "",
    "STORY_LLM_MODEL": "gemini-2.5-flash",
    "GCP_PROJECT_ID": "",
    "MONGODB_DB": "",
    "EDITORIAL_REVIEW_MODEL": "",
}

PAGES = [
    "https://en.wikipedia.org/wiki/Mae_Jemison",
    "https://www.nasa.gov/people/mae-jemison",
    "https://library.example.edu/archives/jemison",
]


def passage(name: str, index: int) -> str:
    """A passage about the legend: third person, dated, no causal wording,
    long enough that two make a section."""
    year = 1970 + index
    return (
        f"{name} began project {index} in {year}, studying how people stay healthy in space and on Earth. "
        f"In {year + 1} {name} shared careful notes with students, nurses and young doctors at clinics and schools. "
        f"Lesson {index} explained stars, medicine and engineering with simple tools that children could hold. "
        f"Teachers described those classes as clear, patient and full of real experiments, and each project "
        f"closed with a public talk about science, health and asking good questions."
    )


def legend_doc(profile_id: str = "mae-jemison", name: str = "Mae Jemison", count: int = 12, quality: str = "ready",
               pages: List[str] = PAGES, **extra: Any) -> Dict[str, Any]:
    sections: Dict[str, List[Dict[str, Any]]] = {"about": [], "background_and_work": [], "milestones": []}
    names = list(sections)
    for index in range(count):
        sections[names[index % 3]].append({
            "chunk_id": f"{profile_id}-c{index}",
            "text": passage(name, index),
            "source_url": pages[index % len(pages)],
        })
    doc = {
        "profile_id": profile_id,
        "professor_name": name,
        "name": {"full": name},
        "about": {
            "short_bio": (
                f"{name} is an internationally recognized physician, engineer and astronaut whose interdisciplinary "
                "contributions to biomedical instrumentation and aerospace medicine transformed institutional practice."
            ),
            "field_of_study": "Medicine and spaceflight",
        },
        "rag_context": {"section_chunks": {k: v for k, v in sections.items() if v}},
        "daily_story_profile": {"status": quality, "quality_score": 80 if quality == "ready" else 20},
        **extra,
    }
    return _identified(doc)


def _blocks_after(prompt: str, label: str) -> str:
    return prompt.split(label, 1)[1] if label in prompt else ""


def fake_evergreen_writer(name: str):
    """A stand-in for the model that writes an evergreen story: it reads the
    passages and allowed links from the prompt it is given, and writes the
    evergreen sections from them, each cited in its first sentence."""

    def write(prompt: str, max_tokens: int = 0) -> Dict[str, Any]:
        chunks = re.findall(r"\[chunk_id=([^\]]+)\]\[section=[^\]]+\]\n(.+)", prompt)
        urls = re.findall(r"^- (https?://\S+)$", _blocks_after(prompt, "Allowed source URLs").split("Required JSON")[0], flags=re.M)
        assert chunks, "the prompt carries its passages"
        sections = []
        for index, heading in enumerate(worker_module.EVERGREEN_SECTION_BLUEPRINT):
            paragraphs = []
            for offset in (0, 1):
                chunk_id, text = chunks[(2 * index + offset) % len(chunks)]
                paragraphs.append(f"[chunk:{chunk_id}] {text}")
            sections.append(f"## {heading}\n" + "\n\n".join(paragraphs))
        return {
            "title": f"How {name} Studied Health in Space",
            "standfirst": f"A story about the life and work of {name}, from the sources about her.",
            "article_markdown": "\n\n".join(sections),
            "trend_source_url": "https://news.example.org/should-be-cleared",
            "source_urls": urls,
            "claim_evidence_map": [
                {"claim": text.split(". ")[0], "chunk_ids": [chunk_id], "support_summary": "Stated in the passage."}
                for chunk_id, text in chunks[:4]
            ],
            "used_chunk_ids": [chunk_id for chunk_id, _ in chunks[:6]],
            "editor_notes": ["written"],
        }

    return write


class LocalDatabase:
    """A fresh database on the local test MongoDB, dropped afterwards."""

    def __init__(self) -> None:
        from pymongo import MongoClient

        self.name = f"legend_evergreen_test_{uuid.uuid4().hex[:12]}"
        self.uri = f"{LOCAL_MONGO}/{self.name}"
        self.client = MongoClient(LOCAL_MONGO)
        self.db = self.client[self.name]

    def close(self) -> None:
        self.client.drop_database(self.name)
        self.client.close()


def make_worker(uri: str, *, writer=None, structured=None, read_only: bool = False, **kwargs: Any) -> worker_module.DailyStoryWorker:
    """A worker on the local test database whose model calls are stubs: it
    is made without a model, then told it has one."""
    worker = worker_module.DailyStoryWorker(
        use_llm=False, trends_enabled=False, mongodb_uri=uri, load_env_file=False, read_only=read_only,
        model_name="gemini-2.5-flash", **kwargs,
    )
    if writer or structured:
        worker.use_llm = True
        worker._call_llm_json = writer or mock.Mock(side_effect=AssertionError("no writer expected"))
        worker._call_structured_json = structured or mock.Mock(side_effect=AssertionError("no judge expected"))
    return worker


# ── Pure pieces ──────────────────────────────────────────────────────────────


class ReadOnlyWrapperTests(unittest.TestCase):
    def test_reads_pass_and_every_write_is_refused_before_it_is_sent(self) -> None:
        collection = mock.Mock()
        collection.name = "legend_scholars"
        wrapped = ReadOnlyCollection(collection, "test")
        wrapped.find({"a": 1})
        collection.find.assert_called_once_with({"a": 1})
        for method in ("insert_one", "insert_many", "update_one", "update_many", "replace_one", "delete_one",
                       "delete_many", "bulk_write", "create_index", "drop", "find_one_and_update", "watch"):
            with self.assertRaises(ReadOnlyViolation, msg=method):
                getattr(wrapped, method)
        with self.assertRaises(ReadOnlyViolation):
            wrapped.aggregate([{"$match": {}}, {"$facet": {"x": [{"$merge": "elsewhere"}]}}])
        with self.assertRaises(ReadOnlyViolation):
            wrapped.anything = 1
        self.assertEqual(write_stage_in([{"$group": {"_id": None}}]), None)
        self.assertEqual(write_stage_in([{"$out": "x"}]), "$out")

    def test_a_database_hands_out_only_read_only_collections(self) -> None:
        db = ReadOnlyDatabase({"legend_scholars": mock.Mock()}, "test")
        self.assertIsInstance(db["legend_scholars"], ReadOnlyCollection)
        for method in ("command", "create_collection", "drop_collection"):
            with self.assertRaises(ReadOnlyViolation):
                getattr(db, method)


class SourceLicenceTests(unittest.TestCase):
    def test_what_is_known_about_a_page_s_terms(self) -> None:
        self.assertEqual(source_licences.licence_for("https://en.wikipedia.org/wiki/X")["id"], "CC-BY-SA-4.0")
        self.assertEqual(source_licences.licence_for("https://www.nasa.gov/people/x")["status"], source_licences.OPEN)
        self.assertEqual(source_licences.licence_for("https://www.nytimes.com/2026/a.html")["status"], source_licences.REFUSED)
        unknown = source_licences.licence_for("https://library.example.edu/x")
        self.assertEqual(unknown["status"], source_licences.UNKNOWN, "not checked is not the same as refused")
        self.assertEqual(unknown["host"], "library.example.edu")
        self.assertEqual(source_licences.licence_for("https://fakewikipedia.org.evil.com/x")["status"], source_licences.UNKNOWN)


class EvergreenTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = mock.patch.dict(os.environ, {**QUIET_ENV, "MONGODB_URI": "mongodb://fake/test"})
        self.env.start()
        self.addCleanup(self.env.stop)
        from test_editorial_pipeline import _story_worker

        self.worker = _story_worker()

    def test_the_evergreen_brief_has_its_own_sections_and_no_news(self) -> None:
        chunks = [{"chunk_id": f"c{i}", "section": "about", "text": passage("Mae Jemison", i)} for i in range(4)]
        prompt = self.worker._build_prompt("Mae Jemison", "health in space", chunks, None, {}, PAGES, True, mode="evergreen")
        for heading in worker_module.EVERGREEN_SECTION_BLUEPRINT:
            self.assertIn(heading, prompt)
        self.assertNotIn("Why This Matters Now", prompt)
        self.assertNotIn("headline", prompt.lower())
        self.assertIn("says nothing about current events", prompt)
        news = self.worker._build_prompt("Mae Jemison", "health in space", chunks, None, {}, PAGES, True)
        self.assertIn("Why This Matters Now", news, "the news mode is kept")

    def test_the_structure_check_holds_each_story_to_its_own_template(self) -> None:
        chunks = [{"chunk_id": f"c{i}", "section": "about", "text": passage("Mae Jemison", i)} for i in range(12)]
        prompt = self.worker._build_evergreen_prompt("Mae Jemison", "health", chunks, PAGES, True)
        article = fake_evergreen_writer("Mae Jemison")(prompt)["article_markdown"] + "\n\n## Sources\n- " + PAGES[0]
        self.assertEqual(self.worker._validate_standard_article_structure(article, mode="evergreen"), [])
        news_errors = self.worker._validate_standard_article_structure(article)
        self.assertIn("missing_required_section:why_now", news_errors)

    def test_an_evergreen_disclosure_says_what_it_is(self) -> None:
        text = worker_module.DailyStoryWorker._build_disclosure("Mae Jemison", True, "evergreen")
        self.assertIn("AI-generated story about Mae Jemison", text)
        self.assertNotIn("perspective", text)


class CostTests(unittest.TestCase):
    def test_the_arithmetic_is_shown_and_an_unknown_model_is_not_guessed(self) -> None:
        step = model_costs.Step("Writer", "gemini-2.5-flash", 10, 4000, 3500, 24000)
        low, high = step.cost()
        self.assertAlmostEqual(low, 10 * (4000 * 0.30 + 3500 * 2.50) / 1e6)
        self.assertAlmostEqual(high, 10 * (4000 * 0.30 + 24000 * 2.50) / 1e6)
        self.assertIn("10 calls x (4,000 in x $0.30/M + 3,500..24,000 out x $2.50/M)", step.arithmetic())
        self.assertEqual(model_costs.price_of("gemini-2.5-flash-001"), model_costs.PRICES["gemini-2.5-flash"])
        with self.assertRaises(KeyError):
            model_costs.price_of("gemini-9-ultra")
        self.assertEqual(model_costs.price_of("gemini-9-ultra", model_costs.parse_price_overrides(["gemini-9-ultra=1/2"])), (1.0, 2.0))


class ChildBioCheckTests(unittest.TestCase):
    def test_the_reading_grade_is_measured_as_the_learner_api_measures_it(self) -> None:
        # Measured with the learner API's own measureReadability
        # (backend/src/lib/readability.ts) on 2026-11-12.
        self.assertEqual(bios.learner_api_fk_grade(
            "Mae Jemison was a doctor and an astronaut. In 1992 she flew into space. She studied how the body works far from Earth. She wanted kids to love science."
        ), 1.9)
        self.assertEqual(bios.learner_api_fk_grade(
            "Institutional accountability mechanisms systematically differentiate organizational responsibilities across heterogeneous administrative jurisdictions."
        ), 45.6)
        self.assertEqual(bios.learner_api_fk_grade(
            "Dr. Charles Drew found a way to store blood plasma. Blood banks saved many lives in the 1940s. He taught young doctors at Howard University."
        ), 2.9)

    def _calls(self, bios_in_order, unsupported=(), suitable=True):
        """A stub model: each writer call returns the next bio; the checks
        answer as told."""
        queue = list(bios_in_order)
        unsupported = list(unsupported)
        prompts: List[str] = []

        def call(prompt, schema, _max_tokens):
            prompts.append(prompt)
            if schema is bios.BIO_SCHEMA:
                return {"bio": queue.pop(0), "used_passage_ids": ["P1", "P2"]}
            if schema is bios.GROUNDING_SCHEMA:
                return {"unsupported_claims": unsupported.pop(0) if unsupported else [], "reason": "checked"}
            return {"suitable": suitable, "reason": "fits the band" if suitable else "too frightening"}

        return call, prompts

    PASSAGES = [{"id": "P1", "chunk_id": "c1", "section": "about", "text": "x", "url": PAGES[0]},
                {"id": "P2", "chunk_id": "c2", "section": "about", "text": "y", "url": PAGES[1]}]
    EASY = ("Mae Jemison was a doctor. She was also an astronaut. She flew in space. She did tests on her body there. "
            "She liked to teach kids. She wanted them to love science. She told them to ask big questions. "
            "She worked hard all her life. Many kids heard her talks.")
    HARD = ("Mae Jemison, an internationally recognized physician and engineer, contributed interdisciplinary "
            "expertise to biomedical instrumentation, aerospace medicine and institutional educational initiatives "
            "throughout an extraordinarily distinguished professional career spanning numerous organizations.")

    def test_a_bio_that_passes_every_check_is_ready_with_its_record(self) -> None:
        call, prompts = self._calls([self.EASY])
        result = bios.write_bio("Mae Jemison", self.PASSAGES, call)
        self.assertEqual(result["status"], "ready")
        self.assertLessEqual(result["readability"]["flesch_kincaid_grade"], 5.0)
        self.assertLessEqual(result["readability"]["learner_api_fk_grade"], 5.0)
        self.assertEqual([p["chunk_id"] for p in result["passages"]], ["c1", "c2"])
        self.assertEqual(len(prompts), 3, "writer, grounding, suitability")

    def test_too_hard_is_revised_with_its_numbers(self) -> None:
        call, prompts = self._calls([self.HARD, self.EASY])
        result = bios.write_bio("Mae Jemison", self.PASSAGES, call)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["attempts"], 2)
        self.assertIn("must read at grade 5 or below", prompts[1])

    def test_an_unsupported_fact_is_taken_out_and_an_unsuitable_bio_is_not_stored(self) -> None:
        call, prompts = self._calls([self.EASY, self.EASY], unsupported=[["She was born in Alabama."]])
        self.assertEqual(bios.write_bio("Mae Jemison", self.PASSAGES, call)["status"], "ready")
        self.assertTrue(any("Remove this claim" in p and "Alabama" in p for p in prompts))
        call, _ = self._calls([self.EASY], suitable=False)
        self.assertEqual(bios.write_bio("Mae Jemison", self.PASSAGES, call)["status"], "not_suitable")
        call, _ = self._calls([self.HARD, self.HARD, self.HARD])
        self.assertEqual(bios.write_bio("Mae Jemison", self.PASSAGES, call)["status"], "failed")


class EvergreenTopicRulingTests(unittest.TestCase):
    def test_an_evergreen_story_is_ruled_on_as_a_life_not_a_headline(self) -> None:
        story = {"story_mode": "evergreen", "content": {"title": "A life", "standfirst": "s", "article_markdown": "## Who They Were\nText."}}
        prompt = personalize.topic_prompt(story, "grade_3_5")
        self.assertIn("life and work of a historical scholar", prompt)
        self.assertNotIn("News headline", prompt)
        self.assertIn(personalize.TOPIC_SUITABILITY_RULES["grade_3_5"], prompt)
        news = personalize.topic_prompt({"content": {}, "trend_issue": {"title": "Election"}}, "grade_3_5")
        self.assertIn("News headline it answers: Election", news)


# ── With the local test database ────────────────────────────────────────────


@needs_local_mongo
class EvergreenWritingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = mock.patch.dict(os.environ, QUIET_ENV)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.local = LocalDatabase()
        self.addCleanup(self.local.close)
        self.local.db["legend_scholars"].insert_many([
            legend_doc(),
            legend_doc("katherine-johnson", "Katherine Johnson"),
            legend_doc("thin-legend", "Thin Legend", count=3),
            legend_doc("blocked-legend", "Blocked Legend", quality="needs_data_repair"),
        ])

    def _args(self, **overrides: Any) -> argparse.Namespace:
        values = {
            "legend_ids": "mae-jemison,katherine-johnson",
            "legend_ids_file": "",
            "stories_each": 2,
            "limit": None,
            "dry_run": False,
            "date": TODAY,
            "model": None,
            "max_context_chunks": 12,
            "ignore_profile_quality": False,
            "price": [],
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def _run(self, **overrides: Any) -> str:
        """legend_evergreen.run against the local database, its model a stub."""
        real = worker_module.DailyStoryWorker

        def factory(**kwargs: Any):
            worker = real(**{**kwargs, "use_llm": False})
            if not kwargs.get("read_only"):
                worker.use_llm = True
                writers = {"mae-jemison": fake_evergreen_writer("Mae Jemison"), "katherine-johnson": fake_evergreen_writer("Katherine Johnson")}

                def write(prompt: str, max_tokens: int = 0):
                    name = "Katherine Johnson" if "Katherine Johnson" in prompt.split("\n", 1)[0] else "Mae Jemison"
                    return writers["katherine-johnson" if name == "Katherine Johnson" else "mae-jemison"](prompt, max_tokens)

                worker._call_llm_json = write
                worker._call_structured_json = mock.Mock(side_effect=AssertionError("a passing draft needs no repair"))
            return worker

        out = io.StringIO()
        with mock.patch.dict(os.environ, {evergreen.URI_ENV: self.local.uri}), \
                mock.patch.object(evergreen, "DailyStoryWorker", side_effect=factory), \
                contextlib.redirect_stdout(out):
            self.last_code = evergreen.run(self._args(**overrides))
        return out.getvalue()

    def _stories(self) -> List[Dict[str, Any]]:
        return list(self.local.db["legend_scholar_daily_stories"].find({}, {"_id": 0}).sort("story_key", 1))

    def test_a_chosen_list_gets_n_evergreen_stories_each_through_every_check(self) -> None:
        self._run()
        self.assertEqual(self.last_code, 0)
        stories = self._stories()
        self.assertEqual(
            [s["story_key"] for s in stories],
            ["katherine-johnson:evergreen:1", "katherine-johnson:evergreen:2", "mae-jemison:evergreen:1", "mae-jemison:evergreen:2"],
        )
        for story in stories:
            self.assertEqual(story["status"], "pending_review", story["safety"]["errors"])
            self.assertTrue(story["safety"]["validation_passed"])
            self.assertEqual(story["story_mode"], "evergreen")
            self.assertEqual(story["story_date"], TODAY)
            self.assertEqual(story["trend_issue"]["selected"], {})
            self.assertEqual(story["trend_issue"]["verified_source_url"], "", "no news item, whatever the writer said")
            self.assertIn("AI-generated story about", story["content"]["disclosure"])
            record = story["source_record"]
            self.assertTrue(record["all_allowed"])
            self.assertTrue(record["all_cited_attributed"])
            self.assertIsNone(record["news_item"])
            by_host = {row["licence"]["host"]: row["licence"]["status"] for row in record["pages"]}
            self.assertEqual(by_host["en.wikipedia.org"], "open")
            self.assertEqual(by_host["library.example.edu"], "unknown")
            for row in record["pages"]:
                self.assertIn(row["url"], story["content"]["article_markdown"], "every page it cites is credited under Sources")
        mae = [s for s in stories if s["scholar"]["profile_id"] == "mae-jemison"]
        self.assertNotEqual(mae[0]["evergreen"]["focus"], mae[1]["evergreen"]["focus"], "each slot has its own focus")

    def test_running_it_again_writes_nothing_and_a_failed_slot_is_written_again(self) -> None:
        self._run()
        first = {s["story_key"]: s["generation"]["generated_at"] for s in self._stories()}
        self._run()
        self.assertEqual({s["story_key"]: s["generation"]["generated_at"] for s in self._stories()}, first)
        self.local.db["legend_scholar_daily_stories"].update_one(
            {"story_key": "mae-jemison:evergreen:2"}, {"$set": {"status": "failed_validation"}}
        )
        output = self._run()
        self.assertIn('"generated": 1', output)
        after = {s["story_key"]: s for s in self._stories()}
        self.assertEqual(after["mae-jemison:evergreen:2"]["status"], "pending_review")
        self.assertEqual(after["mae-jemison:evergreen:1"]["generation"]["generated_at"], first["mae-jemison:evergreen:1"])

    def test_the_limit_caps_a_run_and_the_next_run_resumes(self) -> None:
        self._run(limit=3)
        self.assertEqual(len(self._stories()), 3)
        self._run(limit=3)
        self.assertEqual(len(self._stories()), 4)

    def test_legends_the_gates_keep_out_are_not_written_and_say_why(self) -> None:
        output = self._run(legend_ids="blocked-legend,thin-legend,no-such-legend")
        self.assertEqual(self._stories(), [])
        self.assertIn("profile_quality_blocked", output)
        self.assertIn('"thin-legend": "too_few_passages:3"', output)
        self.assertIn('"no-such-legend": "not_found"', output)
        self.assertEqual(self.last_code, 0)

    def test_a_dry_run_writes_nothing_calls_no_model_and_prices_the_real_prompts(self) -> None:
        names_before = sorted(self.local.db.list_collection_names())
        output = self._run(dry_run=True)
        self.assertEqual(sorted(self.local.db.list_collection_names()), names_before)
        self.assertEqual(self._stories(), [])
        self.assertIn("Would write 4 stories", output)
        self.assertIn("COST ESTIMATE, 4 stories", output)
        self.assertRegex(output, r"Writer: the story \[gemini-2\.5-flash\]: 4 calls x \([\d,]+ in x \$0\.30/M")
        self.assertIn("AI editor review [gemini-2.5-pro]", output)
        self.assertIn("Total: $", output)

    def test_a_dry_run_cannot_write_even_by_mistake(self) -> None:
        worker = make_worker(self.local.uri, read_only=True)
        self.addCleanup(worker.close)
        with self.assertRaises(ReadOnlyViolation):
            worker.stories_collection.insert_one({"story_key": "x"})

    def test_the_nightly_run_with_no_news_item_writes_an_evergreen_story(self) -> None:
        worker = make_worker(self.local.uri, writer=fake_evergreen_writer("Mae Jemison"), structured=mock.Mock())
        self.addCleanup(worker.close)
        result = worker.run(
            scholar_id="mae-jemison", date_value=date.fromisoformat(TODAY), topic_override=None,
            max_scholars=1, max_context_chunks=12, dry_run=False,
        )
        self.assertEqual(result["summary"]["evergreen"], 1)
        self.assertNotIn("degraded_no_trend_data", result["summary"])
        self.assertEqual(result["summary"]["generated"], 1, result["errors"])
        story = self.local.db["legend_scholar_daily_stories"].find_one({"story_key": f"mae-jemison:{TODAY}"})
        self.assertEqual(story["story_mode"], "evergreen")
        self.assertIsNone(story["evergreen"]["slot"])
        self.assertEqual(story["status"], "pending_review")

    def test_the_retry_stage_writes_a_failed_evergreen_slot_again_in_its_slot(self) -> None:
        self._run(legend_ids="mae-jemison", stories_each=1)
        self.local.db["legend_scholar_daily_stories"].update_one(
            {"story_key": "mae-jemison:evergreen:1"}, {"$set": {"status": "failed_validation"}}
        )
        worker = make_worker(self.local.uri, writer=fake_evergreen_writer("Mae Jemison"), structured=mock.Mock())
        with mock.patch.object(retry_stage, "DailyStoryWorker", return_value=worker):
            result = retry_stage.run_retry_failed(
                story_date=TODAY, statuses=["failed_validation"], reuse_topic=True,
                max_context_chunks=12, dry_run=False, no_llm=False,
            )
        self.assertEqual(result["summary"]["regenerated"], 1, result["errors"])
        keys = [s["story_key"] for s in self._stories()]
        self.assertEqual(keys, ["mae-jemison:evergreen:1"], "the slot is written again, no night's story added")
        self.assertEqual(self._stories()[0]["status"], "pending_review")

    def test_the_versions_stage_takes_evergreen_stories_of_any_day(self) -> None:
        stories = self.local.db["legend_scholar_daily_stories"]
        stories.insert_many([
            {"story_key": "a:evergreen:1", "story_date": "2026-11-10", "status": "pending_review", "story_mode": "evergreen", "scholar": {"profile_id": "a"}},
            {"story_key": "b:evergreen:1", "story_date": "2026-11-12", "status": "pending_review", "story_mode": "evergreen", "scholar": {"profile_id": "b"}},
            {"story_key": "c:evergreen:1", "story_date": "2026-11-12", "status": "pending_review", "story_mode": "evergreen", "scholar": {"profile_id": "c"}, "personalization": {"ready": True}},
            {"story_key": "d:2026-11-12", "story_date": "2026-11-12", "status": "pending_review", "story_mode": "news", "scholar": {"profile_id": "d"}},
        ])
        with mock.patch.dict(os.environ, {"MONGODB_URI": self.local.uri}), mock.patch.object(worker_module, "load_dotenv"):
            result = personalize.run_personalization(story_date="2026-11-01", max_stories=10, dry_run=True, no_llm=False, story_mode="evergreen")
        self.assertEqual(result["summary"]["stories_scanned"], 2, "unsettled evergreen stories, whatever their date")


@needs_local_mongo
class ChildBioRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = mock.patch.dict(os.environ, QUIET_ENV)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.local = LocalDatabase()
        self.addCleanup(self.local.close)
        self.local.db["legend_scholars"].insert_many([legend_doc(), legend_doc("katherine-johnson", "Katherine Johnson")])

    def _run(self, **overrides: Any) -> str:
        values = {"legend_ids": "mae-jemison,katherine-johnson", "legend_ids_file": "", "limit": None, "force": False,
                  "dry_run": False, "model": None, "price": []}
        values.update(overrides)
        real = worker_module.DailyStoryWorker
        easy = ChildBioCheckTests.EASY

        def structured(prompt, schema, _max):
            if schema is bios.BIO_SCHEMA:
                return {"bio": easy, "used_passage_ids": ["P1", "P2"]}
            if schema is bios.GROUNDING_SCHEMA:
                return {"unsupported_claims": [], "reason": "all stated"}
            return {"suitable": True, "reason": "a life in science"}

        def factory(**kwargs: Any):
            worker = real(**{**kwargs, "use_llm": False})
            if not kwargs.get("read_only"):
                worker.use_llm = True
                worker._call_structured_json = structured
            return worker

        out = io.StringIO()
        with mock.patch.dict(os.environ, {bios.URI_ENV: self.local.uri}), \
                mock.patch.object(bios, "DailyStoryWorker", side_effect=factory), \
                contextlib.redirect_stdout(out):
            self.last_code = bios.run(argparse.Namespace(**values))
        return out.getvalue()

    def test_the_child_bio_is_stored_beside_the_adult_one_and_not_written_twice(self) -> None:
        self._run()
        self.assertEqual(self.last_code, 0)
        doc = self.local.db["legend_scholars"].find_one({"profile_id": "mae-jemison"})
        child = doc["about"]["bio_by_band"]["grade_3_5"]
        self.assertEqual(child["status"], "ready")
        self.assertEqual(child["text"], ChildBioCheckTests.EASY)
        self.assertLessEqual(child["readability"]["learner_api_fk_grade"], 5.0)
        self.assertTrue(doc["about"]["short_bio"].startswith("Mae Jemison is an internationally"), "the adult bio is kept")
        self.assertEqual(child["checks"]["suitability"]["band"], "grade_3_5")
        output = self._run()
        self.assertEqual(output.count('"outcome": "current"'), 2)
        self.assertEqual(self.local.db["legend_child_bio_runs"].count_documents({}), 2)

    def test_a_dry_run_writes_nothing_and_prices_the_real_prompts(self) -> None:
        output = self._run(dry_run=True)
        doc = self.local.db["legend_scholars"].find_one({"profile_id": "mae-jemison"})
        self.assertNotIn("bio_by_band", doc["about"])
        self.assertNotIn("legend_child_bio_runs", self.local.db.list_collection_names())
        self.assertIn("Would write 2 bios", output)
        self.assertRegex(output, r"Bio writer \[gemini-2\.5-flash\]: 2 calls x \([\d,]+ in")


@needs_local_mongo
class SelectionReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.local = LocalDatabase()
        self.addCleanup(self.local.close)
        wrong = legend_doc("wrong-person", "Wrong Person")
        wrong["identity_review"]["eligible"] = False
        self.local.db["legend_scholars"].insert_many([
            legend_doc(count=30),
            legend_doc("thin-legend", "Thin Legend", count=3),
            legend_doc("one-page", "One Page", count=10, pages=["https://library.example.edu/only"]),
            legend_doc("blocked-legend", "Blocked Legend", quality="needs_data_repair"),
            wrong,
        ])
        self.local.db["legend_scholar_daily_stories"].insert_many([
            {"story_key": "mae-jemison:2026-10-01", "status": "pending_review", "scholar": {"profile_id": "mae-jemison"},
             "audience_suitability": {"grade_3_5": {"suitable": True}, "grade_k_2": {"suitable": False}},
             "audience_variants": [{"band_key": "grade_3_5", "status": "ready", "validation": {"ok": True}}]},
            {"story_key": "one-page:2026-10-01", "status": "pending_review", "scholar": {"profile_id": "one-page"},
             "review": {"legal_hold": True},
             "audience_suitability": {"grade_3_5": {"suitable": False, "reason": "A killing is not for this age."}},
             "audience_variants": [{"band_key": "grade_3_5", "status": "not_suitable"}]},
        ])

    def _report(self):
        from readonly_mongo import connect_read_only

        client, db = connect_read_only(self.local.uri, label="test", db_name=self.local.name)
        self.addCleanup(client.close)
        return selection.build_report(db), db

    def test_cleared_legends_are_ranked_with_reasons_and_the_others_say_why_not(self) -> None:
        report, _ = self._report()
        self.assertEqual(report["legends_total"], 5)
        self.assertEqual(report["identity_cleared"], 4, "a legend its review did not clear is not offered")
        rows = {row["profile_id"]: row for row in report["rows"]}
        self.assertEqual([r["profile_id"] for r in report["rows"] if not r["blocked"]], ["mae-jemison"])
        self.assertIn("creditable passages", " ".join(rows["thin-legend"]["blocked"]))
        self.assertIn("one source page", " ".join(rows["one-page"]["blocked"]))
        self.assertIn("profile quality needs_data_repair", " ".join(rows["blocked-legend"]["blocked"]))
        mae = rows["mae-jemison"]
        self.assertEqual(mae["sources"]["creditable_passages"], 30)
        self.assertEqual(mae["sources"]["pages"], 3)
        self.assertAlmostEqual(mae["sources"]["open_share"], 0.667, places=2)
        self.assertEqual(mae["stories"]["ruled"]["grade_3_5"], {"suitable": 1, "not_suitable": 0})
        self.assertGreater(mae["bio"]["fk_grade"], 12, "the adult bio is adult prose")
        text = selection.format_report(report)
        self.assertIn("1 of 1 stories ruled fine for grades 3-5 (0 of 1 for K-2)", text)
        self.assertIn("Creative Commons Attribution-ShareAlike 4.0", text)
        self.assertGreater(rows["mae-jemison"]["score"]["fit"], rows["one-page"]["score"]["fit"])
        self.assertIn("ruled unsuitable because: A killing is not for this age.", selection.reasons_line(rows["one-page"]))

    def test_it_reads_only_and_will_not_run_without_its_own_address(self) -> None:
        _, db = self._report()
        with self.assertRaises(ReadOnlyViolation):
            db["legend_scholars"].update_one({}, {"$set": {"x": 1}})
        err = io.StringIO()
        with mock.patch.dict(os.environ, {selection.URI_ENV: ""}), mock.patch("sys.argv", ["legend_selection_report.py"]), contextlib.redirect_stderr(err):
            self.assertEqual(selection.main(), 2)
        self.assertIn(selection.URI_ENV, err.getvalue())


if __name__ == "__main__":
    unittest.main()
