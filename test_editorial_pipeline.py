"""Offline checks for the nightly editorial pipeline's pure logic.

The admin scheduler runs six of this repo's scripts every night from a
container (see requirements-editorial.txt). daily_story_personalize.py was
missing from the repository, so the image never had it and the audience
personalisation stage failed every night. These tests cover what can be
checked without a database or a model: which grade band a reader falls in,
how text is measured, what makes a variant unservable, and which database the
scripts resolve to.

Run with:  python -m unittest test_editorial_pipeline
"""
import os
import unittest
from unittest import mock

import daily_story_personalize as personalize
from config.mongodb_utils import DEFAULT_DB_NAME, resolve_mongo_db_name


class GradeBandTests(unittest.TestCase):
    def test_every_school_grade_has_one_band(self) -> None:
        expected = {0: "grade_k_2", 2: "grade_k_2", 3: "grade_3_5", 5: "grade_3_5", 6: "grade_6_8", 8: "grade_6_8", 9: "grade_9_12", 12: "grade_9_12"}
        for grade, band in expected.items():
            self.assertEqual(personalize._band_for_grade(grade), band, grade)
        self.assertIsNone(personalize._band_for_grade(None))

    def test_bands_are_ordered_and_do_not_leave_gaps(self) -> None:
        bands = list(personalize.GRADE_BANDS.values())
        for lower, upper in zip(bands, bands[1:]):
            self.assertEqual(upper["min_level"], lower["max_level"] + 1)
            self.assertLessEqual(lower["max_sentence_words"], upper["max_sentence_words"])

    def test_a_reader_grade_is_read_from_any_known_field_and_never_invented(self) -> None:
        self.assertEqual(personalize._resolve_grade({"gradeLevel": "4"}), 4)
        self.assertEqual(personalize._resolve_grade({"grade_level": 7}), 7)
        self.assertEqual(personalize._resolve_grade({"personalizationConfig": {"gradeLevel": 10}}), 10)
        self.assertIsNone(personalize._resolve_grade({"gradeLevel": 14}), "beyond school is not a school band")
        self.assertIsNone(personalize._resolve_grade({"gradeLevel": "college"}))
        self.assertIsNone(personalize._resolve_grade({}))


class ReadabilityTests(unittest.TestCase):
    def test_harder_text_measures_higher(self) -> None:
        easy = personalize.readability_metrics("The cat sat on the mat. It was a warm day. The dog ran to the park.")
        hard = personalize.readability_metrics(
            "Institutional accountability mechanisms systematically differentiate organizational responsibilities "
            "across heterogeneous administrative jurisdictions, necessitating comprehensive interdisciplinary evaluation."
        )
        self.assertLess(easy["flesch_kincaid_grade"], hard["flesch_kincaid_grade"])
        self.assertEqual(easy["sentence_count"], 3)


class VariantValidationTests(unittest.TestCase):
    def _variant(self, words: int, **overrides):
        sentence = "The scientist checks the river water every day."
        per = len(sentence.split())
        article = " ".join([sentence] * max(1, words // per))
        variant = {
            "title": "River watch",
            "summary": "A scientist studies a river.",
            "article_markdown": article + " The gauge shows the water level. Sediment is sand in the water.",
            "teacher_note": "Ask what the scientist measures.",
            "glossary": [{"term": "gauge"}, {"term": "sediment"}],
            "key_takeaways": ["a", "b", "c"],
            "comprehension_questions": [{}, {}, {}],
            "used_evidence_ids": ["e1", "e2"],
        }
        variant.update(overrides)
        return variant

    def test_a_variant_inside_its_band_has_no_blockers(self) -> None:
        blockers, metrics = personalize._validate_variant(self._variant(400), "grade_3_5", {"e1", "e2", "e3"})
        self.assertEqual(blockers, [], metrics)

    def test_what_makes_a_variant_unservable(self) -> None:
        cases = {
            "missing_title": {"title": ""},
            "question_count_out_of_range": {"comprehension_questions": [{}]},
            "glossary_count_out_of_range": {"glossary": [{"term": "gauge"}]},
            "unsupported_evidence_ids": {"used_evidence_ids": ["e1", "nope"]},
            "glossary_term_missing_from_article": {"glossary": [{"term": "gauge"}, {"term": "photosynthesis"}]},
        }
        for expected, overrides in cases.items():
            blockers, _ = personalize._validate_variant(self._variant(400, **overrides), "grade_3_5", {"e1", "e2"})
            self.assertTrue(any(b.startswith(expected) for b in blockers), f"{expected}: {blockers}")

    def test_length_and_level_are_held_to_the_band(self) -> None:
        blockers, _ = personalize._validate_variant(self._variant(60), "grade_3_5", {"e1", "e2"})
        self.assertTrue(any(b.startswith("word_count_out_of_range") for b in blockers))
        hard = self._variant(
            400,
            article_markdown=(
                "Institutional accountability mechanisms systematically differentiate organizational responsibilities. " * 40
                + "The gauge shows it. Sediment is there."
            ),
        )
        blockers, _ = personalize._validate_variant(hard, "grade_3_5", {"e1", "e2"})
        self.assertTrue(any(b.startswith("readability_out_of_range") for b in blockers), blockers)


class FakeClient:
    def __init__(self, names):
        self.names = names

    def list_database_names(self):
        return list(self.names)


class DatabaseNameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("MONGODB_DB", None)

    def tearDown(self) -> None:
        self.env.stop()

    def test_named_database_wins(self) -> None:
        self.assertEqual(resolve_mongo_db_name("mongodb+srv://h/archivyn?retryWrites=true"), "archivyn")
        self.assertEqual(resolve_mongo_db_name("mongodb+srv://h/"), DEFAULT_DB_NAME)

    def test_explicit_setting_wins_over_the_uri(self) -> None:
        os.environ["MONGODB_DB"] = "chosen"
        self.assertEqual(resolve_mongo_db_name("mongodb+srv://h/archivyn"), "chosen")

    def test_with_a_client_an_existing_named_database_is_used(self) -> None:
        client = FakeClient(["admin", "ngo_profiles", "archivyn"])
        self.assertEqual(resolve_mongo_db_name("mongodb+srv://h/archivyn", client=client), "archivyn")
        self.assertEqual(resolve_mongo_db_name("mongodb+srv://h/missing", client=client), DEFAULT_DB_NAME)

    def test_an_unnamed_fallback_is_announced(self) -> None:
        client = FakeClient(["admin", "local", "other"])
        with mock.patch("sys.stderr") as stderr:
            self.assertEqual(resolve_mongo_db_name("mongodb+srv://h/missing", client=client), "other")
        self.assertTrue(stderr.write.called, "a database nobody named is not used silently")

    def test_an_unreachable_cluster_falls_back_to_the_named_one(self) -> None:
        class Broken:
            def list_database_names(self):
                raise RuntimeError("down")

        self.assertEqual(resolve_mongo_db_name("mongodb+srv://h/archivyn", client=Broken()), "archivyn")


class TemplateDraftTests(unittest.TestCase):
    """A template is never sent to review (2026-10-29: 29 of the first 30
    legend drafts in the queue were templates, rated "every check passed")."""

    def test_a_template_draft_fails_and_says_which(self) -> None:
        import daily_story_worker as worker_module

        error = worker_module.template_failure(["source_url_enforced", "llm_non_json_fallback_generated"])
        self.assertEqual(error, "template_not_article:llm_non_json_fallback_generated")
        self.assertIsNotNone(worker_module.template_failure(["strict_validation_fallback_pass"]))

    def test_a_real_draft_or_a_dry_run_is_not_failed(self) -> None:
        import daily_story_worker as worker_module

        self.assertIsNone(worker_module.template_failure(["json_repair_pass", "reader_sanitized"]))
        self.assertIsNone(worker_module.template_failure(None))
        self.assertIsNone(worker_module.template_failure(["llm_non_json_fallback_generated"], dry_run=True))


class FakeCollection:
    def __init__(self, stories):
        self.stories = stories
        self.updates = []

    def find(self, _filter):
        return self

    def sort(self, _order):
        return self

    def limit(self, _n):
        return list(self.stories)

    def update_one(self, selector, update):
        self.updates.append((selector, update))


class FakeWorker:
    model_name = "gemini-test"

    def __init__(self, *args, **kwargs):
        self.stories_collection = FakeCollection([STORY])

    def close(self):
        pass


STORY = {
    "_id": "s1",
    "story_key": "k1",
    "status": "pending_review",
    "content": {
        "title": "Historical Lens on a drone strike",
        "standfirst": "A strike killed 41 people.",
        "article_markdown": "## Why This Matters Now\nThe strike [chunk:a] ...",
    },
    "trend_issue": {"title": "Drone strike by paramilitary kills 41"},
    "claim_evidence_map": [{"chunk_ids": ["a"]}],
}


class TopicSuitabilityTests(unittest.TestCase):
    """Whether a band may meet the story's news at all is ruled on before a
    version is written; a gentle rewrite of a killing is still a killing."""

    def test_every_band_has_rules(self) -> None:
        self.assertEqual(set(personalize.TOPIC_SUITABILITY_RULES), set(personalize.GRADE_BANDS))

    def test_a_ruling_is_reused_only_for_the_same_text_and_rules(self) -> None:
        verdict = {"suitable": False, "reason": "x", "canonical_content_hash": "h1", "policy_version": personalize.TOPIC_POLICY_VERSION}
        self.assertEqual(personalize.current_topic_verdict({"grade_k_2": verdict}, "grade_k_2", "h1"), verdict)
        self.assertIsNone(personalize.current_topic_verdict({"grade_k_2": verdict}, "grade_k_2", "h2"))
        self.assertIsNone(personalize.current_topic_verdict({"grade_k_2": {**verdict, "policy_version": "old"}}, "grade_k_2", "h1"))
        self.assertIsNone(personalize.current_topic_verdict({}, "grade_k_2", "h1"))

    def test_the_judge_rules_per_band_and_a_band_without_rules_is_never_suitable(self) -> None:
        with mock.patch.object(personalize, "_call_structured", return_value={"suitable": False, "reason": "A killing is not for this age."}) as call:
            verdict = personalize.judge_topic_for_band(FakeWorker(), STORY, "grade_k_2", "h1")
        self.assertFalse(verdict["suitable"])
        self.assertEqual(verdict["reason"], "A killing is not for this age.")
        self.assertEqual(verdict["policy_version"], personalize.TOPIC_POLICY_VERSION)
        self.assertIn("Drone strike by paramilitary kills 41", call.call_args.args[1])
        with mock.patch.dict(personalize.TOPIC_SUITABILITY_RULES, {}, clear=True):
            self.assertFalse(personalize.judge_topic_for_band(FakeWorker(), STORY, "grade_k_2", "h1")["suitable"])
        with mock.patch.object(personalize, "_call_structured", return_value={"reason": "no verdict"}):
            with self.assertRaises(RuntimeError):
                personalize.judge_topic_for_band(FakeWorker(), STORY, "grade_k_2", "h1")

    def test_an_unsuitable_band_gets_no_version_and_the_ruling_is_kept(self) -> None:
        worker = FakeWorker()
        audience = {"band_keys": ["grade_k_2"]}
        with mock.patch.object(personalize, "DailyStoryWorker", return_value=worker),                 mock.patch.object(personalize, "resolve_active_audience", return_value=audience),                 mock.patch.object(personalize, "_call_structured", return_value={"suitable": False, "reason": "Not for this age."}) as call:
            result = personalize.run_personalization(story_date="2026-10-29", max_stories=5, dry_run=False, no_llm=False)
        self.assertEqual(result["summary"]["not_suitable"], 1)
        self.assertEqual(call.call_count, 1, "only the ruling: no version was written")
        update = worker.stories_collection.updates[0][1]["$set"]
        self.assertEqual([v["status"] for v in update["audience_variants"]], ["not_suitable"])
        self.assertFalse(update["audience_suitability"]["grade_k_2"]["suitable"])
        self.assertTrue(update["personalization.ready"], "a band ruled out is settled")


if __name__ == "__main__":
    unittest.main()
