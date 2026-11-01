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
from datetime import date
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

    def test_the_article_budget_leaves_room_for_the_model_s_thinking(self) -> None:
        import daily_story_worker as worker_module

        # The model thinks within the same budget; 6200 cut drafts short.
        self.assertGreaterEqual(worker_module.STRICT_MAX_OUTPUT_TOKENS, 16000)
        self.assertGreaterEqual(worker_module.DEFAULT_MAX_OUTPUT_TOKENS, 8000)

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


def _field(doc, path):
    for part in path.split("."):
        if not isinstance(doc, dict):
            return None
        doc = doc.get(part)
    return doc


def _matches(doc, query):
    for path, condition in query.items():
        value = _field(doc, path)
        if not isinstance(condition, dict):
            if value != condition:
                return False
            continue
        for op, arg in condition.items():
            if op == "$ne":
                ok = value != arg
            elif op == "$nin":
                ok = value not in arg
            elif op == "$gte":
                ok = value is not None and value >= arg
            elif op == "$lt":
                ok = value is not None and value < arg
            else:
                raise NotImplementedError(op)
            if not ok:
                return False
    return True


class FakeCursor(list):
    def sort(self, *_args, **_kwargs):
        return self

    def limit(self, n):
        return FakeCursor(self[:n])


class FakeMongoCollection:
    """The query operators the story worker uses, over documents in memory."""

    def __init__(self, docs=()):
        self.docs = [dict(d) for d in docs]

    def create_index(self, *_args, **_kwargs):
        pass

    def find(self, query=None, _projection=None):
        return FakeCursor(d for d in self.docs if _matches(d, query or {}))

    def find_one(self, query=None, _projection=None):
        found = self.find(query)
        return found[0] if found else None

    def insert_one(self, doc):
        self.docs.append(doc)

    def update_one(self, *_args, **_kwargs):
        pass


class FakeDatabase(dict):
    def __missing__(self, name):
        self[name] = FakeMongoCollection()
        return self[name]


class FakeMongoClient:
    def __init__(self, db):
        self.db = db

    def __getitem__(self, _name):
        return self.db

    def close(self):
        pass


TODAY = "2026-10-30"


def _issue(title, url):
    import daily_story_worker as worker_module

    return {
        "title": title,
        "summary": "",
        "url": url,
        "source": url.split("/")[2],
        "published_at": "2026-10-30T06:00:00+00:00",
        "tokens": worker_module.DailyStoryWorker._phrase_tokens(title),
        "text": title,
    }


# One morning's pool, best match first for TOPIC: one event worded by two
# feeds, then two other events.
MAZARI_BBC = _issue("Pakistan jails human rights lawyers Imaan Mazari and her husband over posts", "https://www.bbc.co.uk/news/articles/mazari")
MAZARI_AJ = _issue("Pakistani human rights lawyers Imaan Mazari, husband jailed over posts", "https://www.aljazeera.com/news/mazari")
VENEZUELA = _issue("Venezuela opposition lawyers challenge election in court", "https://www.nytimes.com/venezuela-election")
KENYA = _issue("Kenya teachers end strike after pay deal", "https://www.bbc.co.uk/news/articles/kenya")
POOL = [MAZARI_BBC, MAZARI_AJ, VENEZUELA, KENYA]
TOPIC = "human rights lawyers courts democracy"


def _story(profile_id, story_date, issue=None, title="", status="pending_review"):
    return {
        "story_key": f"{profile_id}:{story_date}",
        "story_date": story_date,
        "status": status,
        "scholar": {"profile_id": profile_id},
        "trend_issue": {"selected": dict(issue or {})},
        "content": {"title": title or f"A legend on {(issue or {}).get('title', '')}"},
    }


def _story_worker(stories=(), scholars=(), **env):
    import daily_story_worker as worker_module

    db = FakeDatabase()
    db["legend_scholar_daily_stories"] = FakeMongoCollection(stories)
    db["legend_scholars"] = FakeMongoCollection(scholars)
    settings = {
        "MONGODB_URI": "mongodb://fake/test",
        "STORY_ML_ENABLED": "0",
        "STORY_ENFORCE_PROFILE_QUALITY": "",
        "STORY_MAX_LEGENDS_PER_EVENT": "",
        "STORY_TOPIC_REPEAT_WINDOW_DAYS": "",
        **env,
    }
    with mock.patch.dict(os.environ, settings), \
            mock.patch.object(worker_module, "load_dotenv"), \
            mock.patch.object(worker_module, "create_mongo_client", return_value=FakeMongoClient(db)), \
            mock.patch.object(worker_module, "resolve_mongo_db_name", return_value="test"):
        return worker_module.DailyStoryWorker(use_llm=False, trends_enabled=True)


class TrendVarietyTests(unittest.TestCase):
    """2026-10-29: nearly all ~29 legend drafts were on two news events, some
    with near identical titles. An event carries at most two legends a day, a
    legend does not repeat an event within 30 days, and a legend with nothing
    fresh left sits the day out."""

    def _pick(self, worker, profile_id):
        fresh, _ = worker._fresh_trend_issues(POOL, profile_id=profile_id, story_date=TODAY)
        picked = worker._select_trending_issue(TOPIC, fresh)
        if picked:
            worker.stories_collection.insert_one(_story(profile_id, TODAY, picked))
        return picked

    def test_one_feed_s_wording_of_an_event_is_the_same_event(self) -> None:
        import daily_story_worker as worker_module

        same = worker_module.DailyStoryWorker._same_news_event
        self.assertTrue(same(MAZARI_BBC, MAZARI_AJ))
        self.assertTrue(same(MAZARI_BBC, {"title": "Pakistan Sentences Rights Lawyer Imaan Mazari to Prison"}))
        self.assertFalse(same(MAZARI_BBC, VENEZUELA))
        self.assertFalse(same(VENEZUELA, KENYA))

    def test_a_day_s_legends_are_spread_at_most_two_to_an_event(self) -> None:
        self.assertEqual(_story_worker().max_legends_per_event, 2, "the default")
        # A draft that failed never reached review and holds no place.
        worker = _story_worker([_story("legend-failed", TODAY, MAZARI_BBC, status="failed_validation")])
        picks = [self._pick(worker, f"legend-{n}") for n in range(7)]
        titles = [p["title"] if p else None for p in picks]
        self.assertEqual(
            titles,
            [MAZARI_BBC["title"], MAZARI_BBC["title"], VENEZUELA["title"], VENEZUELA["title"], KENYA["title"], KENYA["title"], None],
            "the best match until it holds two legends (in any feed's wording), then the next best",
        )

    def test_the_daily_cap_is_set_by_its_variable(self) -> None:
        worker = _story_worker([_story("legend-a", TODAY, MAZARI_AJ)], STORY_MAX_LEGENDS_PER_EVENT="1")
        self.assertEqual(self._pick(worker, "legend-b")["title"], VENEZUELA["title"])

    def test_a_legend_does_not_repeat_an_event_or_a_title_within_the_window(self) -> None:
        stories = [
            # The same event in another feed's wording, ten days ago.
            _story("legend-c", "2026-10-20", {"title": "Pakistan Sentences Rights Lawyer Imaan Mazari to Prison", "url": "https://www.nytimes.com/mazari"}),
            # A title essentially the same as today's Kenya headline.
            _story("legend-c", "2026-10-25", title="Kenya's teachers end their strike after a pay deal"),
            # Older than the window, and a draft that failed: neither counts.
            _story("legend-c", "2026-09-20", VENEZUELA),
            _story("legend-c", "2026-10-27", VENEZUELA, status="failed_validation"),
            # Another legend's history is not this legend's.
            _story("legend-x", "2026-10-29", VENEZUELA),
        ]
        worker = _story_worker(stories)
        fresh, blocked = worker._fresh_trend_issues(POOL, profile_id="legend-c", story_date=TODAY)
        self.assertEqual([i["title"] for i in fresh], [VENEZUELA["title"]])
        self.assertEqual(blocked, {"at_event_cap": 0, "repeats_legend": 3})
        self.assertEqual(worker._select_trending_issue(TOPIC, fresh)["title"], VENEZUELA["title"])

    def test_a_legend_with_no_fresh_event_is_skipped_with_the_reason(self) -> None:
        stories = [
            _story("legend-a", TODAY, MAZARI_BBC),
            _story("legend-b", TODAY, MAZARI_AJ),
            _story("legend-d", TODAY, VENEZUELA),
            _story("legend-e", TODAY, VENEZUELA),
            _story("legend-c", "2026-10-25", KENYA),
        ]
        scholars = [{"profile_id": "legend-c", "professor_name": "Ida B. Wells"}, {"profile_id": "legend-f", "professor_name": "Carter G. Woodson"}]
        worker = _story_worker(stories, scholars)
        chunks = [{"chunk_id": f"c{i}", "section": "about", "text": "x"} for i in range(4)]
        with mock.patch.object(worker, "_fetch_trending_issues", return_value=list(POOL)), \
                mock.patch.object(worker, "_choose_topic", return_value=(TOPIC, "test", 0)) as choose_topic, \
                mock.patch.object(worker, "_extract_context_chunks", return_value=chunks), \
                mock.patch.object(worker, "_build_style_profile", return_value={}), \
                mock.patch.object(worker, "_collect_allowed_source_urls", return_value=[]), \
                mock.patch.object(worker, "_select_trending_issue", wraps=worker._select_trending_issue) as select, \
                mock.patch.object(worker, "_generate_story_payload", side_effect=RuntimeError("no model in tests")):
            result = worker.run(
                scholar_id=None,
                date_value=date(2026, 10, 30),
                topic_override=None,
                max_scholars=5,
                max_context_chunks=12,
                dry_run=True,
            )

        self.assertEqual(result["summary"]["skipped_no_fresh_trend"], 1)
        reason = next(e for e in result["errors"] if e.startswith("legend-c:"))
        self.assertEqual(
            reason,
            "legend-c:skipped_no_fresh_trend:of 4 trend issues, 3 are on events already carried by 2 legends today "
            "and 1 repeat this legend's stories of the last 30 days",
        )
        self.assertEqual(choose_topic.call_count, 1, "only legend-f went on to be written")
        self.assertEqual(choose_topic.call_args.kwargs["profile_id"], "legend-f")
        # legend-f is written on the one event still open, not on a full one.
        self.assertEqual([i["title"] for i in select.call_args.args[1]], [KENYA["title"]])
        self.assertFalse([s for s in worker.stories_collection.docs if s["story_date"] == TODAY and s["scholar"]["profile_id"] == "legend-c"])


class PersonaCheckTests(unittest.TestCase):
    """2026-11-01: all six drafts failed persona_violation on \\b(?:we|our|ours|us)\\b,
    run on lowercased text, so naming the US read as the writer saying "us"."""

    def _persona_errors(self, article, assume_deceased=True):
        worker = _story_worker()
        payload = {"title": "A legend on sanctions", "standfirst": "x" * 30, "article_markdown": article}
        _, errors, *_ = worker._validate_story(
            name="Derrick Bell",
            payload=payload,
            context_chunks=[],
            trend_issue=None,
            allowed_source_urls=[],
            allowed_chunk_ids=set(),
            assume_deceased=assume_deceased,
        )
        return [e for e in errors if e.startswith("persona_violation")]

    def test_the_united_states_is_not_the_writer(self) -> None:
        article = (
            "## Why This Matters Now\nThe US imposed new U.S. sanctions; USA Today reported the vote, "
            "and Bell read US courts closely [chunk:us-law-1].\n\n## Sources\n"
            "- https://www.nytimes.com/2026/10/31/us/politics/sanctions.html\n- https://example.org/about-us\n"
        )
        self.assertEqual(self._persona_errors(article), [])

    def test_first_person_plural_still_fails_for_a_deceased_legend(self) -> None:
        for phrase, word in [("we believe the ruling was wrong", "we"), ("our country owes more", "our"),
                             ("Readers should join us in asking", "us"), ("We should ask", "We"), ("Ours is a law", "Ours")]:
            self.assertEqual(
                self._persona_errors(f"## Why This Matters Now\nBell on the US: {phrase}."),
                [f"persona_violation:first_person_plural:{word}"],
                phrase,
            )
        self.assertEqual(self._persona_errors("## Why\nwe believe it.", assume_deceased=False), [])

    def test_a_scholar_sentence_naming_the_us_can_anchor_a_fallback(self) -> None:
        worker = _story_worker()
        sentence = "Derrick Bell argued that US courts advanced racial reform only when it served other interests."
        chunks = [{"chunk_id": f"c{i}", "section": "about", "text": sentence} for i in range(4)]
        payload = worker._build_strict_validation_fallback_payload(
            name="Derrick Bell", topic="law", trend_issue=None, context_chunks=chunks, allowed_source_urls=[], note="t"
        )
        self.assertIn(sentence, payload["article_markdown"])

    def test_the_eval_counts_the_pronoun_not_the_country(self) -> None:
        import daily_story_eval as story_eval

        self.assertEqual(story_eval._first_person_hits("The US and U.S. sanctions; USA. https://x.org/us/a"), 0)
        self.assertEqual(story_eval._first_person_hits("We think our case is ours; join us."), 4)


if __name__ == "__main__":
    unittest.main()
