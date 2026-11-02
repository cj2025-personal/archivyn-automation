"""Offline checks for the legend identity review (legend_identity.py and the
legend_identity_review stage) and its gate on story generation.

2026-11-01: the passages of two legends that passed the profile-quality gate
were about other people (robert-williams: a basketball player; edward-t-gordon:
the theatre designer Edward Gordon Craig). These tests cover how pages are
grouped and hashed, who a legend is checked against, how the model's page
verdicts are counted into a status, and what the story worker does with it.

Run with:  python -m unittest discover -p "test_*.py"
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import legend_identity as identity
import legend_identity_review as stage
from test_editorial_pipeline import KENYA, TODAY, _identified, _run_day, _story_worker

PSYCH = "https://psychology.example.edu/robert-l-williams"
HOOPS = "https://sports.example.com/robert-williams-basketball"


def _chunk(chunk_id, text, url=""):
    return {"chunk_id": chunk_id, "text": text, "primary_source_url": url, "source_urls": [url] if url else []}


def _passage(words):
    return " ".join(words for _ in range(12))


def _legend(profile_id="robert-williams", name="Robert Williams", pages=None, **extra):
    pages = pages if pages is not None else {
        "biography": [
            _chunk("b1", _passage("Robert L. Williams, a psychologist, coined the term Ebonics in 1973."), PSYCH),
            _chunk("b2", _passage("Williams created the BITCH-100 test to show cultural bias in testing."), PSYCH + "/?utm=feed"),
        ],
        "career": [
            _chunk("c1", _passage("Robert Williams, a centre, was drafted by the Boston Celtics in 2018."), HOOPS),
        ],
    }
    return {"_id": f"oid-{profile_id}", "profile_id": profile_id, "professor_name": name, "rag_context": {
        "section_chunks": pages,
        "section_text": {"biography": "aggregated text from every page"},
        "section_summaries": {"biography": "a summary of every page"},
    }, "links_and_media": {"references": [HOOPS]}, **extra}


def _verdict(**pages):
    return {
        "pages": [{"page_id": pid, "verdict": v, "person_described": "", "reason": "r"} for pid, v in pages.items()],
        "intended_person": {"full_name": "Robert L. Williams", "field": "Psychology", "years": "1930-2020", "known_for": "Ebonics"},
        "name_check": "correct",
        "usual_spelling": "",
    }


def _review(doc, verdict, anchor=None):
    pages, beyond = identity.source_pages(doc)
    page_ids = {f"P{n + 1}": p["key"] for n, p in enumerate(pages)}
    return identity.build_review(
        doc=doc, anchor=anchor or identity.anchor_for(doc),
        pages=pages, beyond=beyond, verdict=verdict, page_ids=page_ids, model="m",
        doc_sources_hash=identity.sources_hash(doc), review_input_hash="h", reviewed_at="2026-11-01T00:00:00+00:00",
    )


class SourcePageTests(unittest.TestCase):
    def test_passages_are_grouped_by_the_page_they_came_from(self) -> None:
        pages, beyond = identity.source_pages(_legend())
        self.assertEqual(beyond, [])
        self.assertEqual([(p["key"], p["passages"]) for p in pages], [
            ("psychology.example.edu/robert-l-williams", 2),  # the tracking query is the same page
            ("sports.example.com/robert-williams-basketball", 1),
        ])
        self.assertTrue(pages[0]["excerpts"][0].startswith("Robert L. Williams, a psychologist"))
        self.assertLessEqual(len(pages[0]["excerpts"][0]), identity.EXCERPT_CHARS)
        self.assertEqual(identity.page_key("http://www.Example.org/a/"), identity.page_key("https://example.org/a?x=1#y"))
        self.assertEqual(identity.chunk_page_key({"text": "x"}), identity.UNATTRIBUTED)

    def test_the_sources_hash_changes_with_any_passage_or_its_page(self) -> None:
        doc = _legend()
        base = identity.sources_hash(doc)
        self.assertEqual(identity.sources_hash(_legend()), base)
        changed = _legend()
        changed["rag_context"]["section_chunks"]["career"][0]["text"] += " More."
        self.assertNotEqual(identity.sources_hash(changed), base)
        moved = _legend()
        moved["rag_context"]["section_chunks"]["career"][0]["primary_source_url"] = PSYCH
        self.assertNotEqual(identity.sources_hash(moved), base)

    def test_pages_beyond_the_limit_are_never_shown_and_never_allowed(self) -> None:
        many = {"s": [_chunk(f"c{n}", _passage(f"Passage {n} on the legend's work."), f"https://p{n}.example.org/") for n in range(identity.MAX_PAGES + 2)]}
        doc = _legend(pages=many)
        pages, beyond = identity.source_pages(doc)
        self.assertEqual((len(pages), len(beyond)), (identity.MAX_PAGES, 2))
        review = _review(doc, _verdict(**{f"P{n + 1}": identity.ABOUT_LEGEND for n in range(identity.MAX_PAGES)}))
        self.assertEqual(len(review["allowed_page_keys"]), identity.MAX_PAGES)
        self.assertEqual([p["verdict"] for p in review["pages"]][-2:], [identity.NOT_REVIEWED] * 2)


class AnchorTests(unittest.TestCase):
    def test_a_confirmed_identity_then_the_roster_then_the_name(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            roster = Path(folder)
            (roster / "robert-williams.meta.json").write_text(json.dumps({"profile_name": "Robert Williams", "field": "Psychology"}), encoding="utf-8")
            doc = _legend()
            self.assertEqual(identity.anchor_for(doc, roster), {"name": "Robert Williams", "field": "Psychology", "from": "roster"})
            confirmed = {**doc, "identity": {"expected": {"name": "Robert Lee Williams II", "field": "Psychology"}}}
            self.assertEqual(identity.anchor_for(confirmed, roster)["from"], "administrator")
            unknown = _legend(profile_id="3c457bc8-2bf9-4ce7-82d2-e1e8db8e1f38", name="Jewel Cobb")
            self.assertEqual(identity.anchor_for(unknown, roster), {"name": "Jewel Cobb", "field": "", "from": "name_only"})
            self.assertEqual(identity.anchor_for(_legend(profile_id="../etc/passwd"), roster)["from"], "name_only")

    def test_the_shipped_roster_knows_the_two_wrong_person_legends(self) -> None:
        self.assertEqual(identity.anchor_for(_legend())["field"], "Psychology")
        self.assertEqual(identity.anchor_for(_legend(profile_id="edward-t-gordon", name="Edward T. Gordon"))["field"], "Psychology")


class StatusTests(unittest.TestCase):
    def test_the_status_is_counted_from_the_page_verdicts(self) -> None:
        a, d, n, u = identity.ABOUT_LEGEND, identity.DIFFERENT_PERSON, identity.NOT_ABOUT_A_PERSON, identity.UNCLEAR
        self.assertEqual(identity.identity_status([]), identity.NO_SOURCES)
        self.assertEqual(identity.identity_status([a, n, u]), identity.VERIFIED)
        self.assertEqual(identity.identity_status([a, d]), identity.MIXED)
        self.assertEqual(identity.identity_status([d, n]), identity.WRONG_PERSON)
        self.assertEqual(identity.identity_status([n, u]), identity.UNVERIFIED)

    def test_only_pages_about_the_legend_are_allowed_and_bad_verdicts_count_as_unclear(self) -> None:
        review = _review(_legend(), _verdict(P1=identity.ABOUT_LEGEND, P2=identity.DIFFERENT_PERSON))
        self.assertEqual(review["status"], identity.MIXED)
        self.assertTrue(review["eligible"])
        self.assertEqual(review["allowed_page_keys"], ["psychology.example.edu/robert-l-williams"])
        self.assertEqual(review["attention"], ["1 of its source pages describe someone else; they are left out of its stories."])
        nonsense = _review(_legend(), {**_verdict(P1="yes, probably", P9=identity.ABOUT_LEGEND), "name_check": "maybe"})
        self.assertEqual([p["verdict"] for p in nonsense["pages"]], [identity.UNCLEAR, identity.UNCLEAR])
        self.assertEqual((nonsense["status"], nonsense["eligible"], nonsense["name_check"]), (identity.UNVERIFIED, False, "unknown"))

    def test_a_wrong_person_or_a_misspelled_name_is_not_written_about(self) -> None:
        wrong = _review(_legend(), _verdict(P1=identity.DIFFERENT_PERSON, P2=identity.DIFFERENT_PERSON))
        self.assertEqual((wrong["status"], wrong["eligible"]), (identity.WRONG_PERSON, False))
        self.assertIn("Its stored sources describe someone else.", wrong["attention"])
        misspelled = _review(_legend(name="Marie Daley"), {**_verdict(P1=identity.ABOUT_LEGEND, P2=identity.ABOUT_LEGEND), "name_check": "misspelled", "usual_spelling": "Marie Maynard Daly"})
        self.assertEqual((misspelled["status"], misspelled["eligible"]), (identity.VERIFIED, False))
        self.assertIn("Its name looks misspelled (usually: Marie Maynard Daly).", misspelled["attention"])
        nothing = _review(_legend(pages={}), _verdict(), anchor={"name": "Jewel Cobb", "field": "", "from": "name_only"})
        self.assertEqual(nothing["status"], identity.NO_SOURCES)
        self.assertIn("No roster entry or confirmed identity: checked against its name alone.", nothing["attention"])


class EligibilityTests(unittest.TestCase):
    def test_a_review_applies_only_to_the_sources_and_policy_it_judged(self) -> None:
        doc = _legend()
        self.assertEqual(identity.story_eligibility(doc), (False, "not_reviewed"))
        doc["identity_review"] = _review(doc, _verdict(P1=identity.ABOUT_LEGEND, P2=identity.DIFFERENT_PERSON))
        self.assertEqual(identity.story_eligibility(doc), (True, identity.MIXED))
        doc["rag_context"]["section_chunks"]["career"].append(_chunk("c2", _passage("A new page, collected later."), "https://new.example.org/"))
        self.assertEqual(identity.story_eligibility(doc), (False, "stale"))
        old = _legend()
        old["identity_review"] = {**_review(old, _verdict(P1=identity.ABOUT_LEGEND)), "policy_version": "legend-identity-v0"}
        self.assertEqual(identity.story_eligibility(old), (False, "stale"))
        misspelled = _legend()
        misspelled["identity_review"] = _review(misspelled, {**_verdict(P1=identity.ABOUT_LEGEND), "name_check": "misspelled"})
        self.assertEqual(identity.story_eligibility(misspelled), (False, "name_misspelled"))

    def test_a_corrected_name_or_identity_makes_the_review_stale_and_a_bad_link_never_raises(self) -> None:
        doc = _legend()
        doc["identity_review"] = _review(doc, _verdict(P1=identity.ABOUT_LEGEND, P2=identity.ABOUT_LEGEND))
        self.assertTrue(identity.story_eligibility(doc)[0])
        renamed = {**doc, "professor_name": "Robert L. Williams"}
        self.assertEqual(identity.story_eligibility(renamed), (False, "stale"))
        corrected = {**doc, "identity": {"expected": {"name": "Robert Lee Williams II", "field": "Psychology"}}}
        self.assertEqual(identity.story_eligibility(corrected), (False, "stale"))
        broken = _legend()
        broken["rag_context"]["section_chunks"]["career"][0]["primary_source_url"] = "https://[bad/path"
        self.assertTrue(identity.page_key("https://[bad/path").startswith("unparsed:"))
        broken["identity_review"] = _review(broken, _verdict(P1=identity.ABOUT_LEGEND, P2=identity.ABOUT_LEGEND))
        self.assertTrue(identity.story_eligibility(broken)[0])
        with mock.patch.object(identity, "sources_hash", side_effect=RuntimeError("unreadable")):
            self.assertEqual(identity.story_eligibility(doc), (False, "check_failed"))

    def test_the_judge_sees_every_passage_and_unlinked_ones_together(self) -> None:
        pages = {"s": [
            _chunk("u1", _passage("An unlinked passage about the psychologist's testing work.")),
            _chunk("u2", _passage("Another unlinked passage, about a basketball season.")),
            {**_chunk("s1", _passage("A passage whose only link is the singular field.")), "source_url": PSYCH},
            _chunk("x", "too short to judge", PSYCH),
        ]}
        shown, beyond = identity.source_pages(_legend(pages=pages))
        self.assertEqual(beyond, [])
        by_key = {p["key"]: p for p in shown}
        self.assertEqual(by_key[identity.UNATTRIBUTED]["passages"], 2)
        self.assertEqual(len(by_key[identity.UNATTRIBUTED]["excerpts"]), 2, "both unlinked passages are shown")
        self.assertEqual(by_key["psychology.example.edu/robert-l-williams"]["passages"], 1)
        string_links = {"text": _passage("A passage."), "source_urls": HOOPS}
        self.assertEqual(identity.chunk_page_key(string_links), "sports.example.com/robert-williams-basketball")
        many = {"s": [_chunk(f"c{n}", "x" * 400, "https://big.example.org/") for n in range(200)]}
        shown, beyond = identity.source_pages(_legend(pages=many))
        self.assertEqual((shown, beyond), ([], ["big.example.org"]), "a page too big to show is not reviewed")

    def test_a_kept_passage_loses_its_links_to_excluded_pages(self) -> None:
        pages = {"biography": [{
            **_chunk("b1", _passage("Robert L. Williams, a psychologist, coined the term Ebonics in 1973."), PSYCH),
            "source_urls": [PSYCH, HOOPS],
            "source_refs": [{"source_url": PSYCH, "score": 0.9}, {"source_url": HOOPS, "score": 0.4}],
            "source_url": HOOPS,
        }], "career": [_chunk("c1", _passage("Robert Williams was drafted by the Boston Celtics in 2018."), HOOPS)]}
        doc = _legend(pages=pages)
        doc["identity_review"] = _review(doc, _verdict(P1=identity.ABOUT_LEGEND, P2=identity.DIFFERENT_PERSON))
        [kept] = identity.restricted_to_reviewed_sources(doc)["rag_context"]["section_chunks"]["biography"]
        self.assertEqual(kept["source_urls"], [PSYCH])
        self.assertEqual([r["source_url"] for r in kept["source_refs"]], [PSYCH])
        self.assertEqual((kept["primary_source_url"], kept["source_url"]), (PSYCH, ""))
        self.assertNotIn(HOOPS, json.dumps(identity.restricted_to_reviewed_sources(doc)))

    def test_the_writer_sees_only_the_allowed_pages_and_nothing_built_from_all_of_them(self) -> None:
        doc = _legend()
        doc["identity_review"] = _review(doc, _verdict(P1=identity.ABOUT_LEGEND, P2=identity.DIFFERENT_PERSON))
        restricted = identity.restricted_to_reviewed_sources(doc)
        kept = [c["chunk_id"] for chunks in restricted["rag_context"]["section_chunks"].values() for c in chunks]
        self.assertEqual(kept, ["b1", "b2"])
        self.assertEqual((restricted["rag_context"]["section_text"], restricted["rag_context"]["section_summaries"], restricted["links_and_media"]), ({}, {}, {}))
        self.assertIn("c1", json.dumps(doc), "the stored legend is not changed")
        clean = _legend(pages={"bio": doc["rag_context"]["section_chunks"]["biography"]})
        clean["identity_review"] = _review(clean, _verdict(P1=identity.ABOUT_LEGEND))
        self.assertEqual(identity.restricted_to_reviewed_sources(clean)["rag_context"]["section_text"], {"biography": "aggregated text from every page"})


class GateTests(unittest.TestCase):
    """The story worker writes only about legends an identity review lets it."""

    def test_a_legend_without_an_eligible_review_is_not_a_candidate(self) -> None:
        ready = {"daily_story_profile": {"status": "ready", "quality_score": 84}}
        wrong = _legend(profile_id="robert-williams", **ready)
        wrong["identity_review"] = _review(wrong, _verdict(P1=identity.DIFFERENT_PERSON, P2=identity.DIFFERENT_PERSON))
        unreviewed = _legend(profile_id="never-reviewed", identity_review=None, **ready)
        fine = _legend(profile_id="carter-g-woodson", **ready)
        worker = _story_worker(scholars=[wrong, unreviewed, fine], STORY_ENFORCE_PROFILE_QUALITY="1")
        docs, kept_out = worker._story_candidates(TODAY, 25)
        self.assertEqual([d["profile_id"] for d in docs], ["carter-g-woodson"])
        self.assertEqual(kept_out, {"profile_quality": 0, "identity": 2})

    def test_a_run_skips_a_stale_review_and_writes_a_mixed_legend_from_its_own_pages_only(self) -> None:
        stale = _legend(profile_id="stale-legend")
        stale["identity_review"] = _review(stale, _verdict(P1=identity.ABOUT_LEGEND, P2=identity.ABOUT_LEGEND))
        stale["rag_context"]["section_chunks"]["biography"][0]["text"] += " Edited since."
        mixed = _legend(profile_id="mixed-legend")
        mixed["identity_review"] = _review(mixed, _verdict(P1=identity.ABOUT_LEGEND, P2=identity.DIFFERENT_PERSON))
        worker = _story_worker(scholars=[stale, mixed])
        seen = []

        def context(scholar_doc, topic, max_context_chunks):
            seen.append((scholar_doc["profile_id"], sorted(c["chunk_id"] for cs in scholar_doc["rag_context"]["section_chunks"].values() for c in cs)))
            return [{"chunk_id": f"c{i}", "section": "about", "text": "x"} for i in range(4)]

        result = _run_day(worker, _extract_context_chunks=mock.Mock(side_effect=context))
        self.assertEqual(result["summary"]["skipped_identity"], 1, "the stale review takes no slot")
        self.assertEqual(seen, [("mixed-legend", ["b1", "b2"])])

    def test_a_stale_review_gives_up_its_slot_to_the_next_legend(self) -> None:
        stale = _legend(profile_id="aaa-stale")
        stale["identity_review"] = _review(stale, _verdict(P1=identity.ABOUT_LEGEND, P2=identity.ABOUT_LEGEND))
        stale["rag_context"]["section_chunks"]["biography"][0]["text"] += " Edited since."
        fine = [_legend(profile_id=f"fine-{n}") for n in range(2)]
        worker = _story_worker(scholars=[stale, *fine])
        docs, kept_out = worker._story_candidates(TODAY, 2)
        self.assertEqual(sorted(d["profile_id"] for d in docs), ["fine-0", "fine-1"])
        self.assertEqual(kept_out["identity"], 1)


class FakeJudgeWorker:
    """What the stage needs of a worker, over the fake database."""

    def __init__(self, scholars, judge):
        base = _story_worker(scholars=scholars)
        self.db = base.db
        self.scholars_collection = base.scholars_collection
        self.model_name = "gemini-test"
        self._call_structured_json = judge
        self._profile_id = base._profile_id

    def close(self):
        pass


class ReviewStageTests(unittest.TestCase):
    def _judge(self):
        def judge(prompt, schema, max_tokens):
            self.assertIs(schema, stage.IDENTITY_SCHEMA)
            ids = {line.split("]")[0][1:]: line for line in prompt.splitlines() if line.startswith("[P")}
            return _verdict(**{pid: identity.DIFFERENT_PERSON if "sports." in line else identity.ABOUT_LEGEND for pid, line in ids.items()})
        return mock.Mock(side_effect=judge)

    def _run(self, worker, **kwargs):
        options = {"scholar_id": None, "max_scholars": 100, "force": False, "dry_run": False, "concurrency": 2, **kwargs}
        return stage.run_identity_review(worker=worker, **options)

    def test_the_prompt_names_the_legend_we_mean_and_shows_each_page(self) -> None:
        doc = _legend()
        pages, _ = identity.source_pages(doc)
        prompt = stage.judge_prompt(name="Robert Williams", anchor=identity.anchor_for(doc), pages=pages, page_ids={"P1": pages[0]["key"], "P2": pages[1]["key"]})
        self.assertIn("Our roster lists: Robert Williams, field: Psychology.", prompt)
        self.assertIn(f"[P2] {HOOPS} (1 passages stored)", prompt)
        self.assertIn("drafted by the Boston Celtics", prompt)
        self.assertIn("A page about someone in a different field or era is a different person", prompt)

    def test_a_review_is_kept_on_the_legend_logged_and_reused_until_its_sources_change(self) -> None:
        judge = self._judge()
        worker = FakeJudgeWorker([_legend(identity_review=None)], judge)
        result = self._run(worker)
        self.assertEqual((result["status"], result["summary"]["reviewed"], result["summary"]["statuses"]), ("completed", 1, {identity.MIXED: 1}))
        saved = worker.scholars_collection.docs[0]["identity_review"]
        self.assertEqual((saved["status"], saved["eligible"], saved["allowed_page_keys"]), (identity.MIXED, True, ["psychology.example.edu/robert-l-williams"]))
        self.assertEqual(saved["anchor"]["from"], "roster")
        self.assertEqual(len(worker.db["legend_identity_review_events"].docs), 1)
        self.assertEqual(worker.db["legend_identity_review_jobs"].docs[0]["status"], "completed")
        self.assertEqual(result["attention"][0]["profile_id"], "robert-williams")

        again = self._run(worker)
        self.assertEqual((again["summary"]["unchanged"], judge.call_count), (1, 1), "nothing changed: no model call")
        self._run(worker, force=True)
        self.assertEqual(judge.call_count, 2)
        worker.scholars_collection.docs[0]["rag_context"]["section_chunks"]["career"][0]["text"] += " Re-collected."
        self._run(worker)
        self.assertEqual(judge.call_count, 3, "new sources are judged again")

    def test_a_failed_judge_keeps_the_last_review_and_the_run_says_so(self) -> None:
        doc = _legend()
        doc["identity_review"] = _review(doc, _verdict(P1=identity.ABOUT_LEGEND, P2=identity.ABOUT_LEGEND))
        doc["rag_context"]["section_chunks"]["career"][0]["text"] += " Re-collected."
        other = _legend(profile_id="carter-g-woodson", name="Carter G. Woodson", identity_review=None)

        def judge(prompt, schema, max_tokens):
            if "Name as stored: Robert Williams" in prompt:
                raise RuntimeError("429 quota")
            return _verdict(P1=identity.ABOUT_LEGEND, P2=identity.ABOUT_LEGEND)

        worker = FakeJudgeWorker([doc, other], mock.Mock(side_effect=judge))
        result = self._run(worker)
        self.assertEqual((result["status"], result["summary"]["failed"], result["summary"]["reviewed"]), ("completed_with_failures", 1, 1))
        self.assertIn("robert-williams:429 quota", result["errors"])
        kept = worker.scholars_collection.docs[0]
        self.assertEqual(identity.story_eligibility(kept), (False, "stale"), "the old verdict no longer applies to the new sources")
        # Nothing judged at all: the run failed, and story generation waits on it.
        alone = FakeJudgeWorker([_legend(identity_review=None)], mock.Mock(side_effect=RuntimeError("503 unavailable")))
        self.assertEqual(self._run(alone)["status"], "failed")

    def test_administrators_hear_only_of_reasons_that_changed(self) -> None:
        worker = FakeJudgeWorker([_legend(identity_review=None)], self._judge())
        [first] = self._run(worker)["attention"]
        self.assertTrue(first["changed"], "newly found")
        [again] = self._run(worker, force=True)["attention"]
        self.assertEqual((again["outcome"], again["changed"]), ("reviewed", False), "judged again, same reasons")
        [kept] = self._run(worker)["attention"]
        self.assertEqual((kept["outcome"], kept["changed"]), ("unchanged", False))

    def test_one_legend_can_be_reviewed_on_its_own(self) -> None:
        judge = self._judge()
        worker = FakeJudgeWorker([_legend(identity_review=None), _legend(profile_id="carter-g-woodson", name="Carter G. Woodson", identity_review=None)], judge)
        result = self._run(worker, scholar_id="carter-g-woodson")
        self.assertEqual((result["summary"]["attempted"], judge.call_count), (1, 1))
        self.assertIn("Name as stored: Carter G. Woodson", judge.call_args.args[0])
        self.assertIsNone(worker.scholars_collection.docs[0]["identity_review"], "the other legend is untouched")

    def test_a_dry_run_judges_and_writes_nothing(self) -> None:
        worker = FakeJudgeWorker([_legend(identity_review=None)], self._judge())
        result = self._run(worker, dry_run=True)
        self.assertEqual(result["summary"]["reviewed"], 1)
        self.assertIsNone(worker.scholars_collection.docs[0]["identity_review"])
        self.assertEqual(worker.db["legend_identity_review_events"].docs, [])
        self.assertEqual(worker.db["legend_identity_review_jobs"].docs, [])


if __name__ == "__main__":
    unittest.main()
