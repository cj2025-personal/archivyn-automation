"""
Live eval of the story pipeline's models: does the story model write a
story the worker's own checks accept, and the versions model a grades 3-5
version its judges accept?

The worker had tests with a stub for the model and no eval of the model
itself, so a change of model could only be judged in production. Here a
legend written for this eval (Katherine Johnson, from passages about her
well-documented life) is written up by the real model as the nightly job
writes it when there is no news item (an evergreen story), through
DailyStoryWorker.run and its strict validation, then made into its grades
3-5 version by daily_story_personalize, with its topic judge and verifier.
Grades 3-5 only: the pilot's band, and one band keeps the run cheap. One
run is one draw from the model; run it a few times before trusting a model,
and whenever STORY_LLM_MODEL, STORY_VERSION_MODEL or their prompts change.

  python daily_story_writer_eval.py                     # each stage's own model
  python daily_story_writer_eval.py --model gemini-3.8-flash --version-model gemini-3.5-flash-lite

Needs GCP_PROJECT_ID and Google credentials (application default credentials
work); calls go where the worker's go (VERTEX_MODEL_LOCATION). Writes only
to the local test MongoDB at 127.0.0.1:27018, in a fresh database dropped
afterwards, and reads no .env file. Prints each stage's outcome and the
tokens each stage used, thinking apart. Exit 0 when the story and its
version both pass, 1 when either fails, 2 when it could not start.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import date
from typing import Any, Dict, List
from unittest import mock

LOCAL_MONGO = "mongodb://127.0.0.1:27018"
BAND = "grade_3_5"
LEGEND_ID = "katherine-johnson"

PAGES = {
    "wiki": "https://en.wikipedia.org/wiki/Katherine_Johnson",
    "nasa": "https://www.nasa.gov/people/katherine-johnson",
    "library": "https://library.example.edu/archives/katherine-johnson",
}

# Passages written for this eval from Johnson's widely documented life: what
# a legend's stored sections look like, short enough to keep the run cheap.
# A legend who has died, as the writer is told every legend is.
PASSAGES: Dict[str, List[tuple]] = {
    "about": [
        ("wiki", "Creola Katherine Johnson was born on August 26, 1918, in White Sulphur Springs, West Virginia. She showed a strong talent for numbers as a young child and loved to count things. Because the local schools for Black children ended after eighth grade, her family moved during the school year to Institute, West Virginia, so she could attend high school there."),
        ("wiki", "Johnson started high school at the age of 10 and graduated at 14. In 1937, at the age of 18, she graduated with highest honors from West Virginia State College with degrees in mathematics and French. There, the mathematician William W. Schieffelin Claytor added classes in the analytic geometry of space for her."),
        ("library", "After college Johnson worked as a teacher at Black public schools in Virginia. In 1939 she was one of three Black students, and the only woman, chosen to integrate the graduate school of West Virginia University. She left the program to raise her family."),
        ("library", "Johnson retired from NASA in 1986 after 33 years at Langley. After retiring she often spoke to students about mathematics and careers in science. She died on February 24, 2020, at the age of 101, in Newport News, Virginia."),
    ],
    "background_and_work": [
        ("nasa", "In 1953 Johnson joined the National Advisory Committee for Aeronautics, known as NACA, at the Langley laboratory in Hampton, Virginia. She began in the West Area Computing unit, a group of Black women mathematicians who did calculations by hand. The unit was segregated, and its members had to use separate dining and bathroom facilities."),
        ("nasa", "Within weeks Johnson was assigned to the Flight Research Division, where she analyzed data from flight tests and studied how gusts of wind affected airplanes. In 1958 NACA became NASA, the National Aeronautics and Space Administration, and the segregated computing units were closed."),
        ("nasa", "Johnson calculated the flight path for Alan Shepard's 1961 mission, the first flight of an American into space. In 1962, before John Glenn's orbital flight on Friendship 7, Glenn asked that Johnson personally check the numbers from the new electronic computers before he would fly."),
        ("wiki", "In 1960 Johnson and the engineer Ted Skopinski wrote a report on how to place a spacecraft in orbit and bring it down at a chosen place. It was the first time a woman in the Flight Research Division was credited as an author of a research report. Over her career she wrote or co-wrote 26 research reports."),
    ],
    "milestones": [
        ("nasa", "Johnson worked on calculations for the 1969 Apollo 11 mission that landed the first people on the Moon, helping to make sure the lunar module could meet the command module again in orbit. In 1970 she worked on the procedures that helped the crew of Apollo 13 return safely to Earth."),
        ("wiki", "In 2015 President Barack Obama awarded Johnson the Presidential Medal of Freedom, the highest civilian honor in the United States. In 2016 the book Hidden Figures by Margot Lee Shetterly, and the film based on it, told the story of Johnson and her colleagues."),
        ("nasa", "In 2017 NASA opened the Katherine G. Johnson Computational Research Facility in Hampton, Virginia, named in her honor. In 2019 she was awarded the Congressional Gold Medal along with other women of the West Area Computing unit."),
        ("library", "Johnson's work showed that careful hand calculation could be trusted beside the first electronic computers. Students who met her after her retirement heard her advice to ask questions, to like what they do, and to check their work."),
    ],
}

USAGE = {stage: {"calls": 0, "input": 0, "output": 0, "thinking": 0} for stage in ("story", "versions")}
STAGE = ["story"]


def legend_doc() -> Dict[str, Any]:
    """The legend as the pipeline stores one, with a current identity review
    that lets it be written about (as legend_identity_review leaves it)."""
    import legend_identity as identity

    sections = {
        name: [
            {"chunk_id": f"{LEGEND_ID}-{name}-{n}", "text": text, "source_url": PAGES[page]}
            for n, (page, text) in enumerate(rows)
        ]
        for name, rows in PASSAGES.items()
    }
    doc: Dict[str, Any] = {
        "profile_id": LEGEND_ID,
        "professor_name": "Katherine Johnson",
        "name": {"full": "Katherine Johnson"},
        "about": {
            "short_bio": "Katherine Johnson was a NASA mathematician whose calculations of flight paths helped send the first Americans into space and to the Moon.",
            "field_of_study": "Mathematics",
        },
        "rag_context": {"section_chunks": sections},
        "daily_story_profile": {"status": "ready", "quality_score": 80},
    }
    pages, _ = identity.source_pages(doc)
    doc["identity_review"] = {
        "policy_version": identity.IDENTITY_POLICY_VERSION,
        "status": identity.VERIFIED,
        "eligible": True,
        "name_check": "correct",
        "name_reviewed": identity.stored_name(doc),
        "anchor": identity.anchor_for(doc),
        "sources_hash": identity.sources_hash(doc),
        "pages": [{"key": p["key"], "verdict": identity.ABOUT_LEGEND} for p in pages],
        "allowed_page_keys": [p["key"] for p in pages],
    }
    return doc


def _counting(model_class: Any) -> Any:
    class Counted(model_class):  # type: ignore[misc, valid-type]
        def generate_content(self, *args: Any, **kwargs: Any) -> Any:
            response = super().generate_content(*args, **kwargs)
            meta = getattr(response, "usage_metadata", None)
            usage = USAGE[STAGE[0]]
            usage["calls"] += 1
            usage["input"] += int(getattr(meta, "prompt_token_count", 0) or 0)
            usage["output"] += int(getattr(meta, "candidates_token_count", 0) or 0)
            usage["thinking"] += int(getattr(meta, "thoughts_token_count", 0) or 0)
            return response

    return Counted


def run(model: str, version_model: str) -> Dict[str, Any]:
    from pymongo import MongoClient

    import daily_story_personalize as personalize
    import daily_story_worker as worker_module

    name = f"writer_eval_{uuid.uuid4().hex[:12]}"
    uri = f"{LOCAL_MONGO}/{name}"
    client = MongoClient(LOCAL_MONGO, serverSelectionTimeoutMS=2000)
    client.admin.command("ping")
    db = client[name]
    today = date.today()
    env = {
        "MONGODB_URI": uri,
        "MONGODB_DB": "",
        "STORY_LLM_MODEL": model,
        "STORY_VERSION_MODEL": version_model,
        "STORY_ML_ENABLED": "0",
        "STORY_ENFORCE_PROFILE_QUALITY": "",
    }
    report: Dict[str, Any] = {"model": model, "version_model": version_model, "location": os.getenv("VERTEX_MODEL_LOCATION") or worker_module.DEFAULT_MODEL_LOCATION}
    try:
        db["legend_scholars"].insert_one(legend_doc())
        # One active learner in grade 4: the versions stage makes the bands
        # its learners read, so this one band.
        db["users"].insert_one({"role": "student", "isActive": True, "gradeLevel": 4})
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(worker_module, "load_dotenv"), \
                mock.patch.object(worker_module, "GenerativeModel", _counting(worker_module.GenerativeModel)):
            worker = worker_module.DailyStoryWorker(
                use_llm=True, trends_enabled=False, mongodb_uri=uri, load_env_file=False, model_name=model,
            )
            try:
                result = worker.run(
                    scholar_id=LEGEND_ID, date_value=today, topic_override=None,
                    max_scholars=1, max_context_chunks=12, dry_run=False,
                )
            finally:
                worker.close()
            story = db["legend_scholar_daily_stories"].find_one({"scholar.profile_id": LEGEND_ID}) or {}
            safety = story.get("safety") or {}
            article = (story.get("content") or {}).get("article_markdown") or ""
            report["story"] = {
                "passed": story.get("status") == "pending_review" and bool(safety.get("validation_passed")),
                "status": story.get("status"),
                "mode": story.get("story_mode"),
                "words": len(article.split()),
                "title": (story.get("content") or {}).get("title"),
                "errors": (safety.get("errors") or result.get("errors") or [])[:8],
                "editor_notes": (story.get("generation") or {}).get("editor_notes") or (story.get("content") or {}).get("editor_notes") or [],
            }
            if not report["story"]["passed"]:
                report["version"] = {"passed": False, "skipped": "no story to make a version of"}
                return report
            STAGE[0] = "versions"
            versions = personalize.run_personalization(
                story_date=today.isoformat(), max_stories=1, dry_run=False, no_llm=False, scholar_id=LEGEND_ID,
            )
        story = db["legend_scholar_daily_stories"].find_one({"scholar.profile_id": LEGEND_ID}) or {}
        variant = next((v for v in story.get("audience_variants") or [] if v.get("band_key") == BAND), {})
        validation = variant.get("validation") or {}
        suitability = (story.get("audience_suitability") or {}).get(BAND) or {}
        report["version"] = {
            "passed": variant.get("status") == "ready" and bool(validation.get("ok")),
            "status": variant.get("status"),
            "suitable": suitability.get("suitable"),
            "words": len(((variant.get("article_markdown") or "")).split()),
            "readability": variant.get("readability"),
            "blockers": validation.get("blockers") or [],
            "failures": (versions.get("failures") or [])[:4],
        }
        return report
    finally:
        client.drop_database(name)
        client.close()


def main() -> int:
    from daily_story_personalize import version_model
    from daily_story_worker import DEFAULT_STORY_MODEL

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=os.getenv("STORY_LLM_MODEL") or DEFAULT_STORY_MODEL, help="The story model")
    parser.add_argument("--version-model", default=version_model(), help="The grade-band versions' model")
    args = parser.parse_args()
    if not os.getenv("GCP_PROJECT_ID"):
        print("Set GCP_PROJECT_ID.", file=sys.stderr)
        return 2
    try:
        report = run(args.model, args.version_model)
    except Exception as exc:  # the local database or the model unreachable
        print(f"Could not run: {exc}", file=sys.stderr)
        return 2
    report["usage"] = USAGE
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["story"]["passed"] and report["version"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
