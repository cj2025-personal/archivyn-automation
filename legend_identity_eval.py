"""
Live eval of the legend identity judge: fixed cases with known answers, judged
by the real model through the stage's own prompt and schema, the verdicts
counted by legend_identity exactly as in production. Run it whenever the
prompt, the policy or LEGEND_IDENTITY_MODEL changes.

  python legend_identity_eval.py                 # judge model: LEGEND_IDENTITY_MODEL, else STORY_LLM_MODEL
  python legend_identity_eval.py --model gemini-2.5-pro

Needs GCP_PROJECT_ID (and GCP_LOCATION) and Google credentials (application
default credentials work). No database. One model call per case; the cases
are written for this eval, excerpts kept short. Exit 0 when every case
passes, 1 when any fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import legend_identity as identity
from legend_identity_review import IDENTITY_SCHEMA, JUDGE_MAX_TOKENS, judge_prompt

A, D, N, U = identity.ABOUT_LEGEND, identity.DIFFERENT_PERSON, identity.NOT_ABOUT_A_PERSON, identity.UNCLEAR


def _page(url: str, *excerpts: str) -> Dict[str, Any]:
    return {"key": identity.page_key(url), "url": url, "passages": len(excerpts), "excerpts": list(excerpts)}


CASES: List[Dict[str, Any]] = [
    {
        "name": "a basketball player's page under a psychologist's name",
        "stored": "Robert Williams",
        "anchor": {"name": "Robert Williams", "field": "Psychology", "from": "roster"},
        "pages": [
            _page("https://en.wikipedia.org/wiki/Robert_Williams_(basketball)",
                  "Robert Williams III is an American professional basketball player. He played college basketball for Texas A&M and was selected by the Boston Celtics in the first round of the 2018 NBA draft.",
                  "Known for his shot-blocking, Williams was named to the NBA All-Defensive Second Team in 2022."),
            _page("https://psychology.example.edu/history/robert-l-williams",
                  "Robert Lee Williams II was an African-American psychologist who coined the term 'Ebonics' in 1973 and developed the Black Intelligence Test of Cultural Homogeneity to show bias in standardized intelligence testing.",
                  "He was a founding member of the Association of Black Psychologists and taught at Washington University in St. Louis."),
        ],
        "expect": {"status": identity.MIXED, "pages": [D, A], "name_check": ["correct"]},
    },
    {
        "name": "a theatre designer's pages under a psychologist's name",
        "stored": "Edward T. Gordon",
        "anchor": {"name": "Edward T. Gordon", "field": "Psychology", "from": "roster"},
        "pages": [
            _page("https://en.wikipedia.org/wiki/Edward_Gordon_Craig",
                  "Edward Henry Gordon Craig was an English modernist theatre practitioner, actor, director and scenic designer, son of the actress Ellen Terry.",
                  "His idea of the actor as an 'Über-marionette' and his designs of moving screens influenced twentieth-century stagecraft."),
        ],
        "expect": {"status": identity.WRONG_PERSON, "pages": [D], "name_check": ["correct", "unknown"]},
    },
    {
        "name": "the legend's own biography beside a page about a topic",
        "stored": "Carter G. Woodson",
        "anchor": {"name": "Carter G. Woodson", "field": "History", "from": "roster"},
        "pages": [
            _page("https://www.britannica.com/biography/Carter-G-Woodson",
                  "Carter G. Woodson, American historian, author and journalist, founded the Association for the Study of Negro Life and History in 1915 and launched Negro History Week in 1926, the forerunner of Black History Month.",
                  "His 1933 book The Mis-Education of the Negro argued that schools taught Black Americans to accept an inferior place in society."),
            _page("https://www.britannica.com/event/Afrocentrism",
                  "Afrocentrism is a cultural and political movement whose mainly African American adherents regard themselves and all other Black people as sharing a common African heritage.",
                  "The movement grew in the late twentieth century, drawing on writers such as Cheikh Anta Diop and Molefi Kete Asante."),
        ],
        "expect": {"status": identity.VERIFIED, "pages": [A, N], "name_check": ["correct"]},
    },
    {
        "name": "a misspelled surname on the roster",
        "stored": "Marie Daley",
        "anchor": {"name": "Marie Maynard Daley", "field": "Biochemistry", "from": "roster"},
        "pages": [
            _page("https://www.example.org/marie-maynard-daly",
                  "Marie Maynard Daly was an American biochemist and the first Black woman in the United States to earn a PhD in chemistry, from Columbia University in 1947.",
                  "Her research on cholesterol, sugars and proteins helped show the link between high cholesterol and clogged arteries."),
        ],
        "expect": {"status": identity.VERIFIED, "pages": [A], "name_check": ["misspelled"], "usual_spelling_contains": "Daly"},
    },
    {
        "name": "a legend with nothing stored, known by name only",
        "stored": "Jewel Plummer Cobb",
        "anchor": {"name": "Jewel Plummer Cobb", "field": "", "from": "name_only"},
        "pages": [],
        "expect": {"status": identity.NO_SOURCES, "pages": [], "name_check": ["correct"]},
    },
    {
        "name": "a page too thin to tell",
        "stored": "Andrew Brimmer",
        "anchor": {"name": "Andrew Brimmer", "field": "Economics", "from": "roster"},
        "pages": [
            _page("https://www.example.org/search?q=brimmer", "Results for Brimmer. Showing 1 to 10 of 2,341 results. Sort by relevance."),
            _page("https://www.example.org/andrew-brimmer",
                  "Andrew Felton Brimmer was an American economist who in 1966 became the first African American governor of the Federal Reserve Board."),
        ],
        "expect": {"status": identity.VERIFIED, "pages": [[U, N], A], "name_check": ["correct"]},
    },
]


def _judge_worker(model: str):
    """The production call path (DailyStoryWorker._call_structured_json) on a
    bare model, without the database the worker would otherwise open."""
    import vertexai
    from vertexai.generative_models import GenerativeModel

    from daily_story_worker import DailyStoryWorker

    vertexai.init(project=os.environ["GCP_PROJECT_ID"], location=os.getenv("GCP_LOCATION") or "us-central1")
    worker = DailyStoryWorker.__new__(DailyStoryWorker)
    worker.vertex_model = GenerativeModel(model)
    worker.model_name = model
    return worker


def run_case(worker: Any, case: Dict[str, Any], model: str) -> Dict[str, Any]:
    pages = case["pages"]
    page_ids = {f"P{n + 1}": page["key"] for n, page in enumerate(pages)}
    prompt = judge_prompt(name=case["stored"], anchor=case["anchor"], pages=pages, page_ids=page_ids)
    verdict = worker._call_structured_json(prompt, IDENTITY_SCHEMA, JUDGE_MAX_TOKENS)
    doc = {"professor_name": case["stored"], "rag_context": {"section_chunks": {}}}
    review = identity.build_review(
        doc=doc, anchor=case["anchor"], pages=pages, beyond=[], verdict=verdict, page_ids=page_ids,
        model=model, doc_sources_hash="eval", review_input_hash="eval", reviewed_at="eval",
    )
    expect = case["expect"]
    got_pages = [p["verdict"] for p in review["pages"]]
    problems = []
    if review["status"] != expect["status"]:
        problems.append(f"status {review['status']}, expected {expect['status']}")
    for n, (got, want) in enumerate(zip(got_pages, expect["pages"])):
        allowed = want if isinstance(want, list) else [want]
        if got not in allowed:
            problems.append(f"P{n + 1} {got}, expected {' or '.join(allowed)}")
    if review["name_check"] not in expect["name_check"]:
        problems.append(f"name_check {review['name_check']}, expected {' or '.join(expect['name_check'])}")
    if expect.get("usual_spelling_contains") and expect["usual_spelling_contains"] not in review["usual_spelling"]:
        problems.append(f"usual_spelling {review['usual_spelling']!r}")
    return {
        "case": case["name"],
        "passed": not problems,
        "problems": problems,
        "status": review["status"],
        "pages": [{"verdict": p["verdict"], "person": p["person_described"], "reason": p["reason"]} for p in review["pages"]],
        "intended_person": review["intended_person"],
        "name_check": review["name_check"],
        "usual_spelling": review["usual_spelling"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=os.getenv("LEGEND_IDENTITY_MODEL") or os.getenv("STORY_LLM_MODEL") or "gemini-2.5-flash")
    args = parser.parse_args()
    worker = _judge_worker(args.model)
    results = []
    for case in CASES:
        try:
            results.append(run_case(worker, case, args.model))
        except Exception as exc:
            results.append({"case": case["name"], "passed": False, "problems": [f"call failed: {exc}"]})
    passed = sum(1 for r in results if r["passed"])
    print(json.dumps({"model": args.model, "policy": identity.IDENTITY_POLICY_VERSION, "passed": passed, "of": len(results), "results": results}, indent=2))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
