"""
Legend identity review: does each legend's stored source material describe
that legend? (The why, and the rules, are in legend_identity.py.)

For every legend whose sources, anchor, model or policy changed since its last
review, one model call judges each of its source pages against the legend we
mean. The result is kept on the legend (identity_review) and appended to
legend_identity_review_events; the run is recorded in
legend_identity_review_jobs. A legend whose review is unchanged costs nothing,
so the nightly run only pays for new or re-sourced legends.

  python legend_identity_review.py                     # every legend
  python legend_identity_review.py --scholar-id robert-williams
  python legend_identity_review.py --force             # judge again even when unchanged
  python legend_identity_review.py --dry-run           # judge, print, write nothing

Settings: LEGEND_IDENTITY_MODEL (default: the story model, STORY_LLM_MODEL),
LEGEND_IDENTITY_CONCURRENCY (default 4).

Prints the run summary as JSON. Story generation waits on this stage, so the
exit code is about the run, not each legend: 0 when it completed, legends
needing attention and single failures included (a legend whose review failed
keeps its last review and, its sources having changed, stays out of stories;
the summary and the run record name it); 1 when every legend it had to judge
failed (the model unreachable, say); 2 when it could not start.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import legend_identity as identity
from daily_story_worker import DailyStoryWorker, safe_text, utc_now_iso

DEFAULT_CONCURRENCY = 4
JUDGE_MAX_TOKENS = 8192

IDENTITY_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "pages": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "page_id": {"type": "STRING"},
                    "verdict": {"type": "STRING", "enum": list(identity.PAGE_VERDICTS)},
                    "person_described": {"type": "STRING"},
                    "reason": {"type": "STRING"},
                },
                "required": ["page_id", "verdict", "person_described", "reason"],
            },
        },
        "intended_person": {
            "type": "OBJECT",
            "properties": {
                "full_name": {"type": "STRING"},
                "field": {"type": "STRING"},
                "years": {"type": "STRING"},
                "known_for": {"type": "STRING"},
            },
            "required": ["full_name", "field", "years", "known_for"],
        },
        "name_check": {"type": "STRING", "enum": list(identity.NAME_CHECKS)},
        "usual_spelling": {"type": "STRING"},
    },
    "required": ["pages", "intended_person", "name_check", "usual_spelling"],
}


def judge_prompt(*, name: str, anchor: Dict[str, str], pages: List[Dict[str, Any]], page_ids: Dict[str, str]) -> str:
    """The model's brief: the legend we mean, then each page's excerpts."""
    if anchor.get("from") == "administrator":
        meant = f"An administrator confirmed: {anchor['name']}" + (f", {anchor['field']}" if anchor.get("field") else "") + "."
    elif anchor.get("from") == "roster":
        meant = (
            f"Our roster lists: {anchor['name']}" + (f", field: {anchor['field']}" if anchor.get("field") else "")
            + ". (The roster was typed by hand; its spelling may be wrong.)"
        )
    else:
        meant = "No roster entry: only the stored name is known."
    key_to_id = {key: pid for pid, key in page_ids.items()}
    blocks = []
    for page in pages:
        excerpts = "\n\n".join(f'"""\n{e}\n"""' for e in page["excerpts"])
        where = page["url"] or "(no link recorded)"
        blocks.append(f"[{key_to_id[page['key']]}] {where} ({page['passages']} passages stored)\n{excerpts}")
    pages_text = "\n\n".join(blocks) if blocks else "None: nothing is stored for this legend."
    return f"""You check whether the source pages stored for one of our legends describe that legend.
Our legends are notable scholars, scientists, writers and educators, no longer living, whose work we present to school students.

THE LEGEND WE MEAN
Name as stored: {name}
{meant}

THE SOURCE PAGES (excerpts of the passages we stored from each)
{pages_text}

For each page, give its id and one verdict:
- about_legend: the page is about this legend, or is the legend's own work.
- different_person: the page is about someone else, for example another person with the same or a similar name. In person_described, name that person and what they are known for.
- not_about_a_person: the page is about a topic, place, event or organisation, not about one person.
- unclear: the excerpts are too short or too general to tell.
A page about someone in a different field or era is a different person even when the name matches.
Your verdict on a page covers every excerpt shown under it: if any excerpt is about someone other than this legend, rule the page different_person.
Excerpts under "(no link recorded)" were stored without a source; judge them like any page.

Then:
- intended_person: who this legend is: full name as usually written, field, birth and death years, and what they are known for. Use the pages you ruled about_legend and well-established knowledge; leave a part empty rather than guess.
- name_check: "correct" if the stored name is spelled as this person's name is usually written (a shorter form, such as one without a middle name, is correct); "misspelled" if a letter or word is wrong or the parts are in the wrong order; "unknown" if you cannot tell who the legend is.
- usual_spelling: the usual spelling when misspelled, else empty.
Return JSON only."""


def review_legend(worker: DailyStoryWorker, doc: Dict[str, Any], *, model: str, force: bool) -> Tuple[str, Optional[Dict[str, Any]]]:
    """("unchanged", None) when the legend's review still applies, else
    ("reviewed", the new identity_review). Raises when the model fails."""
    name = identity.stored_name(doc)
    anchor = identity.anchor_for(doc)
    pages, beyond = identity.source_pages(doc)
    doc_sources_hash = identity.sources_hash(doc)
    review_input_hash = identity.input_hash(anchor=anchor, doc_sources_hash=doc_sources_hash, model=model, name=name)
    previous = doc.get("identity_review") if isinstance(doc.get("identity_review"), dict) else {}
    if not force and previous.get("input_hash") == review_input_hash and previous.get("policy_version") == identity.IDENTITY_POLICY_VERSION:
        return "unchanged", None
    page_ids = {f"P{n + 1}": page["key"] for n, page in enumerate(pages)}
    prompt = judge_prompt(name=name, anchor=anchor, pages=pages, page_ids=page_ids)
    verdict = worker._call_structured_json(prompt, IDENTITY_SCHEMA, JUDGE_MAX_TOKENS)
    return "reviewed", identity.build_review(
        doc=doc,
        anchor=anchor,
        pages=pages,
        beyond=beyond,
        verdict=verdict,
        page_ids=page_ids,
        model=model,
        doc_sources_hash=doc_sources_hash,
        review_input_hash=review_input_hash,
        reviewed_at=utc_now_iso(),
    )


# What the review reads of a legend: its passages and names, not the text
# built from all pages together (section_text, section_summaries), the
# bulkiest part of a legend and never shown to the judge.
LEGEND_PROJECTION = {"rag_context.section_text": 0, "rag_context.section_summaries": 0}


def _legends(worker: DailyStoryWorker, scholar_id: Optional[str], max_scholars: int) -> List[Dict[str, Any]]:
    if scholar_id:
        found = list(worker.scholars_collection.find({"profile_id": scholar_id}, LEGEND_PROJECTION).limit(1))
        return found or list(worker.scholars_collection.find({"_id": scholar_id}, LEGEND_PROJECTION).limit(1))
    return list(worker.scholars_collection.find({}, LEGEND_PROJECTION).limit(max_scholars))


def _judge_model(worker: DailyStoryWorker) -> str:
    """LEGEND_IDENTITY_MODEL when set, else the worker's story model."""
    wanted = safe_text(os.getenv("LEGEND_IDENTITY_MODEL"))
    if wanted and wanted != worker.model_name:
        import daily_story_worker as worker_module

        worker.vertex_model = worker_module.GenerativeModel(wanted)
        worker.model_name = wanted
    return worker.model_name


def run_identity_review(
    *,
    scholar_id: Optional[str],
    max_scholars: int,
    force: bool,
    dry_run: bool,
    concurrency: int,
    worker: Optional[DailyStoryWorker] = None,
) -> Dict[str, Any]:
    own_worker = worker is None
    worker = worker or DailyStoryWorker(use_llm=True, trends_enabled=False)
    try:
        model = _judge_model(worker)
        events = worker.db["legend_identity_review_events"]
        jobs = worker.db["legend_identity_review_jobs"]
        if not dry_run:
            events.create_index([("profile_id", 1), ("reviewed_at", -1)])
            jobs.create_index("run_id", unique=True)
            jobs.create_index("started_at")

        run_id = f"identity-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        summary: Dict[str, Any] = {"attempted": 0, "reviewed": 0, "unchanged": 0, "failed": 0, "eligible": 0, "statuses": {}, "needs_attention": 0}
        if not dry_run:
            jobs.insert_one({
                "run_id": run_id,
                "started_at": utc_now_iso(),
                "status": "running",
                "config": {"scholar_id": scholar_id, "max_scholars": max_scholars, "force": force, "model": model, "policy_version": identity.IDENTITY_POLICY_VERSION},
            })

        docs = _legends(worker, scholar_id, max_scholars)
        errors: List[str] = []
        attention: List[Dict[str, Any]] = []

        def one(doc: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]], List[str]]:
            """(outcome, new review or None, the attention reasons it replaced)."""
            previous = doc.get("identity_review") if isinstance(doc.get("identity_review"), dict) else {}
            previous_attention = list(previous.get("attention") or [])
            outcome, review = review_legend(worker, doc, model=model, force=force)
            if review is not None and not dry_run:
                worker.scholars_collection.update_one({"_id": doc["_id"]}, {"$set": {"identity_review": review}})
                events.insert_one({"profile_id": worker._profile_id(doc), "run_id": run_id, **review})
            return outcome, review, previous_attention

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {pool.submit(one, doc): doc for doc in docs}
            for future in as_completed(futures):
                doc = futures[future]
                profile_id = worker._profile_id(doc)
                summary["attempted"] += 1
                try:
                    outcome, review, previous_attention = future.result()
                except Exception as exc:  # one legend's failure is reported, never fatal
                    summary["failed"] += 1
                    errors.append(f"{profile_id}:{safe_text(exc)[:240]}")
                    continue
                summary[outcome] += 1
                current = review or doc.get("identity_review") or {}
                status = safe_text(current.get("status")) or "unknown"
                summary["statuses"][status] = summary["statuses"].get(status, 0) + 1
                if current.get("eligible"):
                    summary["eligible"] += 1
                if current.get("attention"):
                    summary["needs_attention"] += 1
                    attention.append({
                        "profile_id": profile_id,
                        "name": identity.stored_name(doc),
                        "status": status,
                        "outcome": outcome,
                        # New to an administrator: judged now, with reasons it did not have before.
                        "changed": review is not None and list(current["attention"]) != previous_attention,
                        "attention": current["attention"],
                    })

        if not errors:
            status = "completed"
        elif summary["reviewed"] or summary["unchanged"]:
            status = "completed_with_failures"
        else:
            status = "failed"
        result = {"run_id": run_id, "status": status, "dry_run": dry_run, "model": model, "summary": summary, "errors": errors[:50], "attention": sorted(attention, key=lambda a: a["profile_id"])}
        if not dry_run:
            jobs.update_one(
                {"run_id": run_id},
                {"$set": {"ended_at": utc_now_iso(), "status": status, "summary": summary, "errors": errors[:200]}},
            )
        return result
    finally:
        if own_worker:
            worker.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check that each legend's stored sources describe that legend.")
    parser.add_argument("--scholar-id", type=str, default=None, help="Review one legend (profile_id).")
    parser.add_argument("--max-scholars", type=int, default=1000)
    parser.add_argument("--force", action="store_true", help="Judge again even when nothing changed.")
    parser.add_argument("--dry-run", action="store_true", help="Judge and print; write nothing.")
    parser.add_argument("--concurrency", type=int, default=int(safe_text(os.getenv("LEGEND_IDENTITY_CONCURRENCY")) or DEFAULT_CONCURRENCY))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        result = run_identity_review(
            scholar_id=args.scholar_id,
            max_scholars=max(1, args.max_scholars),
            force=args.force,
            dry_run=args.dry_run,
            concurrency=args.concurrency,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": safe_text(exc)[:500]}), flush=True)
        return 2
    print(json.dumps(result, indent=2, default=str), flush=True)
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
