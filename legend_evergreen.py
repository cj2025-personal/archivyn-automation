"""Evergreen stories for a chosen list of legends: N stories each.

Pilot plan B2: grade 3-5 readers had no legend story they could be served
(2026-11-12: 212 stored, none approved, 23 on legal hold). The owner and the
professor pick 15 to 20 legends from legend_selection_report.py; this writes
evergreen stories (daily_story_worker STORY_MODE_EVERGREEN) for just those
legends, through the same gates and checks as the nightly run. The stories
then go through the same stages as any other: the grade-band versions
(daily_story_personalize.py --story-mode evergreen), the AI editor's review
(admin auto-review), and at serving the learner API's age screen.

- Idempotent and resumable: legend L's story n is stored under the key
  "L:evergreen:n". A slot holding a story that did not fail is never written
  again; a failed one is written again. Running the same command twice
  writes nothing the second time; a run stopped part way picks up where it
  left off.
- Each slot takes its own focus from the legend's reviewed passages
  (DailyStoryWorker.evergreen_focus_pool), so a legend's stories differ.
- --limit caps how many stories this run writes, across all legends, in the
  order the legends were given.
- --dry-run writes nothing (the database is wrapped read-only, readonly_mongo)
  and calls no model: it prints what would be written and what it would
  cost, from the prompts it would actually send (model_costs).

The database address is read from LEGEND_MONGO_URI only; no .env is loaded.

Running it (PowerShell 5.1, from this folder):
  $env:LEGEND_MONGO_URI = (gcloud secrets versions access latest --secret=MONGODB_URI)
  $env:STORY_LLM_MODEL = 'gemini-2.5-flash'
  python legend_evergreen.py --legend-ids id1,id2 --stories-each 5 --dry-run
  $env:GCP_PROJECT_ID = 'ngo-application-services'
  python legend_evergreen.py --legend-ids id1,id2 --stories-each 5
  Remove-Item Env:LEGEND_MONGO_URI
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

import daily_story_personalize as personalize
import model_costs
from daily_story_worker import (
    EVERGREEN_MAX_ARTICLE_WORDS,
    EVERGREEN_SLOT_KEY,
    FAILED_STORY_STATUSES,
    STORY_MODE_EVERGREEN,
    DailyStoryWorker,
    parse_date,
    safe_text,
    utc_now_iso,
)
import legend_identity

URI_ENV = "LEGEND_MONGO_URI"
RUNS_COLLECTION = "evergreen_story_runs"
# The AI editor's model is the admin service's (editorialAutoReview.js
# EDITORIAL_REVIEW_MODEL, default gemini-2.5-pro); set the same variable here
# if production sets another.
DEFAULT_EDITOR_MODEL = "gemini-2.5-pro"

# Sizes the estimate assumes for what does not exist before the run: the
# story (the template's longest, EVERGREEN_MAX_ARTICLE_WORDS words at about
# six characters a word), and each step's expected visible answer. Caps are
# the calls' own limits in the code that makes them: the writer's is the
# worker's max_output_tokens; a structured call (a judge, a repair, a band
# version) is given at least 8,192 (DailyStoryWorker._call_structured_json).
STORY_CHARS = EVERGREEN_MAX_ARTICLE_WORDS * 6
STRUCTURED_CAP = 8192
WRITER_OUTPUT_TOKENS = 3500                      # the story, its claim map and fields as JSON
SECTION_REPAIR_OUTPUT = (1500, STRUCTURED_CAP)   # daily_story_worker._repair_section_citations
TOPIC_JUDGE_OUTPUT = (80, STRUCTURED_CAP)        # daily_story_personalize.judge_topic_for_band
VARIANT_OUTPUT = (1600, STRUCTURED_CAP)          # a band version as JSON
VERIFIER_OUTPUT = (150, STRUCTURED_CAP)
VARIANT_ATTEMPTS = (1, 3)              # run_personalization tries a band up to three times a run
EDITOR_OUTPUT = (400, 16384)           # editorialAutoReview maxOutputTokens
EDITOR_PROMPT_CHARS = 3000 + STORY_CHARS + 8000 + 6000  # instructions, story, citations, claim map


def parse_ids(inline: str, path: str) -> List[str]:
    ids: List[str] = []
    for chunk in [inline or ""] + ([open(path, encoding="utf-8").read()] if path else []):
        for raw in chunk.replace("\n", ",").split(","):
            text = raw.strip()
            if text and not text.startswith("#") and text not in ids:
                ids.append(text)
    return ids


def existing_slots(worker: DailyStoryWorker, profile_id: str) -> Dict[int, str]:
    """The legend's evergreen slots already holding a story, and its status."""
    found: Dict[int, str] = {}
    for story in worker.stories_collection.find(
        {"scholar.profile_id": profile_id, "story_mode": STORY_MODE_EVERGREEN, "evergreen.slot": {"$gte": 1}},
        {"evergreen.slot": 1, "status": 1, "story_key": 1},
    ):
        slot = int(((story.get("evergreen") or {}).get("slot")) or 0)
        if slot and story.get("story_key") == EVERGREEN_SLOT_KEY.format(profile_id=profile_id, slot=slot):
            found[slot] = safe_text(story.get("status"))
    return found


def build_plan(worker: DailyStoryWorker, legend_ids: List[str], stories_each: int, limit: Optional[int]) -> List[Dict[str, Any]]:
    """For each legend, in order: whether it can be written about now, its
    slots 1..stories_each, and which of them this run writes (at most limit
    in all)."""
    rows: List[Dict[str, Any]] = []
    budget = limit if limit is not None else float("inf")
    for profile_id in legend_ids:
        doc = worker.scholars_collection.find_one({"profile_id": profile_id})
        row: Dict[str, Any] = {"profile_id": profile_id, "name": "", "blocked": None, "slots": [], "doc": doc}
        rows.append(row)
        if not doc:
            row["blocked"] = "not_found"
            continue
        row["name"] = worker._professor_name(doc)
        row["blocked"] = worker.evergreen_gate(doc)
        if row["blocked"]:
            continue
        creditable = worker._with_creditable_passages(legend_identity.restricted_to_reviewed_sources(doc))
        passages = sum(
            1
            for chunks in ((creditable.get("rag_context") or {}).get("section_chunks") or {}).values()
            for chunk in chunks
            if len(legend_identity.chunk_text(chunk)) >= 80
        )
        if passages < 4:
            # The writer needs four passages it can cite and credit.
            row["blocked"] = f"too_few_passages:{passages}"
            continue
        pool = worker.evergreen_focus_pool(creditable)
        have = existing_slots(worker, profile_id)
        for slot in range(1, stories_each + 1):
            status = have.get(slot)
            focus, source = pool[slot - 1] if slot <= len(pool) else ("", "")
            entry = {"slot": slot, "focus": focus, "focus_source": source, "existing": status or None, "write": False}
            if status and status not in FAILED_STORY_STATUSES:
                entry["state"] = "written"
            elif not focus:
                entry["state"] = "no_distinct_focus"
            elif budget <= 0:
                entry["state"] = "over_limit"
            else:
                entry["state"] = "rewrite_failed" if status else "new"
                entry["write"] = True
                budget -= 1
            row["slots"].append(entry)
    return rows


def audience_band_count(worker: DailyStoryWorker) -> int:
    """How many grade bands the versions stage writes for: those of the
    active learners (personalize.resolve_active_audience reads only)."""
    return len(personalize.resolve_active_audience(worker)["band_keys"])


def estimate(worker: DailyStoryWorker, rows: List[Dict[str, Any]], max_context_chunks: int, writer_model: str, editor_model: str, bands: int):
    """(steps if every draft passes, repair steps if every draft needs both
    repairs, stories counted, stories that would fail for want of passages)."""
    prompt_chars: List[int] = []
    passage_chars: List[int] = []
    short = 0
    for row in rows:
        for entry in row["slots"]:
            if not entry["write"]:
                continue
            inputs = worker.evergreen_inputs(row["doc"], entry["focus"], max_context_chunks)
            entry["passages"] = len(inputs["context_chunks"])
            entry["source_pages"] = len(inputs["allowed_source_urls"])
            entry["prompt_chars"] = len(inputs["prompt"])
            if entry["passages"] < 4:
                entry["state"] = "insufficient_context"
                short += 1
                continue
            prompt_chars.append(len(inputs["prompt"]))
            passage_chars.append(sum(min(len(c["text"]), 1200) for c in inputs["context_chunks"]))
    stories = len(prompt_chars)
    if not stories:
        return [], [], 0, short
    mean = lambda values: int(round(sum(values) / len(values)))  # noqa: E731
    writer_in = model_costs.tokens_for(mean(prompt_chars))
    # A band version's prompt: its brief (~1,800 characters), the story and
    # up to twelve evidence excerpts of 350 characters; the verifier adds the
    # version itself (~5,000 characters as JSON).
    variant_in = model_costs.tokens_for(1800 + STORY_CHARS + 12 * 350)
    verifier_in = model_costs.tokens_for(900 + STORY_CHARS + 12 * 350 + 5000)
    judge_in = model_costs.tokens_for(1600 + 4000)
    low_attempts, high_attempts = VARIANT_ATTEMPTS
    base = [
        model_costs.Step("Writer: the story", writer_model, stories, writer_in, WRITER_OUTPUT_TOKENS, worker.max_output_tokens,
                         note=f"input from the {stories} real prompts, mean {mean(prompt_chars):,} characters"),
        model_costs.Step("Versions: topic ruling per band", writer_model, stories * bands, judge_in, *TOPIC_JUDGE_OUTPUT,
                         note=f"{bands} bands"),
        model_costs.Step("Versions: band version", writer_model, stories * bands * low_attempts, variant_in, *VARIANT_OUTPUT,
                         note=f"1 attempt; up to {high_attempts} when one fails its checks"),
        model_costs.Step("Versions: verifier", writer_model, stories * bands * low_attempts, verifier_in, *VERIFIER_OUTPUT),
        model_costs.Step("AI editor review", editor_model, stories, model_costs.tokens_for(EDITOR_PROMPT_CHARS), *EDITOR_OUTPUT),
    ]
    extra_attempts = high_attempts - low_attempts
    repairs = [
        model_costs.Step("Repair: failing sections", writer_model, stories, model_costs.tokens_for(mean(passage_chars) + 4 * 800 + 1500), *SECTION_REPAIR_OUTPUT),
        model_costs.Step("Repair: whole story", writer_model, stories, writer_in + model_costs.tokens_for(STORY_CHARS + 6000 + 1200),
                         WRITER_OUTPUT_TOKENS, int(worker.max_output_tokens * 0.9)),
        model_costs.Step("Versions: further attempts", writer_model, stories * bands * extra_attempts, variant_in, *VARIANT_OUTPUT),
        model_costs.Step("Versions: their verifier", writer_model, stories * bands * extra_attempts, verifier_in, *VERIFIER_OUTPUT),
    ]
    return base, repairs, stories, short


def print_plan(rows: List[Dict[str, Any]], stories_each: int) -> None:
    print(f"EVERGREEN PLAN  {len(rows)} legends, {stories_each} stories each")
    for row in rows:
        label = f"{row['profile_id']} ({row['name'] or 'not found'})"
        if row["blocked"]:
            print(f"  {label}: not written: {row['blocked']}")
            continue
        print(f"  {label}:")
        for entry in row["slots"]:
            extra = ""
            if "passages" in entry:
                extra = f"  [{entry['passages']} passages from {entry['source_pages']} pages, prompt {entry['prompt_chars']:,} chars]"
            focus = f" on \"{entry['focus']}\"" if entry["focus"] else ""
            print(f"    story {entry['slot']}: {entry['state']}{focus}{extra}")


def run(args: argparse.Namespace) -> int:
    uri = safe_text(os.environ.get(URI_ENV))
    if not uri:
        print(f"Set {URI_ENV} to the database to write to (see this script's header).", file=sys.stderr)
        return 2
    model = safe_text(args.model) or safe_text(os.environ.get("STORY_LLM_MODEL"))
    if not model:
        print("Say which model writes: --model or STORY_LLM_MODEL (production's job sets gemini-2.5-flash).", file=sys.stderr)
        return 2
    legend_ids = parse_ids(args.legend_ids, args.legend_ids_file)
    if not legend_ids:
        print("Name the legends: --legend-ids id1,id2 or --legend-ids-file.", file=sys.stderr)
        return 2
    overrides = model_costs.parse_price_overrides(args.price)
    editor_model = safe_text(os.environ.get("EDITORIAL_REVIEW_MODEL")) or DEFAULT_EDITOR_MODEL

    worker = DailyStoryWorker(
        model_name=model,
        use_llm=not args.dry_run,
        trends_enabled=False,
        enforce_profile_quality=not args.ignore_profile_quality,
        mongodb_uri=uri,
        load_env_file=False,
        read_only=args.dry_run,
    )
    try:
        rows = build_plan(worker, legend_ids, args.stories_each, args.limit)
        story_date = parse_date(args.date).isoformat()
        if args.dry_run:
            bands = audience_band_count(worker)
            base, repairs, stories, short = estimate(worker, rows, args.max_context_chunks, model, editor_model, bands)
            print_plan(rows, args.stories_each)
            print(f"\nWould write {stories} stories dated {story_date}; {short} would fail for want of passages. Nothing was written and no model was called.")
            if stories:
                print()
                print(model_costs.report(f"COST ESTIMATE, {stories} stories (every draft passes first time)", base, overrides))
                print(model_costs.report("If every draft also needed both repairs and every version three attempts, add", repairs, overrides))
            return 0

        run_id = f"evergreen-{uuid.uuid4()}"
        runs = worker.db[RUNS_COLLECTION]
        runs.insert_one({
            "run_id": run_id,
            "started_at": utc_now_iso(),
            "status": "running",
            "config": {
                "legend_ids": legend_ids,
                "stories_each": args.stories_each,
                "limit": args.limit,
                "story_date": story_date,
                "model": model,
                "enforce_profile_quality": not args.ignore_profile_quality,
                "max_context_chunks": args.max_context_chunks,
            },
        })
        summary = {key: 0 for key in ("generated", "skipped_existing", "skipped_identity", "skipped_profile_quality", "failed_no_context", "failed_validation", "failed_generation", "evergreen")}
        errors: List[str] = []
        outcomes: List[Dict[str, Any]] = []
        for row in rows:
            for entry in row["slots"]:
                if not entry["write"]:
                    continue
                before = dict(summary)
                status = worker.write_evergreen_slot(
                    scholar_doc=row["doc"],
                    slot=entry["slot"],
                    focus=entry["focus"],
                    focus_source=entry["focus_source"],
                    story_date=story_date,
                    run_id=run_id,
                    dry_run=False,
                    max_context_chunks=args.max_context_chunks,
                    summary=summary,
                    all_errors=errors,
                )
                changed = [key for key in summary if summary[key] != before[key] and key != "evergreen"]
                outcomes.append({
                    "story_key": EVERGREEN_SLOT_KEY.format(profile_id=row["profile_id"], slot=entry["slot"]),
                    "focus": entry["focus"],
                    "stored_status": status,
                    "outcome": changed[0] if changed else "nothing",
                })
        result = {
            "run_id": run_id,
            "status": "completed" if summary["failed_validation"] + summary["failed_generation"] + summary["failed_no_context"] == 0 else "completed_with_failures",
            "summary": summary,
            "blocked": {row["profile_id"]: row["blocked"] for row in rows if row["blocked"]},
            "outcomes": outcomes,
            "errors": errors[:100],
        }
        runs.update_one({"run_id": run_id}, {"$set": {**result, "ended_at": utc_now_iso()}})
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(
            "\nNext: the grade-band versions (python daily_story_personalize.py --story-mode evergreen --max-stories "
            f"{max(1, len(outcomes))}), then the AI editor (admin auto-review). A failed story is written again by "
            "running this command again."
        )
        return 0 if result["status"] == "completed" else 1
    finally:
        worker.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write evergreen stories for a chosen list of legends.")
    parser.add_argument("--legend-ids", default="", help="Comma-separated legend profile ids, in the order to write them")
    parser.add_argument("--legend-ids-file", default="", help="A file of legend profile ids, one per line or comma-separated")
    parser.add_argument("--stories-each", type=int, required=True, help="Evergreen stories each legend should have")
    parser.add_argument("--limit", type=int, default=None, help="Write at most this many stories in this run")
    parser.add_argument("--dry-run", action="store_true", help="Write nothing and call no model; print the plan and its cost")
    parser.add_argument("--date", default=None, help="Story date for what is written (default: today, UTC)")
    parser.add_argument("--model", default=None, help="The writer's model (default: STORY_LLM_MODEL)")
    parser.add_argument("--max-context-chunks", type=int, default=12)
    parser.add_argument("--ignore-profile-quality", action="store_true", help="Do not hold legends to the profile-quality gate")
    parser.add_argument("--price", action="append", default=[], help="MODEL=INPUT/OUTPUT USD per 1M tokens, for a model model_costs does not list")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.stories_each < 1:
        print("--stories-each must be at least 1.", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 0:
        print("--limit cannot be negative.", file=sys.stderr)
        return 2
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
