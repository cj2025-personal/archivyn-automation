"""A short biography of each chosen legend that a grade 3-5 reader can read.

Pilot plan B6: a legend's page showed its adult short bio to every reader;
across the 73 cleared legends those read at a median Flesch-Kincaid grade of
12.9 (2026-11-12). This writes, for the legends named, a second short bio at
a grade 3-5 reading level, from the legend's stored, identity-reviewed
passages only, and stores it beside the adult one, which is kept:

  about.bio_by_band.grade_3_5 = {text, status: "ready", checks, ...}

The learner API shows it to readers in grades 5 and below, when present
(user-dashboard-api lib/legendBio).

Before it is stored a bio must pass every check (decisions are the model's;
counting is code's):
- reading level: Flesch-Kincaid grade at or below 5.0, measured both as this
  pipeline measures its grade-band versions (daily_story_personalize
  readability_metrics) and as the learner API measures (learner_api_fk_grade,
  a copy of its lib/readability measureReadability), and 40 to 110 words;
- grounded: a model rules that every fact in it is stated in the passages;
- suitable: a model rules on it under the grade 3-5 topic rules the story
  versions are held to (daily_story_personalize.TOPIC_SUITABILITY_RULES).
A bio that fails on length or reading level is revised (up to three tries);
one with an unsupported fact is written again without it; one ruled
unsuitable is not stored. The learner API's age screen still reads it before
showing it.

Idempotent and resumable: a stored bio records a hash of what it was written
from (the passages, this policy, the model); a legend whose stored bio has
the same hash is skipped unless --force.

The database address is read from LEGEND_MONGO_URI only; no .env is loaded.
--dry-run writes nothing (the database is wrapped read-only) and calls no
model: it prints which legends would get a bio and what it would cost, from
the prompts it would actually send.

Running it (PowerShell 5.1, from this folder):
  $env:LEGEND_MONGO_URI = (gcloud secrets versions access latest --secret=MONGODB_URI)
  $env:STORY_LLM_MODEL = 'gemini-2.5-flash'
  python legend_child_bio.py --legend-ids id1,id2 --dry-run
  $env:GCP_PROJECT_ID = 'ngo-application-services'
  python legend_child_bio.py --legend-ids id1,id2
  Remove-Item Env:LEGEND_MONGO_URI
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import daily_story_personalize as personalize
import legend_identity
import model_costs
from daily_story_worker import SECTION_PRIORITY, DailyStoryWorker, creditable_source, safe_text, utc_now_iso
from legend_evergreen import parse_ids

URI_ENV = "LEGEND_MONGO_URI"
BIO_BAND = "grade_3_5"
BIO_FIELD = f"about.bio_by_band.{BIO_BAND}"
BIO_POLICY_VERSION = "legend-child-bio-v1"
RUNS_COLLECTION = "legend_child_bio_runs"
MAX_FK_GRADE = 5.0
MIN_WORDS = 40
MAX_WORDS = 110
MAX_PASSAGES = 12
PASSAGE_CHARS = 900
MAX_ATTEMPTS = 3
STRUCTURED_CAP = 8192  # a structured call's least output budget (DailyStoryWorker._call_structured_json)

BIO_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "bio": {"type": "STRING"},
        "used_passage_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["bio", "used_passage_ids"],
}
GROUNDING_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "unsupported_claims": {"type": "ARRAY", "items": {"type": "STRING"}},
        "reason": {"type": "STRING"},
    },
    "required": ["unsupported_claims", "reason"],
}
SUITABILITY_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {"suitable": {"type": "BOOLEAN"}, "reason": {"type": "STRING"}},
    "required": ["suitable", "reason"],
}


# ── Reading level ────────────────────────────────────────────────────────────

def _api_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    trimmed = re.sub(r"^y", "", re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w))
    groups = re.findall(r"[aeiouy]{1,2}", trimmed)
    return max(1, len(groups) if groups else 1)


def learner_api_fk_grade(text: str) -> Optional[float]:
    """Flesch-Kincaid grade exactly as the learner API measures a text
    (user-dashboard-api backend/src/lib/readability.ts measureReadability:
    words split on spaces, sentences on . ! ?, the same syllable rule), so a
    bio stored as at or below grade 5 is measured so where it is shown."""
    words = [w for w in str(text or "").strip().split() if w]
    sentences = [s.strip() for s in re.split(r"[.!?]+", str(text or "")) if s.strip()]
    if not words or not sentences:
        return None
    syllables = sum(_api_syllables(w) for w in words)
    grade = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
    return round(grade, 1)


def reading_check(text: str) -> Dict[str, Any]:
    pipeline = personalize.readability_metrics(text)
    api_grade = learner_api_fk_grade(text)
    words = pipeline["word_count"]
    problems: List[str] = []
    if not MIN_WORDS <= words <= MAX_WORDS:
        problems.append(f"word_count:{words}")
    worst = max(pipeline["flesch_kincaid_grade"], api_grade if api_grade is not None else 99.0)
    if worst > MAX_FK_GRADE:
        problems.append(f"reading_grade:{worst}")
    return {
        "ok": not problems,
        "problems": problems,
        "words": words,
        "flesch_kincaid_grade": pipeline["flesch_kincaid_grade"],
        "learner_api_fk_grade": api_grade,
        "average_sentence_words": pipeline["average_sentence_words"],
        "max_grade": MAX_FK_GRADE,
    }


# ── What a bio is written from ───────────────────────────────────────────────

def bio_passages(doc: Dict[str, Any]) -> List[Dict[str, str]]:
    """The legend's reviewed passages a bio may use, life before works:
    passages of pages its identity review ruled about the legend, whose page
    may be credited (creditable_source), the biographical sections first,
    longer ones first."""
    restricted = legend_identity.restricted_to_reviewed_sources(doc)
    sections = (restricted.get("rag_context") or {}).get("section_chunks") or {}
    found: List[Tuple[int, int, str, Dict[str, Any]]] = []
    for section, chunks in sections.items():
        for chunk in chunks:
            text = legend_identity.chunk_text(chunk)
            links = legend_identity.chunk_links(chunk)
            if len(text) < legend_identity.MIN_PASSAGE_CHARS or not links or not creditable_source(links[0]):
                continue
            found.append((SECTION_PRIORITY.get(safe_text(section).lower(), 99), -len(text), safe_text(section), {**chunk, "_url": links[0]}))
    found.sort(key=lambda row: (row[0], row[1]))
    passages = []
    for index, (_, _, section, chunk) in enumerate(found[:MAX_PASSAGES], start=1):
        passages.append({
            "id": f"P{index}",
            "chunk_id": safe_text(chunk.get("chunk_id")) or f"{section}:{index}",
            "section": section,
            "text": legend_identity.chunk_text(chunk)[:PASSAGE_CHARS],
            "url": chunk["_url"],
        })
    return passages


def input_hash(name: str, passages: List[Dict[str, str]], model: str) -> str:
    payload = {
        "policy": BIO_POLICY_VERSION,
        "model": model,
        "name": name,
        "passages": [[p["chunk_id"], hashlib.sha256(p["text"].encode("utf-8")).hexdigest()] for p in passages],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _passage_block(passages: List[Dict[str, str]]) -> str:
    return "\n\n".join(f"[{p['id']}] ({p['section']}) {p['text']}" for p in passages)


def writer_prompt(name: str, passages: List[Dict[str, str]], previous: str = "", fixes: Optional[List[str]] = None) -> str:
    revise = ""
    if previous:
        revise = (
            "\nYour last bio did not pass its checks. Write it again, fixing these:\n"
            + "\n".join(f"- {fix}" for fix in (fixes or []))
            + f"\nThe last bio:\n{previous}\n"
        )
    return f"""Write a short biography of {name} for children in grades 3 to 5 (about 8 to 11 years old).

Use only facts stated in the passages below; add nothing else, not even well-known facts.
- {MIN_WORDS} to {MAX_WORDS} words, in 4 to 7 short sentences.
- Short, everyday words, sentences of about 8 to 12 words; it must read at grade 5 or below.
- Say who {name} was, what they did or studied, and one thing their work helped people know or do.
- Past tense; third person; no quotations; no dates or numbers the passages do not give.
- Leave out violence, cruelty or death in detail; if the passages are mostly about such things, say so plainly in one gentle sentence or not at all.
- used_passage_ids: the IDs of the passages the bio uses.
{revise}
Passages:
{_passage_block(passages)}

Return JSON: {{"bio": "...", "used_passage_ids": ["P1", "P2"]}}"""


def grounding_prompt(name: str, passages: List[Dict[str, str]], bio: str) -> str:
    return f"""You check a short children's biography of {name} against its sources.
List every claim in the biography that the passages do not state (a fact, a date, a number, a place,
a description of what {name} did or thought). A plain restatement of a passage is supported; a fact
that is true but not in the passages is not. Return JSON: unsupported_claims (an empty list when
every claim is supported) and reason (one sentence).

Passages:
{_passage_block(passages)}

Biography:
{bio}"""


def suitability_prompt(name: str, bio: str) -> str:
    band = personalize.GRADE_BANDS[BIO_BAND]
    return f"""You decide whether this short biography of {name} may be shown to {band['label']} readers
(typically ages {band['typical_age_min']} to {band['typical_age_max']}) on a children's reading platform.

Rules for this band:
{personalize.TOPIC_SUITABILITY_RULES[BIO_BAND]}

Decide on what it says and what a reader would have to engage with. Return JSON only: suitable (true
or false) and reason (one sentence a reviewer can read).

Biography:
{bio}"""


# ── Writing one bio ──────────────────────────────────────────────────────────

Caller = Callable[[str, Dict[str, Any], int], Dict[str, Any]]


def write_bio(name: str, passages: List[Dict[str, str]], call: Caller) -> Dict[str, Any]:
    """Writes and checks one bio. Returns {"status": "ready", ...the record}
    or {"status": "failed"|"not_suitable", "attempts": [...]}; never raises
    for a model's answer, only for a model that cannot be reached."""
    ids = {p["id"]: p for p in passages}
    attempts: List[Dict[str, Any]] = []
    previous, fixes = "", []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        reply = call(writer_prompt(name, passages, previous, fixes), BIO_SCHEMA, 2048)
        bio = re.sub(r"\s+", " ", safe_text(reply.get("bio"))).strip()
        used = list(dict.fromkeys(
            safe_text(i).upper() for i in (reply.get("used_passage_ids") or []) if safe_text(i).upper() in ids
        ))
        reading = reading_check(bio)
        row: Dict[str, Any] = {"attempt": attempt, "bio": bio, "reading": reading, "used_passage_ids": used}
        attempts.append(row)
        fixes = []
        if not bio:
            fixes.append("The bio was empty.")
        if not used:
            fixes.append("Name the passages the bio uses in used_passage_ids.")
        for problem in reading["problems"]:
            if problem.startswith("word_count"):
                fixes.append(f"It had {reading['words']} words; it must have {MIN_WORDS} to {MAX_WORDS}.")
            else:
                fixes.append(
                    f"It read at grade {problem.split(':', 1)[1]}; it must read at grade {MAX_FK_GRADE:g} or below. "
                    "Use shorter sentences and shorter, everyday words."
                )
        if fixes:
            previous = bio
            continue

        grounding = call(grounding_prompt(name, passages, bio), GROUNDING_SCHEMA, 1024)
        unsupported = [safe_text(c) for c in (grounding.get("unsupported_claims") or []) if safe_text(c)]
        row["grounding"] = {"unsupported_claims": unsupported, "reason": safe_text(grounding.get("reason"))[:400]}
        if unsupported:
            previous = bio
            fixes = [f"Remove this claim, which the passages do not state: {claim}" for claim in unsupported[:6]]
            continue

        verdict = call(suitability_prompt(name, bio), SUITABILITY_SCHEMA, 1024)
        if not isinstance(verdict.get("suitable"), bool):
            raise RuntimeError("suitability judge returned no verdict")
        row["suitability"] = {"suitable": verdict["suitable"], "reason": safe_text(verdict.get("reason"))[:400]}
        if not verdict["suitable"]:
            return {"status": "not_suitable", "attempts": attempts}
        return {
            "status": "ready",
            "text": bio,
            "readability": reading,
            "checks": {
                "reading_level": {"ok": True, "max_grade": MAX_FK_GRADE},
                "grounded": row["grounding"],
                "suitability": {**row["suitability"], "band": BIO_BAND, "policy_version": personalize.TOPIC_POLICY_VERSION},
            },
            "passages": [
                {"id": ids[i]["id"], "chunk_id": ids[i]["chunk_id"], "url": ids[i]["url"]} for i in used
            ],
            "attempts": attempt,
        }
    return {"status": "failed", "attempts": attempts}


# ── The run ──────────────────────────────────────────────────────────────────

def plan(worker: DailyStoryWorker, legend_ids: List[str], force: bool, model: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    budget = limit if limit is not None else float("inf")
    for profile_id in legend_ids:
        doc = worker.scholars_collection.find_one({"profile_id": profile_id})
        row: Dict[str, Any] = {"profile_id": profile_id, "doc": doc, "name": "", "state": "", "write": False}
        rows.append(row)
        if not doc:
            row["state"] = "not_found"
            continue
        row["name"] = worker._professor_name(doc)
        identity_ok, reason = legend_identity.story_eligibility(doc)
        if not identity_ok:
            row["state"] = f"identity_blocked:{reason}"
            continue
        passages = bio_passages(doc)
        row["passages"] = passages
        if len(passages) < 2:
            row["state"] = "too_few_passages"
            continue
        row["input_hash"] = input_hash(row["name"], passages, model)
        stored = (((doc.get("about") or {}).get("bio_by_band") or {}).get(BIO_BAND) or {})
        if not force and stored.get("status") == "ready" and stored.get("input_hash") == row["input_hash"]:
            row["state"] = "current"
            continue
        if budget <= 0:
            row["state"] = "over_limit"
            continue
        budget -= 1
        row["state"] = "rewrite" if stored else "new"
        row["write"] = True
    return rows


def estimate(rows: List[Dict[str, Any]], model: str) -> Tuple[List[model_costs.Step], List[model_costs.Step], int]:
    writing = [row for row in rows if row["write"]]
    if not writing:
        return [], [], 0
    bio_chars = MAX_WORDS * 6
    mean = lambda values: int(round(sum(values) / len(values)))  # noqa: E731
    writer_in = model_costs.tokens_for(mean([len(writer_prompt(r["name"], r["passages"])) for r in writing]))
    grounding_in = model_costs.tokens_for(mean([len(grounding_prompt(r["name"], r["passages"], "x" * bio_chars)) for r in writing]))
    suitability_in = model_costs.tokens_for(len(suitability_prompt("A Legend", "x" * bio_chars)))
    n = len(writing)
    base = [
        model_costs.Step("Bio writer", model, n, writer_in, 200, STRUCTURED_CAP, note="input from the real prompts"),
        model_costs.Step("Grounding check", model, n, grounding_in, 60, STRUCTURED_CAP),
        model_costs.Step("Age suitability check", model, n, suitability_in, 60, STRUCTURED_CAP),
    ]
    more = MAX_ATTEMPTS - 1
    retries = [
        model_costs.Step("Bio writer, further tries", model, n * more, writer_in + model_costs.tokens_for(bio_chars + 400), 200, STRUCTURED_CAP),
        model_costs.Step("Grounding check, further tries", model, n * more, grounding_in, 60, STRUCTURED_CAP),
    ]
    return base, retries, n


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

    worker = DailyStoryWorker(
        model_name=model,
        use_llm=not args.dry_run,
        trends_enabled=False,
        mongodb_uri=uri,
        load_env_file=False,
        read_only=args.dry_run,
    )
    try:
        rows = plan(worker, legend_ids, args.force, model, args.limit)
        if args.dry_run:
            print(f"CHILD BIO PLAN  {len(rows)} legends, band {BIO_BAND}, at or below grade {MAX_FK_GRADE:g}")
            for row in rows:
                detail = f", {len(row.get('passages') or [])} passages" if row.get("passages") else ""
                print(f"  {row['profile_id']} ({row['name'] or 'not found'}): {row['state']}{detail}")
            base, retries, n = estimate(rows, model)
            print(f"\nWould write {n} bios. Nothing was written and no model was called.")
            if n:
                print()
                print(model_costs.report(f"COST ESTIMATE, {n} bios (each passes on its first try)", base, overrides))
                print(model_costs.report(f"If every bio needed all {MAX_ATTEMPTS} tries, add", retries, overrides))
            return 0

        run_id = f"child-bio-{uuid.uuid4()}"
        results: List[Dict[str, Any]] = []
        for row in rows:
            if not row["write"]:
                results.append({"profile_id": row["profile_id"], "outcome": row["state"]})
                continue
            try:
                outcome = write_bio(row["name"], row["passages"], worker._call_structured_json)
            except Exception as exc:
                results.append({"profile_id": row["profile_id"], "outcome": "error", "error": safe_text(exc)[:300]})
                continue
            if outcome["status"] == "ready":
                record = {
                    **outcome,
                    "band": BIO_BAND,
                    "policy_version": BIO_POLICY_VERSION,
                    "model": model,
                    "input_hash": row["input_hash"],
                    "written_at": utc_now_iso(),
                    "run_id": run_id,
                }
                # Only the child bio's own field: the adult bio is kept.
                worker.scholars_collection.update_one({"_id": row["doc"]["_id"]}, {"$set": {BIO_FIELD: record}})
                results.append({
                    "profile_id": row["profile_id"],
                    "outcome": "stored",
                    "text": record["text"],
                    "flesch_kincaid_grade": record["readability"]["flesch_kincaid_grade"],
                    "learner_api_fk_grade": record["readability"]["learner_api_fk_grade"],
                })
            else:
                results.append({"profile_id": row["profile_id"], "outcome": outcome["status"], "attempts": outcome["attempts"]})
        worker.db[RUNS_COLLECTION].insert_one({
            "run_id": run_id,
            "at": utc_now_iso(),
            "model": model,
            "policy_version": BIO_POLICY_VERSION,
            "results": results,
        })
        stored = sum(1 for r in results if r["outcome"] == "stored")
        failed = [r for r in results if r["outcome"] in ("failed", "error")]
        print(json.dumps({"run_id": run_id, "stored": stored, "results": results}, indent=2, ensure_ascii=False))
        return 0 if not failed else 1
    finally:
        worker.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write grade 3-5 bios for chosen legends.")
    parser.add_argument("--legend-ids", default="", help="Comma-separated legend profile ids")
    parser.add_argument("--legend-ids-file", default="", help="A file of legend profile ids, one per line or comma-separated")
    parser.add_argument("--limit", type=int, default=None, help="Write at most this many bios in this run")
    parser.add_argument("--force", action="store_true", help="Write again even where a current bio is stored")
    parser.add_argument("--dry-run", action="store_true", help="Write nothing and call no model; print the plan and its cost")
    parser.add_argument("--model", default=None, help="The model (default: STORY_LLM_MODEL)")
    parser.add_argument("--price", action="append", default=[], help="MODEL=INPUT/OUTPUT USD per 1M tokens, for a model model_costs does not list")
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
