"""Targeted profile repair: more source pages for named legends, so the
profile-quality gate can call them ready and stories can be written about
them (pilot: child-friendly scientists for grades 3 to 5).

Why it collects. A legend is "ready" only with 20+ stored passages and a
score of 60+ (profile_quality_refresh.evaluate_profile; see
legend_profile_diagnosis.py for each legend's reasons). Scoring again changes
nothing: the stored scores matched a recomputation on 2026-11-01. Only more
passages about the legend, from pages fetched under today's rules, help.

Two modes.

APPEND (the default). The diagnosis of 2026-11-13 found that each legend's
stored URL list holds pages that gave no stored passage (Chappelle 4, Fuller
5, Du Bois 8 with en.wikipedia and loc.gov among them), and that a full
re-collection would make them worse: the present chunker titles sections
after their pages, so a profile collected again names no core section and
loses its coverage points (64 becomes 50). An append keeps what is stored and
adds to it:
  1. URLs: the stored list's URLs whose page gave no stored passage, then
     Wikipedia's article and external links (the collector's own lookup and
     filters, collect._wikipedia_seed_sync) that are not stored pages yet. No
     search. Each URL must pass the collector's skip rules and be a page a
     story could credit (creditable_source).
  2. Pages and passages exactly as in a full repair (steps 2 and 3 below).
  3. Append (appended_rag_context): every stored passage, its section title
     and its order are kept as they are; the new passages follow, under their
     pages' titles; one whose text hash (the pipeline's compute_text_hash)
     matches a passage already held is skipped.
  4. Write, review identity: as steps 4 and 5 below. The identity rules tie a
     review to the whole set of judged pages (legend_identity.sources_hash),
     so a review of the new pages alone would not count as current: the
     identity stage judges the legend again, in one call.

FULL RE-COLLECTION (--recollect). For each legend named, with the pipeline's
own code:
  1. URLs: the legend's stored list (legendary_scholars/final/<id>.txt),
     then what collect_legendary_scholar_urls.collect_for_scholar finds now
     (Wikipedia's API: the article and its external links; the Brave Search
     API, searched even though Wikipedia gave enough seeds). Each URL must
     pass the collector's skip rules and be a page a story could credit
     (creditable_source); at most --max-urls in all.
  2. Pages: unified_pipeline.UnifiedPipeline fetches them through the page
     worker (one honest name, robots.txt on every page and redirect, a
     refusal final, sites whose terms forbid automated access never asked)
     and cuts each page into passages with the per-source chunker, under a
     legend run's filters (apply_legend_filter_defaults). No cloud model: LLM
     cleaning, the page role filter and the legacy model chunker are refused
     before a page is fetched (refuse_model_paths); nothing goes to Pinecone.
  3. Candidate: LocalChunkedMongoSync.build_document, without its model
     summaries, turns the fetched pages into the rag_context a legend sync
     stores, and evaluate_profile scores it.
  4. Write, only when the candidate passes the gate (ready, 60+) and has four
     creditable passages from two pages: the legend's rag_context and its
     score where the gate reads it (profile_quality_refresh.store_quality:
     daily_story_profile and daily_story_profile_quality). Nothing else on
     the legend changes: a whole-document sync would also replace its bios
     (about, with the grade 3-5 bio), name, display and admin fields. The
     previous rag_context, label and identity review are kept in
     legend_profile_repair_backups. A candidate that falls short is not
     written; the stored profile stays as it was.
  5. Identity review: new sources make the legend's identity review stale,
     which keeps it out of stories, so legend_identity_review judges it again
     (one model call; a legend whose review still applies costs nothing).

Resumable and idempotent: each legend's collected URLs and fetched pages are
kept under --work-dir/<id>/ with a state file (one for each mode), and a
later run reuses them (no new lookup, no new fetch) unless --fresh. A legend
whose profile is already ready is left alone (only a stale label is
corrected), so a second run writes nothing. A write is refused if the
legend's rag_context changed since it was read.

It needs the full requirements.txt (pandas, and Playwright for pages that
need a browser), as a legend run does; the editorial job image has neither.

The database address is read from LEGEND_MONGO_URI only, and no .env is
loaded (refuse_env_files). --dry-run writes nothing to the database (it is
wrapped read-only) and calls no model: it prints each legend's state, the
URLs it would fetch, what would be written, and the cost, with the
arithmetic. How many passages a page gives is known only once it is fetched,
so --dry-run --fetch also collects and fetches the pages (through the same
page worker, under the same rules) into --work-dir and shows each legend's
passage count after appending and whether it would pass; the run then reuses
those pages. A plain --dry-run fetches nothing and saves nothing.

Running it (PowerShell 5.1, from this folder):
  $env:LEGEND_MONGO_URI = (gcloud secrets versions access latest --secret=MONGODB_URI)
  $env:STORY_LLM_MODEL = 'gemini-2.5-flash'
  python legend_profile_repair.py --legend-ids-file pilot_ids.txt --dry-run --fetch
  $env:GCP_PROJECT_ID = 'ngo-application-services'
  python legend_profile_repair.py --legend-ids-file pilot_ids.txt
  Remove-Item Env:LEGEND_MONGO_URI
A full re-collection also wants $env:BRAVE_SEARCH_API and --recollect.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def refuse_env_files() -> None:
    """Make every load_dotenv in this process a no-op.

    unified_pipeline.py and api/utils/llm_text_cleaner.py load .env as they
    are imported. In a working copy that file points at production, and its
    settings can switch on page filters that call a model
    (PROFILE_ROLE_FILTER_ENABLED, USE_LLM_CLEANING). This run is given its
    database and settings explicitly, so the file is refused before any of
    those modules is imported."""
    import dotenv

    def no_env_file(*_args: Any, **_kwargs: Any) -> bool:
        return False

    dotenv.load_dotenv = no_env_file


refuse_env_files()

# Imported after refuse_env_files, on purpose.
import collect_legendary_scholar_urls as collect  # noqa: E402
import legend_identity  # noqa: E402
import legend_identity_review as identity_stage  # noqa: E402
import legend_profile_diagnosis as diagnosis  # noqa: E402
import legend_selection_report as selection  # noqa: E402
import model_costs  # noqa: E402
import profile_quality_refresh as quality  # noqa: E402
from api.utils.source_guardrails import compute_text_hash  # noqa: E402
from daily_story_worker import DailyStoryWorker, creditable_source, safe_text, utc_now_iso  # noqa: E402
from legend_evergreen import parse_ids  # noqa: E402

URI_ENV = "LEGEND_MONGO_URI"
REPAIR_POLICY_VERSION = "legend-profile-repair-v1"
APPEND_POLICY_VERSION = "legend-profile-append-v1"
RUNS_COLLECTION = "legend_profile_repair_runs"
BACKUPS_COLLECTION = "legend_profile_repair_backups"
DEFAULT_WORK_DIR = Path("output") / "profile_repair"
APPEND, RECOLLECT = "append", "recollect"
STATE_FILES = {APPEND: "append-state.json", RECOLLECT: "state.json"}
POLICIES = {APPEND: APPEND_POLICY_VERSION, RECOLLECT: REPAIR_POLICY_VERSION}
PAGES_DIRS = {APPEND: "append-pages", RECOLLECT: "pages"}

# What the Wikipedia seed costs (collect._wikipedia_seed_sync): title searches
# until a result's name matches (one to four), then the page's address and
# its external links.
WIKIPEDIA_REQUESTS = (3, 6)
# What it can give: the article and at most 8 of its external links
# (collect._filter_wikipedia_extlinks keeps 8).
WIKIPEDIA_SEED_URLS = 1 + 8
# The page worker's reach from one seed (playwright_subprocess_worker.run):
# its CV documents, and up to 4 linked personal sites, each read two levels
# deep with up to 5 subpages a level (1 + 5 + 25 pages).
LINKED_SITES_PER_SEED = 4
PAGES_PER_LINKED_SITE = 1 + 5 + 25
# Each page the judge sees adds its id, link and passage count above its
# excerpts (identity_stage.judge_prompt); a generous allowance per page.
JUDGE_PAGE_LINE_CHARS = 200
# The visible answer per judged page (page id, verdict, person, reason), and
# for the legend as a whole (intended person, name check).
JUDGE_ANSWER_TOKENS_PER_PAGE = 60
JUDGE_ANSWER_TOKENS_BASE = 150
# _call_structured_json's later budgets for a reply cut off at its limit.
JUDGE_RETRY_CAPS = (16384, 32768)


# ── URLs ─────────────────────────────────────────────────────────────────────

def skip_reason(url: str) -> Optional[str]:
    """Why a URL is not fetched by a repair, or None when it may be. The
    page worker refuses what robots.txt or a site's terms forbid as it
    fetches; these are the rules that can be checked without asking."""
    if collect._legendary_should_skip(url):
        return "skipped by the collector's rules (not a web page, an excluded or terms-forbidden site, or a noise domain)"
    if not creditable_source(url):
        return "a story could not credit it (creditable_source)"
    return None


def screen_urls(urls: List[str]) -> Tuple[List[str], List[Dict[str, str]]]:
    kept: List[str] = []
    dropped: List[Dict[str, str]] = []
    for url in urls:
        reason = skip_reason(url)
        if reason:
            dropped.append({"url": url, "reason": reason})
        else:
            kept.append(url)
    return kept, dropped


def _url_key(url: str) -> str:
    """One page, whatever its scheme or tracking parameters."""
    return collect._normalize_url(url).split("://", 1)[-1]


def merge_urls(stored: List[str], found: List[str], max_urls: int) -> List[str]:
    """The stored list first (those pages were the legend's sources), then
    what was found now, without repeats, at most max_urls."""
    merged: List[str] = []
    seen = set()
    for url in [*stored, *found]:
        key = _url_key(url)
        if key in seen or skip_reason(url):
            continue
        seen.add(key)
        merged.append(url)
        if len(merged) >= max_urls:
            break
    return merged


def urls_hash(urls: List[str]) -> str:
    return hashlib.sha256("\n".join(urls).encode("utf-8")).hexdigest()


# ── Saved work ───────────────────────────────────────────────────────────────

def load_state(path: Path, *, profile_id: str, name: str, policy: str = REPAIR_POLICY_VERSION) -> Dict[str, Any]:
    """A legend's saved work, if it was done for this legend under this
    name and policy (a name change means a different search)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("policy") != policy:
        return {}
    if data.get("profile_id") != profile_id or data.get("name") != name:
        return {}
    return data


def save_state(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)  # a crash mid-write leaves the last good state


# ── The plan ─────────────────────────────────────────────────────────────────

def plan(worker: DailyStoryWorker, legend_ids: List[str], *, url_lists_dir: Path, work_dir: Path, min_quality: int, fresh: bool, mode: str = RECOLLECT) -> List[Dict[str, Any]]:
    """Each legend's state: not_found; ready (nothing to do); relabel (its
    passages already score ready, only the stored label is stale); repair."""
    rows: List[Dict[str, Any]] = []
    for requested in legend_ids:
        row: Dict[str, Any] = {"profile_id": requested, "doc": None, "state": "not_found", "name": "", "mode": mode}
        rows.append(row)
        doc = diagnosis.find_legend(worker.scholars_collection, requested)
        if not doc:
            continue
        profile_id = worker._profile_id(doc)
        now = quality.evaluate_profile(worker, doc)
        label = diagnosis.stored_label(doc, worker.profile_quality_collection.find_one({"profile_id": profile_id}))
        anchor = legend_identity.anchor_for(doc)
        row.update(profile_id=profile_id, doc=doc, now=now, label=label, name=anchor["name"])
        if now["status"] == "ready" and now["score"] >= min_quality:
            row["state"] = "ready" if (label["status"], label["score"]) == (now["status"], now["score"]) else "relabel"
            continue
        row["state"] = "repair"
        roster = diagnosis.read_roster(url_lists_dir / f"{profile_id}.meta.json")
        field = anchor.get("field") or safe_text(((doc.get("about") or {}).get("field_of_study")))
        row["scholar"] = collect.ScholarRow(
            row_number=int(roster.get("row_number") or 0),
            scholar_id=profile_id,
            full_name=anchor["name"],
            field=field,
            slug=profile_id,
        )
        row["profile_url"] = safe_text(roster.get("profile_url"))
        row["stored_urls"], row["dropped_urls"] = screen_urls(diagnosis.read_url_list(url_lists_dir / f"{profile_id}.txt"))
        if mode == APPEND:
            # The pages that already gave stored passages, keyed as the
            # diagnosis and the identity rules key a page (page_key).
            row["stored_keys"] = {page["key"] for page in diagnosis.page_table(doc)[0]}
            row["append_urls"] = [u for u in row["stored_urls"] if legend_identity.page_key(u) not in row["stored_keys"]]
            row["queries"] = []
        else:
            row["queries"] = collect._build_queries(row["scholar"])
        row["state_path"] = work_dir / profile_id / STATE_FILES[mode]
        row["saved"] = {} if fresh else load_state(row["state_path"], profile_id=profile_id, name=anchor["name"], policy=POLICIES[mode])
    return rows


def search_key_set() -> bool:
    # brave_search reads the same variable and searches nothing without it.
    return bool(safe_text(os.getenv("BRAVE_SEARCH_API")))


def saved_collection(row: Dict[str, Any]) -> Dict[str, Any]:
    """The saved URL collection a run would reuse. A re-collection made
    without a search key holds only the stored list and Wikipedia's links, so
    once a key is set it is collected again (an append never searches)."""
    collected = (row.get("saved") or {}).get("collected") or {}
    if row.get("mode") == RECOLLECT and collected and not collected.get("search_ran") and search_key_set():
        return {}
    return collected


def saved_fetch(row: Dict[str, Any]) -> Dict[str, Any]:
    """The saved fetch a run would reuse: one that finished, of the URLs
    saved as collected."""
    collected, fetched = saved_collection(row), (row.get("saved") or {}).get("fetched") or {}
    if collected and fetched.get("status") == "success" and fetched.get("urls_hash") == collected.get("urls_hash"):
        return fetched
    return {}


# ── The steps ────────────────────────────────────────────────────────────────

def collect_urls(row: Dict[str, Any], *, max_urls: int, sleep_s: float, per_query: int) -> Dict[str, Any]:
    result = asyncio.run(collect.collect_for_scholar(
        row["scholar"], max_urls=max_urls, sleep_s=sleep_s, per_query=per_query, always_search=True,
    ))
    found, found_dropped = screen_urls(result["kept_urls"])
    urls = merge_urls(row["stored_urls"], found, max_urls)
    return {
        "at": utc_now_iso(),
        "search_ran": search_key_set(),
        "queries": len(result["queries"]),
        "raw_results": result["raw_results"],
        "rejected_count": result["rejected_count"],
        "found_urls": found,
        "found_dropped": found_dropped,
        "urls": urls,
        "urls_hash": urls_hash(urls),
    }


def collect_append_urls(row: Dict[str, Any], *, max_urls: int) -> Dict[str, Any]:
    """The pages an append fetches: the stored list's URLs whose page gave no
    stored passage, then Wikipedia's article and external links (looked up
    and filtered by the collector) that are not stored pages yet. No search."""
    wikipedia, wikipedia_dropped = screen_urls(collect._wikipedia_seed_sync(row["scholar"]))
    urls: List[str] = []
    seen = set()
    for url in [*row["append_urls"], *wikipedia]:
        if legend_identity.page_key(url) in row["stored_keys"] or _url_key(url) in seen or skip_reason(url):
            continue
        seen.add(_url_key(url))
        urls.append(url)
        if len(urls) >= max_urls:
            break
    from_list = {_url_key(u) for u in row["append_urls"]}
    return {
        "at": utc_now_iso(),
        "from_stored_list": [u for u in urls if _url_key(u) in from_list],
        "from_wikipedia": [u for u in urls if _url_key(u) not in from_list],
        "wikipedia_dropped": wikipedia_dropped,
        "urls": urls,
        "urls_hash": urls_hash(urls),
    }


def refuse_model_paths(pipeline: Any) -> None:
    """Refuse every way the unified pipeline can call a cloud model, before a
    page is fetched: the LLM cleaner and the page role filter must not exist,
    and the legacy chunker (which asks a model to section a page's text, and
    is used only when a page gave no per-source passages) raises instead."""
    if getattr(pipeline, "llm_cleaner", None) is not None:
        raise RuntimeError("LLM cleaning is on (USE_LLM_CLEANING?); a profile repair calls no model")
    # The cleaning service is made once a process; one made earlier with a
    # model would clean every page with it.
    if getattr(getattr(pipeline.cleaning_service, "pipeline", None), "use_llm_cleaning", False):
        raise RuntimeError("the cleaning service cleans with a model; a profile repair calls no model")
    if getattr(pipeline, "profile_role_filter_client", None) is not None:
        raise RuntimeError("the page role filter has a model client (PROFILE_ROLE_FILTER_ENABLED?); a profile repair calls no model")
    chunker = getattr(pipeline, "chunking_pipeline", None)
    if chunker is None:
        raise RuntimeError("the per-source chunker is missing: chunks.json would not be written")

    def legacy_chunker_refused(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("the legacy chunker asks a model to section the text; a profile repair calls no model")

    chunker.process_profile = legacy_chunker_refused


def make_pipeline(out_dir: Path, scraper: Any = None) -> Any:
    import unified_pipeline

    unified_pipeline.apply_legend_filter_defaults()
    pipeline = unified_pipeline.UnifiedPipeline(
        output_dir=str(out_dir),
        chunking_output_dir=str(out_dir / "chunked_profiles"),
        # chunks.json is written only with this on; from per-source passages
        # it is built without a model (build_chunks_from_source_chunks).
        use_llm_chunking=True,
        use_llm_cleaning=False,
        incremental_sync_enabled=False,
        source_registry_path=str(out_dir / "source_registry.jsonl"),
    )
    refuse_model_paths(pipeline)
    if scraper is not None:
        pipeline.scraper = scraper
    return pipeline


def fetch_pages(row: Dict[str, Any], collected: Dict[str, Any], legend_dir: Path, scraper: Any = None) -> Dict[str, Any]:
    urls = collected["urls"]
    out_dir = legend_dir / f"{PAGES_DIRS[row.get('mode') or RECOLLECT]}-{collected['urls_hash'][:12]}"
    pipeline = make_pipeline(out_dir, scraper)
    profile_url = row["profile_url"] if row["profile_url"] in urls else urls[0]
    summary = asyncio.run(pipeline.run_from_urls(
        urls=urls, profile_name=row["name"], profile_url=profile_url, profile_id=row["profile_id"],
    ))
    result = (summary.get("results") or [{}])[0]
    return {
        "at": utc_now_iso(),
        "urls_hash": collected["urls_hash"],
        "status": result.get("status"),
        "error": result.get("error"),
        "chunks_path": result.get("chunks_path"),
        "profiles_root": str(out_dir / "profiles"),
        "seed_urls_total": result.get("seed_urls_total", len(urls)),
        "seed_urls_successful": result.get("seed_urls_successful", 0),
        "seed_urls_failed": result.get("seed_urls_failed", 0),
    }


def fetched_rag_context(worker: DailyStoryWorker, fetched: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The rag_context a legend sync would store from the fetched pages alone
    (LocalChunkedMongoSync.build_document, no model). Reads local files only."""
    if fetched.get("status") != "success" or not fetched.get("chunks_path"):
        return None
    from sync_local_chunked_to_mongodb import LocalChunkedMongoSync

    builder = LocalChunkedMongoSync(use_llm=False, database=worker.db)
    built = builder.build_document(Path(fetched["chunks_path"]), Path(fetched["profiles_root"]))
    return built[1]["rag_context"] if built else None


def _scored(worker: DailyStoryWorker, row: Dict[str, Any], rag_context: Dict[str, Any]) -> Dict[str, Any]:
    """How the gate would score the legend holding rag_context."""
    doc = {**row["doc"], "rag_context": rag_context}
    pages, totals = diagnosis.page_table(doc)
    return {
        "rag_context": rag_context,
        "doc": doc,
        "quality": quality.evaluate_profile(worker, doc),
        "pages": len(pages),
        "creditable_passages": totals.get("creditable_passages", 0),
        "creditable_pages": totals.get("creditable_pages", 0),
    }


def build_candidate(worker: DailyStoryWorker, row: Dict[str, Any], fetched: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A full re-collection's candidate: the fetched pages' rag_context in
    place of the stored one."""
    rag_context = fetched_rag_context(worker, fetched)
    return _scored(worker, row, rag_context) if rag_context is not None else None


def _chunks(rag_context: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    sections = rag_context.get("section_chunks")
    return sections if isinstance(sections, dict) else {}


def appended_rag_context(stored: Dict[str, Any], new: Dict[str, Any], *, record: Dict[str, Any]) -> Tuple[Dict[str, Any], int, int]:
    """(the stored rag_context with the new pages' passages added after its
    own, passages added, duplicates skipped).

    Everything stored is copied, not rebuilt: each passage, its section title
    and its place stay as they were. A new passage goes at the end of the
    section its page is titled as (a new section, or after the passages of
    one already so titled), numbered after the last stored one. One whose
    text hash (compute_text_hash, as the chunker dedupes) matches a passage
    already held, stored or new, is skipped. chunk_count becomes the number
    of passages held, which is what the gate counts."""
    merged = copy.deepcopy(stored or {})
    if not isinstance(merged.get("section_chunks"), dict):
        merged["section_chunks"] = {}
    sections = merged["section_chunks"]
    held = [c for chunks in sections.values() if isinstance(chunks, list) for c in chunks if isinstance(c, dict)]
    seen = {compute_text_hash(legend_identity.chunk_text(c)) for c in held if legend_identity.chunk_text(c)}
    ids = {safe_text(c.get("chunk_id")) for c in held}
    orders = [c.get("order") for c in held if isinstance(c.get("order"), int)]
    next_order = max(orders) + 1 if orders else len(held)
    added_texts: Dict[str, List[str]] = {}
    new_ids: Dict[str, str] = {}
    added = duplicates = 0
    for section, chunks in _chunks(new).items():
        for chunk in chunks if isinstance(chunks, list) else []:
            text = legend_identity.chunk_text(chunk) if isinstance(chunk, dict) else ""
            digest = compute_text_hash(text)
            if not text or digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            item = copy.deepcopy(chunk)
            chunk_id = safe_text(item.get("chunk_id")) or f"appended:{next_order}"
            if chunk_id in ids:
                chunk_id = f"{chunk_id}:appended:{next_order}"
            new_ids[safe_text(chunk.get("chunk_id"))] = chunk_id
            ids.add(chunk_id)
            item["chunk_id"], item["order"] = chunk_id, next_order
            next_order += 1
            target = sections.get(section)
            if not isinstance(target, list):
                target = sections[section] = []
            target.append(item)
            added_texts.setdefault(section, []).append(text)
            added += 1

    merged["chunk_count"] = sum(len(chunks) for chunks in sections.values() if isinstance(chunks, list))
    if isinstance(merged.get("sections_available"), list):
        merged["sections_available"] += [s for s in added_texts if s not in merged["sections_available"]]
    if isinstance(merged.get("section_text"), dict):
        for section, texts in added_texts.items():
            before = safe_text(merged["section_text"].get(section))
            merged["section_text"][section] = "\n\n".join(([before] if before else []) + texts)
    if isinstance(merged.get("source_catalog"), list):
        known = {safe_text(s.get("source_url")) for s in merged["source_catalog"] if isinstance(s, dict)}
        merged["source_catalog"] += [s for s in new.get("source_catalog") or [] if isinstance(s, dict) and safe_text(s.get("source_url")) not in known]
    if isinstance(merged.get("chunk_source_map"), dict):
        for old_id, refs in (new.get("chunk_source_map") or {}).items():
            if old_id in new_ids:
                merged["chunk_source_map"][new_ids[old_id]] = refs
    merged["appended"] = list(merged.get("appended") or []) + [{**record, "passages": added, "duplicates_skipped": duplicates}]
    return merged, added, duplicates


def build_append_candidate(worker: DailyStoryWorker, row: Dict[str, Any], fetched: Dict[str, Any], collected: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """An append's candidate: the stored rag_context with the fetched pages'
    passages added after its own."""
    new = fetched_rag_context(worker, fetched)
    if new is None:
        return None
    stored = row["doc"].get("rag_context") or {}
    merged, added, duplicates = appended_rag_context(
        stored, new, record={"at": fetched["at"], "policy": APPEND_POLICY_VERSION, "urls": collected["urls"]},
    )
    candidate = _scored(worker, row, merged)
    candidate.update(added=added, duplicates=duplicates, before=diagnosis.counted_passages(row["doc"]))
    return candidate


def candidate_for(worker: DailyStoryWorker, row: Dict[str, Any], fetched: Dict[str, Any], collected: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if row["mode"] == APPEND:
        return build_append_candidate(worker, row, fetched, collected)
    return build_candidate(worker, row, fetched)


def shortfalls(candidate: Dict[str, Any], min_quality: int) -> List[str]:
    """Why a candidate would not make the legend writable (empty: it would)."""
    scored = candidate["quality"]
    out: List[str] = []
    if scored["status"] != "ready":
        out.append(f"it would score {scored['status']} {scored['score']} with {scored['signals']['chunk_count']} passages")
    elif scored["score"] < min_quality:
        out.append(f"it would score {scored['score']}, under the gate's {min_quality}")
    if candidate["creditable_passages"] < selection.MIN_PASSAGES:
        out.append(f"{candidate['creditable_passages']} creditable passages (the writer needs {selection.MIN_PASSAGES})")
    if candidate["creditable_pages"] < selection.MIN_PAGES:
        out.append(f"{candidate['creditable_pages']} creditable pages (the writer needs {selection.MIN_PAGES})")
    return out


def write_candidate(worker: DailyStoryWorker, row: Dict[str, Any], candidate: Dict[str, Any], run_id: str) -> None:
    doc = row["doc"]
    backup = worker.db[BACKUPS_COLLECTION].insert_one({
        "profile_id": row["profile_id"],
        "run_id": run_id,
        "at": utc_now_iso(),
        "policy": POLICIES[row.get("mode") or RECOLLECT],
        "previous": {k: doc.get(k) for k in ("rag_context", "daily_story_profile", "identity_review")},
    })
    # Only if rag_context is still what was read: an append built on an older
    # copy would otherwise drop passages written in between.
    written = worker.scholars_collection.update_one(
        {"_id": doc["_id"], "rag_context": doc.get("rag_context")},
        {"$set": {"rag_context": candidate["rag_context"]}},
    )
    if written.matched_count != 1:
        worker.db[BACKUPS_COLLECTION].delete_one({"_id": backup.inserted_id})
        raise RuntimeError("its rag_context changed since this run read it; nothing was written, run again")
    quality.store_quality(worker.profile_quality_collection, worker.scholars_collection, candidate["quality"], run_id=run_id, update_scholar_docs=True)


def review_identity(worker: DailyStoryWorker, profile_id: str) -> Dict[str, Any]:
    """Judge the legend's sources again if its review no longer applies (the
    identity stage reuses a review whose input is unchanged)."""
    doc = diagnosis.find_legend(worker.scholars_collection, profile_id)
    eligible, reason = legend_identity.story_eligibility(doc)
    if reason not in ("not_reviewed", "stale"):
        return {"outcome": "current", "eligible": eligible, "reason": reason}
    result = identity_stage.run_identity_review(
        scholar_id=profile_id, max_scholars=1, force=False, dry_run=False, concurrency=1, worker=worker,
    )
    eligible, reason = legend_identity.story_eligibility(diagnosis.find_legend(worker.scholars_collection, profile_id))
    return {"outcome": result["status"], "eligible": eligible, "reason": reason, "errors": result.get("errors") or []}


def writable_now(worker: DailyStoryWorker, profile_id: str, min_quality: int) -> Dict[str, Any]:
    """The story writer's gates on the stored legend, by their own code."""
    doc = diagnosis.find_legend(worker.scholars_collection, profile_id)
    gate_ok, status, score = worker._profile_quality_state(doc)
    blocked = selection.blockers(doc, selection.source_signals(doc), {"status": status, "score": score}, min_quality)
    return {"quality_gate": gate_ok, "status": status, "score": score, "blocked_by": blocked, "writable": not blocked}


def prepare(row: Dict[str, Any], args: argparse.Namespace, scraper: Any = None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """(state, collected, fetched) for a legend to repair: its saved work
    where that still applies, else the URLs collected and the pages fetched
    now, each saved as soon as it is done. Writes to --work-dir only, never
    to the database, so --dry-run --fetch can do it too."""
    state = dict(row["saved"]) or {"policy": POLICIES[row["mode"]], "profile_id": row["profile_id"], "name": row["name"]}
    collected = saved_collection(row)
    if not collected:
        if row["mode"] == APPEND:
            collected = collect_append_urls(row, max_urls=args.max_urls)
        else:
            collected = collect_urls(row, max_urls=args.max_urls, sleep_s=args.sleep_seconds, per_query=args.per_query_results)
        state["collected"], state["fetched"] = collected, {}
        save_state(row["state_path"], state)
    fetched = state.get("fetched") or {}
    # A fetch that failed (the network down, say) is tried again; one that
    # finished is kept, whatever it gave.
    if collected["urls"] and (fetched.get("urls_hash") != collected["urls_hash"] or fetched.get("status") != "success"):
        fetched = fetch_pages(row, collected, row["state_path"].parent, scraper)
        state["fetched"] = fetched
        save_state(row["state_path"], state)
    row["saved"] = state
    return state, collected, fetched


def repair_one(worker: DailyStoryWorker, row: Dict[str, Any], args: argparse.Namespace, run_id: str, scraper: Any = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"profile_id": row["profile_id"], "name": row["name"], "state": row["state"]}
    if row["state"] == "not_found":
        out["outcome"] = "not_found"
        return out
    if row["state"] == "relabel":
        quality.store_quality(worker.profile_quality_collection, worker.scholars_collection, row["now"], run_id=run_id, update_scholar_docs=True)
        out["outcome"] = "relabeled"
    elif row["state"] == "ready":
        out["outcome"] = "already_ready"
    else:
        out["mode"] = row["mode"]
        state, collected, fetched = prepare(row, args, scraper)
        out["urls"] = len(collected["urls"])
        if row["mode"] == RECOLLECT:
            out["search_ran"] = collected["search_ran"]
        else:
            out["fetch"] = collected["urls"]
        if not collected["urls"]:
            out["outcome"] = "no_urls"
            return out
        out["seeds"] = {k: fetched.get(k) for k in ("seed_urls_total", "seed_urls_successful", "seed_urls_failed")}
        candidate = candidate_for(worker, row, fetched, collected)
        if candidate is None:
            out.update(outcome="nothing_fetched", error=fetched.get("error"))
            return out
        out["candidate"] = {
            "status": candidate["quality"]["status"],
            "score": candidate["quality"]["score"],
            "passages": candidate["quality"]["signals"]["chunk_count"],
            "creditable_passages": candidate["creditable_passages"],
            "pages": candidate["pages"],
        }
        if row["mode"] == APPEND:
            out["candidate"].update(before=candidate["before"], added=candidate["added"], duplicates_skipped=candidate["duplicates"])
        missing = shortfalls(candidate, args.min_quality)
        if missing:
            out.update(outcome="not_written", still_short=missing)
            return out
        write_candidate(worker, row, candidate, run_id)
        state["written"] = {"run_id": run_id, "at": utc_now_iso()}
        save_state(row["state_path"], state)
        out["outcome"] = "written"
    out["identity"] = review_identity(worker, row["profile_id"])
    out["now"] = writable_now(worker, row["profile_id"], args.min_quality)
    return out


# ── The dry run ──────────────────────────────────────────────────────────────

def judge_input_tokens(row: Dict[str, Any], candidate: Optional[Dict[str, Any]]) -> Tuple[int, int, str]:
    """(input tokens, pages judged, how known) for one identity review: the
    real prompt when the pages are known (a saved fetch, or the legend as it
    is), else the most the judge can be shown."""
    doc = candidate["doc"] if candidate else row["doc"]
    if candidate or row["state"] != "repair":
        pages, _ = legend_identity.source_pages(doc)
        page_ids = {f"P{n + 1}": page["key"] for n, page in enumerate(pages)}
        prompt = identity_stage.judge_prompt(name=legend_identity.stored_name(doc), anchor=legend_identity.anchor_for(doc), pages=pages, page_ids=page_ids)
        return model_costs.tokens_for(len(prompt)), len(pages), "its prompt"
    fixed = identity_stage.judge_prompt(name=legend_identity.stored_name(doc), anchor=legend_identity.anchor_for(doc), pages=[], page_ids={})
    chars = len(fixed) + legend_identity.JUDGE_CHAR_BUDGET + legend_identity.MAX_PAGES * JUDGE_PAGE_LINE_CHARS
    return model_costs.tokens_for(chars), legend_identity.MAX_PAGES, "the most the judge is shown, as its pages are not fetched yet"


def _fetch_list(urls: List[str], indent: str = "        ") -> List[str]:
    return [f"{indent}- {url}" for url in urls]


def _recollect_lines(worker: DailyStoryWorker, row: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[str], Optional[Dict[str, Any]], int, bool]:
    """(lines, candidate if the pages are fetched, seed pages at most, needs a search)."""
    lines: List[str] = []
    dropped = row["dropped_urls"]
    lines.append(
        f"      URLs: {len(row['stored_urls']) + len(dropped)} in its stored list"
        + (f" ({len(dropped)} left out: " + "; ".join(f"{d['url']} - {d['reason']}" for d in dropped) + ")" if dropped else "")
        + f", then Wikipedia's article and links and {len(row['queries'])} searches; at most {args.max_urls} pages in all"
    )
    collected = saved_collection(row)
    fetched = saved_fetch(row)
    if fetched:
        candidate = build_candidate(worker, row, fetched)
        if candidate:
            short = shortfalls(candidate, args.min_quality)
            scored = candidate["quality"]
            lines.append(
                f"      saved work: pages fetched {fetched['at']} ({len(collected['urls'])} URLs): would be {scored['status']} {scored['score']}"
                f" with {scored['signals']['chunk_count']} passages, {candidate['creditable_passages']} creditable from {candidate['creditable_pages']} pages: "
                + ("would write" if not short else "would not write: " + "; ".join(short))
            )
        else:
            lines.append(f"      saved work: a fetch {fetched['at']} that gave no passages ({fetched.get('error') or 'no chunks'}): would not write")
        return lines, candidate, 0, False
    if collected:
        lines.append(f"      saved work: {len(collected['urls'])} URLs collected {collected['at']}: would fetch them (no new search)")
        return lines, None, len(collected["urls"]), False
    note = " (a collection made without a search key is redone)" if row["saved"].get("collected") else ""
    lines.append(f"      saved work: none{note}")
    return lines, None, args.max_urls, True


def _append_lines(worker: DailyStoryWorker, row: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[str], Optional[Dict[str, Any]], int, bool]:
    """(lines, candidate if the pages are fetched, seed pages at most, needs a Wikipedia lookup)."""
    lines: List[str] = []
    listed = len(row["stored_urls"]) + len(row["dropped_urls"])
    gave = len(row["stored_urls"]) - len(row["append_urls"])
    lines.append(f"      stored list: {listed} URLs; {gave} gave stored passages; {len(row['dropped_urls'])} left out by the rules"
                 + "".join(f"\n        x {d['url']} - {d['reason']}" for d in row["dropped_urls"]))
    count = diagnosis.counted_passages(row["doc"])
    points = row["now"]["points"]
    least = diagnosis.least_passages(points["name"] + points["coverage"] + points["avg_len"], args.min_quality)
    collected = saved_collection(row)
    fetched = saved_fetch(row)
    if fetched:
        lines.append(f"      fetched {fetched['at']}: {len(collected['urls'])} URLs ({len(collected['from_stored_list'])} from the stored list, {len(collected['from_wikipedia'])} from Wikipedia):")
        lines += _fetch_list(collected["urls"])
        candidate = build_append_candidate(worker, row, fetched, collected)
        if candidate is None:
            lines.append(f"      after appending: {count} passages, as now (the pages gave none: {fetched.get('error') or 'no chunks'}): would not write")
            return lines, None, 0, False
        scored = candidate["quality"]
        short = shortfalls(candidate, args.min_quality)
        lines.append(
            f"      after appending: {candidate['before']} + {candidate['added']} new"
            + (f" ({candidate['duplicates']} duplicates skipped)" if candidate["duplicates"] else "")
            + f" = {scored['signals']['chunk_count']} passages, {scored['status']} {scored['score']}"
            f", {candidate['creditable_passages']} creditable from {candidate['creditable_pages']} pages: "
            + ("WOULD PASS and be written" if not short else "would not pass, not written: " + "; ".join(short))
        )
        return lines, candidate, 0, False
    if collected:
        lines.append(f"      would fetch these {len(collected['urls'])} (collected {collected['at']}; {len(collected['from_wikipedia'])} from Wikipedia):")
        lines += _fetch_list(collected["urls"])
        seeds, lookup = len(collected["urls"]), False
    else:
        lines.append(f"      would fetch the {len(row['append_urls'])} that gave none:")
        lines += _fetch_list(row["append_urls"])
        lines.append("        + Wikipedia's article and links that are not stored pages yet (looked up when fetching)")
        seeds, lookup = len(row["append_urls"]) + WIKIPEDIA_SEED_URLS, True
    if least is None:
        need = "no number of new passages makes it ready with its titles and lengths as now"
    else:
        need = f"it passes only with at least {max(0, least - count)} new passages ({least} in all, with titles and lengths as now)"
    lines.append(f"      after appending: {count} + what these pages give; {need}. Exact count: --dry-run --fetch")
    return lines, None, seeds, lookup


def dry_run_report(worker: DailyStoryWorker, rows: List[Dict[str, Any]], args: argparse.Namespace, model: str) -> str:
    overrides = model_costs.parse_price_overrides(args.price)
    fetching = bool(getattr(args, "fetch", False))
    mode = rows[0]["mode"] if rows else APPEND
    lines = [
        f"PROFILE REPAIR PLAN  {len(rows)} legends, {'appending to their stored passages' if mode == APPEND else 'full re-collection (--recollect)'}"
        + ("  (dry run: nothing written to the database, no model asked; pages fetched into --work-dir by --fetch)" if fetching
           else "  (dry run: nothing written, fetched or asked of a model)")
    ]
    review_steps: List[model_costs.Step] = []
    retry_steps: List[model_costs.Step] = []
    lookups = 0
    seeds_cap = 0
    searching: List[Dict[str, Any]] = []
    for row in rows:
        if row["state"] == "not_found":
            lines.append(f"  {row['profile_id']}: not found")
            continue
        now = row["now"]
        head = f"  {row['profile_id']} ({row['name']}): {row['state']}"
        candidate = None
        if row["state"] == "ready":
            lines.append(f"{head}, {now['status']} {now['score']}: nothing to fetch or write")
        elif row["state"] == "relabel":
            lines.append(f"{head}: stored {row['label']['status']} {row['label']['score']}, its passages score {now['status']} {now['score']}: would correct the label only")
        else:
            reasons = diagnosis.diagnose(row["doc"], min_quality=args.min_quality)["reasons"]
            lines.append(f"{head}, {now['status']} {now['score']}: {'; '.join(reasons) or 'not ready'}")
            if row.get("prepare_error"):
                lines.append(f"      the fetch failed: {row['prepare_error']}")
            more, candidate, seeds, lookup = (_append_lines if row["mode"] == APPEND else _recollect_lines)(worker, row, args)
            lines += more
            seeds_cap += seeds
            lookups += int(lookup)
            if lookup and row["mode"] == RECOLLECT:
                searching.append(row)
        # Who the identity stage would judge: the legends this run makes
        # ready whose review would then be stale or missing.
        eligible, why = legend_identity.story_eligibility(row["doc"])
        if row["state"] == "repair" or why in ("not_reviewed", "stale"):
            if row["state"] == "repair" and candidate is not None and shortfalls(candidate, args.min_quality):
                continue  # fetched, and it would not be written: no review follows
            tokens, pages, known = judge_input_tokens(row, candidate)
            low = JUDGE_ANSWER_TOKENS_BASE + JUDGE_ANSWER_TOKENS_PER_PAGE * pages
            review_steps.append(model_costs.Step(f"Identity review, {row['profile_id']}", model, 1, tokens, low, identity_stage.JUDGE_MAX_TOKENS, note=f"input from {known}"))
            retry_steps.append(model_costs.Step(f"Identity review, {row['profile_id']}, two more tries", model, 1, 2 * tokens, 0, sum(JUDGE_RETRY_CAPS)))

    lines += ["", "WEB (counted; " + ("what is left to fetch after --fetch)" if fetching else "nothing here is fetched in a dry run)")]
    lines.append(
        f"  Wikipedia API: {WIKIPEDIA_REQUESTS[0]} to {WIKIPEDIA_REQUESTS[1]} requests a legend x {lookups} legends to look up"
        f" = {WIKIPEDIA_REQUESTS[0] * lookups} to {WIKIPEDIA_REQUESTS[1] * lookups}"
    )
    if mode == APPEND:
        lines.append("  Brave Search API: none (an append does not search)")
    else:
        queries = sum(len(r["queries"]) for r in searching)
        lines.append(
            f"  Brave Search API: {' + '.join(str(len(r['queries'])) for r in searching) or '0'} = {queries} queries, one request each"
            + ("" if search_key_set() else "  (BRAVE_SEARCH_API is not set: none would run, and only the stored lists and Wikipedia would be fetched)")
        )
    lines.append(
        f"  Seed pages: at most {seeds_cap}; robots.txt once per site."
        f" From a seed the page worker can also read its CV documents and up to {LINKED_SITES_PER_SEED} linked sites of up to"
        f" {PAGES_PER_LINKED_SITE} pages, a ceiling of {seeds_cap} x {LINKED_SITES_PER_SEED} x {PAGES_PER_LINKED_SITE}"
        f" = {seeds_cap * LINKED_SITES_PER_SEED * PAGES_PER_LINKED_SITE:,} more requests plus CV documents"
    )
    lines += ["", "MODELS"]
    lines.append("  Fetching and chunking: none. Pages are cut by the per-source chunker; LLM cleaning, the page role filter and the legacy model chunker are refused before any fetch; no summaries, no embeddings, no Pinecone.")
    if review_steps:
        lines.append(model_costs.report(
            f"  COST ESTIMATE, at most {len(review_steps)} identity reviews (one for each legend this run could leave ready with a stale or missing review; each answered on its first try)",
            review_steps, overrides,
        ))
        lines.append(model_costs.report("  If every review were cut off at its limit and tried twice more, add", retry_steps, overrides))
    else:
        lines.append("  No identity review would be needed.")
    lines += [
        "",
        "WRITES (a repair only when its candidate passes the gate)",
        "  legend_scholars: rag_context" + (" (stored passages and titles kept as they are, new ones after them)" if mode == APPEND else "")
        + " and daily_story_profile; identity_review by the identity stage",
        "  daily_story_profile_quality (each legend's row); legend_identity_review_events and _jobs",
        f"  {BACKUPS_COLLECTION} (each legend's previous rag_context, label and identity review); {RUNS_COLLECTION} (this run)",
        f"  on disk: {Path(args.work_dir).resolve()}\\<id>\\ (URLs, fetched pages, state)",
    ]
    return "\n".join(lines)


# ── Entry point ──────────────────────────────────────────────────────────────

def run(args: argparse.Namespace, *, worker: Optional[DailyStoryWorker] = None, scraper: Any = None) -> int:
    model = safe_text(args.model) or safe_text(os.environ.get("LEGEND_IDENTITY_MODEL")) or safe_text(os.environ.get("STORY_LLM_MODEL"))
    if not model:
        print("Say which model judges identity: --model, LEGEND_IDENTITY_MODEL or STORY_LLM_MODEL (production's job sets gemini-2.5-flash).", file=sys.stderr)
        return 2
    legend_ids = parse_ids(args.legend_ids, args.legend_ids_file)
    if not legend_ids:
        print("Name the legends: --legend-ids-file (or --legend-ids id1,id2).", file=sys.stderr)
        return 2
    own_worker = worker is None
    if own_worker:
        uri = safe_text(os.environ.get(URI_ENV))
        if not uri:
            print(f"Set {URI_ENV} to the database to repair (see this script's header).", file=sys.stderr)
            return 2
        worker = DailyStoryWorker(
            model_name=model,
            use_llm=not args.dry_run,
            trends_enabled=False,
            mongodb_uri=uri,
            load_env_file=False,
            read_only=args.dry_run,
        )
    # The gate checked as production's story job enforces it.
    worker.enforce_profile_quality = True
    worker.profile_quality_min_score = args.min_quality
    work_dir = Path(args.work_dir).resolve()
    mode = RECOLLECT if args.recollect else APPEND
    try:
        rows = plan(worker, legend_ids, url_lists_dir=Path(args.url_lists_dir), work_dir=work_dir, min_quality=args.min_quality, fresh=args.fresh, mode=mode)
        if args.dry_run:
            if args.fetch:
                for row in rows:
                    if row["state"] != "repair":
                        continue
                    try:
                        prepare(row, args, scraper)
                    except Exception as exc:  # one legend's failed fetch is shown, never fatal to the plan
                        row["prepare_error"] = safe_text(exc)[:300]
            print(dry_run_report(worker, rows, args, model))
            return 0
        run_id = f"profile-repair-{uuid.uuid4()}"
        results: List[Dict[str, Any]] = []
        for row in rows:
            try:
                results.append(repair_one(worker, row, args, run_id, scraper))
            except Exception as exc:  # one legend's failure is reported, never fatal to the rest
                results.append({"profile_id": row["profile_id"], "state": row["state"], "outcome": "error", "error": safe_text(exc)[:400]})
        worker.db[RUNS_COLLECTION].insert_one({
            "run_id": run_id,
            "at": utc_now_iso(),
            "policy": POLICIES[mode],
            "mode": mode,
            "model": model,
            "config": {"max_urls": args.max_urls, "min_quality": args.min_quality, "fresh": args.fresh, "work_dir": str(work_dir)},
            "results": results,
        })
        print(json.dumps({"run_id": run_id, "results": results}, indent=2, ensure_ascii=False, default=str))
        return 1 if any(r["outcome"] == "error" for r in results) else 0
    finally:
        if own_worker:
            worker.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add passages from more source pages to named legends so their profiles pass the quality gate.")
    parser.add_argument("--legend-ids-file", default="", help="A file of legend profile ids, one per line or comma-separated")
    parser.add_argument("--legend-ids", default="", help="Comma-separated legend profile ids")
    parser.add_argument("--recollect", action="store_true",
                        help="Full re-collection (stored list, Wikipedia and Brave search) that replaces the stored passages; the default appends to them")
    parser.add_argument("--dry-run", action="store_true", help="Write nothing to the database and ask no model; print the plan and its cost")
    parser.add_argument("--fetch", action="store_true",
                        help="With --dry-run: also collect and fetch the pages into --work-dir (reused by the run) to show the passage counts and whether each would pass")
    parser.add_argument("--max-urls", type=int, default=collect.DEFAULT_MAX_URLS, help="Pages a legend at most, its stored list included (the collector's default)")
    parser.add_argument("--sleep-seconds", type=float, default=collect.DEFAULT_SLEEP_S, help="Pause between searches (the collector's default)")
    parser.add_argument("--per-query-results", type=int, default=collect.DEFAULT_PER_QUERY, help="Results a search (the collector's default)")
    parser.add_argument("--min-quality", type=int, default=quality.READY_MIN_SCORE, help="The score the nightly gate wants (STORY_PROFILE_MIN_QUALITY_SCORE)")
    parser.add_argument("--url-lists-dir", default=str(legend_identity.ROSTER_DIR), help="Where the legends' stored <id>.txt URL lists and <id>.meta.json roster entries are")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR), help="Where each legend's collected URLs, fetched pages and state are kept")
    parser.add_argument("--fresh", action="store_true", help="Search and fetch again, ignoring saved work")
    parser.add_argument("--model", default=None, help="The identity judge (default: LEGEND_IDENTITY_MODEL, else STORY_LLM_MODEL)")
    parser.add_argument("--price", action="append", default=[], help="MODEL=INPUT/OUTPUT USD per 1M tokens, for a model model_costs does not list")
    return parser


def main() -> int:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
        # As run_legend_scholar_pipeline.py: the page worker runs in a thread
        # beside the event loop.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
