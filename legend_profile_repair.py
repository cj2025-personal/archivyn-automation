"""Targeted profile repair: more source pages for named legends, so the
profile-quality gate can call them ready and stories can be written about
them (pilot: child-friendly scientists for grades 3 to 5).

Why it collects. A legend is "ready" only with 20+ stored passages and a
score of 60+ (profile_quality_refresh.evaluate_profile; see
legend_profile_diagnosis.py for each legend's reasons). Scoring again changes
nothing: the stored scores matched a recomputation on 2026-11-01. The
legends named here were collected from Wikipedia and its external links
alone (the first collection stops searching at four seeds), and their pages
were fetched before the lawful-sources rules of 2026-11-08. Only more pages
about the legend, fetched under today's rules, add passages.

What it does for each legend named, with the pipeline's own code:
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
kept under --work-dir/<id>/ with a state file, and a later run reuses them
(no new search, no new fetch) unless --fresh. A legend whose profile is
already ready is left alone (only a stale label is corrected), so a second
run writes nothing.

It needs the full requirements.txt (pandas, and Playwright for pages that
need a browser), as a legend run does; the editorial job image has neither.

The database address is read from LEGEND_MONGO_URI only, and no .env is
loaded (refuse_env_files). --dry-run writes nothing (the database is wrapped
read-only, nothing is saved to disk), fetches nothing and calls no model: it
prints each legend's state, what would be fetched and written, and the cost,
with the arithmetic.

Running it (PowerShell 5.1, from this folder):
  $env:LEGEND_MONGO_URI = (gcloud secrets versions access latest --secret=MONGODB_URI)
  $env:STORY_LLM_MODEL = 'gemini-2.5-flash'
  python legend_profile_repair.py --legend-ids-file pilot_ids.txt --dry-run
  $env:GCP_PROJECT_ID = 'ngo-application-services'
  $env:BRAVE_SEARCH_API = '<the Brave Search API key>'
  python legend_profile_repair.py --legend-ids-file pilot_ids.txt
  Remove-Item Env:LEGEND_MONGO_URI
"""
from __future__ import annotations

import argparse
import asyncio
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
from daily_story_worker import DailyStoryWorker, creditable_source, safe_text, utc_now_iso  # noqa: E402
from legend_evergreen import parse_ids  # noqa: E402

URI_ENV = "LEGEND_MONGO_URI"
REPAIR_POLICY_VERSION = "legend-profile-repair-v1"
RUNS_COLLECTION = "legend_profile_repair_runs"
BACKUPS_COLLECTION = "legend_profile_repair_backups"
DEFAULT_WORK_DIR = Path("output") / "profile_repair"
STATE_FILE = "state.json"

# What the Wikipedia seed costs (collect._wikipedia_seed_sync): title searches
# until a result's name matches (one to four), then the page's address and
# its external links.
WIKIPEDIA_REQUESTS = (3, 6)
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

def load_state(path: Path, *, profile_id: str, name: str) -> Dict[str, Any]:
    """A legend's saved work, if it was done for this legend under this
    name and policy (a name change means a different search)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("policy") != REPAIR_POLICY_VERSION:
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

def plan(worker: DailyStoryWorker, legend_ids: List[str], *, url_lists_dir: Path, work_dir: Path, min_quality: int, fresh: bool) -> List[Dict[str, Any]]:
    """Each legend's state: not_found; ready (nothing to do); relabel (its
    passages already score ready, only the stored label is stale); repair."""
    rows: List[Dict[str, Any]] = []
    for requested in legend_ids:
        row: Dict[str, Any] = {"profile_id": requested, "doc": None, "state": "not_found", "name": ""}
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
        row["queries"] = collect._build_queries(row["scholar"])
        row["state_path"] = work_dir / profile_id / STATE_FILE
        row["saved"] = {} if fresh else load_state(row["state_path"], profile_id=profile_id, name=anchor["name"])
    return rows


def search_key_set() -> bool:
    # brave_search reads the same variable and searches nothing without it.
    return bool(safe_text(os.getenv("BRAVE_SEARCH_API")))


def saved_collection(row: Dict[str, Any]) -> Dict[str, Any]:
    """The saved URL collection a run would reuse. One made without a search
    key holds only the stored list and Wikipedia's links, so once a key is
    set it is collected again."""
    collected = (row.get("saved") or {}).get("collected") or {}
    if collected and not collected.get("search_ran") and search_key_set():
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
    out_dir = legend_dir / f"pages-{collected['urls_hash'][:12]}"
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


def build_candidate(worker: DailyStoryWorker, row: Dict[str, Any], fetched: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The rag_context a legend sync would store from the fetched pages, and
    how the gate would score the legend holding it. Reads local files only."""
    if fetched.get("status") != "success" or not fetched.get("chunks_path"):
        return None
    from sync_local_chunked_to_mongodb import LocalChunkedMongoSync

    builder = LocalChunkedMongoSync(use_llm=False, database=worker.db)
    built = builder.build_document(Path(fetched["chunks_path"]), Path(fetched["profiles_root"]))
    if not built:
        return None
    rag_context = built[1]["rag_context"]
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
    worker.db[BACKUPS_COLLECTION].insert_one({
        "profile_id": row["profile_id"],
        "run_id": run_id,
        "at": utc_now_iso(),
        "policy": REPAIR_POLICY_VERSION,
        "previous": {k: doc.get(k) for k in ("rag_context", "daily_story_profile", "identity_review")},
    })
    worker.scholars_collection.update_one({"_id": doc["_id"]}, {"$set": {"rag_context": candidate["rag_context"]}})
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
        state = dict(row["saved"]) or {"policy": REPAIR_POLICY_VERSION, "profile_id": row["profile_id"], "name": row["name"]}
        collected = saved_collection(row)
        if not collected:
            collected = collect_urls(row, max_urls=args.max_urls, sleep_s=args.sleep_seconds, per_query=args.per_query_results)
            state["collected"], state["fetched"] = collected, {}
            save_state(row["state_path"], state)
        out["urls"] = len(collected["urls"])
        out["search_ran"] = collected["search_ran"]
        if not collected["urls"]:
            out["outcome"] = "no_urls"
            return out
        fetched = state.get("fetched") or {}
        # A fetch that failed (the network down, say) is tried again; one that
        # finished is kept, whatever it gave.
        if fetched.get("urls_hash") != collected["urls_hash"] or fetched.get("status") != "success":
            fetched = fetch_pages(row, collected, row["state_path"].parent, scraper)
            state["fetched"] = fetched
            save_state(row["state_path"], state)
        out["seeds"] = {k: fetched.get(k) for k in ("seed_urls_total", "seed_urls_successful", "seed_urls_failed")}
        candidate = build_candidate(worker, row, fetched)
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


def dry_run_report(worker: DailyStoryWorker, rows: List[Dict[str, Any]], args: argparse.Namespace, model: str) -> str:
    overrides = model_costs.parse_price_overrides(args.price)
    lines = [f"PROFILE REPAIR PLAN  {len(rows)} legends  (dry run: nothing written, fetched or asked of a model)"]
    repairs = [r for r in rows if r["state"] == "repair"]
    review_steps: List[model_costs.Step] = []
    retry_steps: List[model_costs.Step] = []
    to_search = 0
    seeds_cap = 0
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
            elif collected:
                lines.append(f"      saved work: {len(collected['urls'])} URLs collected {collected['at']}: would fetch them (no new search)")
                seeds_cap += len(collected["urls"])
            else:
                note = " (a collection made without a search key is redone)" if row["saved"].get("collected") else ""
                lines.append(f"      saved work: none{note}")
                to_search += 1
                seeds_cap += args.max_urls
        # Who the identity stage would judge: the legends this run makes
        # ready whose review would then be stale or missing.
        eligible, why = legend_identity.story_eligibility(row["doc"])
        if row["state"] == "repair" or why in ("not_reviewed", "stale"):
            tokens, pages, known = judge_input_tokens(row, candidate)
            low = JUDGE_ANSWER_TOKENS_BASE + JUDGE_ANSWER_TOKENS_PER_PAGE * pages
            review_steps.append(model_costs.Step(f"Identity review, {row['profile_id']}", model, 1, tokens, low, identity_stage.JUDGE_MAX_TOKENS, note=f"input from {known}"))
            retry_steps.append(model_costs.Step(f"Identity review, {row['profile_id']}, two more tries", model, 1, 2 * tokens, 0, sum(JUDGE_RETRY_CAPS)))

    searching = [r for r in repairs if not saved_collection(r)]
    queries = sum(len(r["queries"]) for r in searching)
    brave_on = search_key_set()
    lines += ["", "WEB (counted; nothing here is fetched in a dry run)"]
    lines.append(
        f"  Wikipedia API: {WIKIPEDIA_REQUESTS[0]} to {WIKIPEDIA_REQUESTS[1]} requests a legend x {to_search} legends to search"
        f" = {WIKIPEDIA_REQUESTS[0] * to_search} to {WIKIPEDIA_REQUESTS[1] * to_search}"
    )
    lines.append(
        f"  Brave Search API: {' + '.join(str(len(r['queries'])) for r in searching) or '0'} = {queries} queries, one request each"
        + ("" if brave_on else "  (BRAVE_SEARCH_API is not set: none would run, and only the stored lists and Wikipedia would be fetched)")
    )
    lines.append(
        f"  Seed pages: at most {seeds_cap} ({args.max_urls} a legend to search, or its saved list); robots.txt once per site."
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
        "  legend_scholars: rag_context and daily_story_profile; identity_review by the identity stage",
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
    try:
        rows = plan(worker, legend_ids, url_lists_dir=Path(args.url_lists_dir), work_dir=work_dir, min_quality=args.min_quality, fresh=args.fresh)
        if args.dry_run:
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
            "policy": REPAIR_POLICY_VERSION,
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
    parser = argparse.ArgumentParser(description="Collect more source pages for named legends so their profiles pass the quality gate.")
    parser.add_argument("--legend-ids-file", default="", help="A file of legend profile ids, one per line or comma-separated")
    parser.add_argument("--legend-ids", default="", help="Comma-separated legend profile ids")
    parser.add_argument("--dry-run", action="store_true", help="Write, fetch and ask nothing; print the plan and its cost")
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
