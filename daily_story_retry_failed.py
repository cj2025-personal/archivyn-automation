"""
Retry failed daily stories for a given story date.

This job finds failed story documents, removes the blocked rows (same story_key),
and re-runs generation for those scholars only.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from daily_story_worker import DailyStoryWorker, parse_date, safe_text, utc_now_iso


DEFAULT_RETRY_STATUSES = ["failed_validation", "failed_generation"]


def run_retry_failed(
    *,
    story_date: str,
    statuses: List[str],
    reuse_topic: bool,
    max_context_chunks: int,
    dry_run: bool,
    no_llm: bool,
) -> Dict[str, Any]:
    worker = DailyStoryWorker(
        use_llm=not no_llm,
        trends_enabled=True,
    )
    try:
        retry_jobs = worker.db["daily_story_retry_jobs"]
        retry_jobs.create_index("run_id", unique=True)
        retry_jobs.create_index("started_at")

        run_id = f"retry-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        started_at = utc_now_iso()
        job_doc = {
            "run_id": run_id,
            "started_at": started_at,
            "ended_at": None,
            "status": "running",
            "config": {
                "story_date": story_date,
                "statuses": statuses,
                "reuse_topic": reuse_topic,
                "max_context_chunks": max_context_chunks,
                "dry_run": dry_run,
                "no_llm": no_llm,
            },
            "summary": {
                "found": 0,
                "retried": 0,
                "regenerated": 0,
                "still_failed": 0,
            },
            "errors": [],
        }
        retry_jobs.insert_one(job_doc)

        failed_docs = list(
            worker.stories_collection.find(
                {
                    "story_date": story_date,
                    "status": {"$in": statuses},
                }
            )
        )
        summary = job_doc["summary"]
        errors: List[str] = []
        summary["found"] = len(failed_docs)

        for failed in failed_docs:
            story_key = safe_text(failed.get("story_key"))
            scholar = failed.get("scholar") or {}
            profile_id = safe_text(scholar.get("profile_id"))
            if not story_key or not profile_id:
                errors.append("missing_story_key_or_profile_id")
                continue

            topic_override = safe_text(failed.get("topic")) if reuse_topic else None

            # Remove failed row to unblock unique story_key before retry.
            worker.stories_collection.delete_one({"story_key": story_key})
            summary["retried"] += 1

            result = worker.run(
                scholar_id=profile_id,
                date_value=parse_date(story_date),
                topic_override=topic_override,
                max_scholars=1,
                max_context_chunks=max_context_chunks,
                dry_run=dry_run,
            )
            gen_count = int((result.get("summary") or {}).get("generated") or 0)
            if gen_count > 0:
                summary["regenerated"] += 1
            else:
                summary["still_failed"] += 1
                errors.append(f"{profile_id}:retry_not_generated")

        final_status = "completed"
        if summary["found"] == 0:
            final_status = "completed_no_failures"
        elif summary["regenerated"] == 0:
            final_status = "completed_no_recoveries"

        retry_jobs.update_one(
            {"run_id": run_id},
            {
                "$set": {
                    "ended_at": utc_now_iso(),
                    "status": final_status,
                    "summary": summary,
                    "errors": errors,
                }
            },
        )
        return {
            "run_id": run_id,
            "status": final_status,
            "summary": summary,
            "errors": errors,
        }
    finally:
        worker.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retry failed daily story rows for one date.")
    parser.add_argument("--date", type=str, default=None, help="Story date in YYYY-MM-DD (default: today UTC)")
    parser.add_argument(
        "--statuses",
        type=str,
        default=",".join(DEFAULT_RETRY_STATUSES),
        help="Comma-separated story statuses to retry",
    )
    parser.add_argument("--max-context-chunks", type=int, default=12)
    parser.add_argument("--no-reuse-topic", action="store_true", help="Do not reuse failed story topic")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    story_date = parse_date(args.date).isoformat()
    statuses = [s.strip() for s in safe_text(args.statuses).split(",") if s.strip()]
    result = run_retry_failed(
        story_date=story_date,
        statuses=statuses or DEFAULT_RETRY_STATUSES,
        reuse_topic=not args.no_reuse_topic,
        max_context_chunks=max(4, args.max_context_chunks),
        dry_run=args.dry_run,
        no_llm=args.no_llm,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

