"""
Trend ingestion worker for daily story generation.

Fetches trending issues from configured providers and stores normalized records
in MongoDB collection `daily_story_trend_issues` for downstream story jobs.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from daily_story_worker import DailyStoryWorker, utc_now_iso


def parse_date_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def run_trend_ingestion(*, provider: str, max_items: int, dry_run: bool) -> Dict[str, Any]:
    worker = DailyStoryWorker(
        use_llm=False,
        trends_enabled=True,
        trend_provider=provider,
    )
    try:
        jobs = worker.db["daily_story_trend_jobs"]
        jobs.create_index("run_id", unique=True)
        jobs.create_index("started_at")

        run_id = f"trend-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        started_at = utc_now_iso()
        job_doc = {
            "run_id": run_id,
            "started_at": started_at,
            "ended_at": None,
            "status": "running",
            "config": {
                "provider": provider,
                "max_items": max_items,
                "dry_run": dry_run,
            },
            "summary": {
                "fetched": 0,
                "cached": 0,
            },
            "errors": [],
        }
        jobs.insert_one(job_doc)

        # Force live fetch for this job; store into cache unless dry_run.
        original_max = worker.trend_max_items
        worker.trend_max_items = max_items
        issues = worker._fetch_trending_issues(prefer_cache=False)
        worker.trend_max_items = original_max

        if not dry_run and issues:
            worker._upsert_trending_issue_cache(issues)

        summary = {
            "fetched": len(issues),
            "cached": len(issues) if (issues and not dry_run) else 0,
        }
        status = "completed" if issues else "completed_no_data"
        jobs.update_one(
            {"run_id": run_id},
            {
                "$set": {
                    "ended_at": utc_now_iso(),
                    "status": status,
                    "summary": summary,
                }
            },
        )
        return {
            "run_id": run_id,
            "date": parse_date_utc(),
            "status": status,
            "summary": summary,
            "sample_titles": [i.get("title") for i in issues[:5]],
        }
    finally:
        worker.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and cache trending issues for daily story jobs.")
    parser.add_argument(
        "--provider",
        type=str,
        default="rss",
        help="Trend provider: rss | newsapi | gdelt | auto",
    )
    parser.add_argument("--max-items", type=int, default=40, help="Max issues to fetch")
    parser.add_argument("--dry-run", action="store_true", help="Fetch but do not write cache")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_trend_ingestion(
        provider=args.provider.strip().lower(),
        max_items=max(1, args.max_items),
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

