"""
Daily story cron suite orchestrator.

Runs end-to-end pipeline for one day:
1) profile quality refresh
2) trend ingestion
3) story generation
4) retry failed stories
5) quality evaluation
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from daily_story_eval import run_eval
from daily_story_retry_failed import run_retry_failed
from daily_story_worker import DailyStoryWorker, parse_date
from profile_quality_refresh import run_quality_refresh
from trend_ingestion_worker import run_trend_ingestion


def run_suite(
    *,
    story_date: str,
    max_scholars: int,
    max_context_chunks: int,
    trend_provider: str,
    enforce_profile_quality: bool,
    profile_quality_min_score: int,
    dry_run: bool,
    no_llm: bool,
    disable_trends: bool,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "story_date": story_date,
        "steps": {},
    }

    quality_result = run_quality_refresh(
        max_scholars=max_scholars,
        scholar_id=None,
        update_scholar_docs=True,
    )
    output["steps"]["profile_quality_refresh"] = quality_result

    trend_result = (
        {"status": "skipped", "reason": "disable_trends=true"}
        if disable_trends
        else run_trend_ingestion(provider=trend_provider, max_items=60, dry_run=dry_run)
    )
    output["steps"]["trend_ingestion"] = trend_result

    worker = DailyStoryWorker(
        use_llm=not no_llm,
        trends_enabled=not disable_trends,
        trend_provider=trend_provider,
        enforce_profile_quality=enforce_profile_quality,
        profile_quality_min_score=profile_quality_min_score,
    )
    try:
        gen_result = worker.run(
            scholar_id=None,
            date_value=parse_date(story_date),
            topic_override=None,
            max_scholars=max_scholars,
            max_context_chunks=max_context_chunks,
            dry_run=dry_run,
        )
    finally:
        worker.close()
    output["steps"]["daily_story_generate"] = gen_result

    retry_result = run_retry_failed(
        story_date=story_date,
        statuses=["failed_validation", "failed_generation"],
        reuse_topic=True,
        max_context_chunks=max_context_chunks,
        dry_run=dry_run,
        no_llm=no_llm,
    )
    output["steps"]["daily_story_retry_failed"] = retry_result

    eval_result = run_eval(story_date=story_date, window_days=1)
    output["steps"]["daily_story_eval"] = eval_result

    output["status"] = "completed"
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full daily story cron suite.")
    parser.add_argument("--date", type=str, default=None, help="Story date in YYYY-MM-DD (default: today UTC)")
    parser.add_argument("--max-scholars", type=int, default=25)
    parser.add_argument("--max-context-chunks", type=int, default=12)
    parser.add_argument("--trend-provider", type=str, default="rss")
    parser.add_argument("--enforce-profile-quality", action="store_true")
    parser.add_argument("--profile-quality-min-score", type=int, default=60)
    parser.add_argument("--disable-trends", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    story_date = parse_date(args.date).isoformat()
    result = run_suite(
        story_date=story_date,
        max_scholars=max(1, args.max_scholars),
        max_context_chunks=max(4, args.max_context_chunks),
        trend_provider=args.trend_provider.strip().lower(),
        enforce_profile_quality=args.enforce_profile_quality,
        profile_quality_min_score=max(0, args.profile_quality_min_score),
        dry_run=args.dry_run,
        no_llm=args.no_llm,
        disable_trends=args.disable_trends,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

