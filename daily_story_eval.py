"""
Daily story quality evaluation job.

Computes quality metrics for generated stories and stores results in
`daily_story_quality_events`.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from daily_story_worker import DailyStoryWorker, parse_date, safe_text, utc_now_iso
from daily_story_personalize import GRADE_BANDS


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", safe_text(text)))


def _first_person_hits(text: str) -> int:
    reader_text = re.sub(r"https?://\S+", " ", safe_text(text), flags=re.I)
    return len(re.findall(r"\b(i|me|my|mine|we|our|ours|us)\b", reader_text.lower()))


def run_eval(*, story_date: str, window_days: int) -> Dict[str, Any]:
    worker = DailyStoryWorker(use_llm=False, trends_enabled=False)
    try:
        eval_col = worker.db["daily_story_quality_events"]
        eval_col.create_index("event_key", unique=True)
        eval_col.create_index([("created_at", -1)])
        jobs = worker.db["daily_story_eval_jobs"]
        jobs.create_index("run_id", unique=True)
        jobs.create_index("started_at")

        run_id = f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        started_at = utc_now_iso()
        jobs.insert_one(
            {
                "run_id": run_id,
                "started_at": started_at,
                "ended_at": None,
                "status": "running",
                "config": {
                    "story_date": story_date,
                    "window_days": window_days,
                },
            }
        )

        end_date = parse_date(story_date)
        start_date = end_date - timedelta(days=max(0, window_days - 1))
        q = {
            "story_date": {
                "$gte": start_date.isoformat(),
                "$lte": end_date.isoformat(),
            }
        }
        stories = list(
            worker.stories_collection.find(
                q,
                {
                    "_id": 0,
                    "story_date": 1,
                    "status": 1,
                    "topic_selection.source": 1,
                    "content.article_markdown": 1,
                    "content.title": 1,
                    "citations": 1,
                    "safety.errors": 1,
                    "safety.warnings": 1,
                    "trend_issue.selected.title": 1,
                    "audience_contract": 1,
                    "audience_variants": 1,
                    "personalization": 1,
                },
            )
        )

        status_counts: Dict[str, int] = {}
        topic_source_counts: Dict[str, int] = {}
        total_words = 0
        total_citations = 0
        citation_ge_4 = 0
        first_person_issues = 0
        trend_bound = 0
        validation_error_total = 0
        validation_warning_total = 0
        title_counts: Dict[str, int] = {}
        required_variant_total = 0
        ready_variant_total = 0
        valid_variant_total = 0
        personalization_ready_stories = 0
        stories_missing_audience_contract = 0
        variants_by_band: Dict[str, Dict[str, int]] = {
            key: {"required": 0, "ready": 0, "failed": 0}
            for key in GRADE_BANDS
        }

        for story in stories:
            status = safe_text(story.get("status")) or "unknown"
            status_counts[status] = status_counts.get(status, 0) + 1

            source = safe_text(((story.get("topic_selection") or {}).get("source")))
            source = source or "unknown"
            topic_source_counts[source] = topic_source_counts.get(source, 0) + 1

            text = safe_text((story.get("content") or {}).get("article_markdown"))
            title = safe_text((story.get("content") or {}).get("title")).lower()
            if title:
                title_counts[title] = title_counts.get(title, 0) + 1
            wc = _word_count(text)
            total_words += wc

            citations = story.get("citations") or []
            c_count = len(citations) if isinstance(citations, list) else 0
            total_citations += c_count
            if c_count >= 4:
                citation_ge_4 += 1

            if _first_person_hits(text) > 0:
                first_person_issues += 1

            trend_title = safe_text((((story.get("trend_issue") or {}).get("selected") or {}).get("title")))
            if trend_title:
                trend_bound += 1

            errs = ((story.get("safety") or {}).get("errors")) or []
            warns = ((story.get("safety") or {}).get("warnings")) or []
            if isinstance(errs, list):
                validation_error_total += len(errs)
            if isinstance(warns, list):
                validation_warning_total += len(warns)

            audience = story.get("audience_contract") or {}
            required_bands = [
                key for key in (audience.get("band_keys") or [])
                if key in GRADE_BANDS
            ]
            if not required_bands:
                stories_missing_audience_contract += 1
            variant_map = {
                safe_text(item.get("band_key")): item
                for item in (story.get("audience_variants") or [])
                if isinstance(item, dict)
            }
            story_ready = bool(required_bands)
            for band_key in required_bands:
                required_variant_total += 1
                variants_by_band[band_key]["required"] += 1
                variant = variant_map.get(band_key) or {}
                is_ready = safe_text(variant.get("status")) == "ready"
                is_valid = bool((variant.get("validation") or {}).get("ok"))
                if is_ready:
                    ready_variant_total += 1
                    variants_by_band[band_key]["ready"] += 1
                else:
                    variants_by_band[band_key]["failed"] += 1
                if is_valid:
                    valid_variant_total += 1
                story_ready = story_ready and is_ready and is_valid
            if story_ready and bool((story.get("personalization") or {}).get("ready")):
                personalization_ready_stories += 1

        total = len(stories)
        avg_words = round(total_words / total, 2) if total else 0.0
        avg_citations = round(total_citations / total, 2) if total else 0.0
        citation_coverage = round(citation_ge_4 / total, 4) if total else 0.0
        trend_coverage = round(trend_bound / total, 4) if total else 0.0
        first_person_rate = round(first_person_issues / total, 4) if total else 0.0
        variant_coverage = round(ready_variant_total / required_variant_total, 4) if required_variant_total else 0.0
        variant_validation_rate = round(valid_variant_total / required_variant_total, 4) if required_variant_total else 0.0
        personalization_ready_rate = round(personalization_ready_stories / total, 4) if total else 0.0
        unique_title_ratio = round(len(title_counts) / total, 4) if total else 0.0
        duplicate_title_count = sum(count - 1 for count in title_counts.values() if count > 1)

        event = {
            "event_key": f"{start_date.isoformat()}:{end_date.isoformat()}",
            "window": {
                "date_from": start_date.isoformat(),
                "date_to": end_date.isoformat(),
                "days": window_days,
            },
            "metrics": {
                "total_stories": total,
                "status_counts": status_counts,
                "avg_word_count": avg_words,
                "avg_citation_count": avg_citations,
                "citation_coverage_ge_4": citation_coverage,
                "trend_coverage": trend_coverage,
                "first_person_rate": first_person_rate,
                "validation_error_total": validation_error_total,
                "validation_warning_total": validation_warning_total,
                "topic_source_counts": topic_source_counts,
                "unique_title_ratio": unique_title_ratio,
                "duplicate_title_count": duplicate_title_count,
                "required_variant_total": required_variant_total,
                "ready_variant_total": ready_variant_total,
                "variant_coverage": variant_coverage,
                "variant_validation_rate": variant_validation_rate,
                "personalization_ready_stories": personalization_ready_stories,
                "personalization_ready_rate": personalization_ready_rate,
                "stories_missing_audience_contract": stories_missing_audience_contract,
                "variants_by_band": variants_by_band,
            },
            "run_id": run_id,
            "created_at": utc_now_iso(),
        }

        eval_col.update_one({"event_key": event["event_key"]}, {"$set": event}, upsert=True)

        status = "completed" if total > 0 else "completed_no_data"
        jobs.update_one(
            {"run_id": run_id},
            {"$set": {"ended_at": utc_now_iso(), "status": status, "summary": event["metrics"]}},
        )

        return {
            "run_id": run_id,
            "status": status,
            "window": event["window"],
            "metrics": event["metrics"],
        }
    finally:
        worker.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate quality metrics for daily scholar stories.")
    parser.add_argument("--date", type=str, default=None, help="End date YYYY-MM-DD (default: today UTC)")
    parser.add_argument("--window-days", type=int, default=1, help="Rolling window length")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    story_date = parse_date(args.date).isoformat()
    result = run_eval(story_date=story_date, window_days=max(1, args.window_days))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
