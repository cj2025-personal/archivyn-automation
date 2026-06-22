"""
Profile quality refresh for daily story generation.

Computes readiness scores for `legend_scholars` profiles and stores results in
`daily_story_profile_quality`.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from daily_story_worker import DailyStoryWorker, safe_text, utc_now_iso


CORE_SECTIONS = [
    "biography",
    "about",
    "background",
    "legacy",
    "publications",
]


def _section_coverage(sections: List[str]) -> int:
    lower_sections = [safe_text(s).lower() for s in sections]
    hits = 0
    for core in CORE_SECTIONS:
        if any(core in sec for sec in lower_sections):
            hits += 1
    return hits


def _avg_chunk_len(section_chunks: Dict[str, Any]) -> float:
    lengths: List[int] = []
    if not isinstance(section_chunks, dict):
        return 0.0
    for chunks in section_chunks.values():
        if not isinstance(chunks, list):
            continue
        for c in chunks:
            if not isinstance(c, dict):
                continue
            txt = safe_text(c.get("text")) or safe_text(c.get("summary"))
            if txt:
                lengths.append(len(txt))
            if len(lengths) >= 500:
                break
    if not lengths:
        return 0.0
    return float(sum(lengths) / len(lengths))


def evaluate_profile(worker: DailyStoryWorker, scholar_doc: Dict[str, Any]) -> Dict[str, Any]:
    profile_id = worker._profile_id(scholar_doc)
    raw_name = (
        safe_text(scholar_doc.get("professor_name"))
        or safe_text((scholar_doc.get("name") or {}).get("full"))
        or safe_text((scholar_doc.get("name") or {}).get("display"))
        or safe_text(scholar_doc.get("name"))
    )
    normalized_name = worker._professor_name(scholar_doc)
    name_quality = 1 if worker._looks_like_clean_name(normalized_name) else 0

    rag_context = scholar_doc.get("rag_context") or {}
    section_chunks = rag_context.get("section_chunks") or {}
    sections_available = rag_context.get("sections_available") or list(section_chunks.keys())
    chunk_count = int(rag_context.get("chunk_count") or 0)
    if chunk_count <= 0 and isinstance(section_chunks, dict):
        chunk_count = sum(len(v) for v in section_chunks.values() if isinstance(v, list))
    coverage = _section_coverage(sections_available if isinstance(sections_available, list) else [])
    avg_len = _avg_chunk_len(section_chunks)

    score = 0
    score += 20 if name_quality else 0
    if chunk_count >= 80:
        score += 40
    elif chunk_count >= 40:
        score += 30
    elif chunk_count >= 20:
        score += 20
    elif chunk_count >= 10:
        score += 10

    if coverage >= 4:
        score += 20
    elif coverage == 3:
        score += 14
    elif coverage == 2:
        score += 8

    if avg_len >= 420:
        score += 20
    elif avg_len >= 250:
        score += 14
    elif avg_len >= 140:
        score += 8

    status = "ready"
    if score < 60 or chunk_count < 20:
        status = "needs_data_repair"
    if not profile_id:
        status = "invalid_profile"

    return {
        "profile_id": profile_id,
        "name_raw": raw_name,
        "name_normalized": normalized_name,
        "score": score,
        "status": status,
        "signals": {
            "name_quality": name_quality,
            "chunk_count": chunk_count,
            "section_coverage": coverage,
            "avg_chunk_len": round(avg_len, 1),
        },
    }


def run_quality_refresh(*, max_scholars: int, scholar_id: str | None, update_scholar_docs: bool) -> Dict[str, Any]:
    worker = DailyStoryWorker(use_llm=False, trends_enabled=False)
    try:
        quality_col = worker.db["daily_story_profile_quality"]
        quality_col.create_index("profile_id", unique=True)
        quality_col.create_index([("refreshed_at", -1)])
        jobs = worker.db["daily_story_profile_quality_jobs"]
        jobs.create_index("run_id", unique=True)
        jobs.create_index("started_at")

        run_id = f"quality-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        started_at = utc_now_iso()
        job_doc = {
            "run_id": run_id,
            "started_at": started_at,
            "ended_at": None,
            "status": "running",
            "config": {
                "max_scholars": max_scholars,
                "scholar_id": scholar_id,
                "update_scholar_docs": update_scholar_docs,
            },
            "summary": {
                "attempted": 0,
                "ready": 0,
                "needs_data_repair": 0,
                "invalid_profile": 0,
            },
            "errors": [],
        }
        jobs.insert_one(job_doc)

        scholars = list(worker._iter_scholars(scholar_id=scholar_id, max_scholars=max_scholars))
        summary = job_doc["summary"]
        errors: List[str] = []

        for doc in scholars:
            summary["attempted"] += 1
            result = evaluate_profile(worker, doc)
            status = safe_text(result.get("status")) or "invalid_profile"
            if status not in summary:
                summary[status] = 0
            summary[status] += 1
            profile_id = safe_text(result.get("profile_id"))
            if not profile_id:
                errors.append("missing_profile_id")
                continue

            quality_doc = {
                "profile_id": profile_id,
                "name_raw": result["name_raw"],
                "name_normalized": result["name_normalized"],
                "quality_score": result["score"],
                "status": status,
                "signals": result["signals"],
                "run_id": run_id,
                "refreshed_at": utc_now_iso(),
            }
            quality_col.update_one({"profile_id": profile_id}, {"$set": quality_doc}, upsert=True)

            if update_scholar_docs:
                worker.scholars_collection.update_one(
                    {"profile_id": profile_id},
                    {
                        "$set": {
                            "daily_story_profile.quality_score": result["score"],
                            "daily_story_profile.status": status,
                            "daily_story_profile.name_normalized": result["name_normalized"],
                            "daily_story_profile.refreshed_at": utc_now_iso(),
                        }
                    },
                )

        final_status = "completed" if summary.get("attempted", 0) > 0 else "completed_no_data"
        jobs.update_one(
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
    parser = argparse.ArgumentParser(description="Refresh profile quality readiness for daily story jobs.")
    parser.add_argument("--max-scholars", type=int, default=1000)
    parser.add_argument("--scholar-id", type=str, default=None)
    parser.add_argument(
        "--update-scholar-docs",
        action="store_true",
        help="Write quality fields back into legend_scholars.daily_story_profile",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_quality_refresh(
        max_scholars=max(1, args.max_scholars),
        scholar_id=args.scholar_id,
        update_scholar_docs=args.update_scholar_docs,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

