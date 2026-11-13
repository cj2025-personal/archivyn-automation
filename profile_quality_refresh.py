"""
Profile quality refresh for daily story generation.

Computes readiness scores for `legend_scholars` profiles and stores results in
`daily_story_profile_quality`.
"""
from __future__ import annotations

import argparse
import json
import uuid
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

# The score's parts and the bars a profile must clear to be "ready". They are
# named here, not written inline, so legend_profile_diagnosis.py can say what
# a profile lacks from the very rules that score it.
NAME_POINTS = 20
# (at least this much, points), highest first.
CHUNK_POINTS: Tuple[Tuple[int, int], ...] = ((80, 40), (40, 30), (20, 20), (10, 10))
COVERAGE_POINTS: Tuple[Tuple[int, int], ...] = ((4, 20), (3, 14), (2, 8))
AVG_LEN_POINTS: Tuple[Tuple[int, int], ...] = ((420, 20), (250, 14), (140, 8))
READY_MIN_SCORE = 60
READY_MIN_CHUNKS = 20


def tier_points(value: float, tiers: Tuple[Tuple[int, int], ...]) -> int:
    for floor, points in tiers:
        if value >= floor:
            return points
    return 0


def core_sections_found(sections: List[str]) -> List[str]:
    """The CORE_SECTIONS named in any section title (a substring match)."""
    lower_sections = [safe_text(s).lower() for s in sections]
    return [core for core in CORE_SECTIONS if any(core in sec for sec in lower_sections)]


def _section_coverage(sections: List[str]) -> int:
    return len(core_sections_found(sections))


def score_points(*, name_quality: int, chunk_count: int, section_coverage: int, avg_chunk_len: float) -> Dict[str, int]:
    """Each part of the score. avg_chunk_len is the unrounded mean: the
    stored signal is rounded, and rounding can cross a bar."""
    return {
        "name": NAME_POINTS if name_quality else 0,
        "chunks": tier_points(chunk_count, CHUNK_POINTS),
        "coverage": tier_points(section_coverage, COVERAGE_POINTS),
        "avg_len": tier_points(avg_chunk_len, AVG_LEN_POINTS),
    }


def status_for(*, score: int, chunk_count: int, profile_id: str) -> str:
    if not profile_id:
        return "invalid_profile"
    if score < READY_MIN_SCORE or chunk_count < READY_MIN_CHUNKS:
        return "needs_data_repair"
    return "ready"


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
    sections = sections_available if isinstance(sections_available, list) else []
    coverage = _section_coverage(sections)
    avg_len = _avg_chunk_len(section_chunks)

    points = score_points(name_quality=name_quality, chunk_count=chunk_count, section_coverage=coverage, avg_chunk_len=avg_len)
    score = sum(points.values())
    status = status_for(score=score, chunk_count=chunk_count, profile_id=profile_id)

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
        # Not stored (quality_record keeps the signals); for the diagnosis.
        "points": points,
        "core_sections": core_sections_found(sections),
    }


def quality_record(result: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    """The daily_story_profile_quality row for one scored profile."""
    return {
        "profile_id": safe_text(result.get("profile_id")),
        "name_raw": result["name_raw"],
        "name_normalized": result["name_normalized"],
        "quality_score": result["score"],
        "status": safe_text(result.get("status")) or "invalid_profile",
        "signals": result["signals"],
        "run_id": run_id,
        "refreshed_at": utc_now_iso(),
    }


def store_quality(quality_col: Any, scholars_col: Any, result: Dict[str, Any], *, run_id: str, update_scholar_docs: bool) -> None:
    """Record a profile's score where the story gate reads it: the quality
    row, and with update_scholar_docs the legend's own daily_story_profile
    (which the gate reads first). Shared by the nightly refresh and
    legend_profile_repair.py, so both label a profile the same way."""
    record = quality_record(result, run_id)
    profile_id = record["profile_id"]
    quality_col.update_one({"profile_id": profile_id}, {"$set": record}, upsert=True)
    if update_scholar_docs:
        scholars_col.update_one(
            {"profile_id": profile_id},
            {
                "$set": {
                    "daily_story_profile.quality_score": result["score"],
                    "daily_story_profile.status": record["status"],
                    "daily_story_profile.name_normalized": result["name_normalized"],
                    "daily_story_profile.refreshed_at": utc_now_iso(),
                }
            },
        )


def run_quality_refresh(*, max_scholars: int, scholar_id: str | None, update_scholar_docs: bool) -> Dict[str, Any]:
    worker = DailyStoryWorker(use_llm=False, trends_enabled=False)
    try:
        quality_col = worker.db["daily_story_profile_quality"]
        quality_col.create_index("profile_id", unique=True)
        quality_col.create_index([("refreshed_at", -1)])
        jobs = worker.db["daily_story_profile_quality_jobs"]
        jobs.create_index("run_id", unique=True)
        jobs.create_index("started_at")

        run_id = f"quality-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}-{uuid.uuid4().hex[:8]}"
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

            store_quality(
                quality_col,
                worker.scholars_collection,
                result,
                run_id=run_id,
                update_scholar_docs=update_scholar_docs,
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
