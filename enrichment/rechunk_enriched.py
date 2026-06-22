"""
Re-chunking module for enriched profiles.

After enrichment data is collected, this module:
1. Reads the original profile text + enrichment text
2. Merges them into a unified document with clear section markers
3. Runs the existing ProfileChunkingPipeline on the merged text
4. Optionally uploads the new chunks to Pinecone
5. Optionally re-syncs the profile to MongoDB with richer summaries

This is designed to be run AFTER run_enrichment_pipeline.py has
collected data for all (or a batch of) professors.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def merge_profile_with_enrichment(
    profile_dir: Path,
) -> Optional[Dict]:
    """
    Read original profile JSON + enrichment_text.txt and produce
    a merged text document suitable for chunking.

    Returns dict with {profile_id, name, profile_url, merged_text, enriched}
    or None if no enrichment data exists.
    """
    profile_json = profile_dir / f"{profile_dir.name}.json"
    enrichment_text_cleaned = profile_dir / "enrichment_text_cleaned.txt"
    enrichment_text_file = profile_dir / "enrichment_text.txt"
    enrichment_json_file = profile_dir / "enrichment.json"

    if not profile_json.exists():
        return None
    if not enrichment_text_file.exists():
        return None

    try:
        profile_data = json.loads(profile_json.read_text(encoding="utf-8"))
    except Exception:
        return None

    original_text = profile_data.get("clean_text", "") or profile_data.get("raw_text", "")
    # Prefer cleaned enrichment text when available
    if enrichment_text_cleaned.exists():
        enrichment_text = enrichment_text_cleaned.read_text(encoding="utf-8")
    else:
        enrichment_text = enrichment_text_file.read_text(encoding="utf-8")

    # Build merged document with clear structure
    sections = []

    # Section 1: Original profile data
    sections.append("=" * 60)
    sections.append(f"PROFESSOR: {profile_data.get('name', 'Unknown')}")
    sections.append(f"University: Ohio State University")
    sections.append(f"Profile URL: {profile_data.get('profile_url', '')}")
    sections.append("=" * 60)
    sections.append("")
    sections.append("--- ORIGINAL PROFILE ---")
    sections.append(original_text)

    # Section 2: Enrichment data (already well-structured from orchestrator)
    sections.append("")
    sections.append("--- ENRICHMENT DATA FROM PUBLIC SOURCES ---")
    sections.append(enrichment_text)

    merged_text = "\n".join(sections)

    # Read confidence score if available
    confidence = None
    if enrichment_json_file.exists():
        try:
            enr_data = json.loads(enrichment_json_file.read_text(encoding="utf-8"))
            confidence = enr_data.get("confidence", {}).get("overall_confidence")
        except Exception:
            pass

    return {
        "profile_id": profile_data.get("profile_id", profile_dir.name),
        "name": profile_data.get("name", ""),
        "profile_url": profile_data.get("profile_url", ""),
        "merged_text": merged_text,
        "enriched": True,
        "confidence": confidence,
        "original_text_len": len(original_text),
        "enrichment_text_len": len(enrichment_text),
        "total_text_len": len(merged_text),
    }


def rechunk_single_profile(
    profile_dir: Path,
    chunking_pipeline,
    output_dir: Path,
) -> Optional[Dict]:
    """
    Re-chunk a single enriched profile.
    Returns chunk stats or None on failure.
    """
    merged = merge_profile_with_enrichment(profile_dir)
    if not merged:
        return None

    profile_id = merged["profile_id"]

    try:
        result = chunking_pipeline.process_text(
            text=merged["merged_text"],
            profile_id=profile_id,
            metadata={
                "name": merged["name"],
                "profile_url": merged["profile_url"],
                "enriched": True,
            },
        )

        return {
            "profile_id": profile_id,
            "name": merged["name"],
            "original_text_len": merged["original_text_len"],
            "enrichment_text_len": merged["enrichment_text_len"],
            "total_text_len": merged["total_text_len"],
            "confidence": merged["confidence"],
            "chunks_generated": True,
        }

    except Exception as e:
        logger.error("Re-chunking failed for %s: %s", profile_id, e)
        return None


def rechunk_all_enriched(
    profiles_dir: str,
    chunking_output_dir: str,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
    limit: Optional[int] = None,
    start_from: int = 0,
    min_confidence: float = 0.0,
) -> Dict:
    """
    Re-chunk all enriched profiles.

    Args:
        profiles_dir: Path to profiles directory
        chunking_output_dir: Where to write chunked output
        llm_provider: LLM provider for chunking
        llm_model: LLM model for chunking
        limit: Max profiles to process
        start_from: 0-based index to start from
        min_confidence: Skip profiles below this confidence score

    Returns:
        Summary dict with stats
    """
    from profile_chunking_pipeline import ProfileChunkingPipeline

    profiles_path = Path(profiles_dir)
    output_path = Path(chunking_output_dir)

    pipeline = ProfileChunkingPipeline(
        output_dir=str(output_path),
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    profile_dirs = sorted([d for d in profiles_path.iterdir() if d.is_dir()])
    processed = 0
    skipped = 0
    failed = 0
    total_original_chars = 0
    total_enrichment_chars = 0

    start_time = time.perf_counter()

    for i, pdir in enumerate(profile_dirs[start_from:]):
        if limit and processed >= limit:
            break

        # Only process profiles that have enrichment data
        if not (pdir / "enrichment_text.txt").exists():
            skipped += 1
            continue

        # Check confidence threshold
        if min_confidence > 0:
            enr_json = pdir / "enrichment.json"
            if enr_json.exists():
                try:
                    enr_data = json.loads(enr_json.read_text(encoding="utf-8"))
                    conf = enr_data.get("confidence", {}).get("overall_confidence", 0)
                    if conf < min_confidence:
                        skipped += 1
                        continue
                except Exception:
                    pass

        result = rechunk_single_profile(pdir, pipeline, output_path)
        if result:
            processed += 1
            total_original_chars += result["original_text_len"]
            total_enrichment_chars += result["enrichment_text_len"]
            logger.info(
                "[%d] Re-chunked: %s (orig=%d chars, enrichment=%d chars)",
                processed, result["name"],
                result["original_text_len"], result["enrichment_text_len"],
            )
        else:
            failed += 1

    elapsed = time.perf_counter() - start_time

    summary = {
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "total_original_chars": total_original_chars,
        "total_enrichment_chars": total_enrichment_chars,
        "avg_enrichment_ratio": (
            total_enrichment_chars / max(total_original_chars, 1)
        ),
        "elapsed_seconds": round(elapsed, 1),
    }

    logger.info("Re-chunking complete: %s", json.dumps(summary))
    return summary
