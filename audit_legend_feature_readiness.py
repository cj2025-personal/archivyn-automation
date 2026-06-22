"""
Audit legendary scholar profiles in MongoDB for demo-readiness across AI features.

This script is designed for the `legend_scholars` collection but is generic enough
to run against future legend sets in the same document shape.

It evaluates readiness for:
  1. Editorial / historically grounded perspective generation
  2. AI podcast / multi-speaker discussion generation
  3. Grade-level chatbot answers
  4. Profile card / profile page summaries

It does not mutate MongoDB. It produces a scored report plus prioritized
improvement recommendations.

Examples:
  python audit_legend_feature_readiness.py --names "John Hope Franklin" "Samella Lewis" "Carter G. Woodson"
  python audit_legend_feature_readiness.py --all --limit 25 --output-json output/legend_audits/sample.json
  python audit_legend_feature_readiness.py --profile-ids john-hope-franklin samella-lewis
  python audit_legend_feature_readiness.py --names "John Hope Franklin" --apply
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv

from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name


DEFAULT_COLLECTION = "legend_scholars"
DEFAULT_JSON_PATH = "output/legend_audits/legend_readiness_report.json"
DEFAULT_MD_PATH = "output/legend_audits/legend_readiness_report.md"

TRUSTED_SOURCE_DOMAINS = {
    "wikipedia.org",
    "blackpast.org",
    "britannica.com",
    "encyclopedia.com",
    "biography.com",
    "thehistorymakers.org",
    "poetryfoundation.org",
    "loc.gov",
    "archives.gov",
    "si.edu",
    "nps.gov",
    "pbs.org",
    "history.com",
    "smithsonianmag.com",
    "harvard.edu",
    "duke.edu",
}

GENERIC_SECTION_PATTERNS = (
    "home",
    "page not found",
    "personal website",
    "website subpage",
    "upcoming events",
    "recent news",
    "news",
    "directions",
    "contact",
    "asset type",
    "technical information",
    "access request",
)

SEMANTIC_SECTION_PATTERNS = {
    "biography": ("biography", "bio", "about", "life", "early life", "profile"),
    "background": ("background", "education", "career", "academic career", "positions"),
    "legacy": ("legacy", "impact", "influence", "tribute", "honor", "memorial"),
    "works": ("publications", "books", "works", "writings", "articles", "papers"),
    "voice": ("interview", "oral history", "quote", "speech", "lecture", "conversation", "keynote"),
}

VOICE_TEXT_PATTERNS = (
    "interviewer:",
    "interview:",
    "question:",
    "answer:",
    "he said",
    "she said",
)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def normalize_name(value: str) -> str:
    text = safe_text(value).lower()
    return re.sub(r"\s+", " ", text)


def clamp_score(value: float) -> int:
    return max(0, min(100, int(round(value))))


def points_from_thresholds(value: float, thresholds: Sequence[Tuple[float, int]]) -> int:
    for minimum, points in thresholds:
        if value >= minimum:
            return points
    return 0


def truthy_text_len(value: Any) -> int:
    return len(safe_text(value))


def meaningful_list_count(value: Any) -> int:
    if isinstance(value, list):
        count = 0
        for item in value:
            if isinstance(item, dict):
                if any(safe_text(v) for v in item.values()):
                    count += 1
            elif safe_text(item):
                count += 1
        return count
    if safe_text(value):
        return 1
    return 0


def normalize_domain(url: str) -> str:
    text = safe_text(url)
    if not text:
        return ""
    try:
        parsed = urlparse(text)
    except Exception:
        return ""
    host = parsed.netloc.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def domain_is_trusted(domain: str) -> bool:
    if not domain:
        return False
    return any(domain == item or domain.endswith("." + item) for item in TRUSTED_SOURCE_DOMAINS)


def readiness_label(score: int) -> str:
    if score >= 80:
        return "demo_ready"
    if score >= 65:
        return "demo_ready_with_guardrails"
    if score >= 50:
        return "limited_demo_use"
    return "not_ready"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Recommendation:
    priority: int
    category: str
    message: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "priority": self.priority,
            "category": self.category,
            "message": self.message,
        }


def collect_group_docs(
    coll: Any,
    *,
    names: Optional[List[str]],
    profile_ids: Optional[List[str]],
    all_docs: bool,
    limit: Optional[int],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    projection = {
        "_id": 1,
        "profile_id": 1,
        "professor_name": 1,
        "name": 1,
        "about": 1,
        "background_and_work": 1,
        "milestones": 1,
        "publications": 1,
        "links_and_media": 1,
        "metadata": 1,
        "display": 1,
        "rag_context": 1,
        "admin": 1,
        "daily_story_profile": 1,
    }
    groups: List[Tuple[str, List[Dict[str, Any]]]] = []

    if names:
        for name in names:
            query = {"professor_name": {"$regex": f"^{re.escape(name)}$", "$options": "i"}}
            docs = list(coll.find(query, projection))
            groups.append((name, docs))
        return groups

    if profile_ids:
        fetched = list(coll.find({"profile_id": {"$in": profile_ids}}, projection))
        by_id = {safe_text(doc.get("profile_id")): doc for doc in fetched}
        for profile_id in profile_ids:
            doc = by_id.get(profile_id)
            label = safe_text(doc.get("professor_name")) if doc else profile_id
            groups.append((label, [doc] if doc else []))
        return groups

    if all_docs:
        cursor = coll.find({}, projection).sort("professor_name", 1)
        if limit:
            cursor = cursor.limit(limit)
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for doc in cursor:
            key = normalize_name(safe_text(doc.get("professor_name")) or safe_text((doc.get("name") or {}).get("full")))
            grouped[key].append(doc)
        for docs in grouped.values():
            docs.sort(key=lambda d: safe_text(d.get("profile_id")))
            label = safe_text(docs[0].get("professor_name")) or safe_text((docs[0].get("name") or {}).get("full")) or "Unknown"
            groups.append((label, docs))
        groups.sort(key=lambda item: normalize_name(item[0]))
        return groups

    return []


def load_quality_docs(quality_coll: Any, profile_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    ids = [profile_id for profile_id in profile_ids if profile_id]
    if not ids:
        return {}
    docs = list(
        quality_coll.find(
            {"profile_id": {"$in": ids}},
            {"_id": 0, "profile_id": 1, "quality_score": 1, "status": 1, "signals": 1, "refreshed_at": 1},
        )
    )
    return {safe_text(doc.get("profile_id")): doc for doc in docs}


def collect_source_stats(section_chunks: Dict[str, Any], source_catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
    unique_urls = set()
    unique_domains = set()
    trusted_domains = set()
    chunk_count = 0
    chunks_with_primary = 0
    chunks_with_source_urls = 0
    chunks_with_refs = 0

    for chunks in section_chunks.values():
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            chunk_count += 1
            primary_url = safe_text(chunk.get("primary_source_url"))
            if primary_url:
                chunks_with_primary += 1
                unique_urls.add(primary_url)
            source_urls = chunk.get("source_urls") or []
            if isinstance(source_urls, list) and source_urls:
                chunks_with_source_urls += 1
                for url in source_urls:
                    if safe_text(url):
                        unique_urls.add(safe_text(url))
            refs = chunk.get("source_refs") or []
            if isinstance(refs, list) and refs:
                chunks_with_refs += 1
                for ref in refs:
                    if not isinstance(ref, dict):
                        continue
                    url = safe_text(ref.get("source_url"))
                    if url:
                        unique_urls.add(url)

    for src in source_catalog:
        if not isinstance(src, dict):
            continue
        for key in ("source_url", "resolved_url"):
            url = safe_text(src.get(key))
            if url:
                unique_urls.add(url)

    for url in unique_urls:
        domain = normalize_domain(url)
        if not domain:
            continue
        unique_domains.add(domain)
        if domain_is_trusted(domain):
            trusted_domains.add(domain)

    return {
        "chunk_count_seen": chunk_count,
        "chunks_with_primary_url": chunks_with_primary,
        "chunks_with_source_urls": chunks_with_source_urls,
        "chunks_with_source_refs": chunks_with_refs,
        "unique_source_urls": len(unique_urls),
        "unique_source_domains": len(unique_domains),
        "trusted_source_domains": len(trusted_domains),
        "source_catalog_count": len(source_catalog),
        "primary_url_coverage_ratio": round(chunks_with_primary / max(1, chunk_count), 4),
        "source_url_coverage_ratio": round(chunks_with_source_urls / max(1, chunk_count), 4),
        "source_ref_coverage_ratio": round(chunks_with_refs / max(1, chunk_count), 4),
    }


def classify_sections(sections_available: List[str], section_chunks: Dict[str, Any]) -> Dict[str, Any]:
    section_names = [safe_text(item) for item in sections_available if safe_text(item)]
    if not section_names and isinstance(section_chunks, dict):
        section_names = [safe_text(item) for item in section_chunks.keys() if safe_text(item)]

    semantic_hits: Dict[str, int] = {key: 0 for key in SEMANTIC_SECTION_PATTERNS}
    generic_count = 0
    voice_signal = False

    for name in section_names:
        lowered = name.lower()
        if any(pattern in lowered for pattern in GENERIC_SECTION_PATTERNS):
            generic_count += 1
        for category, patterns in SEMANTIC_SECTION_PATTERNS.items():
            if any(pattern in lowered for pattern in patterns):
                semantic_hits[category] += 1
                if category == "voice":
                    voice_signal = True

    if not voice_signal:
        sample_texts: List[str] = []
        for chunks in section_chunks.values():
            if not isinstance(chunks, list):
                continue
            for chunk in chunks[:2]:
                if not isinstance(chunk, dict):
                    continue
                txt = safe_text(chunk.get("text")) or safe_text(chunk.get("summary"))
                if txt:
                    sample_texts.append(txt[:500].lower())
            if len(sample_texts) >= 6:
                break
        voice_signal = any(pattern in "\n".join(sample_texts) for pattern in VOICE_TEXT_PATTERNS)

    total_sections = len(section_names)
    semantic_category_count = sum(1 for value in semantic_hits.values() if value > 0)
    generic_ratio = round(generic_count / max(1, total_sections), 4)
    return {
        "section_count": total_sections,
        "semantic_hits": semantic_hits,
        "semantic_category_count": semantic_category_count,
        "generic_section_count": generic_count,
        "generic_section_ratio": generic_ratio,
        "voice_signal": voice_signal,
        "sample_sections": section_names[:12],
    }


def extract_structured_stats(doc: Dict[str, Any]) -> Dict[str, Any]:
    about = doc.get("about") or {}
    background = doc.get("background_and_work") or {}
    publications = doc.get("publications") or {}
    milestones = doc.get("milestones") or []

    return {
        "short_bio_len": truthy_text_len(about.get("short_bio")),
        "long_bio_len": truthy_text_len(about.get("long_bio")),
        "field_present": bool(safe_text(about.get("field_of_study"))),
        "institution_present": bool(safe_text(about.get("institution"))),
        "background_summary_len": truthy_text_len(background.get("background_summary")),
        "education_item_count": meaningful_list_count(background.get("education_summary")),
        "research_focus_count": meaningful_list_count(background.get("research_focus")),
        "current_work_len": truthy_text_len(background.get("current_work")),
        "methodology_count": meaningful_list_count(background.get("methodology")),
        "milestones_count": meaningful_list_count(milestones),
        "featured_publications_count": meaningful_list_count(publications.get("featured_publications")),
        "total_publications_count": int(publications.get("total_publications_count") or 0),
    }


def score_profile_card(stats: Dict[str, Any], section_stats: Dict[str, Any], source_stats: Dict[str, Any]) -> int:
    score = 0
    score += points_from_thresholds(stats["short_bio_len"], ((180, 22), (100, 16), (40, 8)))
    score += points_from_thresholds(stats["long_bio_len"], ((700, 18), (350, 12), (140, 6)))
    score += 8 if stats["field_present"] else 0
    score += points_from_thresholds(stats["background_summary_len"], ((140, 14), (80, 10), (40, 5)))
    score += points_from_thresholds(stats["featured_publications_count"], ((4, 14), (2, 10), (1, 5)))
    score += points_from_thresholds(stats["milestones_count"], ((4, 10), (2, 6), (1, 3)))
    score += points_from_thresholds(source_stats["unique_source_urls"], ((6, 8), (3, 5), (1, 2)))
    score += 6 if section_stats["generic_section_ratio"] <= 0.35 else 2 if section_stats["generic_section_ratio"] <= 0.6 else 0
    return clamp_score(score)


def score_chatbot(
    stats: Dict[str, Any],
    section_stats: Dict[str, Any],
    source_stats: Dict[str, Any],
    provenance_ratio: float,
) -> int:
    score = 0
    score += points_from_thresholds(source_stats["chunk_count_seen"], ((80, 24), (40, 18), (20, 13), (10, 7)))
    score += points_from_thresholds(source_stats["unique_source_urls"], ((10, 18), (6, 13), (3, 8), (1, 4)))
    score += points_from_thresholds(source_stats["unique_source_domains"], ((8, 12), (5, 8), (3, 5), (1, 2)))
    score += points_from_thresholds(section_stats["semantic_category_count"], ((4, 16), (3, 12), (2, 7), (1, 3)))
    score += points_from_thresholds(stats["long_bio_len"] + stats["background_summary_len"], ((900, 10), (450, 7), (180, 4)))
    score += points_from_thresholds(provenance_ratio, ((0.95, 10), (0.75, 7), (0.5, 4)))
    score += 10 if section_stats["generic_section_ratio"] <= 0.3 else 5 if section_stats["generic_section_ratio"] <= 0.55 else 0
    return clamp_score(score)


def score_editorial(
    stats: Dict[str, Any],
    section_stats: Dict[str, Any],
    source_stats: Dict[str, Any],
    provenance_ratio: float,
    duplicate_count: int,
) -> int:
    score = 0
    score += points_from_thresholds(source_stats["chunk_count_seen"], ((100, 24), (60, 18), (30, 12), (15, 6)))
    score += points_from_thresholds(source_stats["unique_source_urls"], ((12, 18), (8, 13), (5, 8), (2, 4)))
    score += points_from_thresholds(section_stats["semantic_category_count"], ((4, 20), (3, 14), (2, 8), (1, 3)))
    score += 14 if section_stats["voice_signal"] else 4
    score += points_from_thresholds(stats["long_bio_len"] + stats["current_work_len"], ((900, 10), (500, 7), (180, 4)))
    score += points_from_thresholds(provenance_ratio, ((0.95, 8), (0.75, 5), (0.5, 3)))
    score += 10 if section_stats["generic_section_ratio"] <= 0.25 else 4 if section_stats["generic_section_ratio"] <= 0.5 else 0
    if duplicate_count > 1:
        score -= 8
    return clamp_score(score)


def score_podcast(
    stats: Dict[str, Any],
    section_stats: Dict[str, Any],
    source_stats: Dict[str, Any],
    provenance_ratio: float,
    duplicate_count: int,
) -> int:
    score = 0
    score += points_from_thresholds(source_stats["chunk_count_seen"], ((120, 20), (70, 15), (40, 10), (20, 5)))
    score += points_from_thresholds(source_stats["unique_source_urls"], ((15, 14), (10, 10), (6, 6), (3, 3)))
    score += 22 if section_stats["voice_signal"] else 3
    score += points_from_thresholds(section_stats["semantic_category_count"], ((4, 18), (3, 13), (2, 7), (1, 3)))
    score += points_from_thresholds(stats["long_bio_len"], ((750, 10), (400, 7), (180, 4)))
    score += points_from_thresholds(stats["featured_publications_count"] + stats["milestones_count"], ((8, 8), (5, 5), (2, 3)))
    score += points_from_thresholds(provenance_ratio, ((0.95, 4), (0.75, 3), (0.5, 2)))
    score += 4 if section_stats["generic_section_ratio"] <= 0.25 else 1 if section_stats["generic_section_ratio"] <= 0.5 else 0
    if duplicate_count > 1:
        score -= 10
    return clamp_score(score)


def build_recommendations(
    *,
    doc: Dict[str, Any],
    duplicate_count: int,
    quality_doc: Optional[Dict[str, Any]],
    feature_scores: Dict[str, int],
    structured_stats: Dict[str, Any],
    section_stats: Dict[str, Any],
    source_stats: Dict[str, Any],
) -> List[Recommendation]:
    recommendations: List[Recommendation] = []
    profile_id = safe_text(doc.get("profile_id"))

    if duplicate_count > 1:
        recommendations.append(
            Recommendation(
                10,
                "dedupe",
                "Consolidate duplicate profiles for this legend and choose one canonical profile_id before demo retrieval.",
            )
        )

    if source_stats["chunk_count_seen"] < 40:
        recommendations.append(
            Recommendation(
                20,
                "evidence_depth",
                "Collect more long-form source material; target at least 40 to 60 grounded chunks for stronger editorial and podcast generation.",
            )
        )

    if source_stats["unique_source_urls"] < 8 or source_stats["unique_source_domains"] < 5:
        recommendations.append(
            Recommendation(
                30,
                "source_diversity",
                "Increase source diversity with archives, biography/reference pages, oral histories, interviews, lectures, and institutional tributes.",
            )
        )

    if section_stats["semantic_category_count"] < 3 or section_stats["generic_section_ratio"] > 0.4:
        recommendations.append(
            Recommendation(
                40,
                "section_quality",
                "Re-collect URLs that produce stronger biography, legacy, background, and works sections instead of generic home/news/event pages.",
            )
        )

    if not section_stats["voice_signal"]:
        recommendations.append(
            Recommendation(
                50,
                "voice_grounding",
                "Add interviews, oral histories, speeches, lectures, or quoted first-person material to support podcast and perspective-generation features.",
            )
        )

    if structured_stats["education_item_count"] == 0 or structured_stats["research_focus_count"] == 0:
        recommendations.append(
            Recommendation(
                60,
                "structured_fields",
                "Backfill thin structured fields such as education_summary and research_focus from existing chunk evidence for cleaner profile cards and chatbot answers.",
            )
        )

    if not quality_doc:
        recommendations.append(
            Recommendation(
                70,
                "operational_readiness",
                f"Run `python profile_quality_refresh.py --scholar-id {profile_id} --update-scholar-docs` so the profile is formally scored and tracked by downstream jobs.",
            )
        )

    if feature_scores["podcast"] < 65:
        recommendations.append(
            Recommendation(
                80,
                "podcast_scope",
                "For demo podcasts, keep this legend moderator-led and limit turn length unless more voice-rich sources are added.",
            )
        )

    if feature_scores["editorial"] < 65:
        recommendations.append(
            Recommendation(
                90,
                "editorial_scope",
                "Use historically informed synthesis with explicit grounding rather than strong persona imitation until section quality and source breadth improve.",
            )
        )

    recommendations.sort(key=lambda item: (item.priority, item.category))
    return recommendations[:8]


def evaluate_doc(doc: Dict[str, Any], duplicate_count: int, quality_doc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    rag = doc.get("rag_context") or {}
    section_chunks = rag.get("section_chunks") or {}
    sections_available = rag.get("sections_available") or list(section_chunks.keys())
    source_catalog = rag.get("source_catalog") or []
    provenance = rag.get("provenance") or {}

    source_stats = collect_source_stats(section_chunks, source_catalog)
    section_stats = classify_sections(list(sections_available), section_chunks)
    structured_stats = extract_structured_stats(doc)
    provenance_ratio = float(provenance.get("coverage_ratio") or source_stats["source_ref_coverage_ratio"] or 0.0)

    feature_scores = {
        "profile_card": score_profile_card(structured_stats, section_stats, source_stats),
        "chatbot": score_chatbot(structured_stats, section_stats, source_stats, provenance_ratio),
        "editorial": score_editorial(structured_stats, section_stats, source_stats, provenance_ratio, duplicate_count),
        "podcast": score_podcast(structured_stats, section_stats, source_stats, provenance_ratio, duplicate_count),
    }
    overall_demo_score = clamp_score(sum(feature_scores.values()) / len(feature_scores))

    recommendations = build_recommendations(
        doc=doc,
        duplicate_count=duplicate_count,
        quality_doc=quality_doc,
        feature_scores=feature_scores,
        structured_stats=structured_stats,
        section_stats=section_stats,
        source_stats=source_stats,
    )

    return {
        "profile_id": safe_text(doc.get("profile_id")),
        "professor_name": safe_text(doc.get("professor_name")) or safe_text((doc.get("name") or {}).get("full")),
        "duplicate_count_for_name": duplicate_count,
        "overall_demo_score": overall_demo_score,
        "overall_readiness": readiness_label(overall_demo_score),
        "feature_scores": {
            feature: {
                "score": score,
                "readiness": readiness_label(score),
            }
            for feature, score in feature_scores.items()
        },
        "structured_stats": structured_stats,
        "section_stats": section_stats,
        "source_stats": source_stats,
        "provenance": {
            "coverage_ratio": round(provenance_ratio, 4),
            "raw": provenance,
        },
        "quality_job": quality_doc or None,
        "recommendations": [item.as_dict() for item in recommendations],
    }


def choose_best_profile(profile_reports: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not profile_reports:
        return None
    return max(
        profile_reports,
        key=lambda item: (
            item["overall_demo_score"],
            item["feature_scores"]["editorial"]["score"],
            item["feature_scores"]["chatbot"]["score"],
        ),
    )


def build_group_report(group_name: str, docs: List[Dict[str, Any]], quality_docs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    duplicate_count = len(docs)
    profile_reports = [
        evaluate_doc(doc, duplicate_count, quality_docs.get(safe_text(doc.get("profile_id"))))
        for doc in docs
    ]
    profile_reports.sort(key=lambda item: (-item["overall_demo_score"], item["profile_id"]))
    best_profile = choose_best_profile(profile_reports)

    group_recommendations: List[Recommendation] = []
    if duplicate_count > 1 and best_profile:
        group_recommendations.append(
            Recommendation(
                10,
                "canonical_profile",
                f"Use `{best_profile['profile_id']}` as the demo-default profile for {group_name} until duplicates are merged.",
            )
        )
    if best_profile and best_profile["feature_scores"]["podcast"]["score"] < 65:
        group_recommendations.append(
            Recommendation(
                20,
                "feature_scoping",
                "Treat podcast generation as a constrained demo feature for this legend unless more voice-rich evidence is added.",
            )
        )

    return {
        "group_name": group_name,
        "match_count": duplicate_count,
        "best_profile_id": best_profile["profile_id"] if best_profile else None,
        "best_profile_overall_demo_score": best_profile["overall_demo_score"] if best_profile else None,
        "group_recommendations": [item.as_dict() for item in sorted(group_recommendations, key=lambda item: item.priority)],
        "profiles": profile_reports,
    }


def persist_report_to_scholar_docs(
    *,
    db: Any,
    source_collection: str,
    group_reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    scholars_coll = db[source_collection]
    run_id = f"legend-feature-readiness-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
    summary = {
        "groups": len(group_reports),
        "profiles": 0,
        "scholar_docs_updated": 0,
    }

    errors: List[str] = []
    evaluated_at = utc_now_iso()
    for group in group_reports:
        group_name = group["group_name"]
        match_count = int(group.get("match_count") or 0)
        best_profile_id = safe_text(group.get("best_profile_id"))
        best_score = group.get("best_profile_overall_demo_score")
        group_recommendations = group.get("group_recommendations") or []

        for profile in group.get("profiles") or []:
            summary["profiles"] += 1
            profile_id = safe_text(profile.get("profile_id"))
            if not profile_id:
                errors.append(f"{group_name}:missing_profile_id")
                continue

            scholar_update = {
                "ai_feature_readiness.version": "legend_feature_readiness_v1",
                "ai_feature_readiness.run_id": run_id,
                "ai_feature_readiness.evaluated_at": evaluated_at,
                "ai_feature_readiness.professor_name": safe_text(profile.get("professor_name")),
                "ai_feature_readiness.group_name": group_name,
                "ai_feature_readiness.duplicate_count_for_name": int(profile.get("duplicate_count_for_name") or 0),
                "ai_feature_readiness.best_profile_id_for_group": best_profile_id,
                "ai_feature_readiness.best_profile_overall_demo_score_for_group": best_score,
                "ai_feature_readiness.match_count_for_group": match_count,
                "ai_feature_readiness.overall_demo_score": int(profile.get("overall_demo_score") or 0),
                "ai_feature_readiness.overall_readiness": safe_text(profile.get("overall_readiness")),
                "ai_feature_readiness.feature_scores": profile.get("feature_scores") or {},
                "ai_feature_readiness.structured_stats": profile.get("structured_stats") or {},
                "ai_feature_readiness.section_stats": profile.get("section_stats") or {},
                "ai_feature_readiness.source_stats": profile.get("source_stats") or {},
                "ai_feature_readiness.provenance": profile.get("provenance") or {},
                "ai_feature_readiness.quality_job": profile.get("quality_job"),
                "ai_feature_readiness.recommendations": profile.get("recommendations") or [],
                "ai_feature_readiness.group_recommendations": group_recommendations,
                "admin.ai_feature_readiness.version": "legend_feature_readiness_v1",
                "admin.ai_feature_readiness.run_id": run_id,
                "admin.ai_feature_readiness.evaluated_at": evaluated_at,
                "admin.ai_feature_readiness.overall_demo_score": int(profile.get("overall_demo_score") or 0),
                "admin.ai_feature_readiness.overall_readiness": safe_text(profile.get("overall_readiness")),
            }
            result = scholars_coll.update_one({"profile_id": profile_id}, {"$set": scholar_update})
            if result.matched_count:
                summary["scholar_docs_updated"] += 1
            else:
                errors.append(f"{profile_id}:scholar_doc_not_found")

    return {
        "run_id": run_id,
        "status": "completed",
        "summary": summary,
        "errors": errors,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Legend Feature Readiness Audit")
    lines.append("")
    lines.append(f"- Database: `{report['database']}`")
    lines.append(f"- Collection: `{report['collection']}`")
    lines.append(f"- Groups audited: `{report['group_count']}`")
    lines.append("")

    for group in report["groups"]:
        lines.append(f"## {group['group_name']}")
        lines.append("")
        lines.append(f"- Matches: `{group['match_count']}`")
        lines.append(f"- Best profile: `{group['best_profile_id']}`")
        lines.append(f"- Best demo score: `{group['best_profile_overall_demo_score']}`")
        if group["group_recommendations"]:
            lines.append("- Group recommendations:")
            for item in group["group_recommendations"]:
                lines.append(f"  - {item['message']}")
        lines.append("")

        for profile in group["profiles"]:
            lines.append(f"### {profile['profile_id']}")
            lines.append("")
            lines.append(
                f"- Overall: `{profile['overall_demo_score']}` ({profile['overall_readiness']})"
            )
            lines.append(
                "- Feature scores: "
                f"profile_card=`{profile['feature_scores']['profile_card']['score']}`, "
                f"chatbot=`{profile['feature_scores']['chatbot']['score']}`, "
                f"editorial=`{profile['feature_scores']['editorial']['score']}`, "
                f"podcast=`{profile['feature_scores']['podcast']['score']}`"
            )
            lines.append(
                "- Evidence: "
                f"chunks=`{profile['source_stats']['chunk_count_seen']}`, "
                f"urls=`{profile['source_stats']['unique_source_urls']}`, "
                f"domains=`{profile['source_stats']['unique_source_domains']}`, "
                f"trusted_domains=`{profile['source_stats']['trusted_source_domains']}`"
            )
            lines.append(
                "- Sections: "
                f"semantic_categories=`{profile['section_stats']['semantic_category_count']}`, "
                f"generic_ratio=`{profile['section_stats']['generic_section_ratio']}`, "
                f"voice_signal=`{profile['section_stats']['voice_signal']}`"
            )
            lines.append(
                "- Structured fields: "
                f"short_bio_len=`{profile['structured_stats']['short_bio_len']}`, "
                f"long_bio_len=`{profile['structured_stats']['long_bio_len']}`, "
                f"featured_publications=`{profile['structured_stats']['featured_publications_count']}`, "
                f"milestones=`{profile['structured_stats']['milestones_count']}`"
            )
            if profile["recommendations"]:
                lines.append("- Recommendations:")
                for item in profile["recommendations"]:
                    lines.append(f"  - {item['message']}")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def print_console_summary(report: Dict[str, Any]) -> None:
    print(f"Database: {report['database']}")
    print(f"Collection: {report['collection']}")
    print(f"Groups audited: {report['group_count']}")
    print("")
    for group in report["groups"]:
        print(f"{group['group_name']}: best={group['best_profile_id']} score={group['best_profile_overall_demo_score']} matches={group['match_count']}")
        for profile in group["profiles"]:
            feature_scores = profile["feature_scores"]
            print(
                "  "
                f"{profile['profile_id']}: overall={profile['overall_demo_score']} "
                f"card={feature_scores['profile_card']['score']} "
                f"chatbot={feature_scores['chatbot']['score']} "
                f"editorial={feature_scores['editorial']['score']} "
                f"podcast={feature_scores['podcast']['score']}"
            )
        print("")


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--names", nargs="*", default=None, help="Exact professor_name values to audit.")
    parser.add_argument("--profile-ids", nargs="*", default=None, help="Specific profile_id values to audit.")
    parser.add_argument("--all", action="store_true", help="Audit the entire collection, grouped by professor_name.")
    parser.add_argument("--limit", type=int, default=None, help="Limit groups when using --all.")
    parser.add_argument("--apply", action="store_true", help="Write ai_feature_readiness onto the existing scholar documents in MongoDB.")
    parser.add_argument("--output-json", default=None, help="Optional JSON report path.")
    parser.add_argument("--output-md", default=None, help="Optional Markdown report path.")
    parser.add_argument("--write-default-output", action="store_true", help="Write JSON and Markdown reports under output/legend_audits.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.names and not args.profile_ids and not args.all:
        raise SystemExit("Provide --names, --profile-ids, or --all.")

    load_dotenv(dotenv_path=".env")
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        raise SystemExit("MONGODB_URI not found in environment.")

    client = create_mongo_client(mongodb_uri)
    try:
        db_name = resolve_mongo_db_name(mongodb_uri)
        db = client[db_name]
        coll = db[args.collection]
        quality_coll = db["daily_story_profile_quality"]

        groups = collect_group_docs(
            coll,
            names=args.names,
            profile_ids=args.profile_ids,
            all_docs=args.all,
            limit=args.limit,
        )

        group_reports: List[Dict[str, Any]] = []
        for group_name, docs in groups:
            docs = [doc for doc in docs if doc]
            quality_docs = load_quality_docs(quality_coll, [safe_text(doc.get("profile_id")) for doc in docs])
            group_reports.append(build_group_report(group_name, docs, quality_docs))

        report = {
            "database": db_name,
            "collection": args.collection,
            "group_count": len(group_reports),
            "groups": group_reports,
        }

        mongo_persist_result = None
        if args.apply:
            mongo_persist_result = persist_report_to_scholar_docs(
                db=db,
                source_collection=args.collection,
                group_reports=group_reports,
            )
            report["mongo_persist_result"] = mongo_persist_result

        print_console_summary(report)
        if mongo_persist_result:
            print(
                "Mongo persist: "
                f"run_id={mongo_persist_result['run_id']} "
                f"scholar_docs_updated={mongo_persist_result['summary']['scholar_docs_updated']}"
            )

        output_json = args.output_json
        output_md = args.output_md
        if args.write_default_output:
            output_json = output_json or DEFAULT_JSON_PATH
            output_md = output_md or DEFAULT_MD_PATH

        if output_json:
            ensure_parent(output_json)
            Path(output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Wrote JSON report to {output_json}")

        if output_md:
            ensure_parent(output_md)
            Path(output_md).write_text(render_markdown(report), encoding="utf-8")
            print(f"Wrote Markdown report to {output_md}")

        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
