"""Generate grade-banded access variants for grounded daily editorial stories.

This worker is invoked by the existing Node cron coordinator immediately after
canonical story generation/retry. It never receives student identifiers. It
derives aggregate grade-band demand from active users and stores one reusable,
review-gated variant per represented band on the canonical story document.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from daily_story_worker import DailyStoryWorker, GenerationConfig, parse_date, safe_text, utc_now_iso


AUDIENCE_CONTRACT_VERSION = "active-student-audience-v1"
PROMPT_VERSION = "editorial-access-ladder-v3"
READY_COMPATIBLE_PROMPT_VERSIONS = {
    "editorial-access-ladder-v1",
    "editorial-access-ladder-v2",
    PROMPT_VERSION,
}
MAX_TOTAL_GENERATION_ATTEMPTS = max(1, int(os.getenv("EDITORIAL_PERSONALIZATION_MAX_TOTAL_ATTEMPTS", "9")))

GRADE_BANDS: Dict[str, Dict[str, Any]] = {
    "grade_k_2": {
        "label": "Kindergarten to Grade 2",
        "min_level": 0,
        "max_level": 2,
        "typical_age_min": 5,
        "typical_age_max": 8,
        "min_grade": 0.0,
        "max_grade": 3.5,
        "max_sentence_words": 10.0,
        "min_words": 180,
        "max_words": 350,
        "min_evidence": 2,
        "guidance": "Use very short sentences, concrete words, and immediate explanations of unfamiliar terms.",
    },
    "grade_3_5": {
        "label": "Grades 3 to 5",
        "min_level": 3,
        "max_level": 5,
        "typical_age_min": 8,
        "typical_age_max": 11,
        "min_grade": 2.5,
        "max_grade": 6.75,
        "max_sentence_words": 14.0,
        "min_words": 300,
        "max_words": 650,
        "min_evidence": 2,
        "guidance": "Use short sentences, define new words, and connect abstract ideas to familiar settings without adding facts.",
    },
    "grade_6_8": {
        "label": "Grades 6 to 8",
        "min_level": 6,
        "max_level": 8,
        "typical_age_min": 11,
        "typical_age_max": 14,
        "min_grade": 5.0,
        "max_grade": 9.0,
        "max_sentence_words": 18.0,
        "min_words": 450,
        "max_words": 1100,
        "min_evidence": 3,
        "guidance": "Use middle-school language, brief definitions, clear chunks, and explicit transitions.",
    },
    "grade_9_12": {
        "label": "Grades 9 to 12",
        "min_level": 9,
        "max_level": 12,
        "typical_age_min": 14,
        "typical_age_max": 18,
        "min_grade": 8.0,
        "max_grade": 13.5,
        "max_sentence_words": 22.0,
        "min_words": 650,
        "max_words": 1400,
        "min_evidence": 4,
        "guidance": "Use high-school academic language, preserve nuance, and define technical terms on first use.",
    },
}

VARIANT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "article_markdown": {"type": "string"},
        "glossary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"term": {"type": "string"}, "definition": {"type": "string"}},
                "required": ["term", "definition"],
            },
        },
        "key_takeaways": {"type": "array", "items": {"type": "string"}},
        "comprehension_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer_guidance": {"type": "string"},
                },
                "required": ["question", "answer_guidance"],
            },
        },
        "teacher_note": {"type": "string"},
        "used_evidence_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "title",
        "summary",
        "article_markdown",
        "glossary",
        "key_takeaways",
        "comprehension_questions",
        "teacher_note",
        "used_evidence_ids",
    ],
}

VERIFIER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "blockers": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ok", "blockers", "notes"],
}


def _strip_markup(value: Any) -> str:
    text = safe_text(value)
    text = re.sub(r"https?://\S+", " ", text, flags=re.I)
    text = re.sub(r"\[[^\]]+\]\([^\)]+\)", " ", text)
    text = re.sub(r"[#*_`>|~]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _syllables(value: str) -> int:
    word = re.sub(r"[^a-z]", "", safe_text(value).lower())
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", word)
    word = re.sub(r"^y", "", word)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", word)))


def readability_metrics(value: Any) -> Dict[str, Any]:
    text = _strip_markup(value)
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    sentence_count = max(1, len(re.findall(r"[.!?]+(?:\s|$)", text)))
    syllable_count = sum(_syllables(word) for word in words)
    word_count = len(words)
    average_sentence_words = word_count / sentence_count
    grade = 0.0
    if word_count:
        grade = max(0.0, 0.39 * average_sentence_words + 11.8 * (syllable_count / word_count) - 15.59)
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "syllable_count": syllable_count,
        "average_sentence_words": round(average_sentence_words, 2),
        "flesch_kincaid_grade": round(grade, 2),
        "algorithm": "flesch_kincaid_heuristic_v1",
    }


def _resolve_grade(user: Dict[str, Any]) -> Optional[int]:
    personalization = user.get("personalizationConfig") or {}
    for value in (user.get("gradeLevel"), user.get("grade_level"), personalization.get("gradeLevel")):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= parsed <= 12:
            return parsed
    return None


def _band_for_grade(grade: Optional[int]) -> Optional[str]:
    if grade is None:
        return None
    if grade <= 2:
        return "grade_k_2"
    if grade <= 5:
        return "grade_3_5"
    if grade <= 8:
        return "grade_6_8"
    return "grade_9_12"


def resolve_active_audience(worker: DailyStoryWorker) -> Dict[str, Any]:
    users = list(
        worker.db["users"].find(
            {"$or": [{"isActive": True}, {"is_active": True}], "role": {"$in": ["k12", "student"]}},
            {
                "_id": 0,
                "gradeLevel": 1,
                "grade_level": 1,
                "personalizationConfig": 1,
                "age": 1,
                "dateOfBirth": 1,
                "birthDate": 1,
                "dob": 1,
            },
        )
    )
    counts: Dict[str, int] = {}
    valid_users = 0
    explicit_age_users = 0
    for user in users:
        if any(user.get(field) not in (None, "") for field in ("age", "dateOfBirth", "birthDate", "dob")):
            explicit_age_users += 1
        band_key = _band_for_grade(_resolve_grade(user))
        if not band_key:
            continue
        valid_users += 1
        counts[band_key] = counts.get(band_key, 0) + 1
    counts = {key: counts[key] for key in GRADE_BANDS if counts.get(key, 0) > 0}
    resolved_band_keys = list(counts) if counts else list(GRADE_BANDS)
    fingerprint = hashlib.sha256(
        json.dumps(
            {"band_counts": counts, "valid_users": valid_users},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "contract_version": AUDIENCE_CONTRACT_VERSION,
        "source": "users",
        "active_user_count": len(users),
        "users_with_valid_grade": valid_users,
        "users_with_explicit_age": explicit_age_users,
        "age_personalization_basis": "typical_age_range_derived_from_grade_band",
        "band_keys": resolved_band_keys,
        "band_counts": counts,
        "fingerprint": fingerprint,
        "resolved_at": utc_now_iso(),
    }


def _content_hash(story: Dict[str, Any]) -> str:
    content = story.get("content") or {}
    payload = "\n".join([safe_text(content.get("title")), safe_text(content.get("article_markdown"))])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coerce_variant(payload: Dict[str, Any]) -> Dict[str, Any]:
    glossary = payload.get("glossary") if isinstance(payload.get("glossary"), list) else []
    questions_raw = payload.get("comprehension_questions", payload.get("comprehensionQuestions"))
    questions = questions_raw if isinstance(questions_raw, list) else []
    takeaways_raw = payload.get("key_takeaways", payload.get("keyTakeaways"))
    evidence_raw = payload.get("used_evidence_ids", payload.get("usedEvidenceIds"))
    return {
        "title": safe_text(payload.get("title")),
        "summary": safe_text(payload.get("summary")),
        "article_markdown": safe_text(payload.get("article_markdown", payload.get("articleMarkdown"))),
        "glossary": [
            {"term": safe_text(item.get("term")), "definition": safe_text(item.get("definition"))}
            for item in glossary if isinstance(item, dict) and safe_text(item.get("term"))
        ],
        "key_takeaways": [safe_text(item) for item in (takeaways_raw or []) if safe_text(item)],
        "comprehension_questions": [
            {
                "question": safe_text(item.get("question")),
                "answer_guidance": safe_text(item.get("answer_guidance", item.get("answerGuidance"))),
            }
            for item in questions if isinstance(item, dict) and safe_text(item.get("question"))
        ],
        "teacher_note": safe_text(payload.get("teacher_note", payload.get("teacherNote"))),
        "used_evidence_ids": list(dict.fromkeys(
            safe_text(item) for item in (evidence_raw or []) if safe_text(item)
        )),
    }


def _preserve_canonical_disclosure(story: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    disclosure = safe_text((story.get("content") or {}).get("disclosure"))
    article = safe_text(candidate.get("article_markdown"))
    if not disclosure or disclosure.lower() in article.lower():
        return False
    candidate["article_markdown"] = f"> {disclosure}\n\n{article}".strip()
    return True


def _call_structured(worker: DailyStoryWorker, prompt: str, schema: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    if not worker.vertex_model or GenerationConfig is None:
        raise RuntimeError("Vertex model is not available for editorial personalization")
    last_error: Optional[Exception] = None
    last_finish_hint = ""
    last_content = ""
    token_budget = max(8192, max_tokens)
    for attempt, (temperature, enforce_schema, enforce_json) in enumerate((
        (0.15, True, True),
        (0.05, False, True),
        (0.0, False, False),
    )):
        try:
            if enforce_schema:
                config = GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=token_budget,
                    response_mime_type="application/json",
                    response_schema=schema,
                )
            elif enforce_json:
                config = GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=token_budget,
                    response_mime_type="application/json",
                )
            else:
                config = GenerationConfig(temperature=temperature, max_output_tokens=token_budget)
            response = worker.vertex_model.generate_content(prompt, generation_config=config)
            last_content = worker._flatten_vertex_text(response)
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                last_finish_hint = safe_text(getattr(candidates[0], "finish_reason", ""))
            parsed = worker._extract_json_object(last_content)
            if isinstance(parsed, dict):
                return parsed
            last_error = RuntimeError("model returned invalid JSON")
        except Exception as exc:  # pragma: no cover - external SDK behavior
            last_error = exc
            time.sleep(1.25 * (2 ** attempt))
    preview = re.sub(r"\s+", " ", safe_text(last_content))[:240]
    raise RuntimeError(
        f"Structured personalization call failed: {last_error}; "
        f"finish_reason={last_finish_hint or 'n/a'}; content_preview={preview or 'empty'}"
    )


def _evidence_rows(story: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for citation in story.get("citations") or []:
        if not isinstance(citation, dict):
            continue
        evidence_id = safe_text(citation.get("chunk_id"))
        excerpt = safe_text(citation.get("evidence_excerpt"))
        if evidence_id and excerpt:
            rows.append({"evidence_id": evidence_id, "excerpt": excerpt})
    return rows


def _build_prompt(story: Dict[str, Any], band_key: str, blockers: Iterable[str]) -> str:
    band = GRADE_BANDS[band_key]
    content = story.get("content") or {}
    evidence = _evidence_rows(story)
    evidence_blob = "\n".join(f"[{row['evidence_id']}] {row['excerpt']}" for row in evidence)
    return f"""Adapt a grounded editorial for {band['label']}.

This is an access-layer rewrite of a canonical, human-review-gated article. Preserve its topic,
historical nuance, uncertainty, and conclusions. Do not add facts, identities, motivations, examples,
or advice that are absent from the canonical article and evidence excerpts. Do not address or profile
an individual student. Do not impersonate the scholar. Use respectful, non-condescending language.

Audience guidance: {band['guidance']}
Developmental context: typical ages {band['typical_age_min']}-{band['typical_age_max']}; this is a grade-derived range, not an individual student's age.
Hard contract:
- {band['min_words']}-{band['max_words']} article words
- Flesch-Kincaid estimate {band['min_grade']}-{band['max_grade']}
- average sentence length no more than {band['max_sentence_words']} words
- 3-6 glossary terms used in the article (at least 2 must survive deterministic verification)
- 3-5 grounded takeaways and 3-5 comprehension questions
- cite at least {band['min_evidence']} supplied evidence IDs in used_evidence_ids
- before returning, count words in article_markdown and revise until every numeric contract above is satisfied
- prefer short, direct sentences and concrete verbs; do not preserve canonical sentence structure when it is too dense
- return JSON only

Previous blockers to correct: {' | '.join(blockers) or 'none'}

Canonical title: {safe_text(content.get('title'))}
Canonical standfirst: {safe_text(content.get('standfirst'))}
Canonical article:
{safe_text(content.get('article_markdown'))}

Allowed evidence excerpts:
{evidence_blob}
"""


def _validate_variant(variant: Dict[str, Any], band_key: str, allowed_ids: set[str]) -> Tuple[List[str], Dict[str, Any]]:
    band = GRADE_BANDS[band_key]
    blockers: List[str] = []
    for field in ("title", "summary", "article_markdown", "teacher_note"):
        if not safe_text(variant.get(field)):
            blockers.append(f"missing_{field}")
    if not 2 <= len(variant.get("glossary") or []) <= 6:
        blockers.append("glossary_count_out_of_range")
    if not 3 <= len(variant.get("key_takeaways") or []) <= 5:
        blockers.append("takeaway_count_out_of_range")
    if not 3 <= len(variant.get("comprehension_questions") or []) <= 5:
        blockers.append("question_count_out_of_range")

    metrics = readability_metrics(variant.get("article_markdown"))
    if metrics["word_count"] < band["min_words"] or metrics["word_count"] > band["max_words"]:
        blockers.append(f"word_count_out_of_range:{metrics['word_count']}")
    if not band["min_grade"] <= metrics["flesch_kincaid_grade"] <= band["max_grade"]:
        blockers.append(f"readability_out_of_range:{metrics['flesch_kincaid_grade']}")
    if metrics["average_sentence_words"] > band["max_sentence_words"]:
        blockers.append(f"average_sentence_too_long:{metrics['average_sentence_words']}")

    article_lc = _strip_markup(variant.get("article_markdown")).lower()
    for item in variant.get("glossary") or []:
        term = safe_text(item.get("term")).lower()
        if term and term not in article_lc:
            blockers.append(f"glossary_term_missing_from_article:{term}")

    used_ids = variant.get("used_evidence_ids") or []
    invalid_ids = [item for item in used_ids if item not in allowed_ids]
    if invalid_ids:
        blockers.append("unsupported_evidence_ids:" + ",".join(invalid_ids[:5]))
    if len(set(used_ids)) < min(band["min_evidence"], len(allowed_ids)):
        blockers.append(f"insufficient_evidence_ids:{len(set(used_ids))}")
    return blockers, metrics


def _verify_variant(worker: DailyStoryWorker, story: Dict[str, Any], variant: Dict[str, Any], band_key: str) -> Dict[str, Any]:
    evidence_blob = "\n".join(
        f"[{row['evidence_id']}] {row['excerpt']}" for row in _evidence_rows(story)
    )
    prompt = f"""Validate this student editorial adaptation.

Return ok=false if it adds unsupported facts, changes the canonical meaning, impersonates a scholar,
contains unsafe or age-inappropriate framing, or does not actually suit {GRADE_BANDS[band_key]['label']}.
Readability numbers are only one signal; also judge conceptual density, assumed prior knowledge,
language clarity, scaffolding, and the learning task. Return JSON only.

Canonical article:
{safe_text((story.get('content') or {}).get('article_markdown'))}

Evidence excerpts:
{evidence_blob}

Variant:
{json.dumps(variant, ensure_ascii=False)}
"""
    payload = _call_structured(worker, prompt, VERIFIER_SCHEMA, 1400)
    return {
        "ok": bool(payload.get("ok")),
        "blockers": [safe_text(item) for item in (payload.get("blockers") or []) if safe_text(item)],
        "notes": [safe_text(item) for item in (payload.get("notes") or []) if safe_text(item)],
    }


def _existing_variant(story: Dict[str, Any], band_key: str) -> Optional[Dict[str, Any]]:
    for variant in story.get("audience_variants") or []:
        if isinstance(variant, dict) and safe_text(variant.get("band_key")) == band_key:
            return variant
    return None


def _replace_variant(variants: List[Dict[str, Any]], next_variant: Dict[str, Any]) -> List[Dict[str, Any]]:
    filtered = [item for item in variants if safe_text(item.get("band_key")) != next_variant["band_key"]]
    filtered.append(next_variant)
    order = {key: index for index, key in enumerate(GRADE_BANDS)}
    return sorted(filtered, key=lambda item: order.get(safe_text(item.get("band_key")), len(order)))


def run_personalization(*, story_date: str, max_stories: int, dry_run: bool, no_llm: bool, scholar_id: str = "") -> Dict[str, Any]:
    worker = DailyStoryWorker(use_llm=not (dry_run or no_llm), trends_enabled=False)
    try:
        audience = resolve_active_audience(worker)
        story_filter: Dict[str, Any] = {
            "story_date": story_date,
            "status": {"$in": ["pending_review", "generated"]},
        }
        if safe_text(scholar_id):
            story_filter["scholar.profile_id"] = safe_text(scholar_id)
        stories = list(
            worker.stories_collection.find(
                story_filter
            ).sort([("_id", 1)]).limit(max(1, max_stories))
        )
        summary = {
            "stories_scanned": len(stories),
            "bands_required": len(audience["band_keys"]),
            "variants_required": len(stories) * len(audience["band_keys"]),
            "generated": 0,
            "skipped_current": 0,
            "failed": 0,
        }
        failures: List[Dict[str, str]] = []

        if dry_run or no_llm:
            return {
                "status": "dry_run" if dry_run else "completed_no_llm",
                "story_date": story_date,
                "audience": audience,
                "summary": summary,
                "failures": failures,
            }

        for story in stories:
            canonical_hash = _content_hash(story)
            variants = list(story.get("audience_variants") or [])
            allowed_ids = {row["evidence_id"] for row in _evidence_rows(story)}
            for band_key in audience["band_keys"]:
                existing = _existing_variant(story, band_key)
                existing_hash_is_current = bool(
                    existing and existing.get("canonical_content_hash") == canonical_hash
                )
                existing_prompt_is_compatible = bool(
                    existing and existing.get("prompt_version") in READY_COMPATIBLE_PROMPT_VERSIONS
                )
                if existing_hash_is_current and existing_prompt_is_compatible:
                    upgraded_candidate = _coerce_variant(existing or {})
                    disclosure_added = _preserve_canonical_disclosure(story, upgraded_candidate)
                    if disclosure_added or existing.get("prompt_version") != PROMPT_VERSION:
                        upgraded_candidate["used_evidence_ids"] = sorted(allowed_ids)
                        normalized_article = _strip_markup(upgraded_candidate.get("article_markdown")).lower()
                        upgraded_candidate["glossary"] = [
                            item for item in upgraded_candidate.get("glossary") or []
                            if safe_text(item.get("definition"))
                            and safe_text(item.get("term")).lower() in normalized_article
                        ]
                        upgrade_blockers, upgrade_metrics = _validate_variant(
                            upgraded_candidate,
                            band_key,
                            allowed_ids,
                        )
                        if not upgrade_blockers and existing.get("status") == "ready" and (existing.get("validation") or {}).get("ok"):
                            next_variant = {
                                **(existing or {}),
                                **upgraded_candidate,
                                "readability": upgrade_metrics,
                                "prompt_version": PROMPT_VERSION,
                                "validation": {
                                    **((existing or {}).get("validation") or {}),
                                    "ok": True,
                                    "blockers": [],
                                    "notes": list(dict.fromkeys(
                                        list(((existing or {}).get("validation") or {}).get("notes") or [])
                                        + ["canonical_disclosure_preserved_v3"]
                                    )),
                                    "validated_at": utc_now_iso(),
                                },
                            }
                            variants = _replace_variant(variants, next_variant)
                            summary["skipped_current"] += 1
                            continue
                        if not upgrade_blockers and existing.get("status") == "failed":
                            upgrade_verifier = _verify_variant(worker, story, upgraded_candidate, band_key)
                            if upgrade_verifier["ok"] and not upgrade_verifier["blockers"]:
                                next_variant = {
                                    **(existing or {}),
                                    **upgraded_candidate,
                                    "readability": upgrade_metrics,
                                    "status": "ready",
                                    "prompt_version": PROMPT_VERSION,
                                    "validation": {
                                        "ok": True,
                                        "blockers": [],
                                        "notes": upgrade_verifier["notes"],
                                        "validated_at": utc_now_iso(),
                                        "validator": "deterministic_plus_vertex_v1",
                                    },
                                }
                                variants = _replace_variant(variants, next_variant)
                                summary["generated"] += 1
                                continue
                if (
                    existing
                    and existing.get("status") == "ready"
                    and existing.get("canonical_content_hash") == canonical_hash
                    and existing.get("prompt_version") in READY_COMPATIBLE_PROMPT_VERSIONS
                ):
                    summary["skipped_current"] += 1
                    continue

                existing_is_current = bool(
                    existing
                    and existing.get("canonical_content_hash") == canonical_hash
                    and existing.get("prompt_version") == PROMPT_VERSION
                )
                prior_attempts = int(existing.get("generation_attempts") or 0) if existing_is_current else 0
                prior_blockers: List[str] = [
                    safe_text(item)
                    for item in (((existing or {}).get("validation") or {}).get("blockers") or [])
                    if safe_text(item)
                ] if existing_is_current else []
                next_variant: Optional[Dict[str, Any]] = None
                should_generate = True
                if not allowed_ids:
                    should_generate = False
                    next_variant = {
                        "band_key": band_key,
                        "band_label": GRADE_BANDS[band_key]["label"],
                        "min_grade_level": GRADE_BANDS[band_key]["min_level"],
                        "max_grade_level": GRADE_BANDS[band_key]["max_level"],
                        "typical_age_min": GRADE_BANDS[band_key]["typical_age_min"],
                        "typical_age_max": GRADE_BANDS[band_key]["typical_age_max"],
                        "status": "failed",
                        "validation": {
                            "ok": False,
                            "blockers": ["canonical_story_has_no_evidence_ids"],
                            "notes": [],
                            "validated_at": utc_now_iso(),
                            "validator": "deterministic_plus_vertex_v1",
                        },
                        "generation_attempts": 0,
                        "canonical_content_hash": canonical_hash,
                        "prompt_version": PROMPT_VERSION,
                        "generated_at": utc_now_iso(),
                        "source_story_key": safe_text(story.get("story_key")),
                    }
                elif prior_attempts >= MAX_TOTAL_GENERATION_ATTEMPTS:
                    should_generate = False
                    candidate = _coerce_variant(existing or {})
                    _preserve_canonical_disclosure(story, candidate)
                    candidate["used_evidence_ids"] = sorted(allowed_ids)
                    normalized_article = _strip_markup(candidate.get("article_markdown")).lower()
                    candidate["glossary"] = [
                        item for item in candidate.get("glossary") or []
                        if safe_text(item.get("definition"))
                        and safe_text(item.get("term")).lower() in normalized_article
                    ]
                    deterministic_blockers, metrics = _validate_variant(candidate, band_key, allowed_ids)
                    verifier = {"ok": False, "blockers": [], "notes": []}
                    if not deterministic_blockers:
                        verifier = _verify_variant(worker, story, candidate, band_key)
                    combined_blockers = list(dict.fromkeys(deterministic_blockers + verifier["blockers"]))
                    next_variant = {
                        **(existing or {}),
                        **candidate,
                        "readability": metrics,
                        "status": "ready" if not combined_blockers and verifier["ok"] else "failed",
                        "validation": {
                            "ok": not combined_blockers and verifier["ok"],
                            "blockers": combined_blockers,
                            "notes": verifier["notes"],
                            "validated_at": utc_now_iso(),
                            "validator": "deterministic_plus_vertex_v1",
                        },
                    }

                for local_attempt in range(1, 4) if should_generate else []:
                    attempt = prior_attempts + local_attempt
                    if attempt > MAX_TOTAL_GENERATION_ATTEMPTS:
                        break
                    try:
                        payload = _call_structured(
                            worker,
                            _build_prompt(story, band_key, prior_blockers),
                            VARIANT_SCHEMA,
                            5200,
                        )
                        candidate = _coerce_variant(payload)
                        _preserve_canonical_disclosure(story, candidate)
                        # A variant inherits the canonical story's complete evidence lineage.
                        # Model-provided IDs are never trusted as the source of record.
                        candidate["used_evidence_ids"] = sorted(allowed_ids)
                        normalized_article = _strip_markup(candidate.get("article_markdown")).lower()
                        candidate["glossary"] = [
                            item for item in candidate.get("glossary") or []
                            if safe_text(item.get("definition"))
                            and safe_text(item.get("term")).lower() in normalized_article
                        ]
                        deterministic_blockers, metrics = _validate_variant(candidate, band_key, allowed_ids)
                        candidate["readability"] = metrics
                        verifier = {"ok": False, "blockers": [], "notes": []}
                        if not deterministic_blockers:
                            verifier = _verify_variant(worker, story, candidate, band_key)
                        combined_blockers = list(dict.fromkeys(deterministic_blockers + verifier["blockers"]))
                        next_variant = {
                            **candidate,
                            "band_key": band_key,
                            "band_label": GRADE_BANDS[band_key]["label"],
                            "min_grade_level": GRADE_BANDS[band_key]["min_level"],
                            "max_grade_level": GRADE_BANDS[band_key]["max_level"],
                            "typical_age_min": GRADE_BANDS[band_key]["typical_age_min"],
                            "typical_age_max": GRADE_BANDS[band_key]["typical_age_max"],
                            "status": "ready" if not combined_blockers and verifier["ok"] else "failed",
                            "validation": {
                                "ok": not combined_blockers and verifier["ok"],
                                "blockers": combined_blockers,
                                "notes": verifier["notes"],
                                "validated_at": utc_now_iso(),
                                "validator": "deterministic_plus_vertex_v1",
                            },
                            "generation_attempts": attempt,
                            "canonical_content_hash": canonical_hash,
                            "prompt_version": PROMPT_VERSION,
                            "generated_at": utc_now_iso(),
                            "source_story_key": safe_text(story.get("story_key")),
                            "source_urls": list((story.get("content") or {}).get("source_urls") or []),
                        }
                        if next_variant["status"] == "ready":
                            break
                        prior_blockers = combined_blockers
                    except Exception as exc:  # pragma: no cover - external service behavior
                        prior_blockers = [safe_text(exc)]
                        next_variant = {
                            "band_key": band_key,
                            "band_label": GRADE_BANDS[band_key]["label"],
                            "status": "failed",
                            "validation": {
                                "ok": False,
                                "blockers": prior_blockers,
                                "notes": [],
                                "validated_at": utc_now_iso(),
                                "validator": "deterministic_plus_vertex_v1",
                            },
                            "generation_attempts": attempt,
                            "canonical_content_hash": canonical_hash,
                            "prompt_version": PROMPT_VERSION,
                            "generated_at": utc_now_iso(),
                            "source_story_key": safe_text(story.get("story_key")),
                        }

                if next_variant is None:
                    continue
                variants = _replace_variant(variants, next_variant)
                if next_variant["status"] == "ready":
                    summary["generated"] += 1
                else:
                    summary["failed"] += 1
                    failures.append({
                        "story_key": safe_text(story.get("story_key")),
                        "band_key": band_key,
                        "error": " | ".join(next_variant.get("validation", {}).get("blockers") or [])[:1200],
                    })

            worker.stories_collection.update_one(
                {"_id": story["_id"]},
                {"$set": {
                    "audience_contract": audience,
                    "audience_variants": variants,
                    "personalization.updated_at": utc_now_iso(),
                    "personalization.prompt_version": PROMPT_VERSION,
                    "personalization.ready": all(
                        _existing_variant({"audience_variants": variants}, key)
                        and _existing_variant({"audience_variants": variants}, key).get("status") == "ready"
                        for key in audience["band_keys"]
                    ),
                }},
            )

        status = "completed" if not failures else "completed_with_failures"
        return {
            "status": status,
            "story_date": story_date,
            "audience": audience,
            "summary": summary,
            "failures": failures[:50],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        worker.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate grade-banded editorial access variants.")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--max-stories", type=int, default=25)
    parser.add_argument("--scholar-id", type=str, default="", help="Optional profile_id for bounded recovery")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_personalization(
        story_date=parse_date(args.date).isoformat(),
        max_stories=max(1, args.max_stories),
        dry_run=args.dry_run,
        no_llm=args.no_llm,
        scholar_id=args.scholar_id,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result.get("status") == "completed_with_failures" else 0


if __name__ == "__main__":
    raise SystemExit(main())
