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
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from daily_story_worker import DailyStoryWorker, parse_date, safe_text, utc_now_iso


AUDIENCE_CONTRACT_VERSION = "active-student-audience-v1"
PROMPT_VERSION = "editorial-access-ladder-v4"
READY_COMPATIBLE_PROMPT_VERSIONS = {
    "editorial-access-ladder-v1",
    "editorial-access-ladder-v2",
    "editorial-access-ladder-v3",
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
    """Puts the canonical story's AI disclosure at the top of the version's
    article. An empty article stays empty: given the disclosure alone, a
    version the model never wrote read as a 22-word article (2026-11-01)."""
    disclosure = safe_text((story.get("content") or {}).get("disclosure"))
    article = safe_text(candidate.get("article_markdown"))
    if not disclosure or not article or disclosure.lower() in article.lower():
        return False
    candidate["article_markdown"] = f"> {disclosure}\n\n{article}".strip()
    return True


def _without_disclosure(article: Any, disclosure: str) -> str:
    """The version's article without the disclosure line the code put at its
    top: the reading measures and word counts are of what the model wrote,
    the boilerplate disclosure (~22 words) not counted against its limit."""
    text = safe_text(article)
    disclosure = safe_text(disclosure)
    if not disclosure:
        return text
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lstrip(">").strip().lower() == disclosure.lower():
            return "\n".join(lines[index + 1:]).strip()
        break
    return text


def _call_structured(worker: DailyStoryWorker, prompt: str, schema: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    """One whole JSON object from the model (DailyStoryWorker._call_structured_json:
    a reply missing any required field, such as the remains of one cut off at
    the token limit, is never taken as the answer)."""
    try:
        return worker._call_structured_json(prompt, schema, max_tokens)
    except RuntimeError as exc:
        raise RuntimeError(f"Structured personalization call failed: {exc}") from exc


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


# Blockers a complete version can be revised out of, keeping the version: its
# length, reading level and counts. Anything else (a missing field, a model
# verifier's finding) is written again from the canonical article.
REVISABLE_BLOCKER_PREFIXES = (
    "word_count_out_of_range:",
    "readability_out_of_range:",
    "average_sentence_too_long:",
    "glossary_count_out_of_range",
    "glossary_term_missing_from_article:",
    "takeaway_count_out_of_range",
    "question_count_out_of_range",
)


def _story_disclosure(story: Dict[str, Any]) -> str:
    return safe_text((story.get("content") or {}).get("disclosure"))


def _revisable(candidate: Dict[str, Any], blockers: List[str]) -> bool:
    """Whether the next attempt revises this version rather than writing a
    new one: it is complete, and every blocker is one a revision can fix."""
    if not blockers:
        return False
    if any(not safe_text(candidate.get(field)) for field in ("title", "summary", "article_markdown", "teacher_note")):
        return False
    return all(blocker.startswith(REVISABLE_BLOCKER_PREFIXES) for blocker in blockers)


def _revision_fixes(band_key: str, article: str, blockers: Iterable[str]) -> List[str]:
    """What the revision must change, in numbers the model can act on."""
    band = GRADE_BANDS[band_key]
    metrics = readability_metrics(article)
    fixes: List[str] = []
    for blocker in blockers:
        if blocker.startswith("word_count_out_of_range:"):
            words = metrics["word_count"]
            if words > band["max_words"]:
                fixes.append(
                    f"The article has {words} words; the most is {band['max_words']}. Cut it to about "
                    f"{int(band['max_words'] * 0.9)} words by removing repetition and the least important details. Add nothing."
                )
            else:
                fixes.append(
                    f"The article has {words} words; the least is {band['min_words']}. Expand it to about "
                    f"{int(band['min_words'] * 1.15)} words, using only the canonical article and the evidence excerpts."
                )
        elif blocker.startswith("readability_out_of_range:"):
            grade = metrics["flesch_kincaid_grade"]
            if grade > band["max_grade"]:
                fixes.append(f"Its reading level measures {grade}; the most is {band['max_grade']}. Use shorter sentences and plainer words.")
            else:
                fixes.append(f"Its reading level measures {grade}; the least is {band['min_grade']}. Use fuller sentences and the subject's own terms, defined.")
        elif blocker.startswith("average_sentence_too_long:"):
            fixes.append(
                f"Its sentences average {metrics['average_sentence_words']} words; keep the average at or under "
                f"{band['max_sentence_words']} by splitting long sentences."
            )
        elif blocker.startswith("glossary_"):
            fixes.append("Give 3 to 6 glossary terms, each a word or phrase that appears in the article exactly as written there.")
        elif blocker.startswith("takeaway_count_out_of_range"):
            fixes.append("Give 3 to 5 key takeaways.")
        elif blocker.startswith("question_count_out_of_range"):
            fixes.append("Give 3 to 5 comprehension questions, each with answer guidance.")
    return list(dict.fromkeys(fixes))


def _build_revision_prompt(story: Dict[str, Any], band_key: str, candidate: Dict[str, Any], blockers: Iterable[str]) -> str:
    """Revise a complete version that failed only on length, reading level or
    counts, instead of writing it again: a grade 9-12 version 40 words over
    its limit (2026-11-01) needs a trim, not a new draft."""
    band = GRADE_BANDS[band_key]
    content = story.get("content") or {}
    article = _without_disclosure(candidate.get("article_markdown"), _story_disclosure(story))
    version = {
        "title": safe_text(candidate.get("title")),
        "summary": safe_text(candidate.get("summary")),
        "article_markdown": article,
        "glossary": candidate.get("glossary") or [],
        "key_takeaways": candidate.get("key_takeaways") or [],
        "comprehension_questions": candidate.get("comprehension_questions") or [],
        "teacher_note": safe_text(candidate.get("teacher_note")),
        "used_evidence_ids": candidate.get("used_evidence_ids") or [],
    }
    fixes = "\n".join(f"- {fix}" for fix in _revision_fixes(band_key, article, blockers)) or "- none"
    evidence_blob = "\n".join(f"[{row['evidence_id']}] {row['excerpt']}" for row in _evidence_rows(story))
    return f"""Revise this {band['label']} version of a grounded editorial so it meets its contract.
Change only what the fixes below need; keep everything else as it is. Do not add facts, identities,
motivations, examples, or advice absent from the canonical article and evidence excerpts. Do not
impersonate the scholar.

Contract:
- {band['min_words']}-{band['max_words']} article words
- Flesch-Kincaid estimate {band['min_grade']}-{band['max_grade']}
- average sentence length no more than {band['max_sentence_words']} words
- 3-6 glossary terms used in the article, 3-5 takeaways, 3-5 comprehension questions

Fixes needed:
{fixes}

The version to revise (JSON):
{json.dumps(version, ensure_ascii=False)}

Canonical title: {safe_text(content.get('title'))}
Canonical article:
{safe_text(content.get('article_markdown'))}

Allowed evidence excerpts:
{evidence_blob}

Return the whole revised version, every field, as JSON only.
"""


def _validate_variant(
    variant: Dict[str, Any],
    band_key: str,
    allowed_ids: set[str],
    disclosure: str = "",
) -> Tuple[List[str], Dict[str, Any]]:
    """The version's blockers and reading measures. The measures are of the
    article without the disclosure line (_without_disclosure)."""
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

    metrics = readability_metrics(_without_disclosure(variant.get("article_markdown"), disclosure))
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


# ── Whether a band may meet the story's news at all ─────────────────────────
# A legend story answers today's news through a legend's ideas. Whether a band
# may meet that news is decided by the model, per story, under these rules,
# before any version is written. The variant verifier judges only how a
# rewrite is framed, so a drone strike that killed 41 people could be framed
# gently for kindergarten and pass (2026-10-29). A band the topic does not
# suit gets no version, and the reader app shows that band nothing.
TOPIC_POLICY_VERSION = "legend-topic-suitability-v1"
TOPIC_SUITABILITY_RULES: Dict[str, str] = {
    "grade_k_2": (
        "Only what a young child meets in daily life or a picture book: nature, animals, space, "
        "inventions and how things work, art, music, sport, community helpers, kindness and fairness "
        "among friends, celebrations. Never war or armed conflict, terrorism, killing or death by "
        "violence, disasters with casualties, crime, abuse, illness that frightens, political fights "
        "or elections, or anything frightening."
    ),
    "grade_3_5": (
        "Science and discovery, the environment, health in general terms, inventions, culture, sport, "
        "and how civic life works (what laws, voting or courts are for), fairness and rights in "
        "general terms. Not war or armed conflict, terrorism, killings, disasters with casualties, "
        "crime and violence, graphic illness, or partisan political fights."
    ),
    "grade_6_8": (
        "What a middle-school social studies class discusses: elections and government, human "
        "rights, climate, migration, the economy, and conflicts explained through causes and "
        "consequences. Not graphic violence or casualty detail, sexual content, self-harm, or a "
        "story whose subject is the suffering itself rather than understanding it."
    ),
    "grade_9_12": (
        "Most news a high-school current-events class discusses, war and conflict included, handled "
        "analytically and without graphic detail. Not sexual content, detailed violence, self-harm "
        "methods, or content that targets a group of people."
    ),
}
TOPIC_VERDICT_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "suitable": {"type": "BOOLEAN"},
        "reason": {"type": "STRING"},
    },
    "required": ["suitable", "reason"],
}


def topic_prompt(story: Dict[str, Any], band_key: str) -> str:
    band = GRADE_BANDS[band_key]
    content = story.get("content") or {}
    trend = story.get("trend_issue") or {}
    opening = _strip_markup(content.get("article_markdown"))[:1500]
    return f"""You decide whether a news topic may be brought to {band['label']} readers (typically
ages {band['typical_age_min']} to {band['typical_age_max']}) on a children's reading platform, as the subject of
an article that looks at today's news through a historical scholar's ideas.

Rules for this band:
{TOPIC_SUITABILITY_RULES[band_key]}

Decide on the topic and what a reader would have to engage with, not on the writing. An article
can be gentle and still be about something this band should not be given. Return JSON only:
suitable (true or false) and reason (one sentence a reviewer can read).

News headline it answers: {safe_text(trend.get('title')) or 'none'}
Article title: {safe_text(content.get('title'))}
Standfirst: {safe_text(content.get('standfirst'))}
Opening: {opening}
"""


def judge_topic_for_band(worker: DailyStoryWorker, story: Dict[str, Any], band_key: str, canonical_hash: str) -> Dict[str, Any]:
    """The model's ruling on one band. A band with no rules is never suitable."""
    base = {
        "canonical_content_hash": canonical_hash,
        "policy_version": TOPIC_POLICY_VERSION,
        "judged_at": utc_now_iso(),
    }
    if band_key not in TOPIC_SUITABILITY_RULES:
        return {**base, "suitable": False, "reason": "No topic rules for this band.", "judge": "policy"}
    payload = _call_structured(worker, topic_prompt(story, band_key), TOPIC_VERDICT_SCHEMA, 1200)
    if not isinstance(payload.get("suitable"), bool):
        raise RuntimeError("topic judge returned no verdict")
    return {
        **base,
        "suitable": payload["suitable"],
        "reason": safe_text(payload.get("reason"))[:400],
        "judge": safe_text(getattr(worker, "model_name", "")) or "vertex",
    }


def current_topic_verdict(suitability: Dict[str, Any], band_key: str, canonical_hash: str) -> Optional[Dict[str, Any]]:
    """A ruling already made for this story text under these rules, or None."""
    verdict = suitability.get(band_key)
    if (
        isinstance(verdict, dict)
        and isinstance(verdict.get("suitable"), bool)
        and verdict.get("canonical_content_hash") == canonical_hash
        and verdict.get("policy_version") == TOPIC_POLICY_VERSION
    ):
        return verdict
    return None


def not_suitable_variant(story: Dict[str, Any], band_key: str, verdict: Dict[str, Any], canonical_hash: str) -> Dict[str, Any]:
    band = GRADE_BANDS[band_key]
    return {
        "band_key": band_key,
        "band_label": band["label"],
        "min_grade_level": band["min_level"],
        "max_grade_level": band["max_level"],
        "typical_age_min": band["typical_age_min"],
        "typical_age_max": band["typical_age_max"],
        "status": "not_suitable",
        "topic_suitability": verdict,
        "canonical_content_hash": canonical_hash,
        "prompt_version": PROMPT_VERSION,
        "generated_at": utc_now_iso(),
        "source_story_key": safe_text(story.get("story_key")),
    }


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
            "not_suitable": 0,
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
            suitability = dict(story.get("audience_suitability") or {})
            for band_key in audience["band_keys"]:
                verdict = current_topic_verdict(suitability, band_key, canonical_hash)
                if verdict is None:
                    try:
                        verdict = judge_topic_for_band(worker, story, band_key, canonical_hash)
                    except Exception as exc:  # pragma: no cover - external service behavior
                        # No ruling, no version this run: the reader app serves a
                        # band only on a ruling that it suits, so nothing is shown
                        # in the meantime, and the next run asks again.
                        summary["failed"] += 1
                        failures.append({
                            "story_key": safe_text(story.get("story_key")),
                            "band_key": band_key,
                            "error": f"topic_suitability_unjudged:{safe_text(exc)}"[:1200],
                        })
                        continue
                    suitability[band_key] = verdict
                if not verdict["suitable"]:
                    variants = _replace_variant(variants, not_suitable_variant(story, band_key, verdict, canonical_hash))
                    summary["not_suitable"] += 1
                    continue
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
                            _story_disclosure(story),
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
                    deterministic_blockers, metrics = _validate_variant(candidate, band_key, allowed_ids, _story_disclosure(story))
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

                # A complete version that failed only on length, reading level
                # or counts is revised on the next attempt, not written again.
                revise_from: Optional[Dict[str, Any]] = None
                for local_attempt in range(1, 4) if should_generate else []:
                    attempt = prior_attempts + local_attempt
                    if attempt > MAX_TOTAL_GENERATION_ATTEMPTS:
                        break
                    try:
                        payload = _call_structured(
                            worker,
                            _build_revision_prompt(story, band_key, revise_from, prior_blockers)
                            if revise_from is not None
                            else _build_prompt(story, band_key, prior_blockers),
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
                        deterministic_blockers, metrics = _validate_variant(candidate, band_key, allowed_ids, _story_disclosure(story))
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
                        revise_from = candidate if _revisable(candidate, combined_blockers) else None
                    except Exception as exc:  # pragma: no cover - external service behavior
                        prior_blockers = [safe_text(exc)]
                        revise_from = None
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
                    "audience_suitability": suitability,
                    "personalization.updated_at": utc_now_iso(),
                    "personalization.prompt_version": PROMPT_VERSION,
                    # Every band settled: written, or ruled unsuitable.
                    "personalization.ready": all(
                        _existing_variant({"audience_variants": variants}, key)
                        and _existing_variant({"audience_variants": variants}, key).get("status") in ("ready", "not_suitable")
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
