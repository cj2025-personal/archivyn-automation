"""Whether a legend's stored source material describes that legend.

A legend's passages (rag_context.section_chunks) were collected from web pages
found by searching the legend's name, and a common name finds someone else.
On 2026-11-01 the passages of robert-williams (roster: Robert Williams,
psychology) described a basketball player, and those of edward-t-gordon
(psychology) the theatre designer Edward Gordon Craig; both legends passed
the profile-quality gate, which counts passages and cannot tell whose they
are. Names on the roster were misspelled too (Marie Daley for Marie Maynard
Daly, Fraizer for Frazier).

The legend_identity_review stage has a model judge each source page against
the legend we mean (the anchor: an administrator's confirmed identity, else
the roster entry, else the stored name alone) and keeps the verdict on the
legend as identity_review. Code only counts those verdicts into a status and
acts on it: a legend is written about only while its review is current (this
policy, these sources) and eligible, and only from the passages of pages the
model ruled are about the legend or are the legend's own work.

Everything here is pure: no database, no model.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

IDENTITY_POLICY_VERSION = "legend-identity-v1"

# A page's verdict, as the model gives it.
ABOUT_LEGEND = "about_legend"
DIFFERENT_PERSON = "different_person"
NOT_ABOUT_A_PERSON = "not_about_a_person"
UNCLEAR = "unclear"
PAGE_VERDICTS = (ABOUT_LEGEND, DIFFERENT_PERSON, NOT_ABOUT_A_PERSON, UNCLEAR)
# Pages beyond MAX_PAGES (the smallest) are not shown to the model, so their
# passages are never used: a legend with that many pages is rare, and an
# unreviewed page must not reach a story.
NOT_REVIEWED = "not_reviewed"

NAME_CHECKS = ("correct", "misspelled", "unknown")

# A legend's status, counted from its pages' verdicts.
VERIFIED = "verified"            # some pages about the legend, none about anyone else
MIXED = "mixed"                  # some about the legend, some about someone else
WRONG_PERSON = "wrong_person"    # none about the legend, some about someone else
UNVERIFIED = "unverified"        # no page could be confirmed either way
NO_SOURCES = "no_sources"        # nothing stored to check
ELIGIBLE_STATUSES = (VERIFIED, MIXED)

UNATTRIBUTED = "unattributed"
# The judge sees every passage it rules on: each page with all its passages
# (each cut to EXCERPT_CHARS), pages taken most passages first while the
# excerpts fit JUDGE_CHAR_BUDGET and the count MAX_PAGES. A page left out is
# not reviewed and none of its passages is used. A verdict on a page covers
# exactly the passages shown; the judge rules a page different_person if any
# of them is about someone else (judge_prompt).
MAX_PAGES = 40
EXCERPT_CHARS = 500
JUDGE_CHAR_BUDGET = 60_000
# Shorter passages are neither judged nor used.
MIN_PASSAGE_CHARS = 80

ROSTER_DIR = Path(__file__).resolve().parent / "legendary_scholars" / "final"


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ("" if value is None else str(value).strip())


def chunk_text(chunk: Dict[str, Any]) -> str:
    return _text(chunk.get("text")) or _text(chunk.get("summary"))


def page_key(url: Any) -> str:
    """A source page's identity: host and path, lower-case host, without
    query, fragment or trailing slash (so tracking parameters and http/https
    do not split one page in two). A link that cannot be parsed is its own
    page, keyed by its text."""
    text = _text(url)
    if not text:
        return UNATTRIBUTED
    try:
        parsed = urlparse(text if "://" in text else f"https://{text}")
        host = (parsed.hostname or "").lower()
    except ValueError:
        return f"unparsed:{text.lower()[:200]}"
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return f"unparsed:{text.lower()[:200]}"
    path = parsed.path.rstrip("/")
    return f"{host}{path}"


def chunk_links(chunk: Dict[str, Any]) -> List[str]:
    """Every link a passage carries, as the story worker reads them
    (_chunk_source_urls): primary_source_url, source_url, source_urls (a list
    or one string) and each source_refs[].source_url."""
    links: List[str] = []
    for field in ("primary_source_url", "source_url"):
        if _text(chunk.get(field)):
            links.append(_text(chunk.get(field)))
    urls = chunk.get("source_urls")
    if isinstance(urls, str):
        urls = [urls]
    if isinstance(urls, list):
        links += [_text(u) for u in urls if _text(u)]
    refs = chunk.get("source_refs")
    if isinstance(refs, list):
        links += [_text(r.get("source_url")) for r in refs if isinstance(r, dict) and _text(r.get("source_url"))]
    return list(dict.fromkeys(links))


def chunk_page_key(chunk: Dict[str, Any]) -> str:
    """The page a passage came from: its first link (primary source first)."""
    links = chunk_links(chunk)
    return page_key(links[0]) if links else UNATTRIBUTED


def _section_chunks(doc: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    section_chunks = (doc.get("rag_context") or {}).get("section_chunks") or {}
    if not isinstance(section_chunks, dict):
        return {}
    return {
        section: [c for c in chunks if isinstance(c, dict)]
        for section, chunks in section_chunks.items()
        if isinstance(chunks, list)
    }


def source_pages(doc: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """The legend's source pages, most passages first, as the model sees
    them: key, url, passage count, every passage's excerpt, and a digest of
    every passage's text and links. Returns (the pages shown, the keys of the
    pages beyond MAX_PAGES or JUDGE_CHAR_BUDGET)."""
    pages: Dict[str, Dict[str, Any]] = {}
    for section, chunks in _section_chunks(doc).items():
        for chunk in chunks:
            text = chunk_text(chunk)
            if len(text) < MIN_PASSAGE_CHARS:
                continue
            key = chunk_page_key(chunk)
            links = chunk_links(chunk)
            page = pages.setdefault(key, {
                "key": key,
                "url": links[0] if links else "",
                "passages": 0,
                "excerpts": [],
                "_digest": hashlib.sha256(),
            })
            page["passages"] += 1
            page["excerpts"].append(text[:EXCERPT_CHARS])
            page["_digest"].update(f"{section}\x1f{text}\x1f{json.dumps(links)}\x1e".encode("utf-8"))
    ordered = sorted(pages.values(), key=lambda p: (-p["passages"], p["key"]))
    shown: List[Dict[str, Any]] = []
    beyond: List[str] = []
    used = 0
    for page in ordered:
        page["digest"] = page.pop("_digest").hexdigest()
        size = sum(len(e) for e in page["excerpts"])
        if len(shown) < MAX_PAGES and used + size <= JUDGE_CHAR_BUDGET:
            shown.append(page)
            used += size
        else:
            beyond.append(page["key"])
    return shown, beyond


def sources_hash(doc: Dict[str, Any]) -> str:
    """Changes whenever any judged passage, its links, or the page it is
    credited to, does."""
    shown, beyond = source_pages(doc)
    payload = [[p["key"], p["passages"], p["digest"]] for p in shown] + [[k] for k in beyond]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def stored_name(doc: Dict[str, Any]) -> str:
    name = doc.get("name")
    return (
        _text(doc.get("professor_name"))
        or (_text(name.get("full")) if isinstance(name, dict) else "")
        or (_text(name.get("display")) if isinstance(name, dict) else "")
        or (_text(name) if isinstance(name, str) else "")
    )


def anchor_for(doc: Dict[str, Any], roster_dir: Optional[Path] = None) -> Dict[str, str]:
    """Who the legend is meant to be: an administrator's confirmed identity
    (identity.expected on the legend), else the roster entry the legend was
    collected from, else the stored name alone."""
    expected = (doc.get("identity") or {}).get("expected") or {}
    if isinstance(expected, dict) and _text(expected.get("name")):
        return {"name": _text(expected.get("name")), "field": _text(expected.get("field")), "from": "administrator"}
    profile_id = _text(doc.get("profile_id"))
    roster_dir = roster_dir or ROSTER_DIR
    if profile_id and re.fullmatch(r"[a-z0-9][a-z0-9._-]*", profile_id):
        meta_path = roster_dir / f"{profile_id}.meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = None
        if isinstance(meta, dict) and _text(meta.get("profile_name")):
            return {"name": _text(meta.get("profile_name")), "field": _text(meta.get("field")), "from": "roster"}
    return {"name": stored_name(doc), "field": "", "from": "name_only"}


def input_hash(*, anchor: Dict[str, str], doc_sources_hash: str, model: str, name: str) -> str:
    """What a review was given: a review whose input_hash matches is reused."""
    payload = {
        "policy": IDENTITY_POLICY_VERSION,
        "model": model,
        "name": name,
        "anchor": {k: anchor.get(k, "") for k in ("name", "field", "from")},
        "sources": doc_sources_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def identity_status(verdicts: List[str]) -> str:
    """The legend's status from its pages' verdicts (counting only)."""
    if not verdicts:
        return NO_SOURCES
    about = sum(1 for v in verdicts if v == ABOUT_LEGEND)
    other = sum(1 for v in verdicts if v == DIFFERENT_PERSON)
    if about and other:
        return MIXED
    if about:
        return VERIFIED
    if other:
        return WRONG_PERSON
    return UNVERIFIED


def attention_reasons(review: Dict[str, Any]) -> List[str]:
    """Why an administrator should look at this legend (empty: nothing to do)."""
    reasons: List[str] = []
    status = review.get("status")
    if status == WRONG_PERSON:
        reasons.append("Its stored sources describe someone else.")
    elif status == MIXED:
        excluded = sum(1 for p in review.get("pages") or [] if p.get("verdict") == DIFFERENT_PERSON)
        reasons.append(f"{excluded} of its source pages describe someone else; they are left out of its stories.")
    elif status == UNVERIFIED:
        reasons.append("None of its source pages could be confirmed to be about this legend.")
    elif status == NO_SOURCES:
        reasons.append("It has no stored source passages.")
    if review.get("name_check") == "misspelled":
        usual = _text(review.get("usual_spelling"))
        reasons.append(f"Its name looks misspelled{f' (usually: {usual})' if usual else ''}.")
    if (review.get("anchor") or {}).get("from") == "name_only":
        reasons.append("No roster entry or confirmed identity: checked against its name alone.")
    return reasons


def build_review(
    *,
    doc: Dict[str, Any],
    anchor: Dict[str, str],
    pages: List[Dict[str, Any]],
    beyond: List[str],
    verdict: Dict[str, Any],
    page_ids: Dict[str, str],
    model: str,
    doc_sources_hash: str,
    review_input_hash: str,
    reviewed_at: str,
) -> Dict[str, Any]:
    """The identity_review kept on the legend, from the model's verdict.
    A page the model gave no valid verdict for counts as unclear; pages
    beyond MAX_PAGES as not reviewed. Only about_legend pages are allowed."""
    by_page: Dict[str, Dict[str, Any]] = {}
    for row in verdict.get("pages") or []:
        if not isinstance(row, dict):
            continue
        key = page_ids.get(_text(row.get("page_id")).upper())
        if key is None or key in by_page:
            continue
        given = _text(row.get("verdict")).lower()
        by_page[key] = {
            "verdict": given if given in PAGE_VERDICTS else UNCLEAR,
            "person_described": _text(row.get("person_described"))[:200],
            "reason": _text(row.get("reason"))[:300],
        }
    reviewed_pages = []
    for page in pages:
        row = by_page.get(page["key"]) or {"verdict": UNCLEAR, "person_described": "", "reason": "no verdict given"}
        reviewed_pages.append({"key": page["key"], "url": page["url"], "passages": page["passages"], **row})
    reviewed_pages += [{"key": key, "url": "", "passages": None, "verdict": NOT_REVIEWED, "person_described": "", "reason": "not shown to the judge: past the page or size limit"} for key in beyond]

    status = identity_status([p["verdict"] for p in reviewed_pages if p["verdict"] != NOT_REVIEWED])
    name_check = _text(verdict.get("name_check")).lower()
    name_check = name_check if name_check in NAME_CHECKS else "unknown"
    intended = verdict.get("intended_person") if isinstance(verdict.get("intended_person"), dict) else {}
    review = {
        "policy_version": IDENTITY_POLICY_VERSION,
        "status": status,
        "eligible": status in ELIGIBLE_STATUSES and name_check != "misspelled",
        "name_reviewed": stored_name(doc),
        "name_check": name_check,
        "usual_spelling": _text(verdict.get("usual_spelling"))[:120] if name_check == "misspelled" else "",
        "intended_person": {k: _text(intended.get(k))[:200] for k in ("full_name", "field", "years", "known_for")},
        "anchor": anchor,
        "pages": reviewed_pages,
        "allowed_page_keys": [p["key"] for p in reviewed_pages if p["verdict"] == ABOUT_LEGEND],
        "sources_hash": doc_sources_hash,
        "input_hash": review_input_hash,
        "model": model,
        "reviewed_at": reviewed_at,
    }
    review["attention"] = attention_reasons(review)
    return review


def _anchor_key(anchor: Any) -> Tuple[str, str, str]:
    anchor = anchor if isinstance(anchor, dict) else {}
    return tuple(_text(anchor.get(k)) for k in ("name", "field", "from"))  # type: ignore[return-value]


def current_review(doc: Dict[str, Any], roster_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The legend's identity review if it still applies: this policy, the
    sources it judged are the sources stored now, and it judged them against
    the legend's present name and anchor (an administrator's correction makes
    the old review stale)."""
    review = doc.get("identity_review")
    if not isinstance(review, dict) or review.get("policy_version") != IDENTITY_POLICY_VERSION:
        return None
    if review.get("sources_hash") != sources_hash(doc):
        return None
    if _text(review.get("name_reviewed")) != stored_name(doc):
        return None
    if _anchor_key(review.get("anchor")) != _anchor_key(anchor_for(doc, roster_dir)):
        return None
    return review


def story_eligibility(doc: Dict[str, Any]) -> Tuple[bool, str]:
    """Whether a story may be written about this legend now, and if not why:
    not_reviewed, stale (its sources, name, anchor or the policy changed
    since), check_failed (the legend's data could not be read), or the
    status. Never raises: a legend that cannot be checked is not written about."""
    review = doc.get("identity_review")
    if not isinstance(review, dict):
        return False, "not_reviewed"
    try:
        current = current_review(doc)
    except Exception:
        return False, "check_failed"
    if current is None:
        return False, "stale"
    if not review.get("eligible"):
        if review.get("status") in ELIGIBLE_STATUSES and review.get("name_check") == "misspelled":
            return False, "name_misspelled"
        return False, _text(review.get("status")) or "not_eligible"
    return True, _text(review.get("status"))


def _with_allowed_links(chunk: Dict[str, Any], allowed: set) -> Dict[str, Any]:
    """A copy of a kept passage carrying only links to allowed pages: a
    passage's other links (source_urls, source_refs) can point at a page the
    judge ruled about someone else, and they become a story's citations."""
    out = dict(chunk)
    for field in ("primary_source_url", "source_url"):
        if _text(out.get(field)) and page_key(out[field]) not in allowed:
            out[field] = ""
    urls = out.get("source_urls")
    if isinstance(urls, str):
        urls = [urls]
    if isinstance(urls, list):
        out["source_urls"] = [u for u in urls if _text(u) and page_key(u) in allowed]
    refs = out.get("source_refs")
    if isinstance(refs, list):
        out["source_refs"] = [r for r in refs if isinstance(r, dict) and page_key(r.get("source_url")) in allowed]
    return out


def restricted_to_reviewed_sources(doc: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of the legend holding only the passages its current review
    judged and allowed, each with its links to excluded pages removed. Where
    any page was left out, text built from all pages together (section_text,
    section_summaries) and stored links are dropped too, since they may carry
    the other person's details. The stored legend is not changed."""
    review = current_review(doc) or {}
    allowed = set(review.get("allowed_page_keys") or [])
    out = copy.copy(doc)
    out.pop("identity_review", None)  # the verdict's own record (it names the excluded pages)
    rag_context = dict(doc.get("rag_context") or {})
    kept: Dict[str, List[Dict[str, Any]]] = {}
    dropped = 0
    for section, chunks in _section_chunks(doc).items():
        judged = [c for c in chunks if len(chunk_text(c)) >= MIN_PASSAGE_CHARS]
        keep = [_with_allowed_links(c, allowed) for c in judged if chunk_page_key(c) in allowed]
        dropped += len(judged) - len(keep)
        if keep:
            kept[section] = keep
    rag_context["section_chunks"] = kept
    excluded_pages = any(p.get("verdict") != ABOUT_LEGEND for p in review.get("pages") or [])
    if dropped or excluded_pages or not review:
        rag_context["section_text"] = {}
        rag_context["section_summaries"] = {}
        out["links_and_media"] = {}
    out["rag_context"] = rag_context
    return out
