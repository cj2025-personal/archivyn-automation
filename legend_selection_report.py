"""Which legends to write for first, for readers in grades 3 to 5.

Pilot plan B2: the owner and the professor pick 15 to 20 legends to give
grade 3-5 readers stories they can read. This ranks the legends cleared by
their identity review, from what is already stored, and says why, so the
pick is theirs and quick:

- sources: the passages of pages the identity review ruled about the legend
  (legend_identity), how many, from how many pages, with a link to credit;
- licences: how much of that comes from pages whose terms are known to allow
  use with credit (source_licences: an open licence, or a US federal work);
- writable now: the same gates the story writer applies (identity review
  current and eligible, the profile-quality gate, at least four passages to
  write from);
- child suitability already ruled: the grade 3-5 (and K-2) topic rulings
  the versions stage made on the legend's stories, and how its grade 3-5
  versions landed (ready, failed, ruled not suitable);
- what a young reader sees today: the adult short bio's reading grade, and
  whether a grade 3-5 bio is stored.

The score is arithmetic over those signals (SCORE_WEIGHTS); it orders the
list and decides nothing. The field and the reasons are there for the
people choosing.

── READ-ONLY, ENFORCED IN CODE ───────────────────────────────────────────────
It is meant to be run against the production database, so that is not left
to care: every collection is a readonly_mongo.ReadOnlyCollection (find,
find_one, count_documents, distinct and aggregate without $out or $merge;
anything else raises before a command is built), reads go to a secondary
where there is one, and no .env file is loaded: the address comes from
LEGEND_REPORT_MONGO_URI only. A read-only database user makes it hold at the
server too.

Running it (PowerShell 5.1, from this folder):
  $env:LEGEND_REPORT_MONGO_URI = (gcloud secrets versions access latest --secret=MONGODB_URI)
  python legend_selection_report.py
  Remove-Item Env:LEGEND_REPORT_MONGO_URI
LEGEND_REPORT_DB names the database when the address does not (the pipeline's
default is ngo_profiles). --json prints the rows as JSON; --top N shows N.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

import legend_identity
from daily_story_personalize import readability_metrics
from daily_story_worker import creditable_source
from legend_child_bio import BIO_BAND, learner_api_fk_grade
from readonly_mongo import connect_read_only
from source_licences import OPEN, REFUSED, licence_for

URI_ENV = "LEGEND_REPORT_MONGO_URI"
DB_ENV = "LEGEND_REPORT_DB"
BAND = BIO_BAND
YOUNGER_BAND = "grade_k_2"

# The story writer needs at least this many passages (daily_story_worker run
# and write_evergreen_slot); the admin's fixed rules escalate a story citing
# fewer than two source pages (editorialStoryRisk.js).
MIN_PASSAGES = 4
MIN_PAGES = 2
# What "deep enough" means for the score: passages and pages past these add
# nothing more. Five stories a legend need distinct focuses and sources.
FULL_DEPTH_PASSAGES = 60
FULL_PAGES = 6
SCORE_WEIGHTS = {
    "depth": 0.30,
    "pages": 0.15,
    "open_licence": 0.20,
    "ruled_suitable": 0.25,
    "versions_ready": 0.10,
}

# A story is cleared for learners as the learner API rules it
# (legendDailyStoryFeed LEARNER_VISIBLE_STORY_MATCH).
VISIBLE_STATUSES = ("published", "approved")


def _obj(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def cleared_for_learners(story: Dict[str, Any]) -> bool:
    status = _text(story.get("status"))
    return status in VISIBLE_STATUSES or (status == "generated" and _obj(story.get("review")).get("status") == "approved")


def source_signals(doc: Dict[str, Any]) -> Dict[str, Any]:
    """The legend's reviewed passages: counts, pages, sections, licences."""
    restricted = legend_identity.restricted_to_reviewed_sources(doc)
    sections = _obj(_obj(restricted.get("rag_context")).get("section_chunks"))
    passages = 0
    creditable = 0
    chars = 0
    pages: Dict[str, Dict[str, Any]] = {}
    by_licence: Counter = Counter()
    for section, chunks in sections.items():
        for chunk in chunks if isinstance(chunks, list) else []:
            text = legend_identity.chunk_text(chunk)
            if len(text) < legend_identity.MIN_PASSAGE_CHARS:
                continue
            passages += 1
            chars += len(text)
            links = legend_identity.chunk_links(chunk)
            if not links or not creditable_source(links[0]):
                # Nothing the writer may cite and credit (daily_story_worker.creditable_source).
                by_licence["no_link" if not links else "not_creditable"] += 1
                continue
            creditable += 1
            key = legend_identity.chunk_page_key(chunk)
            if key not in pages:
                pages[key] = {"licence": licence_for(links[0]), "passages": 0}
            pages[key]["passages"] += 1
            by_licence[pages[key]["licence"]["status"]] += 1
    licences = Counter()
    for page in pages.values():
        licences[page["licence"]["name"]] += page["passages"]
    return {
        "passages": passages,
        "creditable_passages": creditable,
        "chars": chars,
        "pages": len(pages),
        "sections": sorted(sections),
        "passages_by_licence_status": dict(by_licence),
        "open_share": round(by_licence[OPEN] / creditable, 3) if creditable else 0.0,
        "refused_passages": by_licence[REFUSED],
        "top_licences": licences.most_common(3),
    }


def story_signals(stories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What the pipeline already ruled on the legend's stories for the young bands."""
    out = {
        "stories": len(stories),
        "cleared": 0,
        "on_legal_hold": 0,
        "evergreen": 0,
        "ruled": {BAND: {"suitable": 0, "not_suitable": 0}, YOUNGER_BAND: {"suitable": 0, "not_suitable": 0}},
        "versions": {"ready": 0, "failed": 0, "not_suitable": 0},
        "servable": 0,
        "unsuitable_reasons": [],
    }
    for story in stories:
        cleared = cleared_for_learners(story)
        out["cleared"] += int(cleared)
        out["on_legal_hold"] += int(bool(_obj(story.get("review")).get("legal_hold")))
        out["evergreen"] += int(_text(story.get("story_mode")) == "evergreen")
        rulings = _obj(story.get("audience_suitability"))
        for band in (BAND, YOUNGER_BAND):
            verdict = _obj(rulings.get(band))
            if isinstance(verdict.get("suitable"), bool):
                out["ruled"][band]["suitable" if verdict["suitable"] else "not_suitable"] += 1
                if band == BAND and not verdict["suitable"] and _text(verdict.get("reason")):
                    out["unsuitable_reasons"].append(_text(verdict["reason"])[:160])
        ready = False
        for variant in story.get("audience_variants") or []:
            variant = _obj(variant)
            if variant.get("band_key") != BAND:
                continue
            if variant.get("status") == "ready" and _obj(variant.get("validation")).get("ok") is True:
                out["versions"]["ready"] += 1
                ready = True
            elif variant.get("status") in ("failed", "not_suitable"):
                out["versions"][variant["status"]] += 1
        if cleared and ready and _obj(rulings.get(BAND)).get("suitable") is True:
            out["servable"] += 1
    out["unsuitable_reasons"] = out["unsuitable_reasons"][:2]
    return out


def quality_state(doc: Dict[str, Any], quality: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    meta = _obj(doc.get("daily_story_profile")) or _obj(quality)
    return {"status": _text(meta.get("status")) or "none", "score": int(meta.get("quality_score") or 0)}


def score(sources: Dict[str, Any], stories: Dict[str, Any]) -> Dict[str, float]:
    """Each signal on 0..1, and the weighted fit (0..100). Rulings and
    versions not yet made count as even odds, not as bad ones."""
    ruled = stories["ruled"][BAND]
    judged = ruled["suitable"] + ruled["not_suitable"]
    written = sum(stories["versions"].values())
    parts = {
        "depth": min(1.0, math.log1p(sources["creditable_passages"]) / math.log1p(FULL_DEPTH_PASSAGES)),
        "pages": min(1.0, sources["pages"] / FULL_PAGES),
        "open_licence": sources["open_share"],
        "ruled_suitable": (ruled["suitable"] + 1) / (judged + 2),
        "versions_ready": (stories["versions"]["ready"] + 1) / (written + 2),
    }
    parts["fit"] = round(100 * sum(SCORE_WEIGHTS[k] * parts[k] for k in SCORE_WEIGHTS), 1)
    return parts


def blockers(doc: Dict[str, Any], sources: Dict[str, Any], quality: Dict[str, Any], min_quality: int) -> List[str]:
    """Why no story can be written about the legend now, by the writer's gates."""
    out: List[str] = []
    eligible, reason = legend_identity.story_eligibility(doc)
    if not eligible:
        out.append(f"identity review: {reason}")
    if quality["status"] != "ready" or quality["score"] < min_quality:
        out.append(f"profile quality {quality['status']} {quality['score']} (the nightly gate wants ready, {min_quality}+)")
    if sources["creditable_passages"] < MIN_PASSAGES:
        out.append(f"{sources['creditable_passages']} creditable passages (needs {MIN_PASSAGES})")
    elif sources["pages"] < MIN_PAGES:
        out.append(f"one source page (the AI editor's rules want {MIN_PAGES})")
    return out


def reasons_line(row: Dict[str, Any]) -> str:
    s, t = row["sources"], row["stories"]
    licences = ", ".join(f"{name} {n}" for name, n in s["top_licences"]) or "none"
    ruled = t["ruled"][BAND]
    younger = t["ruled"][YOUNGER_BAND]
    if ruled["suitable"] + ruled["not_suitable"]:
        ruling = f"{ruled['suitable']} of {ruled['suitable'] + ruled['not_suitable']} stories ruled fine for grades 3-5"
    else:
        ruling = "no story ruled on for grades 3-5 yet"
    if younger["suitable"] + younger["not_suitable"]:
        ruling += f" ({younger['suitable']} of {younger['suitable'] + younger['not_suitable']} for K-2)"
    bio = row["bio"]
    bio_text = f"adult bio grade {bio['fk_grade']}" if bio["fk_grade"] is not None else "no adult bio"
    bio_text += "; grade 3-5 bio stored" if bio["child_bio"] else "; no grade 3-5 bio"
    lines = [
        f"{s['creditable_passages']} reviewed passages from {s['pages']} pages ({round(100 * s['open_share'])}% open-licensed; by licence: {licences})",
        f"{ruling}; versions {t['versions']['ready']} ready, {t['versions']['failed']} failed, {t['versions']['not_suitable']} not suitable; {t['servable']} servable now",
        f"profile quality {row['quality']['status']} {row['quality']['score']}; stories {t['stories']} ({t['cleared']} approved, {t['on_legal_hold']} on legal hold, {t['evergreen']} evergreen); {bio_text}",
    ]
    if t["unsuitable_reasons"]:
        lines.append(f"ruled unsuitable because: {t['unsuitable_reasons'][0]}")
    return "\n      ".join(lines)


def build_report(db: Any, *, min_quality: int = 60) -> Dict[str, Any]:
    legends_total = db["legend_scholars"].count_documents({})
    cleared = list(db["legend_scholars"].find({
        "identity_review.eligible": True,
        "identity_review.policy_version": legend_identity.IDENTITY_POLICY_VERSION,
    }))
    ids = [_text(doc.get("profile_id")) or str(doc.get("_id")) for doc in cleared]
    quality_rows = {
        _text(row.get("profile_id")): row
        for row in db["daily_story_profile_quality"].find({"profile_id": {"$in": ids}}, {"profile_id": 1, "status": 1, "quality_score": 1})
    }
    stories_by_legend: Dict[str, List[Dict[str, Any]]] = {}
    projection = {
        "scholar.profile_id": 1, "status": 1, "review.status": 1, "review.legal_hold": 1, "story_mode": 1,
        f"audience_suitability.{BAND}": 1, f"audience_suitability.{YOUNGER_BAND}": 1,
        "audience_variants.band_key": 1, "audience_variants.status": 1, "audience_variants.validation.ok": 1,
    }
    for story in db["legend_scholar_daily_stories"].find({"scholar.profile_id": {"$in": ids}}, projection):
        stories_by_legend.setdefault(_text(_obj(story.get("scholar")).get("profile_id")), []).append(story)

    rows: List[Dict[str, Any]] = []
    for doc, profile_id in zip(cleared, ids):
        about = _obj(doc.get("about"))
        sources = source_signals(doc)
        stories = story_signals(stories_by_legend.get(profile_id, []))
        quality = quality_state(doc, quality_rows.get(profile_id))
        bio = _text(about.get("short_bio"))
        child = _obj(_obj(about.get("bio_by_band")).get(BAND))
        row = {
            "profile_id": profile_id,
            "name": legend_identity.stored_name(doc),
            "field": _text(about.get("field_of_study")) or _text(about.get("current_position")),
            "identity_status": _text(_obj(doc.get("identity_review")).get("status")),
            "sources": sources,
            "stories": stories,
            "quality": quality,
            "bio": {
                "fk_grade": readability_metrics(bio)["flesch_kincaid_grade"] if bio else None,
                "learner_api_fk_grade": learner_api_fk_grade(bio) if bio else None,
                "child_bio": child.get("status") == "ready",
            },
        }
        row["blocked"] = blockers(doc, sources, quality, min_quality)
        row["score"] = score(sources, stories)
        rows.append(row)
    rows.sort(key=lambda r: (bool(r["blocked"]), -r["score"]["fit"], r["name"].lower()))
    return {"database": db.name, "legends_total": legends_total, "identity_cleared": len(cleared), "rows": rows}


def format_report(report: Dict[str, Any], top: Optional[int] = None) -> str:
    rows = report["rows"]
    ready = [r for r in rows if not r["blocked"]]
    blocked = [r for r in rows if r["blocked"]]
    lines = [
        f"LEGEND SELECTION FOR GRADES 3-5  database {report['database']}",
        "Read-only (readonly_mongo). Ranked by a fit score over stored signals; the choice is yours.",
        f"{report['legends_total']} legends; {report['identity_cleared']} cleared by identity review; "
        f"{len(ready)} can be written about now; {len(blocked)} cannot yet.",
        f"Score weights: {', '.join(f'{k} {v:g}' for k, v in SCORE_WEIGHTS.items())} "
        f"(depth saturates at {FULL_DEPTH_PASSAGES} passages, pages at {FULL_PAGES}; rulings not yet made count as even odds).",
    ]
    if not report["legends_total"]:
        lines.append("WARNING: no legends here. Is this the pipeline's database? Check the address or LEGEND_REPORT_DB.")
    lines.append("")
    lines.append("CAN BE WRITTEN ABOUT NOW")
    for rank, row in enumerate(ready[: top or len(ready)], start=1):
        field = f", {row['field']}" if row["field"] else ""
        lines.append(f"  {rank:>2}. {row['name']} [{row['profile_id']}]{field}  fit {row['score']['fit']}")
        lines.append(f"      {reasons_line(row)}")
    lines.append("")
    lines.append("NOT YET (and why)")
    for row in blocked:
        lines.append(f"  - {row['name']} [{row['profile_id']}]: {'; '.join(row['blocked'])}")
    lines.append("")
    lines.append(
        "Next: write the chosen ids to a file, one per line, and give it to legend_child_bio.py and "
        "legend_evergreen.py with --legend-ids-file (each has a --dry-run with its cost)."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank identity-cleared legends for grade 3-5 readers (read-only).")
    parser.add_argument("--top", type=int, default=None, help="Show only the first N that can be written about now")
    parser.add_argument("--json", action="store_true", help="Print the rows as JSON")
    parser.add_argument("--min-quality", type=int, default=60, help="The profile-quality score the nightly gate wants (STORY_PROFILE_MIN_QUALITY_SCORE)")
    args = parser.parse_args()
    uri = (os.environ.get(URI_ENV) or "").strip()
    if not uri:
        print(f"Set {URI_ENV} to the database to report on (see this script's header).", file=sys.stderr)
        return 2
    client, db = connect_read_only(uri, label="legend_selection_report", db_name=(os.environ.get(DB_ENV) or "").strip() or None, app_name="legend-selection-report")
    try:
        report = build_report(db, min_quality=args.min_quality)
    finally:
        client.close()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    else:
        print(format_report(report, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
