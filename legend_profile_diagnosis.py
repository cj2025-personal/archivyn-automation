"""Why a legend's profile is not "ready" for the story pipeline, and what a
repair would need.

The nightly story gate (daily_story_worker._profile_quality_state) writes
about a legend only while its daily_story_profile says status "ready" with a
score of STORY_PROFILE_MIN_QUALITY_SCORE (60) or more. Both come from
profile_quality_refresh.evaluate_profile, which reads rag_context alone:

  score  = name      20 when the stored name passes the worker's name rule
         + passages  10, 20, 30, 40 for 10, 20, 40, 80 or more stored passages
         + coverage  8, 14, 20 when section titles name 2, 3, 4 or more of
                     biography / about / background / legacy / publications
         + length    8, 14, 20 when passages average 140, 250, 420+ characters
  status = "ready" only with a score of 60 or more AND at least 20 passages

so a legend at 64 with 19 passages is still needs_data_repair: the status,
not the score, decides. For each legend named this prints the stored label,
the score recomputed now part by part, every reason the legend is not ready,
and what would change that (how many more passages, with sections as they are
and as a re-collection would name them), beside the story writer's other
gates (identity review; four creditable passages from two pages), the
legend's sources and stored URL list, and what needs a person.

── READ-ONLY, ENFORCED IN CODE ───────────────────────────────────────────────
As legend_selection_report.py: every collection is a
readonly_mongo.ReadOnlyCollection (anything but a read raises before a
command is built), reads go to a secondary where there is one, and no .env
file is loaded: the address comes from LEGEND_REPORT_MONGO_URI only.

Running it (PowerShell 5.1, from this folder):
  $env:LEGEND_REPORT_MONGO_URI = (gcloud secrets versions access latest --secret=MONGODB_URI)
  python legend_profile_diagnosis.py --legend-ids-file pilot_ids.txt
  Remove-Item Env:LEGEND_REPORT_MONGO_URI
--json prints the rows as JSON. --url-lists-dir names where the legends'
stored URL lists (<id>.txt, not kept in git) are; the default is this
folder's legendary_scholars/final.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import legend_identity
import legend_selection_report as selection
import profile_quality_refresh as quality
from daily_story_worker import DailyStoryWorker, creditable_source
from legend_evergreen import parse_ids
from readonly_mongo import connect_read_only
from source_licenses import license_for

URI_ENV = "LEGEND_REPORT_MONGO_URI"
DB_ENV = "LEGEND_REPORT_DB"
CHILD_BIO_BAND = "grade_3_5"
# What DailyStoryWorker._professor_name returns when it can read no name. It
# passes the name rule itself, so the score's 20 name points are given even
# then: the name part never fails, and this is how to see that it should have.
UNREAD_NAME = "Unknown Scholar"

# evaluate_profile scores the name with the story worker's own rule
# (_professor_name, _looks_like_clean_name), which reads only the document.
# A worker made without __init__ has those methods and no database, model or
# .env, so the diagnosis scores exactly as the nightly refresh does.
NAME_RULES = DailyStoryWorker.__new__(DailyStoryWorker)


def _obj(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def least_passages(other_points: int, min_score: int = quality.READY_MIN_SCORE) -> Optional[int]:
    """The fewest passages that make a profile ready when its other parts
    (name, coverage, length) score other_points; None when no number does.
    The passage points only rise at a tier's floor, so the answer is one."""
    floors = sorted({quality.READY_MIN_CHUNKS, *(floor for floor, _ in quality.CHUNK_POINTS if floor >= quality.READY_MIN_CHUNKS)})
    for floor in floors:
        if other_points + quality.tier_points(floor, quality.CHUNK_POINTS) >= max(min_score, quality.READY_MIN_SCORE):
            return floor
    return None


def counted_passages(doc: Dict[str, Any]) -> int:
    """Passages actually held in section_chunks (the score trusts
    rag_context.chunk_count when that is set, and the two can differ)."""
    sections = _obj(_obj(doc.get("rag_context")).get("section_chunks"))
    return sum(len(chunks) for chunks in sections.values() if isinstance(chunks, list))


def page_table(doc: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Every stored passage's page, before any identity review: passages,
    whether a story could credit it, its license. And the totals the story
    writer's source gate counts (creditable passages and pages)."""
    pages: Dict[str, Dict[str, Any]] = {}
    totals = Counter()
    sections = _obj(_obj(doc.get("rag_context")).get("section_chunks"))
    for chunks in sections.values():
        for chunk in chunks if isinstance(chunks, list) else []:
            if not isinstance(chunk, dict):
                continue
            text = legend_identity.chunk_text(chunk)
            if len(text) < legend_identity.MIN_PASSAGE_CHARS:
                totals["too_short"] += 1
                continue
            links = legend_identity.chunk_links(chunk)
            key = legend_identity.page_key(links[0]) if links else legend_identity.UNATTRIBUTED
            page = pages.setdefault(key, {
                "key": key,
                "url": links[0] if links else "",
                "passages": 0,
                "creditable": bool(links) and creditable_source(links[0]),
                "license": license_for(links[0])["status"] if links else "no_link",
            })
            page["passages"] += 1
            totals["passages"] += 1
            if page["creditable"]:
                totals["creditable_passages"] += 1
    totals["creditable_pages"] = sum(1 for p in pages.values() if p["creditable"])
    ordered = sorted(pages.values(), key=lambda p: (-p["passages"], p["key"]))
    return ordered, dict(totals)


def read_url_list(path: Path) -> List[str]:
    """A legend's stored URL list, as the legend run reads it (one URL a
    line, # comments). Not kept in git; empty when the file is absent."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return []
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def read_roster(path: Path) -> Dict[str, Any]:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _name_key(name: str) -> Tuple[str, str]:
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    return (tokens[0], tokens[-1]) if len(tokens) >= 2 else ("", "")


def stored_label(doc: Dict[str, Any], quality_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The label the gate reads: the legend's own daily_story_profile first,
    the quality collection's row only when the legend has none."""
    own = _obj(doc.get("daily_story_profile"))
    if _text(own.get("status")):
        return {"status": _text(own["status"]), "score": int(own.get("quality_score") or 0),
                "from": "legend_scholars.daily_story_profile", "refreshed_at": own.get("refreshed_at")}
    row = _obj(quality_row)
    if _text(row.get("status")):
        return {"status": _text(row["status"]), "score": int(row.get("quality_score") or 0),
                "from": "daily_story_profile_quality", "refreshed_at": row.get("refreshed_at")}
    return {"status": "none", "score": 0, "from": "nowhere: never scored", "refreshed_at": None}


def diagnose(
    doc: Dict[str, Any],
    *,
    quality_row: Optional[Dict[str, Any]] = None,
    stories: Optional[List[Dict[str, Any]]] = None,
    duplicates: Optional[List[Dict[str, Any]]] = None,
    url_list: Optional[List[str]] = None,
    roster: Optional[Dict[str, Any]] = None,
    min_quality: int = quality.READY_MIN_SCORE,
) -> Dict[str, Any]:
    """Everything about one legend that decides whether a story can be
    written about it now, and what would change that. Pure: reads only what
    it is given."""
    profile_id = DailyStoryWorker._profile_id(doc)
    now = quality.evaluate_profile(NAME_RULES, doc)
    signals, points = now["signals"], now["points"]
    label = stored_label(doc, quality_row)
    pages, totals = page_table(doc)
    held = counted_passages(doc)

    reasons: List[str] = []
    if signals["chunk_count"] < quality.READY_MIN_CHUNKS:
        reasons.append(f"{signals['chunk_count']} passages stored; ready needs at least {quality.READY_MIN_CHUNKS}")
    if now["score"] < quality.READY_MIN_SCORE:
        reasons.append(f"score {now['score']} is under {quality.READY_MIN_SCORE}")
    elif now["status"] == "ready" and now["score"] < min_quality:
        reasons.append(f"score {now['score']} is under the gate's {min_quality} (STORY_PROFILE_MIN_QUALITY_SCORE)")
    if not points["name"]:
        reasons.append(f"the stored name {now['name_raw']!r} fails the name rule (read as {now['name_normalized']!r})")
    elif now["name_normalized"] == UNREAD_NAME:
        reasons.append(f"no name could be read from {now['name_raw']!r} or the passages; the fallback {UNREAD_NAME!r} still scores the name's {quality.NAME_POINTS} points")
    if points["coverage"] < quality.COVERAGE_POINTS[0][1]:
        missing = [s for s in quality.CORE_SECTIONS if s not in now["core_sections"]]
        reasons.append(f"section titles name {signals['section_coverage']} of {len(quality.CORE_SECTIONS)} core sections; missing {', '.join(missing)}")
    if points["avg_len"] < quality.AVG_LEN_POINTS[0][1]:
        reasons.append(f"passages average {signals['avg_chunk_len']:g} characters; {quality.AVG_LEN_POINTS[0][0]}+ scores full")
    stored_count = int(_obj(doc.get("rag_context")).get("chunk_count") or 0)
    if stored_count and stored_count != held:
        reasons.append(f"rag_context.chunk_count says {stored_count} but section_chunks holds {held} passages (the score uses {stored_count})")

    needs: List[str] = []
    is_ready = now["status"] == "ready" and now["score"] >= min_quality
    avg = signals["avg_chunk_len"]
    if not is_ready:
        keep = least_passages(points["name"] + points["coverage"] + points["avg_len"], min_quality)
        if keep is None:
            needs.append("no number of passages makes it ready with the name, sections and lengths it has now")
        else:
            more = max(0, keep - signals["chunk_count"])
            text = f" (at its average of {avg:,.0f} characters, about {more * avg:,.0f} characters of new text about the legend)" if more and avg else ""
            needs.append(f"{more} more passages, {keep} in all, with section titles and passage lengths as they are{text}")
        # The present chunker titles each section after its page
        # (profile_chunking_pipeline._section_label_for_source), so a profile
        # collected again usually names no core section and scores 0 there.
        again = least_passages(points["name"] + points["avg_len"], min_quality)
        if again is None:
            needs.append("collected again (sections titled by page, coverage 0): no number of passages makes it ready with this name and passage length")
        else:
            needs.append(f"collected again (sections titled by page, coverage 0): at least {again} passages averaging as now")
        if not points["name"]:
            needs.append("a person to correct the stored name, which costs it 20 points")

    eligible, why = legend_identity.story_eligibility(doc)
    review = _obj(doc.get("identity_review"))
    anchor = legend_identity.anchor_for(doc)
    human: List[str] = []
    if not points["name"]:
        human.append("the stored name fails the name rule: correct professor_name / name.full")
    elif now["name_normalized"] == UNREAD_NAME:
        human.append("its name could not be read: correct professor_name / name.full")
    if review.get("status") == legend_identity.WRONG_PERSON:
        human.append("its stored sources describe someone else: a repair must find pages about the right person")
    if review.get("name_check") == "misspelled":
        usual = _text(review.get("usual_spelling"))
        human.append(f"the identity review says the name is misspelled{f' (usually {usual})' if usual else ''}: an administrator confirms the identity or corrects the name")
    if anchor.get("from") == "name_only":
        human.append("no roster entry or confirmed identity: identity checks and searches have only the stored name")
    if url_list is not None and not url_list and not is_ready:
        human.append("no stored URL list here (legendary_scholars/final/<id>.txt): a repair starts from Wikipedia and search alone")
    for dup in duplicates or []:
        human.append(f"another profile of the same name, {dup['profile_id']} ({dup['passages']} passages, {dup['status']} {dup['score']}): decide which to keep; a repair does not merge them")
    if totals.get("passages") and not totals.get("creditable_passages"):
        human.append("none of its stored pages can be credited in a story")

    stored_keys = {p["key"] for p in pages}
    listed = [{"url": url, "gave_passages": legend_identity.page_key(url) in stored_keys} for url in url_list or []]
    by_status = Counter(_text(s.get("status")) or "unknown" for s in stories or [])
    child = _obj(_obj(_obj(doc.get("about")).get("bio_by_band")).get(CHILD_BIO_BAND))
    source_gate = selection.source_signals(doc)
    return {
        "profile_id": profile_id,
        "name": legend_identity.stored_name(doc),
        "ready": is_ready,
        "stored": label,
        "label_is_stale": (label["status"], label["score"]) != (now["status"], now["score"]),
        "now": {"status": now["status"], "score": now["score"], "signals": signals, "points": points, "core_sections": now["core_sections"]},
        "passages_held": held,
        "reasons": reasons,
        "needs": needs,
        "identity": {
            "eligible": eligible,
            "reason": why,
            "status": _text(review.get("status")) or "not_reviewed",
            "name_check": _text(review.get("name_check")),
            "anchor": anchor,
            "attention": list(review.get("attention") or []),
        },
        "writer_gates": {
            "reviewed_creditable_passages": source_gate["creditable_passages"],
            "reviewed_pages": source_gate["pages"],
            "needs": f"{selection.MIN_PASSAGES} creditable passages from {selection.MIN_PAGES} pages, from pages its identity review allowed",
            "blocked": selection.blockers(doc, source_gate, {"status": label["status"], "score": label["score"]}, min_quality),
        },
        "pages": pages,
        "page_totals": totals,
        "url_list": listed,
        "roster": {k: roster.get(k) for k in ("profile_name", "field", "profile_url", "collector_version")} if roster else {},
        "stories": dict(by_status),
        "child_bio": _text(child.get("status")) or "none",
        "duplicates": duplicates or [],
        "needs_a_person": human,
    }


def find_legend(legends: Any, profile_id: str) -> Optional[Dict[str, Any]]:
    return legends.find_one({"profile_id": profile_id}) or legends.find_one({"_id": profile_id})


def same_name_profiles(legends: Any, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Other legends with the same first and last name (john-hope-franklin
    had a second profile, f0d7838e-..., holding 72 passages in June 2026)."""
    key = _name_key(legend_identity.stored_name(doc))
    if not key[0]:
        return []
    found: List[Dict[str, Any]] = []
    names = {"_id": 1, "profile_id": 1, "professor_name": 1, "name": 1}
    for other in legends.find({}, names):
        if other.get("_id") == doc.get("_id") or _name_key(legend_identity.stored_name(other)) != key:
            continue
        full = legends.find_one({"_id": other["_id"]})
        scored = quality.evaluate_profile(NAME_RULES, full)
        found.append({
            "profile_id": DailyStoryWorker._profile_id(full),
            "name": legend_identity.stored_name(full),
            "passages": scored["signals"]["chunk_count"],
            "status": scored["status"],
            "score": scored["score"],
        })
    return found


def build(db: Any, legend_ids: List[str], *, url_lists_dir: Path, min_quality: int) -> List[Dict[str, Any]]:
    legends = db["legend_scholars"]
    rows: List[Dict[str, Any]] = []
    for profile_id in legend_ids:
        doc = find_legend(legends, profile_id)
        if not doc:
            rows.append({"profile_id": profile_id, "found": False})
            continue
        pid = DailyStoryWorker._profile_id(doc)
        stories = list(db["legend_scholar_daily_stories"].find({"scholar.profile_id": pid}, {"status": 1}))
        row = diagnose(
            doc,
            quality_row=db["daily_story_profile_quality"].find_one({"profile_id": pid}),
            stories=stories,
            duplicates=same_name_profiles(legends, doc),
            url_list=read_url_list(url_lists_dir / f"{pid}.txt"),
            roster=read_roster(url_lists_dir / f"{pid}.meta.json"),
            min_quality=min_quality,
        )
        row["found"] = True
        rows.append(row)
    return rows


def format_row(row: Dict[str, Any]) -> str:
    if not row.get("found"):
        return f"{row['profile_id']}: NOT FOUND in legend_scholars (by profile_id or _id)"
    now, points, signals = row["now"], row["now"]["points"], row["now"]["signals"]
    label = row["stored"]
    lines = [f"{row['profile_id']}  {row['name']}  {'READY' if row['ready'] else 'NOT READY'}"]
    lines.append(f"  stored label: {label['status']} {label['score']} (from {label['from']}, refreshed {label['refreshed_at'] or 'never'})"
                 + ("  << differs from a recomputation: a quality refresh alone would change it" if row["label_is_stale"] else ""))
    lines.append(
        f"  recomputed now: {now['status']} {now['score']} = name {points['name']}"
        f" + passages {points['chunks']} ({signals['chunk_count']} stored)"
        f" + coverage {points['coverage']} ({signals['section_coverage']} of {len(quality.CORE_SECTIONS)}: {', '.join(now['core_sections']) or 'none'})"
        f" + length {points['avg_len']} (average {signals['avg_chunk_len']:g} characters)"
    )
    if row["reasons"]:
        lines.append("  not ready because:" if not row["ready"] else "  notes:")
        lines += [f"    - {reason}" for reason in row["reasons"]]
    if row["needs"]:
        lines.append("  ready would need:")
        lines += [f"    - {need}" for need in row["needs"]]
    identity = row["identity"]
    lines.append(
        f"  identity review: {identity['status']}, {'eligible' if identity['eligible'] else 'not eligible: ' + identity['reason']}"
        f" (checked against {identity['anchor'].get('from')}: {identity['anchor'].get('name')}"
        f"{', ' + identity['anchor']['field'] if identity['anchor'].get('field') else ''})"
    )
    gates = row["writer_gates"]
    lines.append(f"  writer's source gate: {gates['reviewed_creditable_passages']} creditable passages from {gates['reviewed_pages']} reviewed pages (needs {gates['needs']})")
    if gates["blocked"]:
        lines.append(f"  blocked for stories by: {'; '.join(gates['blocked'])}")
    totals = row["page_totals"]
    lines.append(f"  pages ({len(row['pages'])}; {totals.get('creditable_passages', 0)} of {totals.get('passages', 0)} passages creditable):")
    for page in row["pages"][:12]:
        lines.append(f"    {page['passages']:>3}  {page['key']}  [{page['license']}{'' if page['creditable'] else ', not creditable'}]")
    if len(row["pages"]) > 12:
        lines.append(f"    ... and {len(row['pages']) - 12} more")
    listed = row["url_list"]
    if listed:
        gave = [u for u in listed if u["gave_passages"]]
        lines.append(f"  stored URL list ({len(listed)}): {len(gave)} gave stored passages; {len(listed) - len(gave)} gave none:")
        lines += [f"    - {u['url']}" for u in listed if not u["gave_passages"]]
    else:
        lines.append("  stored URL list: none found here")
    roster = row["roster"]
    if roster:
        lines.append(f"  roster: {roster.get('profile_name')} ({roster.get('field') or 'no field'}), collected by {roster.get('collector_version') or 'an unnamed collector'}")
    lines.append(f"  stories: {', '.join(f'{k} {v}' for k, v in sorted(row['stories'].items())) or 'none'}; grade 3-5 bio: {row['child_bio']}")
    if row["needs_a_person"]:
        lines.append("  needs a person:")
        lines += [f"    - {item}" for item in row["needs_a_person"]]
    return "\n".join(lines)


def format_report(rows: List[Dict[str, Any]], database: str) -> str:
    ready = sum(1 for r in rows if r.get("ready"))
    head = [
        f"LEGEND PROFILE DIAGNOSIS  database {database}  (read-only)",
        f"{len(rows)} legends named; {ready} ready by the profile-quality gate.",
        f"Ready = score {quality.READY_MIN_SCORE}+ and {quality.READY_MIN_CHUNKS}+ passages (profile_quality_refresh.status_for);"
        " the gate reads legend_scholars.daily_story_profile first (daily_story_worker._profile_quality_state).",
        "",
    ]
    return "\n".join(head) + "\n\n".join(format_row(r) for r in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Why named legends are not ready for stories (read-only).")
    parser.add_argument("--legend-ids", default="", help="Comma-separated legend profile ids")
    parser.add_argument("--legend-ids-file", default="", help="A file of legend profile ids, one per line or comma-separated")
    parser.add_argument("--url-lists-dir", default=str(legend_identity.ROSTER_DIR), help="Where the legends' <id>.txt URL lists and <id>.meta.json roster entries are")
    parser.add_argument("--min-quality", type=int, default=quality.READY_MIN_SCORE, help="The score the nightly gate wants (STORY_PROFILE_MIN_QUALITY_SCORE)")
    parser.add_argument("--json", action="store_true", help="Print the rows as JSON")
    args = parser.parse_args()
    uri = (os.environ.get(URI_ENV) or "").strip()
    if not uri:
        print(f"Set {URI_ENV} to the database to read (see this script's header).", file=sys.stderr)
        return 2
    legend_ids = parse_ids(args.legend_ids, args.legend_ids_file)
    if not legend_ids:
        print("Name the legends: --legend-ids id1,id2 or --legend-ids-file.", file=sys.stderr)
        return 2
    client, db = connect_read_only(uri, label="legend_profile_diagnosis", db_name=(os.environ.get(DB_ENV) or "").strip() or None, app_name="legend-profile-diagnosis")
    try:
        rows = build(db, legend_ids, url_lists_dir=Path(args.url_lists_dir), min_quality=args.min_quality)
    finally:
        client.close()
    if args.json:
        print(json.dumps({"database": db.name, "rows": rows}, indent=2, ensure_ascii=False, default=str))
    else:
        print(format_report(rows, db.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
