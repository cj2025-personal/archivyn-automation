"""
Backfill scholar image S3 mappings from FacultyImages.images into scholar collections.

Matching strategy:
1. Normalize scholar name + institution
2. Normalize image name + university
3. Accept only unique matches among image documents where:
   - YOLOv8n_human_detection.has_human == True
   - image.status == "uploaded"

The script is preview-first:
- By default it writes debug artifacts only
- Use --write to persist S3 mappings back into MongoDB

Outputs:
- output/image_match_runs/<timestamp>/summary.json
- output/image_match_runs/<timestamp>/<collection>_matches.csv
- output/image_match_runs/<timestamp>/<collection>_ambiguous.csv
- output/image_match_runs/<timestamp>/<collection>_unmatched.csv
- output/image_match_runs/<timestamp>/<collection>_skipped_existing.csv
- output/image_match_runs/<timestamp>/<collection>_debug.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name

load_dotenv()


TITLE_TOKENS = {
    "dr",
    "doctor",
    "prof",
    "professor",
    "mr",
    "mrs",
    "ms",
    "miss",
}

SUFFIX_TOKENS = {
    "jr",
    "sr",
    "ii",
    "iii",
    "iv",
    "phd",
    "md",
    "mba",
}

INSTITUTION_STOPWORDS = {
    "the",
    "of",
    "at",
    "main",
    "campus",
    "university",
}


@dataclass(frozen=True)
class ImageRecord:
    profile_id: str
    name: str
    university: str
    source_url: str
    s3_bucket: str
    s3_key: str
    s3_uri: str
    content_type: str


@dataclass(frozen=True)
class MatchDecision:
    status: str
    method: str
    scholar_name: str
    scholar_institution: str
    normalized_name: str
    normalized_institution: str
    candidates: Tuple[ImageRecord, ...]
    reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill scholar image S3 mappings using normalized name + institution"
    )
    parser.add_argument(
        "--collection",
        action="append",
        default=[],
        help="MongoDB collection to process (repeatable). Default: scholars",
    )
    parser.add_argument(
        "--all-default-collections",
        action="store_true",
        help="Process scholars, archived_scholars, and legend_scholars",
    )
    parser.add_argument(
        "--institution",
        type=str,
        default="",
        help="Optional case-insensitive scholar institution filter",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Optional case-insensitive scholar name filter",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of scholar documents to inspect per collection",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "image_match_runs",
        help="Base output directory for debug artifacts",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional label appended to the run directory name",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist matches back into MongoDB",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing image mapping fields when --write is used",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip documents that already have image mapping fields (default: true)",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Do not skip documents that already have image mapping fields",
    )
    parser.add_argument(
        "--debug-sample-size",
        type=int,
        default=10,
        help="Number of matched/ambiguous/unmatched rows to print to stdout",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: str) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_name(value: str) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    tokens = [token for token in text.split() if token and token not in TITLE_TOKENS]
    while tokens and tokens[-1] in SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens)


def normalize_name_without_initials(value: str) -> str:
    tokens = [token for token in normalize_name(value).split() if len(token) > 1]
    return " ".join(tokens)


def normalize_institution(value: str) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    tokens = [token for token in text.split() if token and token not in INSTITUTION_STOPWORDS]
    return " ".join(tokens)


def make_name_variants(value: str) -> List[Tuple[str, str]]:
    variants: List[Tuple[str, str]] = []
    full = normalize_name(value)
    if full:
        variants.append(("name_institution_normalized", full))
    no_initials = normalize_name_without_initials(value)
    if no_initials and no_initials != full:
        variants.append(("name_without_initials_institution_normalized", no_initials))
    return variants


def safe_display_name(doc: Dict[str, Any]) -> str:
    name = doc.get("name")
    if isinstance(name, dict):
        for key in ("display", "full", "first"):
            value = str(name.get(key) or "").strip()
            if value:
                return value
    return str(doc.get("professor_name") or "").strip()


def safe_institution(doc: Dict[str, Any]) -> str:
    about = doc.get("about") or {}
    metadata = doc.get("metadata") or {}
    for value in (
        about.get("institution"),
        metadata.get("university"),
        metadata.get("institution"),
    ):
        cleaned = str(value or "").strip()
        if cleaned:
            return cleaned
    return ""


def has_existing_image_mapping(doc: Dict[str, Any]) -> bool:
    about = doc.get("about") or {}
    display = doc.get("display") or {}
    image_mapping = doc.get("image_mapping") or {}
    return any(
        str(value or "").strip()
        for value in (
            about.get("avatar_url"),
            display.get("profile_image_url"),
            image_mapping.get("s3_uri"),
            image_mapping.get("s3_key"),
        )
    )


def dedupe_candidates(records: Iterable[ImageRecord]) -> List[ImageRecord]:
    seen: Dict[str, ImageRecord] = {}
    for record in records:
        seen[record.profile_id] = record
    return sorted(seen.values(), key=lambda item: (item.name.lower(), item.profile_id))


def csv_write(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def jsonl_write(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


class ScholarImageBackfill:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        scholar_mongo_uri = os.getenv("MONGODB_URI")
        if not scholar_mongo_uri:
            raise ValueError("MONGODB_URI not found in environment variables")
        image_mongo_uri = os.getenv("MONGO_ATLAS_URI")
        if not image_mongo_uri:
            raise ValueError("MONGO_ATLAS_URI not found in environment variables")

        self.scholar_client = create_mongo_client(scholar_mongo_uri)
        self.scholar_db = self.scholar_client[resolve_mongo_db_name(scholar_mongo_uri)]

        self.image_client = create_mongo_client(image_mongo_uri)
        image_db_name = resolve_mongo_db_name(image_mongo_uri, default="FacultyImages")
        image_collection_name = os.getenv("MONGODB_COLLECTION_NAME", "images")
        self.image_collection = self.image_client[image_db_name][image_collection_name]

        self.run_started_at = utc_now_iso()
        run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        if args.label:
            safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "-", args.label.strip()).strip("-")
            run_stamp = f"{run_stamp}-{safe_label}" if safe_label else run_stamp
        self.run_dir = args.output_dir / run_stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.image_index: Dict[Tuple[str, str], List[ImageRecord]] = defaultdict(list)
        self.image_count = 0

    def build_image_index(self) -> None:
        projection = {
            "_id": 0,
            "profile_id": 1,
            "name": 1,
            "university": 1,
            "source_url": 1,
            "image.s3_bucket": 1,
            "image.s3_key": 1,
            "image.content_type": 1,
            "YOLOv8n_human_detection.has_human": 1,
            "image.status": 1,
        }
        query = {
            "YOLOv8n_human_detection.has_human": True,
            "image.status": "uploaded",
        }

        for doc in self.image_collection.find(query, projection):
            profile_id = str(doc.get("profile_id") or "").strip()
            name = str(doc.get("name") or "").strip()
            university = str(doc.get("university") or "").strip()
            image = doc.get("image") or {}
            bucket = str(image.get("s3_bucket") or "").strip()
            key = str(image.get("s3_key") or "").strip()

            if not profile_id or not name or not university or not bucket or not key:
                continue

            record = ImageRecord(
                profile_id=profile_id,
                name=name,
                university=university,
                source_url=str(doc.get("source_url") or "").strip(),
                s3_bucket=bucket,
                s3_key=key,
                s3_uri=f"s3://{bucket}/{key}",
                content_type=str(image.get("content_type") or "").strip(),
            )
            institution_key = normalize_institution(university)
            if not institution_key:
                continue

            for _, name_key in make_name_variants(name):
                if not name_key:
                    continue
                self.image_index[(name_key, institution_key)].append(record)

            self.image_count += 1

        print(f"[Images] Qualified image records indexed: {self.image_count}")
        print(f"[Images] Unique normalized keys: {len(self.image_index)}")

    def scholar_cursor(self, collection_name: str):
        projection = {
            "_id": 1,
            "profile_id": 1,
            "professor_id": 1,
            "professor_name": 1,
            "name": 1,
            "about": 1,
            "metadata": 1,
            "display": 1,
            "image_mapping": 1,
        }
        cursor = self.scholar_db[collection_name].find({}, projection)
        if self.args.limit and self.args.limit > 0:
            cursor = cursor.limit(self.args.limit)
        return cursor

    def decide_match(self, scholar_name: str, scholar_institution: str) -> MatchDecision:
        normalized_institution = normalize_institution(scholar_institution)
        if not scholar_name:
            return MatchDecision(
                status="unmatched",
                method="none",
                scholar_name=scholar_name,
                scholar_institution=scholar_institution,
                normalized_name="",
                normalized_institution=normalized_institution,
                candidates=(),
                reason="missing_name",
            )
        if not scholar_institution:
            return MatchDecision(
                status="unmatched",
                method="none",
                scholar_name=scholar_name,
                scholar_institution=scholar_institution,
                normalized_name=normalize_name(scholar_name),
                normalized_institution="",
                candidates=(),
                reason="missing_institution",
            )
        if not normalized_institution:
            return MatchDecision(
                status="unmatched",
                method="none",
                scholar_name=scholar_name,
                scholar_institution=scholar_institution,
                normalized_name=normalize_name(scholar_name),
                normalized_institution="",
                candidates=(),
                reason="institution_normalized_empty",
            )

        seen_ambiguous: List[ImageRecord] = []
        for method, name_key in make_name_variants(scholar_name):
            candidates = dedupe_candidates(self.image_index.get((name_key, normalized_institution), []))
            if len(candidates) == 1:
                return MatchDecision(
                    status="matched",
                    method=method,
                    scholar_name=scholar_name,
                    scholar_institution=scholar_institution,
                    normalized_name=name_key,
                    normalized_institution=normalized_institution,
                    candidates=tuple(candidates),
                )
            if len(candidates) > 1:
                seen_ambiguous = candidates
                return MatchDecision(
                    status="ambiguous",
                    method=method,
                    scholar_name=scholar_name,
                    scholar_institution=scholar_institution,
                    normalized_name=name_key,
                    normalized_institution=normalized_institution,
                    candidates=tuple(candidates),
                    reason="multiple_image_candidates",
                )

        return MatchDecision(
            status="unmatched",
            method="none",
            scholar_name=scholar_name,
            scholar_institution=scholar_institution,
            normalized_name=normalize_name(scholar_name),
            normalized_institution=normalized_institution,
            candidates=tuple(seen_ambiguous),
            reason="no_image_candidate",
        )

    def update_document(
        self,
        collection_name: str,
        doc_id: Any,
        match: MatchDecision,
    ) -> None:
        image = match.candidates[0]
        now = utc_now_iso()
        update_payload = {
            "about.avatar_url": image.s3_uri,
            "display.profile_image_url": image.s3_uri,
            "display.last_updated": now,
            "image_mapping.status": "matched",
            "image_mapping.method": match.method,
            "image_mapping.matched_at": now,
            "image_mapping.image_profile_id": image.profile_id,
            "image_mapping.image_name": image.name,
            "image_mapping.image_university": image.university,
            "image_mapping.source_url": image.source_url,
            "image_mapping.s3_bucket": image.s3_bucket,
            "image_mapping.s3_key": image.s3_key,
            "image_mapping.s3_uri": image.s3_uri,
            "image_mapping.has_human": True,
            "image_mapping.image_status": "uploaded",
            "image_mapping.content_type": image.content_type,
            "image_mapping.normalized_name": match.normalized_name,
            "image_mapping.normalized_institution": match.normalized_institution,
        }
        self.scholar_db[collection_name].update_one({"_id": doc_id}, {"$set": update_payload})

    def run_collection(self, collection_name: str) -> Dict[str, Any]:
        collection = self.scholar_db[collection_name]
        total_in_collection = collection.count_documents({})
        print(f"[Collection] {collection_name}: {total_in_collection} total documents")

        matched_rows: List[Dict[str, Any]] = []
        ambiguous_rows: List[Dict[str, Any]] = []
        unmatched_rows: List[Dict[str, Any]] = []
        skipped_rows: List[Dict[str, Any]] = []
        debug_rows: List[Dict[str, Any]] = []

        processed = 0
        written = 0
        skipped_existing = 0

        institution_filter = normalize_institution(self.args.institution)
        name_filter = normalize_name(self.args.name)

        for doc in self.scholar_cursor(collection_name):
            scholar_name = safe_display_name(doc)
            scholar_institution = safe_institution(doc)
            normalized_name = normalize_name(scholar_name)
            normalized_institution = normalize_institution(scholar_institution)

            if institution_filter and institution_filter not in normalized_institution:
                continue
            if name_filter and name_filter not in normalized_name:
                continue

            processed += 1
            scholar_id = str(doc.get("_id"))
            scholar_profile_id = str(doc.get("profile_id") or doc.get("professor_id") or "").strip()
            existing_mapping = has_existing_image_mapping(doc)

            decision = self.decide_match(scholar_name, scholar_institution)

            debug_payload = {
                "collection": collection_name,
                "scholar_id": scholar_id,
                "scholar_profile_id": scholar_profile_id,
                "scholar_name": scholar_name,
                "scholar_institution": scholar_institution,
                "normalized_name": normalized_name,
                "normalized_institution": normalized_institution,
                "decision_status": decision.status,
                "decision_method": decision.method,
                "decision_reason": decision.reason,
                "candidate_count": len(decision.candidates),
                "existing_mapping": existing_mapping,
                "candidates": [
                    {
                        "image_profile_id": image.profile_id,
                        "image_name": image.name,
                        "image_university": image.university,
                        "source_url": image.source_url,
                        "s3_uri": image.s3_uri,
                    }
                    for image in decision.candidates
                ],
            }
            debug_rows.append(debug_payload)

            base_row = {
                "collection": collection_name,
                "scholar_id": scholar_id,
                "scholar_profile_id": scholar_profile_id,
                "scholar_name": scholar_name,
                "scholar_institution": scholar_institution,
                "normalized_name": decision.normalized_name or normalized_name,
                "normalized_institution": decision.normalized_institution or normalized_institution,
                "existing_mapping": existing_mapping,
                "match_method": decision.method,
                "reason": decision.reason,
            }

            if decision.status == "matched":
                image = decision.candidates[0]
                matched_row = {
                    **base_row,
                    "image_profile_id": image.profile_id,
                    "image_name": image.name,
                    "image_university": image.university,
                    "image_source_url": image.source_url,
                    "image_s3_bucket": image.s3_bucket,
                    "image_s3_key": image.s3_key,
                    "image_s3_uri": image.s3_uri,
                }
                if existing_mapping and self.args.skip_existing and not self.args.overwrite:
                    skipped_existing += 1
                    skipped_rows.append({**matched_row, "skip_reason": "existing_image_mapping"})
                    continue
                matched_rows.append(matched_row)
                if self.args.write:
                    self.update_document(collection_name, doc["_id"], decision)
                    written += 1
                continue

            if decision.status == "ambiguous":
                ambiguous_rows.append(
                    {
                        **base_row,
                        "candidate_count": len(decision.candidates),
                        "candidate_profile_ids": " | ".join(image.profile_id for image in decision.candidates),
                        "candidate_s3_uris": " | ".join(image.s3_uri for image in decision.candidates),
                        "candidate_source_urls": " | ".join(image.source_url for image in decision.candidates),
                    }
                )
                continue

            unmatched_rows.append(base_row)

        summary = {
            "collection": collection_name,
            "total_documents": total_in_collection,
            "processed_documents": processed,
            "matched_unique": len(matched_rows),
            "ambiguous": len(ambiguous_rows),
            "unmatched": len(unmatched_rows),
            "skipped_existing": skipped_existing,
            "written": written,
        }

        csv_write(self.run_dir / f"{collection_name}_matches.csv", matched_rows)
        csv_write(self.run_dir / f"{collection_name}_ambiguous.csv", ambiguous_rows)
        csv_write(self.run_dir / f"{collection_name}_unmatched.csv", unmatched_rows)
        csv_write(self.run_dir / f"{collection_name}_skipped_existing.csv", skipped_rows)
        jsonl_write(self.run_dir / f"{collection_name}_debug.jsonl", debug_rows)

        print(f"[Collection] {collection_name} summary")
        print(f"  processed: {processed}")
        print(f"  matched unique: {len(matched_rows)}")
        print(f"  ambiguous: {len(ambiguous_rows)}")
        print(f"  unmatched: {len(unmatched_rows)}")
        print(f"  skipped existing: {skipped_existing}")
        print(f"  written: {written}")

        self.print_samples("matched", matched_rows)
        self.print_samples("ambiguous", ambiguous_rows)
        self.print_samples("unmatched", unmatched_rows)

        return summary

    def print_samples(self, label: str, rows: Sequence[Dict[str, Any]]) -> None:
        sample_size = max(0, int(self.args.debug_sample_size))
        if sample_size == 0 or not rows:
            return
        print(f"[Sample] {label} ({min(sample_size, len(rows))} of {len(rows)})")
        for row in rows[:sample_size]:
            print(" ", json.dumps(row, ensure_ascii=True))

    def run(self, collections: Sequence[str]) -> Dict[str, Any]:
        self.build_image_index()
        summaries = [self.run_collection(name) for name in collections]
        payload = {
            "started_at": self.run_started_at,
            "finished_at": utc_now_iso(),
            "write_enabled": self.args.write,
            "overwrite": self.args.overwrite,
            "skip_existing": self.args.skip_existing,
            "collections": summaries,
            "run_dir": str(self.run_dir),
            "qualified_image_records": self.image_count,
        }
        json_write(self.run_dir / "summary.json", payload)
        print(f"[Run] Debug artifacts written to: {self.run_dir}")
        return payload


def resolve_collections(args: argparse.Namespace) -> List[str]:
    if args.all_default_collections:
        return ["scholars", "archived_scholars", "legend_scholars"]
    if args.collection:
        return args.collection
    return ["scholars"]


def main() -> None:
    args = parse_args()
    collections = resolve_collections(args)
    runner = ScholarImageBackfill(args)
    runner.run(collections)


if __name__ == "__main__":
    main()
