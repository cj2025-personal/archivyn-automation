"""
Full Automation Pipeline Progress Report

Counts how many rows have been processed so far using:
  - local output files
  - Pinecone index data
  - MongoDB scholars collection

Usage:
  python pipeline_progress_report.py path/to/profiles.xlsx
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv

try:
    import pandas as pd
except ImportError:  # pragma: no cover - handled at runtime
    pd = None


def _format_pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{(numerator / denominator * 100):.1f}%"


def _chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _load_excel_urls(
    excel_path: str,
    start_from: int,
    limit: Optional[int],
) -> Tuple[List[str], int, int, Optional[str]]:
    if pd is None:
        return [], 0, 0, "pandas is not installed"
    if not excel_path or not os.path.exists(excel_path):
        return [], 0, 0, f"Excel file not found: {excel_path}"

    df = pd.read_excel(excel_path)

    if "source" in df.columns and "profile_url" not in df.columns:
        df = df.rename(columns={"source": "profile_url"})
    elif "profile_url" not in df.columns:
        url_columns = [col for col in df.columns if "url" in col.lower() or "link" in col.lower()]
        if url_columns:
            df = df.rename(columns={url_columns[0]: "profile_url"})
        else:
            return [], 0, 0, "Excel file must contain a 'source' or 'profile_url' column"

    valid_df = df[df["profile_url"].notna()].reset_index(drop=True)
    total_valid = len(valid_df)

    if start_from > 0:
        valid_df = valid_df.iloc[start_from:].reset_index(drop=True)
    if limit:
        valid_df = valid_df.head(limit)

    scoped_urls = [str(url).strip() for url in valid_df["profile_url"].tolist()]
    return scoped_urls, total_valid, len(scoped_urls), None


def _scan_local_profiles(
    output_dir: Path,
    excel_url_set: Optional[set],
    apply_excel_filter: bool,
) -> Tuple[int, int, Dict[str, str]]:
    profiles_root = output_dir / "profiles"
    if not profiles_root.exists():
        return 0, 0, {}

    matched_profile_ids: Dict[str, str] = {}
    matched_count = 0
    total_json = 0

    for json_path in profiles_root.glob("*/*.json"):
        total_json += 1
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue

        profile_url = str(data.get("profile_url", "")).strip()
        profile_id = str(data.get("profile_id") or json_path.parent.name).strip()
        if not profile_id:
            continue

        if apply_excel_filter:
            if profile_url and excel_url_set and profile_url in excel_url_set:
                matched_count += 1
                matched_profile_ids[profile_id] = profile_url
        else:
            matched_count += 1
            matched_profile_ids[profile_id] = profile_url

    return matched_count, total_json, matched_profile_ids


def _scan_local_chunks(
    chunking_output_dir: Path,
    allowed_profile_ids: Optional[set],
) -> Tuple[int, int, Dict[str, Path]]:
    if not chunking_output_dir.exists():
        return 0, 0, {}

    chunk_files = list(chunking_output_dir.glob("*/chunks.json"))
    total_chunk_files = len(chunk_files)

    matched_chunk_files: Dict[str, Path] = {}
    matched_count = 0
    for chunk_path in chunk_files:
        profile_id = chunk_path.parent.name
        if allowed_profile_ids is None or profile_id in allowed_profile_ids:
            matched_count += 1
            matched_chunk_files[profile_id] = chunk_path

    return matched_count, total_chunk_files, matched_chunk_files


def _build_vector_ids_from_chunks(
    chunk_files: Dict[str, Path]
) -> Tuple[Dict[str, str], int]:
    vector_ids: Dict[str, str] = {}
    skipped_no_chunks = 0

    for profile_id, chunk_path in chunk_files.items():
        try:
            with open(chunk_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            skipped_no_chunks += 1
            continue

        sections = data.get("sections", {})
        chunk_id = ""
        for section_chunks in sections.values():
            for chunk in section_chunks:
                chunk_id = str(chunk.get("chunk_id", "")).strip()
                if chunk_id:
                    break
            if chunk_id:
                break

        if not chunk_id:
            skipped_no_chunks += 1
            continue

        vector_ids[profile_id] = f"chunk_{chunk_id}"

    return vector_ids, skipped_no_chunks


def _fetch_pinecone_vectors(vector_ids: Dict[str, str]) -> Tuple[int, int, Optional[str], Optional[Dict]]:
    try:
        from api.services.vector_db import get_vector_db
        from config.pinecone_config import INDEX_NAME, INDEX_DIMENSION
    except Exception as exc:
        return 0, 0, f"Pinecone import error: {exc}", None

    try:
        vector_db = get_vector_db(index_name=INDEX_NAME, dimension=INDEX_DIMENSION)
        stats = vector_db.index.describe_index_stats()
    except Exception as exc:
        return 0, 0, f"Pinecone connection error: {exc}", None

    vector_id_to_profile = {vector_id: profile_id for profile_id, vector_id in vector_ids.items()}
    found_profiles = set()

    id_list = list(vector_id_to_profile.keys())
    for batch in _chunked(id_list, 100):
        try:
            response = vector_db.index.fetch(ids=batch)
        except Exception:
            continue

        vectors = None
        if hasattr(response, "vectors"):
            vectors = response.vectors
        elif isinstance(response, dict):
            vectors = response.get("vectors")
        if not vectors:
            continue

        for vector_id in vectors.keys():
            profile_id = vector_id_to_profile.get(vector_id)
            if profile_id:
                found_profiles.add(profile_id)

    return len(found_profiles), len(vector_ids), None, stats


def _count_mongo_documents(profile_ids: List[str]) -> Tuple[int, int, Optional[str], Optional[str]]:
    try:
        from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name
    except Exception as exc:
        return 0, 0, f"Mongo import error: {exc}", None

    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        return 0, 0, "MONGODB_URI not set", None

    try:
        client = create_mongo_client(mongodb_uri)
    except Exception as exc:
        return 0, 0, f"Mongo connection error: {exc}", None

    db_name = resolve_mongo_db_name(mongodb_uri)
    collection = client[db_name].scholars

    try:
        total_docs = collection.count_documents({})
    except Exception:
        total_docs = 0

    matched_docs = 0
    if profile_ids:
        for batch in _chunked(profile_ids, 500):
            matched_docs += collection.count_documents({"profile_id": {"$in": batch}})

    try:
        client.close()
    except Exception:
        pass

    return matched_docs, total_docs, None, db_name


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Report progress for full_automation_pipeline.py across local files, Pinecone, and MongoDB."
    )
    parser.add_argument(
        "excel_path",
        nargs="?",
        default="profile.xlsx",
        help="Path to Excel file with profile URLs (default: profile.xlsx)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Base output directory for profiles JSON (default: output)",
    )
    parser.add_argument(
        "--chunking-output-dir",
        default="output/chunked_profiles",
        help="Output directory for chunked profiles (default: output/chunked_profiles)",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start index used for the run you are tracking (default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit used for the run you are tracking (default: no limit)",
    )
    parser.add_argument(
        "--skip-pinecone",
        action="store_true",
        help="Skip Pinecone checks",
    )
    parser.add_argument(
        "--skip-mongodb",
        action="store_true",
        help="Skip MongoDB checks",
    )

    args = parser.parse_args()

    excel_urls, total_valid, scoped_total, excel_error = _load_excel_urls(
        args.excel_path, args.start_from, args.limit
    )
    excel_url_set = set(excel_urls)
    apply_excel_filter = excel_error is None
    expected_rows = scoped_total if excel_error is None else 0

    output_dir = Path(args.output_dir)
    chunking_output_dir = Path(args.chunking_output_dir)

    local_matched, local_total, matched_profile_ids = _scan_local_profiles(
        output_dir, excel_url_set, apply_excel_filter
    )
    matched_profile_id_set = set(matched_profile_ids.keys())

    chunk_matched, chunk_total, matched_chunk_files = _scan_local_chunks(
        chunking_output_dir, matched_profile_id_set if apply_excel_filter else None
    )

    print("=" * 70)
    print("Full Automation Pipeline Progress")
    print("=" * 70)
    print(f"Excel file: {args.excel_path}")
    if excel_error:
        print(f"Excel status: {excel_error}")
    else:
        print(f"Valid URLs in Excel: {total_valid}")
        if args.start_from or args.limit:
            print(f"Scope (start_from={args.start_from}, limit={args.limit or 'none'}): {scoped_total}")

    print("\nLocal files")
    if excel_error:
        print(f"- Profiles JSON: {local_total}")
        print(f"- Chunk files: {chunk_total}")
    else:
        print(
            f"- Profiles JSON (matched): {local_matched} / {expected_rows} "
            f"({_format_pct(local_matched, expected_rows)})"
        )
        print(
            f"- Chunk files (matched): {chunk_matched} / {expected_rows} "
            f"({_format_pct(chunk_matched, expected_rows)})"
        )
        if local_total != local_matched:
            print(f"- Profiles JSON (total in output): {local_total}")
        if chunk_total != chunk_matched:
            print(f"- Chunk files (total in chunk dir): {chunk_total}")

    if not args.skip_pinecone:
        vector_ids, skipped_no_chunks = _build_vector_ids_from_chunks(matched_chunk_files)
        pinecone_found, pinecone_checked, pinecone_error, stats = _fetch_pinecone_vectors(vector_ids)

        print("\nPinecone")
        if pinecone_error:
            print(f"- Status: {pinecone_error}")
        else:
            total_vectors = getattr(stats, "total_vector_count", None)
            namespaces = getattr(stats, "namespaces", None)
            if total_vectors is not None:
                print(f"- Total vectors in index: {total_vectors}")
            if namespaces:
                print(f"- Namespaces: {namespaces}")
            if pinecone_checked:
                print(
                    f"- Profiles with vectors (matched): {pinecone_found} / {pinecone_checked} "
                    f"({_format_pct(pinecone_found, pinecone_checked)})"
                )
            if skipped_no_chunks:
                print(f"- Skipped chunk files without chunk IDs: {skipped_no_chunks}")

    if not args.skip_mongodb:
        mongo_found, mongo_total, mongo_error, db_name = _count_mongo_documents(
            list(matched_profile_id_set)
        )
        print("\nMongoDB")
        if mongo_error:
            print(f"- Status: {mongo_error}")
        else:
            print(f"- Database: {db_name}")
            if excel_error:
                print(f"- Scholars collection documents: {mongo_total}")
            else:
                print(
                    f"- Documents (matched): {mongo_found} / {expected_rows} "
                    f"({_format_pct(mongo_found, expected_rows)})"
                )
                if mongo_total != mongo_found:
                    print(f"- Documents (total in collection): {mongo_total}")

    print("=" * 70)

    # ── Data Quality Dashboard ──
    profiles_dir = Path(args.output_dir) / "profiles" if hasattr(args, "output_dir") else Path("output/osu_faculty_run/profiles")
    if profiles_dir.exists():
        print("\n" + "=" * 70)
        print("DATA QUALITY DASHBOARD")
        print("=" * 70)
        _print_quality_dashboard(profiles_dir)

    return 0


def _print_quality_dashboard(profiles_dir: Path) -> None:
    """Print enrichment data quality metrics."""
    import glob as _glob

    enrichment_files = list(profiles_dir.glob("*/enrichment.json"))
    if not enrichment_files:
        print("  No enrichment data found.")
        return

    total = len(enrichment_files)
    zero_sources = 0
    one_source = 0
    two_sources = 0
    three_plus = 0
    confidence_sum = 0.0
    source_success: Dict[str, int] = {}
    source_attempts: Dict[str, int] = {}
    cleaned_count = 0

    for f in enrichment_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        conf = data.get("confidence", {}).get("overall_confidence", 0)
        confidence_sum += conf

        succ = data.get("summary", {}).get("successful_sources", 0)
        if succ == 0:
            zero_sources += 1
        elif succ == 1:
            one_source += 1
        elif succ == 2:
            two_sources += 1
        else:
            three_plus += 1

        for sname, sdata in data.get("sources", {}).items():
            source_attempts[sname] = source_attempts.get(sname, 0) + 1
            if sdata.get("success"):
                source_success[sname] = source_success.get(sname, 0) + 1

        # Check if cleaned text exists
        cleaned_path = f.parent / "enrichment_text_cleaned.txt"
        if cleaned_path.exists():
            cleaned_count += 1

    total_profiles = len(list(profiles_dir.iterdir()))
    avg_conf = confidence_sum / max(total, 1)

    print(f"\n  Total profiles:              {total_profiles}")
    print(f"  Enriched:                    {total} ({total / max(total_profiles, 1) * 100:.1f}%)")
    print(f"  Cleaned (GPT-4o-mini):       {cleaned_count} ({cleaned_count / max(total, 1) * 100:.1f}%)")
    print(f"  Average confidence:          {avg_conf:.3f}")

    print(f"\n  Source success distribution:")
    print(f"    0 sources (empty):         {zero_sources} ({zero_sources / max(total, 1) * 100:.1f}%)")
    print(f"    1 source:                  {one_source} ({one_source / max(total, 1) * 100:.1f}%)")
    print(f"    2 sources:                 {two_sources} ({two_sources / max(total, 1) * 100:.1f}%)")
    print(f"    3+ sources:                {three_plus} ({three_plus / max(total, 1) * 100:.1f}%)")

    target_3plus = three_plus / max(total, 1) * 100
    print(f"\n  Quality targets:")
    print(f"    3+ sources (target 70%):   {target_3plus:.1f}% {'[PASS]' if target_3plus >= 70 else '[FAIL]'}")
    print(f"    Avg confidence (target 0.5): {avg_conf:.3f} {'[PASS]' if avg_conf >= 0.5 else '[FAIL]'}")

    print(f"\n  Per-source success rates:")
    print(f"  {'Source':<25} {'Success':>7} / {'Attempted':>9}  {'Rate':>6}")
    print(f"  {'-' * 25} {'-' * 7}   {'-' * 9}  {'-' * 6}")
    for sname in sorted(source_attempts.keys()):
        attempted = source_attempts[sname]
        succeeded = source_success.get(sname, 0)
        rate = succeeded / max(attempted, 1) * 100
        bar = "#" * int(rate / 5) + "-" * (20 - int(rate / 5))
        print(f"  {sname:<25} {succeeded:>7} / {attempted:>9}  {rate:>5.1f}% {bar}")

    # AI-readiness estimate
    usable = two_sources + three_plus
    readiness = usable / max(total, 1) * 100
    print(f"\n  AI-readiness (2+ sources):   {usable}/{total} ({readiness:.1f}%)")
    if readiness >= 60:
        print(f"  Verdict: [PASS] Data is usable for professor-specific AI")
    elif readiness >= 40:
        print(f"  Verdict: [WARN] Marginal - many professors will have thin profiles")
    else:
        print(f"  Verdict: [FAIL] Too many empty/thin profiles for reliable AI")


if __name__ == "__main__":
    raise SystemExit(main())
