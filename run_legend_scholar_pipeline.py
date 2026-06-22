"""
Run Legend Scholar URL-list pipelines independently from Excel-based runs.

Behavior:
- Uses URL-list mode only (one .txt file = one profile run).
- Defaults to all .txt files in the final/ directory.
- Uses the TXT filename stem as the scholar profile name unless overridden.
- Writes each run to: output/url_list_runs/<timestamp>-<file-stem>/

Examples:
    python run_legend_scholar_pipeline.py
    python run_legend_scholar_pipeline.py legend-scholar-1.txt legend-scholar-2.txt
    python run_legend_scholar_pipeline.py --skip-pinecone --skip-mongodb
    python run_legend_scholar_pipeline.py --keep-filters
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
from typing import List, Sequence

from dotenv import load_dotenv

from full_automation_pipeline import run_full_pipeline
from unified_pipeline import load_urls_file


@dataclass
class RunResult:
    urls_file: Path
    output_dir: Path
    status: str
    error: str = ""


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    slug = slug.strip("-")
    return slug or "legend-scholar"


def _profile_name_from_stem(stem: str) -> str:
    """Turn the URL-list filename stem into a human-readable subject name.

    Examples:
        carter-g-woodson      -> "Carter G. Woodson"
        web-du-bois           -> "W. E. B. Du Bois"
        ida-stephens-owens    -> "Ida Stephens Owens"
        cheikh-anta-diop      -> "Cheikh Anta Diop"
        ivan-van-sertima      -> "Ivan Van Sertima"

    The previous version produced "Web Du Bois" / "Carter G Woodson"
    (no period after single-letter initials). Single-letter parts now
    get a period so downstream subject-mention filters can match
    forms like "W.E.B. Du Bois" in source pages.
    """
    parts = [part for part in re.split(r"[-_]+", stem.strip()) if part]
    # OSU URL-list files are named "<name-slug>-<8-hex-id>.txt"; the trailing
    # hex token is a profile-id fragment, not part of the person's name.
    # Leaving it in poisons the derived name ("Khaled Boubes 14a85ba0"), which
    # then makes the downstream relevance/subject filters treat the hash as the
    # surname and discard every scraped page.
    if len(parts) > 1 and re.fullmatch(r"[0-9a-f]{8}", parts[-1].lower()):
        parts = parts[:-1]
    if not parts:
        return "Legend Scholar"
    formatted: List[str] = []
    for part in parts:
        if len(part) == 1:
            formatted.append(part.upper() + ".")
        elif part.lower() == "web":
            # "web-du-bois" → "W. E. B." (the "WEB" stem is actually three
            # initials of W.E.B. Du Bois that got collapsed in the filename).
            formatted.append("W. E. B.")
        else:
            formatted.append(part.capitalize())
    return " ".join(formatted)


def _resolve_urls_files(urls_files: Sequence[str]) -> List[Path]:
    if urls_files:
        files = [Path(p) for p in urls_files]
    else:
        final_dir = Path("final")
        if final_dir.exists():
            files = sorted(final_dir.glob("*.txt"))
        else:
            files = sorted(Path(".").glob("legend-scholar-*.txt"))

    resolved: List[Path] = []
    seen = set()
    for file_path in files:
        candidate = file_path.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        resolved.append(candidate)
    return resolved


def _load_urls_file_metadata(urls_file: Path) -> dict:
    meta_path = urls_file.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Legend Scholar URL-list pipeline as a separate workflow"
    )
    parser.add_argument(
        "urls_files",
        nargs="*",
        help="Legend URL list .txt files (default: final/*.txt)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path("output") / "url_list_runs"),
        help="Base directory where per-file run folders are created",
    )
    parser.add_argument(
        "--run-stamp",
        type=str,
        default=None,
        help="Optional run stamp override (default: current timestamp)",
    )
    parser.add_argument(
        "--profile-name",
        type=str,
        default=None,
        help="Optional profile name to force for all selected URL files",
    )
    parser.add_argument(
        "--profile-url",
        type=str,
        default=None,
        help="Optional primary profile URL to force for all selected URL files",
    )
    parser.add_argument(
        "--no-llm-chunking",
        action="store_true",
        help="Disable LLM section-aware chunking",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="openai",
        choices=["openai", "ollama"],
        help="LLM provider used by chunking",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name",
    )
    parser.add_argument(
        "--pinecone-batch-size",
        type=int,
        default=50,
        help="Batch size for Pinecone upload",
    )
    parser.add_argument(
        "--skip-pinecone",
        action="store_true",
        help="Skip Pinecone upload",
    )
    parser.add_argument(
        "--skip-mongodb",
        action="store_true",
        help="Skip MongoDB sync",
    )
    parser.add_argument(
        "--skip-indexes",
        action="store_true",
        help="Skip MongoDB index creation",
    )
    parser.add_argument(
        "--no-incremental-sync",
        action="store_true",
        help="Disable incremental Pinecone/Mongo sync in UnifiedPipeline",
    )
    parser.add_argument(
        "--sync-batch-size",
        type=int,
        default=100,
        help="Profiles per incremental sync batch",
    )
    parser.add_argument(
        "--sync-pinecone-batch-size",
        type=int,
        default=50,
        help="Chunk batch size per incremental Pinecone flush",
    )
    parser.add_argument(
        "--keep-filters",
        action="store_true",
        help="Keep default role/intent/source/content filters enabled",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    urls_files = _resolve_urls_files(args.urls_files)
    if not urls_files:
        print("No Legend Scholar URL files found. Provide files or add .txt files under final/.")
        return 1

    run_stamp = args.run_stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[RunResult] = []
    total = len(urls_files)

    for idx, urls_file in enumerate(urls_files, start=1):
        metadata = _load_urls_file_metadata(urls_file)
        label = _slugify(urls_file.stem)
        profile_name = (
            args.profile_name.strip()
            if isinstance(args.profile_name, str) and args.profile_name.strip()
            else str(metadata.get("profile_name") or "").strip() or _profile_name_from_stem(urls_file.stem)
        )
        profile_id = str(metadata.get("profile_id") or "").strip() or None
        profile_url = args.profile_url or metadata.get("profile_url")
        output_dir = output_root / f"{run_stamp}-{label}"
        chunking_output_dir = output_dir / "chunked_profiles"

        print("=" * 80)
        print(f"[LegendRun] {idx}/{total} | file={urls_file}")
        print(f"[LegendRun] profile_name={profile_name}")
        print(f"[LegendRun] output={output_dir}")
        print("=" * 80)

        try:
            urls = load_urls_file(str(urls_file))
            if not urls:
                raise ValueError(f"No valid URLs in file: {urls_file}")

            await run_full_pipeline(
                excel_path=None,
                urls=urls,
                profile_name=profile_name,
                profile_id=profile_id,
                profile_url=profile_url,
                output_dir=str(output_dir),
                chunking_output_dir=str(chunking_output_dir),
                use_llm_chunking=not args.no_llm_chunking,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                pinecone_batch_size=args.pinecone_batch_size,
                skip_pinecone=args.skip_pinecone,
                skip_mongodb=args.skip_mongodb,
                skip_indexes=args.skip_indexes,
                incremental_sync=not args.no_incremental_sync,
                incremental_sync_batch_size=args.sync_batch_size,
                incremental_sync_pinecone_batch_size=args.sync_pinecone_batch_size,
            )
            results.append(RunResult(urls_file=urls_file, output_dir=output_dir, status="success"))
        except Exception as exc:
            results.append(
                RunResult(
                    urls_file=urls_file,
                    output_dir=output_dir,
                    status="failed",
                    error=str(exc),
                )
            )
            print(f"[LegendRun] failed: {urls_file.name} :: {exc}")

    print("\n" + "=" * 80)
    print("[LegendRun] Summary")
    print("=" * 80)
    success_count = 0
    failed_count = 0
    for res in results:
        if res.status == "success":
            success_count += 1
            print(f"  success | {res.urls_file.name} | {res.output_dir}")
        else:
            failed_count += 1
            print(f"  failed  | {res.urls_file.name} | {res.error}")
    print(f"[LegendRun] successful={success_count} failed={failed_count} total={len(results)}")

    return 0 if failed_count == 0 else 1


def main() -> int:
    load_dotenv()
    args = _parse_args()

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    if not args.keep_filters:
        # Page-level subject filtering now lives in
        # ``_content_mentions_subject`` inside the unified pipeline, which
        # drops whole pages that don't mention the legend. The older
        # per-*line* filter (``PROFILE_TEXT_FILTER_ENABLED``) is too crude
        # for biographical content — it strips lines that are short or
        # don't repeat the subject's name (dates, headings, citations) and
        # left individual sources with too little text to chunk.
        os.environ.setdefault("INTENT_GATING_ENABLED", "0")
        os.environ.setdefault("STRICT_SOURCE_POLICY", "0")
        os.environ.setdefault("PROFILE_ROLE_FILTER_ENABLED", "0")
        os.environ.setdefault("SOURCE_QUALITY_FILTER_ENABLED", "1")
        os.environ.setdefault("PROFILE_RELEVANCE_FILTER_ENABLED", "1")
        os.environ.setdefault("PROFILE_TEXT_FILTER_ENABLED", "0")
        print(
            "[LegendRun] Cross-source intent gate disabled. Page-level "
            "relevance + source-quality filters and the noise-domain "
            "blocklist remain on; the line-level text filter is left off "
            "so biographical pages aren't shredded line-by-line."
        )

    if args.sync_batch_size <= 0:
        raise ValueError("--sync-batch-size must be > 0")
    if args.sync_pinecone_batch_size <= 0:
        raise ValueError("--sync-pinecone-batch-size must be > 0")
    if args.pinecone_batch_size <= 0:
        raise ValueError("--pinecone-batch-size must be > 0")

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
