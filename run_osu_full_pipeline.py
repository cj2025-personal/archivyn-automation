"""
Run the full automation pipeline for OSU.xlsx.

Pipeline:
- Excel -> scraping (profile/CV/personal-site)
- Cleaning + section-aware chunking
- Upload to Pinecone
- Sync to MongoDB
"""

import asyncio
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from full_automation_pipeline import run_full_pipeline


def _configure_logger() -> logging.Logger:
    level_name = os.getenv("OSU_RUN_LOG_LEVEL", "DEBUG").upper()
    level = getattr(logging, level_name, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("osu_full_pipeline_run")


def _env_present(name: str) -> bool:
    return bool(os.getenv(name))


def _read_valid_profile_rows(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(excel_path)

    if "source" in df.columns and "profile_url" not in df.columns:
        df = df.rename(columns={"source": "profile_url"})
    elif "profile_url" not in df.columns:
        url_columns = [col for col in df.columns if "url" in str(col).lower() or "link" in str(col).lower()]
        if url_columns:
            df = df.rename(columns={url_columns[0]: "profile_url"})
        else:
            raise ValueError("Excel file must contain a 'source' or 'profile_url' column")

    valid_df = df[df["profile_url"].notna()].copy()
    valid_df["_excel_row_number"] = valid_df.index + 2
    valid_df = valid_df.reset_index(drop=True)

    if valid_df.empty:
        raise ValueError("No valid profile URLs found in Excel file")

    return valid_df


def _resolve_start_index_from_row_number(
    excel_path: Path,
    start_row_number: int,
) -> tuple[int, int, int]:
    if start_row_number < 2:
        raise ValueError("--start-row-number must be >= 2 because row 1 is the header")

    valid_df = _read_valid_profile_rows(excel_path)
    match_df = valid_df[valid_df["_excel_row_number"] >= start_row_number]
    if match_df.empty:
        last_valid_row = int(valid_df["_excel_row_number"].max())
        raise ValueError(
            f"No valid profile URL found at or after Excel row {start_row_number}. "
            f"Last valid profile row is {last_valid_row}."
        )

    start_from = int(match_df.index[0])
    resolved_row_number = int(match_df.iloc[0]["_excel_row_number"])
    resolved_profile_number = start_from + 1
    return start_from, resolved_profile_number, resolved_row_number


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the OSU full pipeline with optional resume/start controls."
    )
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=Path("OSU.xlsx"),
        help="Path to OSU Excel file (default: OSU.xlsx).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"output\osu_faculty_run"),
        help=r"Output directory (default: output\osu_faculty_run).",
    )
    parser.add_argument(
        "--start-profile-number",
        type=int,
        default=None,
        help=(
            "1-based profile number to start from after URL filtering "
            "(example: 783 starts from the 783rd profile)."
        ),
    )
    parser.add_argument(
        "--start-row-number",
        type=int,
        default=None,
        help=(
            "1-based Excel row number to resume from, including the header row as 1. "
            "If the requested row has no URL, the next valid URL row is used."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of profiles to process from the start point.",
    )
    return parser.parse_args()


async def _main(logger: logging.Logger, args: argparse.Namespace) -> None:
    # Faculty-profile runs are often filtered out by intent gates tuned for speech/testimony corpora.
    logger.debug("Applying default environment flags for faculty-profile ingestion.")
    os.environ.setdefault("INTENT_GATING_ENABLED", "0")
    os.environ.setdefault("STRICT_SOURCE_POLICY", "0")
    os.environ.setdefault("DEBUG_SCRAPER_LINKS", "1")

    excel_path = args.excel_path
    output_dir = args.output_dir
    chunking_output_dir = output_dir / "chunked_profiles"

    if args.start_profile_number is not None and args.start_row_number is not None:
        raise ValueError("Specify only one of --start-profile-number or --start-row-number")

    resolved_row_number = None
    if args.start_row_number is not None:
        start_from, start_profile_number, resolved_row_number = _resolve_start_index_from_row_number(
            excel_path,
            int(args.start_row_number),
        )
    else:
        start_profile_number = int(args.start_profile_number or 1)
        if start_profile_number < 1:
            raise ValueError("--start-profile-number must be >= 1")
        start_from = start_profile_number - 1

    logger.info("OSU full pipeline run starting.")
    logger.debug("Working directory: %s", Path.cwd())
    logger.debug("Python executable: %s", sys.executable)
    logger.debug("Excel path: %s", excel_path.resolve())
    logger.debug("Output directory: %s", output_dir.resolve())
    logger.debug("Chunking output directory: %s", chunking_output_dir.resolve())
    logger.info(
        "Resume config | requested_start_row=%s resolved_start_row=%s "
        "start_profile_number=%s start_from_index=%s limit=%s",
        args.start_row_number,
        resolved_row_number,
        start_profile_number,
        start_from,
        args.limit,
    )
    logger.debug(
        "Runtime flags | INTENT_GATING_ENABLED=%s STRICT_SOURCE_POLICY=%s DEBUG_SCRAPER_LINKS=%s",
        os.getenv("INTENT_GATING_ENABLED"),
        os.getenv("STRICT_SOURCE_POLICY"),
        os.getenv("DEBUG_SCRAPER_LINKS"),
    )
    logger.debug(
        "Secret presence | OPENAI_API_KEY=%s PINECONE_API_KEY=%s MONGODB_URI=%s",
        _env_present("OPENAI_API_KEY"),
        _env_present("PINECONE_API_KEY"),
        _env_present("MONGODB_URI"),
    )

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path.resolve()}")

    start = time.perf_counter()
    logger.info("Invoking run_full_pipeline(...)")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured output directory exists.")

        await run_full_pipeline(
            excel_path=str(excel_path),
            output_dir=str(output_dir),
            chunking_output_dir=str(chunking_output_dir),
            use_llm_chunking=True,
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            limit=args.limit,
            start_from=start_from,
            pinecone_batch_size=50,
            skip_pinecone=False,
            skip_mongodb=False,
            skip_indexes=False,
        )
    except Exception:
        logger.exception("run_full_pipeline failed with an exception.")
        raise
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Pipeline invocation finished in %.2f seconds.", elapsed)
        logger.debug("Output directory exists: %s", output_dir.exists())
        logger.debug("Chunking directory exists: %s", chunking_output_dir.exists())


if __name__ == "__main__":
    load_dotenv()
    args = _parse_args()
    logger = _configure_logger()
    logger.debug("Loaded .env file.")
    if sys.platform == "win32":
        logger.debug("Applying WindowsSelectorEventLoopPolicy.")
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_main(logger, args))
