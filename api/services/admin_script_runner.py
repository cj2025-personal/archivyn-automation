from __future__ import annotations

import ast
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional

from api.services.admin_script_execution import RuntimeJob, execution_service


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DISCOVER_GLOB = "*.py"
MAX_LOG_LINES = 4000
STREAM_POLL_SECONDS = 0.35

SCRIPT_IGNORE = {
    "main.py",
    "start_server.py",
    "playwright_subprocess_worker.py",
    "semantic_splitter.py",
}

DANGER_PATTERNS = (
    "cleanup",
    "clear",
    "delete",
    "drop",
    "archive",
    "migrate",
)

CAUTION_PATTERNS = (
    "backfill",
    "dedupe",
    "fix",
    "sync",
)

CATEGORY_PREFIXES = {
    "run_": "Pipeline",
    "collect_": "Collection",
    "daily_story_": "Editorial",
    "audit_": "Audit",
    "profile_": "Profiles",
    "backfill_": "Backfill",
    "build_": "Reporting",
    "upload_": "Upload",
    "sync_": "Sync",
    "cleanup_": "Cleanup",
    "clear_": "Cleanup",
    "delete_": "Cleanup",
    "fix_": "Repair",
    "migrate_": "Migration",
}

SCOPE_PATTERNS = {
    "osu": "OSU-specific",
    "legend": "Legend-specific",
    "legendary": "Legend-specific",
    "universit": "Multi-university",
    "scholars": "General",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def humanize_stem(stem: str) -> str:
    return " ".join(part.capitalize() for part in stem.replace("-", "_").split("_") if part)


def infer_category(stem: str) -> str:
    for prefix, category in CATEGORY_PREFIXES.items():
        if stem.startswith(prefix):
            return category
    return "Utilities"


def infer_scope(stem: str, docstring: str) -> str:
    haystack = f"{stem}\n{docstring}".lower()
    for token, label in SCOPE_PATTERNS.items():
        if token in haystack:
            return label
    return "General"


def infer_risk(stem: str, docstring: str) -> str:
    haystack = f"{stem}\n{docstring}".lower()
    if any(token in haystack for token in DANGER_PATTERNS):
        return "danger"
    if any(token in haystack for token in CAUTION_PATTERNS):
        return "caution"
    return "safe"


def extract_usage_examples(docstring: str) -> List[str]:
    examples: List[str] = []
    for raw_line in docstring.splitlines():
        line = raw_line.strip()
        if line.startswith("python "):
            examples.append(line)
    return examples[:8]


def summarize_docstring(docstring: str) -> str:
    for raw_line in docstring.splitlines():
        line = raw_line.strip()
        if line:
            return line
    return "No summary available."


def is_runnable_script(path: Path, text: str) -> bool:
    if path.name in SCRIPT_IGNORE or path.name.startswith("_"):
        return False
    if "__main__" in text:
        return True
    if "argparse.ArgumentParser" in text:
        return True
    return False


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def split_multiline(value: Any) -> List[str]:
    return [part.strip() for part in re.split(r"[\r\n]+", clean_string(value)) if part.strip()]


def bool_value(values: Mapping[str, Any], key: str, default: bool = False) -> bool:
    raw = values.get(key, default)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default


def maybe_add_option(args: List[str], flag: str, value: Any) -> None:
    text = clean_string(value)
    if text:
        args.extend([flag, text])


@dataclass
class ScriptDefinition:
    id: str
    filename: str
    path: str
    title: str
    summary: str
    category: str
    scope: str
    risk: str
    usage_examples: List[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "filename": self.filename,
            "path": self.path,
            "title": self.title,
            "summary": self.summary,
            "category": self.category,
            "scope": self.scope,
            "risk": self.risk,
            "usage_examples": self.usage_examples,
        }


@dataclass
class ScriptField:
    id: str
    label: str
    type: str
    help_text: str
    default: Any = ""
    placeholder: str = ""
    required: bool = False
    options: List[Dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "help_text": self.help_text,
            "default": self.default,
            "placeholder": self.placeholder,
            "required": self.required,
            "options": self.options,
        }


@dataclass
class ScriptModule:
    id: str
    script_id: str
    title: str
    subtitle: str
    summary: str
    category: str
    scope: str
    risk: str
    mode: str
    featured: bool
    run_label: str
    success_message: str
    caution_message: str
    indications: List[str]
    fields: List[ScriptField]
    usage_examples: List[str]
    script_filename: str
    script_path: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "script_id": self.script_id,
            "title": self.title,
            "subtitle": self.subtitle,
            "summary": self.summary,
            "category": self.category,
            "scope": self.scope,
            "risk": self.risk,
            "mode": self.mode,
            "featured": self.featured,
            "run_label": self.run_label,
            "success_message": self.success_message,
            "caution_message": self.caution_message,
            "indications": self.indications,
            "fields": [field.as_dict() for field in self.fields],
            "usage_examples": self.usage_examples,
            "script_filename": self.script_filename,
            "script_path": self.script_path,
        }


def approval_required_for_risk(risk: str) -> bool:
    configured = os.getenv("ADMIN_APPROVAL_REQUIRED_RISKS", "danger").strip().lower()
    allowed = {part.strip() for part in configured.split(",") if part.strip()}
    return risk.lower() in allowed


def _build_legend_audit_args(values: Mapping[str, Any]) -> List[str]:
    names = split_multiline(values.get("names"))
    profile_ids = split_multiline(values.get("profile_ids"))
    audit_all = bool_value(values, "audit_all", False)
    if not names and not profile_ids and not audit_all:
        raise ValueError("Provide at least one legend name, profile id, or enable full collection audit.")

    args: List[str] = []
    if names:
        args.extend(["--names", *names])
    if profile_ids:
        args.extend(["--profile-ids", *profile_ids])
    if audit_all:
        args.append("--all")
    maybe_add_option(args, "--limit", values.get("limit"))
    if bool_value(values, "apply", True):
        args.append("--apply")
    maybe_add_option(args, "--output-json", values.get("output_json"))
    maybe_add_option(args, "--output-md", values.get("output_md"))
    return args


def _build_collect_legendary_urls_args(values: Mapping[str, Any]) -> List[str]:
    args: List[str] = []
    maybe_add_option(args, "--excel-path", values.get("excel_path"))
    maybe_add_option(args, "--limit", values.get("limit"))
    maybe_add_option(args, "--start-from", values.get("start_from"))
    maybe_add_option(args, "--max-urls", values.get("max_urls"))
    maybe_add_option(args, "--sleep-seconds", values.get("sleep_seconds"))
    maybe_add_option(args, "--per-query-results", values.get("per_query_results"))
    maybe_add_option(args, "--output-dir", values.get("output_dir"))
    maybe_add_option(args, "--checkpoint-file", values.get("checkpoint_file"))
    maybe_add_option(args, "--log-file", values.get("log_file"))
    maybe_add_option(args, "--slugs-file", values.get("slugs_file"))
    if bool_value(values, "skip_existing", True):
        args.append("--skip-existing")
    return args


def _build_legendary_enrichment_args(values: Mapping[str, Any]) -> List[str]:
    args: List[str] = []
    maybe_add_option(args, "--excel-path", values.get("excel_path"))
    maybe_add_option(args, "--limit", values.get("limit"))
    maybe_add_option(args, "--start-from", values.get("start_from"))
    maybe_add_option(args, "--max-urls", values.get("max_urls"))
    maybe_add_option(args, "--sleep-seconds", values.get("sleep_seconds"))
    maybe_add_option(args, "--per-query-results", values.get("per_query_results"))
    maybe_add_option(args, "--output-dir", values.get("output_dir"))
    maybe_add_option(args, "--checkpoint-file", values.get("checkpoint_file"))
    maybe_add_option(args, "--log-file", values.get("log_file"))
    maybe_add_option(args, "--slugs-file", values.get("slugs_file"))
    maybe_add_option(args, "--batch-size", values.get("batch_size"))
    maybe_add_option(args, "--runs-root", values.get("runs_root"))
    maybe_add_option(args, "--mongo-collection", values.get("mongo_collection"))
    if bool_value(values, "skip_existing", True):
        args.append("--skip-existing")
    if bool_value(values, "resume", False):
        args.append("--resume")
    if bool_value(values, "skip_collection", False):
        args.append("--skip-collection")
    if bool_value(values, "skip_pipeline", False):
        args.append("--skip-pipeline")
    if bool_value(values, "skip_pinecone", False):
        args.append("--skip-pinecone")
    if bool_value(values, "skip_mongodb", False):
        args.append("--skip-mongodb")
    if bool_value(values, "no_mongo_llm", False):
        args.append("--no-mongo-llm")
    return args


def _build_legend_pipeline_args(values: Mapping[str, Any]) -> List[str]:
    args: List[str] = []
    url_files = split_multiline(values.get("urls_files"))
    if url_files:
        args.extend(url_files)
    maybe_add_option(args, "--output-root", values.get("output_root"))
    maybe_add_option(args, "--run-stamp", values.get("run_stamp"))
    maybe_add_option(args, "--profile-name", values.get("profile_name"))
    maybe_add_option(args, "--profile-url", values.get("profile_url"))
    if bool_value(values, "no_llm_chunking", False):
        args.append("--no-llm-chunking")
    maybe_add_option(args, "--llm-provider", values.get("llm_provider"))
    maybe_add_option(args, "--llm-model", values.get("llm_model"))
    maybe_add_option(args, "--pinecone-batch-size", values.get("pinecone_batch_size"))
    if bool_value(values, "skip_pinecone", False):
        args.append("--skip-pinecone")
    if bool_value(values, "skip_mongodb", False):
        args.append("--skip-mongodb")
    if bool_value(values, "skip_indexes", False):
        args.append("--skip-indexes")
    if bool_value(values, "no_incremental_sync", False):
        args.append("--no-incremental-sync")
    maybe_add_option(args, "--sync-batch-size", values.get("sync_batch_size"))
    maybe_add_option(args, "--sync-pinecone-batch-size", values.get("sync_pinecone_batch_size"))
    if bool_value(values, "keep_filters", False):
        args.append("--keep-filters")
    return args


def _build_universities_folder_args(values: Mapping[str, Any]) -> List[str]:
    args: List[str] = []
    maybe_add_option(args, "--input-dir", values.get("input_dir"))
    maybe_add_option(args, "--output-root", values.get("output_root"))
    maybe_add_option(args, "--workbook-glob", values.get("workbook_glob"))
    maybe_add_option(args, "--workbook-name-contains", values.get("workbook_name_contains"))
    maybe_add_option(args, "--limit-workbooks", values.get("limit_workbooks"))
    maybe_add_option(args, "--limit-scholars", values.get("limit_scholars"))
    maybe_add_option(args, "--start-from", values.get("start_from"))
    maybe_add_option(args, "--max-urls", values.get("max_urls"))
    maybe_add_option(args, "--per-query-results", values.get("per_query_results"))
    maybe_add_option(args, "--sleep-seconds", values.get("sleep_seconds"))
    maybe_add_option(args, "--http-timeout-seconds", values.get("http_timeout_seconds"))
    maybe_add_option(args, "--http-retries", values.get("http_retries"))
    if not bool_value(values, "skip_existing", True):
        args.append("--no-skip-existing")
    if bool_value(values, "force", False):
        args.append("--force")
    if bool_value(values, "aggressive", False):
        args.append("--aggressive")
    if bool_value(values, "disable_openalex", False):
        args.append("--disable-openalex")
    if bool_value(values, "dry_run", False):
        args.append("--dry-run")
    if not bool_value(values, "merge_existing", True):
        args.append("--no-merge-existing")
    return args


def _build_profile_quality_refresh_args(values: Mapping[str, Any]) -> List[str]:
    args: List[str] = []
    maybe_add_option(args, "--max-scholars", values.get("max_scholars"))
    maybe_add_option(args, "--scholar-id", values.get("scholar_id"))
    if bool_value(values, "update_scholar_docs", True):
        args.append("--update-scholar-docs")
    return args


def _build_daily_story_worker_args(values: Mapping[str, Any]) -> List[str]:
    args: List[str] = []
    maybe_add_option(args, "--scholar-id", values.get("scholar_id"))
    maybe_add_option(args, "--date", values.get("date"))
    maybe_add_option(args, "--topic", values.get("topic"))
    maybe_add_option(args, "--max-scholars", values.get("max_scholars"))
    maybe_add_option(args, "--max-context-chunks", values.get("max_context_chunks"))
    maybe_add_option(args, "--scholars-collection", values.get("scholars_collection"))
    maybe_add_option(args, "--stories-collection", values.get("stories_collection"))
    maybe_add_option(args, "--jobs-collection", values.get("jobs_collection"))
    maybe_add_option(args, "--model", values.get("model"))
    if bool_value(values, "dry_run", False):
        args.append("--dry-run")
    if bool_value(values, "no_llm", False):
        args.append("--no-llm")
    if bool_value(values, "disable_trends", False):
        args.append("--disable-trends")
    maybe_add_option(args, "--trend-provider", values.get("trend_provider"))
    if bool_value(values, "enforce_profile_quality", False):
        args.append("--enforce-profile-quality")
    if bool_value(values, "ignore_profile_quality", False):
        args.append("--ignore-profile-quality")
    maybe_add_option(args, "--profile-quality-min-score", values.get("profile_quality_min_score"))
    if bool_value(values, "publish_without_review", False):
        args.append("--publish-without-review")
    if bool_value(values, "assume_living", False):
        args.append("--assume-living")
    return args


def _build_daily_story_suite_args(values: Mapping[str, Any]) -> List[str]:
    args: List[str] = []
    maybe_add_option(args, "--date", values.get("date"))
    maybe_add_option(args, "--max-scholars", values.get("max_scholars"))
    maybe_add_option(args, "--max-context-chunks", values.get("max_context_chunks"))
    maybe_add_option(args, "--trend-provider", values.get("trend_provider"))
    if bool_value(values, "enforce_profile_quality", False):
        args.append("--enforce-profile-quality")
    maybe_add_option(args, "--profile-quality-min-score", values.get("profile_quality_min_score"))
    if bool_value(values, "disable_trends", False):
        args.append("--disable-trends")
    if bool_value(values, "dry_run", False):
        args.append("--dry-run")
    if bool_value(values, "no_llm", False):
        args.append("--no-llm")
    return args


MODULE_BUILDERS: Dict[str, Callable[[Mapping[str, Any]], List[str]]] = {
    "legend_readiness_audit": _build_legend_audit_args,
    "collect_legendary_urls": _build_collect_legendary_urls_args,
    "legendary_enrichment": _build_legendary_enrichment_args,
    "legend_pipeline": _build_legend_pipeline_args,
    "universities_folder_pipeline": _build_universities_folder_args,
    "profile_quality_refresh": _build_profile_quality_refresh_args,
    "daily_story_worker": _build_daily_story_worker_args,
    "daily_story_suite": _build_daily_story_suite_args,
}


class AdminScriptRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._catalog_cache: Optional[List[ScriptDefinition]] = None
        self._module_cache: Optional[List[ScriptModule]] = None

    def catalog(self) -> List[ScriptDefinition]:
        with self._lock:
            if self._catalog_cache is None:
                self._catalog_cache = self._discover_scripts()
            return list(self._catalog_cache)

    def refresh_catalog(self) -> List[ScriptDefinition]:
        with self._lock:
            self._catalog_cache = self._discover_scripts()
            self._module_cache = self._build_modules(self._catalog_cache)
            return list(self._catalog_cache)

    def modules(self) -> List[ScriptModule]:
        with self._lock:
            if self._module_cache is None:
                if self._catalog_cache is None:
                    self._catalog_cache = self._discover_scripts()
                self._module_cache = self._build_modules(self._catalog_cache)
            return list(self._module_cache)

    def refresh_modules(self) -> List[ScriptModule]:
        with self._lock:
            self._catalog_cache = self._discover_scripts()
            self._module_cache = self._build_modules(self._catalog_cache)
            return list(self._module_cache)

    def _discover_scripts(self) -> List[ScriptDefinition]:
        scripts: List[ScriptDefinition] = []
        for path in sorted(WORKSPACE_ROOT.glob(DISCOVER_GLOB)):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if not is_runnable_script(path, text):
                continue

            try:
                module = ast.parse(text)
                docstring = ast.get_docstring(module) or ""
            except Exception:
                docstring = ""

            stem = path.stem
            scripts.append(
                ScriptDefinition(
                    id=stem,
                    filename=path.name,
                    path=str(path.relative_to(WORKSPACE_ROOT)),
                    title=humanize_stem(stem),
                    summary=summarize_docstring(docstring),
                    category=infer_category(stem),
                    scope=infer_scope(stem, docstring),
                    risk=infer_risk(stem, docstring),
                    usage_examples=extract_usage_examples(docstring),
                )
            )

        scripts.sort(key=lambda item: (item.category, item.risk, item.title.lower()))
        return scripts

    def _build_modules(self, scripts: List[ScriptDefinition]) -> List[ScriptModule]:
        script_map = {script.id: script for script in scripts}
        modules: List[ScriptModule] = []
        used_scripts: set[str] = set()

        def field(
            field_id: str,
            label: str,
            field_type: str,
            help_text: str,
            *,
            default: Any = "",
            placeholder: str = "",
            required: bool = False,
            options: Optional[List[Dict[str, str]]] = None,
        ) -> ScriptField:
            return ScriptField(
                id=field_id,
                label=label,
                type=field_type,
                help_text=help_text,
                default=default,
                placeholder=placeholder,
                required=required,
                options=options or [],
            )

        curated_specs: List[Dict[str, Any]] = [
            {
                "id": "legend_readiness_audit",
                "script_id": "audit_legend_feature_readiness",
                "title": "Legend Readiness Audit",
                "subtitle": "Score demo readiness for editorial, podcast, chatbot, and profile pages.",
                "summary": "Audits `legend_scholars`, produces reports, and can write readiness metadata back into the same Mongo documents.",
                "category": "Legends",
                "scope": "Legend-specific",
                "risk": "caution",
                "featured": True,
                "run_label": "Run audit",
                "success_message": "Legend readiness metadata was refreshed. Review the report and console for gaps.",
                "caution_message": "When `Apply to Mongo` is enabled, this updates fields on existing `legend_scholars` records.",
                "indications": [
                    "Use names or profile ids for targeted audits.",
                    "Enable full collection mode only when you want a broader sweep.",
                    "Outputs can also be written to JSON and Markdown reports under `output/legend_audits`.",
                ],
                "fields": [
                    field(
                        "names",
                        "Legend Names",
                        "textarea",
                        "One exact `professor_name` per line.",
                        default="John Hope Franklin\nSamella Lewis\nCarter G. Woodson",
                        placeholder="John Hope Franklin\nSamella Lewis",
                    ),
                    field(
                        "profile_ids",
                        "Profile IDs",
                        "textarea",
                        "Optional targeted `profile_id` values, one per line.",
                        placeholder="john-hope-franklin",
                    ),
                    field("audit_all", "Audit Entire Collection", "checkbox", "Group and score every legend in MongoDB.", default=False),
                    field("limit", "Limit Groups", "number", "Optional limit when full collection mode is enabled.", default=25),
                    field("apply", "Apply to Mongo", "checkbox", "Write readiness fields onto the existing legend records.", default=True),
                    field(
                        "output_json",
                        "JSON Report Path",
                        "text",
                        "Optional JSON output path.",
                        default="output/legend_audits/legend_readiness_report.json",
                    ),
                    field(
                        "output_md",
                        "Markdown Report Path",
                        "text",
                        "Optional Markdown output path.",
                        default="output/legend_audits/legend_readiness_report.md",
                    ),
                ],
            },
            {
                "id": "collect_legendary_urls",
                "script_id": "collect_legendary_scholar_urls",
                "title": "Collect Legendary Scholar URLs",
                "subtitle": "Build URL-list files from the canonical legendary workbook.",
                "summary": "Queries DuckDuckGo and trusted biography sources, then writes one `.txt` URL list per legend.",
                "category": "Legends",
                "scope": "Legend-specific",
                "risk": "safe",
                "featured": True,
                "run_label": "Collect URLs",
                "success_message": "Legendary URL collection finished. Review the generated `.txt` files before downstream scraping if needed.",
                "caution_message": "This updates checkpoint and log files in `legendary_scholars/` and rewrites URL-list files when re-run.",
                "indications": [
                    "Use `Slugs File` to target a curated subset.",
                    "Keep `Skip Existing` enabled for normal incremental runs.",
                    "This module prepares input for the legend pipeline; it does not chunk or sync to Mongo by itself.",
                ],
                "fields": [
                    field("excel_path", "Workbook Path", "text", "Canonical legendary workbook.", default="excel/legendary.xlsx", required=True),
                    field("limit", "Limit Legends", "number", "Optional cap on workbook rows."),
                    field("start_from", "Start From Row", "number", "Skip the first N workbook rows.", default=0),
                    field("max_urls", "Max URLs Per Legend", "number", "Cap collected URLs per legend.", default=50),
                    field("per_query_results", "Results Per Query", "number", "DuckDuckGo result cap per query.", default=15),
                    field("sleep_seconds", "Sleep Seconds", "number", "Delay between search requests.", default=2.0),
                    field("output_dir", "Output Directory", "text", "Where `.txt` URL lists are written.", default="legendary_scholars/final"),
                    field("checkpoint_file", "Checkpoint File", "text", "Checkpoint JSON path.", default="legendary_scholars/url_collection_checkpoint.json"),
                    field("log_file", "Log File", "text", "JSONL activity log path.", default="legendary_scholars/url_collection_log.jsonl"),
                    field("slugs_file", "Slugs File", "text", "Optional newline-delimited slug file for targeted runs."),
                    field("skip_existing", "Skip Existing", "checkbox", "Do not recollect legends that already have outputs.", default=True),
                ],
            },
            {
                "id": "legendary_enrichment",
                "script_id": "run_legendary_enrichment",
                "title": "Legendary Enrichment Orchestrator",
                "subtitle": "Run URL collection, scraping, chunking, Pinecone, and Mongo sync as one flow.",
                "summary": "Orchestrates the legend pipeline end to end using MongoDB as the primary store.",
                "category": "Legends",
                "scope": "Legend-specific",
                "risk": "caution",
                "featured": True,
                "run_label": "Run end-to-end",
                "success_message": "Legendary enrichment completed. MongoDB should now reflect the processed legend records.",
                "caution_message": "Avoid reset flags here. This module only exposes the reusable path and leaves destructive resets out of the admin UI.",
                "indications": [
                    "Use `Resume` to retry only unfinished legends after a partial run.",
                    "Use `Skip Collection` when URL lists already look correct and only the downstream pipeline needs to run.",
                    "Keep Mongo sync enabled for demo-facing runs.",
                ],
                "fields": [
                    field("excel_path", "Workbook Path", "text", "Legend workbook source.", default="excel/legendary.xlsx", required=True),
                    field("limit", "Limit Legends", "number", "Optional cap on legends."),
                    field("start_from", "Start From Row", "number", "Skip the first N workbook rows.", default=0),
                    field("max_urls", "Max URLs Per Legend", "number", "Cap collected URLs per legend.", default=50),
                    field("per_query_results", "Results Per Query", "number", "DuckDuckGo result cap per query.", default=15),
                    field("sleep_seconds", "Sleep Seconds", "number", "Delay between search requests.", default=2.0),
                    field("output_dir", "URL Output Directory", "text", "Where legend URL-list files live.", default="legendary_scholars/final"),
                    field("checkpoint_file", "Checkpoint File", "text", "URL collection checkpoint path.", default="legendary_scholars/url_collection_checkpoint.json"),
                    field("log_file", "Log File", "text", "URL collection log path.", default="legendary_scholars/url_collection_log.jsonl"),
                    field("slugs_file", "Slugs File", "text", "Optional newline-delimited legend slug file."),
                    field("batch_size", "Pipeline Batch Size", "number", "Legend URL files per batch.", default=60),
                    field("runs_root", "Runs Output Root", "text", "Root directory for pipeline run folders.", default="output/legend_url_list_runs"),
                    field("mongo_collection", "Mongo Collection", "text", "Destination collection for synced legend records.", default="legend_scholars"),
                    field("skip_existing", "Skip Existing", "checkbox", "Keep previously collected URL files untouched.", default=True),
                    field("resume", "Resume Only Incomplete", "checkbox", "Skip already completed legend run folders.", default=True),
                    field("skip_collection", "Skip URL Collection", "checkbox", "Reuse existing URL lists and go straight to the pipeline.", default=False),
                    field("skip_pipeline", "Skip Pipeline", "checkbox", "Only collect URLs and stop before scraping/chunking.", default=False),
                    field("skip_pinecone", "Skip Pinecone", "checkbox", "Do not upload chunk embeddings.", default=False),
                    field("skip_mongodb", "Skip Mongo Sync", "checkbox", "Do not sync processed legends back to MongoDB.", default=False),
                    field("no_mongo_llm", "Disable Mongo LLM Summaries", "checkbox", "Sync to Mongo without LLM-generated summaries.", default=False),
                ],
            },
            {
                "id": "legend_pipeline",
                "script_id": "run_legend_scholar_pipeline",
                "title": "Legend URL-List Pipeline",
                "subtitle": "Run scraping and chunking directly from legend `.txt` URL lists.",
                "summary": "Processes one or more legend URL-list files and writes run artifacts, chunks, and optional DB sync outputs.",
                "category": "Legends",
                "scope": "Legend-specific",
                "risk": "caution",
                "featured": True,
                "run_label": "Run pipeline",
                "success_message": "Legend URL-list pipeline finished. Inspect the run folder and Mongo sync status in the console.",
                "caution_message": "Leave the URL file list empty to auto-discover `final/*.txt`. Supplying `Profile Name` forces the same name for all selected files.",
                "indications": [
                    "Provide one URL-list file path per line, or leave blank for auto-discovery.",
                    "Use `Skip MongoDB` only for test runs.",
                    "Live console logs include the exact run directory created for each legend.",
                ],
                "fields": [
                    field("urls_files", "URL List Files", "textarea", "Optional `.txt` files, one per line. Blank means auto-discover.", placeholder="final\\john-hope-franklin.txt"),
                    field("output_root", "Output Root", "text", "Base directory for run folders.", default="output/url_list_runs"),
                    field("run_stamp", "Run Stamp Override", "text", "Optional custom timestamp prefix."),
                    field("profile_name", "Forced Profile Name", "text", "Optional shared profile name override."),
                    field("profile_url", "Forced Primary Profile URL", "text", "Optional shared primary profile URL."),
                    field(
                        "llm_provider",
                        "LLM Provider",
                        "select",
                        "Chunking LLM provider.",
                        default="openai",
                        options=[
                            {"label": "OpenAI", "value": "openai"},
                            {"label": "Ollama", "value": "ollama"},
                        ],
                    ),
                    field("llm_model", "LLM Model", "text", "Chunking model name.", default="gpt-4o-mini"),
                    field("pinecone_batch_size", "Pinecone Batch Size", "number", "Chunk upload batch size.", default=50),
                    field("no_llm_chunking", "Disable LLM Chunking", "checkbox", "Use non-LLM chunking mode.", default=False),
                    field("skip_pinecone", "Skip Pinecone", "checkbox", "Do not upload chunks to Pinecone.", default=False),
                    field("skip_mongodb", "Skip MongoDB", "checkbox", "Do not sync the processed records to Mongo.", default=False),
                    field("skip_indexes", "Skip Mongo Indexes", "checkbox", "Skip Mongo index creation during sync.", default=False),
                    field("no_incremental_sync", "Disable Incremental Sync", "checkbox", "Only flush DB stages at the end.", default=False),
                    field("sync_batch_size", "Sync Batch Size", "number", "Profiles per incremental sync batch.", default=100),
                    field("sync_pinecone_batch_size", "Incremental Pinecone Batch Size", "number", "Chunk batch size per incremental Pinecone flush.", default=50),
                    field("keep_filters", "Keep Source Filters", "checkbox", "Retain role/intent/source filters in the pipeline.", default=False),
                ],
            },
            {
                "id": "universities_folder_pipeline",
                "script_id": "run_universities_folder_pipeline",
                "title": "Universities Folder Pipeline",
                "subtitle": "Process every workbook in a folder with resumable checkpoints.",
                "summary": "Scans workbook files under `universities/`, collects scholar-owned URLs, seeds MongoDB, and runs the full pipeline per scholar.",
                "category": "Universities",
                "scope": "Multi-university",
                "risk": "caution",
                "featured": True,
                "run_label": "Run university ingestion",
                "success_message": "University ingestion run finished. Review workbook-level progress and any skipped scholars in the live log.",
                "caution_message": "This is a longer-running job and writes checkpoints plus scholar updates during the run.",
                "indications": [
                    "Use `Dry Run` before a new workbook folder or glob pattern.",
                    "`Skip Existing` stays on by default for safe resumability.",
                    "Turn on `Aggressive` only when normal discovery is missing too many scholar-owned pages.",
                ],
                "fields": [
                    field("input_dir", "Input Directory", "text", "Folder containing workbook files.", default="universities", required=True),
                    field("output_root", "Output Root", "text", "Base output directory for workbook runs.", default="output/universities"),
                    field("workbook_glob", "Workbook Glob", "text", "Glob pattern within the input folder.", default="*.xlsx"),
                    field("workbook_name_contains", "Workbook Name Contains", "text", "Optional filename substring filter."),
                    field("limit_workbooks", "Limit Workbooks", "number", "Optional cap on workbook files."),
                    field("limit_scholars", "Limit Scholars Per Workbook", "number", "Optional cap on scholar rows."),
                    field("start_from", "Start From Row", "number", "Skip the first N rows in each workbook.", default=0),
                    field("max_urls", "Max URLs Per Scholar", "number", "Cap collected URLs per scholar.", default=20),
                    field("per_query_results", "Results Per Query", "number", "DuckDuckGo result cap per query.", default=12),
                    field("sleep_seconds", "Sleep Seconds", "number", "Delay between search requests.", default=2.0),
                    field("http_timeout_seconds", "HTTP Timeout Seconds", "number", "Timeout for API identity calls.", default=25.0),
                    field("http_retries", "HTTP Retries", "number", "Retry count for API identity calls.", default=4),
                    field("skip_existing", "Skip Existing", "checkbox", "Keep completed scholars untouched when signatures match.", default=True),
                    field("force", "Force Reprocess", "checkbox", "Reprocess even with a matching completed checkpoint.", default=False),
                    field("aggressive", "Aggressive Search", "checkbox", "Use broader query sets across identity domains.", default=False),
                    field("disable_openalex", "Disable OpenAlex", "checkbox", "Do not use OpenAlex identity discovery.", default=False),
                    field("dry_run", "Dry Run", "checkbox", "Inspect work without searching or scraping.", default=False),
                    field("merge_existing", "Merge Existing URL Lists", "checkbox", "Merge with existing URL-list files before recollecting.", default=True),
                ],
            },
            {
                "id": "profile_quality_refresh",
                "script_id": "profile_quality_refresh",
                "title": "Profile Quality Refresh",
                "subtitle": "Score scholar readiness for downstream daily story jobs.",
                "summary": "Computes quality signals for `legend_scholars` and can write status fields back into the same records.",
                "category": "Editorial",
                "scope": "Legend-specific",
                "risk": "caution",
                "featured": True,
                "run_label": "Refresh quality",
                "success_message": "Profile quality scores were refreshed. Downstream story jobs can now rely on the updated readiness fields.",
                "caution_message": "With `Update Scholar Docs` enabled, this writes fields under `daily_story_profile` on each matching legend document.",
                "indications": [
                    "Run this before generating story content for stricter quality gating.",
                    "Target one scholar for debugging or use a larger max for broad refreshes.",
                    "The script also persists an audit trail in the quality collections it manages.",
                ],
                "fields": [
                    field("max_scholars", "Max Scholars", "number", "Maximum scholars to evaluate.", default=1000),
                    field("scholar_id", "Scholar ID", "text", "Optional single `profile_id` to target."),
                    field("update_scholar_docs", "Update Scholar Docs", "checkbox", "Write status fields back into `legend_scholars`.", default=True),
                ],
            },
            {
                "id": "daily_story_worker",
                "script_id": "daily_story_worker",
                "title": "Daily Story Generator",
                "subtitle": "Generate scholar-inspired stories with review-aware safeguards.",
                "summary": "Creates one story per scholar per day, with options for dry runs, trend control, quality gates, and review workflow.",
                "category": "Editorial",
                "scope": "Legend-specific",
                "risk": "caution",
                "featured": True,
                "run_label": "Generate stories",
                "success_message": "Daily story generation finished. Review collections and console output for any validation failures or skipped scholars.",
                "caution_message": "This writes job and story records into MongoDB. Use `Dry Run` or `No LLM` for safe validation passes.",
                "indications": [
                    "Use `Dry Run` when validating collection wiring and filters.",
                    "Enable profile-quality enforcement for stricter demo safety.",
                    "Publishing without review changes story status immediately, so reserve that for trusted flows.",
                ],
                "fields": [
                    field("date", "Story Date", "text", "Optional YYYY-MM-DD date. Blank defaults to today UTC.", placeholder="2026-06-22"),
                    field("scholar_id", "Scholar ID", "text", "Optional single `profile_id` to target."),
                    field("topic", "Topic Override", "text", "Optional fixed topic for generation."),
                    field("max_scholars", "Max Scholars", "number", "Used when no scholar id is provided.", default=25),
                    field("max_context_chunks", "Max Context Chunks", "number", "Context chunks provided to the generator.", default=12),
                    field("scholars_collection", "Scholars Collection", "text", "Mongo collection containing scholars.", default="legend_scholars"),
                    field("stories_collection", "Stories Collection", "text", "Mongo collection for generated stories.", default="legend_scholar_daily_stories"),
                    field("jobs_collection", "Jobs Collection", "text", "Mongo collection for story jobs.", default="daily_story_jobs"),
                    field("model", "Model Override", "text", "Optional Vertex model override."),
                    field("dry_run", "Dry Run", "checkbox", "Generate deterministic placeholder content without an LLM call.", default=False),
                    field("no_llm", "Disable LLM", "checkbox", "Bypass LLM generation.", default=False),
                    field("disable_trends", "Disable Trends", "checkbox", "Generate from scholar corpus only.", default=False),
                    field(
                        "trend_provider",
                        "Trend Provider",
                        "select",
                        "Optional trend provider override.",
                        default="",
                        options=[
                            {"label": "Auto", "value": ""},
                            {"label": "RSS", "value": "rss"},
                            {"label": "NewsAPI", "value": "newsapi"},
                            {"label": "GDELT", "value": "gdelt"},
                        ],
                    ),
                    field("enforce_profile_quality", "Enforce Profile Quality", "checkbox", "Require ready profiles before generation.", default=False),
                    field("ignore_profile_quality", "Ignore Profile Quality", "checkbox", "Bypass readiness checks entirely.", default=False),
                    field("profile_quality_min_score", "Profile Quality Min Score", "number", "Optional minimum score threshold."),
                    field("publish_without_review", "Publish Without Review", "checkbox", "Mark valid stories as generated immediately.", default=False),
                    field("assume_living", "Assume Living", "checkbox", "Relax historical figure safety constraints.", default=False),
                ],
            },
            {
                "id": "daily_story_suite",
                "script_id": "daily_story_suite",
                "title": "Daily Story Suite",
                "subtitle": "Run quality refresh, trend ingestion, generation, retries, and evaluation together.",
                "summary": "Orchestrates the full daily story cron sequence for a given date with shared controls for trends, dry runs, and quality thresholds.",
                "category": "Editorial",
                "scope": "Legend-specific",
                "risk": "caution",
                "featured": True,
                "run_label": "Run full suite",
                "success_message": "Daily story suite finished. Inspect the step-level console output for any generation or retry failures.",
                "caution_message": "This coordinates multiple Mongo-writing steps. Use `Dry Run` first when validating a new environment.",
                "indications": [
                    "Best suited for scheduled editorial runs.",
                    "Pair with quality enforcement for stricter demo-safe story generation.",
                    "The final console output includes step-level JSON summaries.",
                ],
                "fields": [
                    field("date", "Story Date", "text", "Optional YYYY-MM-DD date. Blank defaults to today UTC.", placeholder="2026-06-22"),
                    field("max_scholars", "Max Scholars", "number", "Maximum scholars to process.", default=25),
                    field("max_context_chunks", "Max Context Chunks", "number", "Context chunks per story generation.", default=12),
                    field(
                        "trend_provider",
                        "Trend Provider",
                        "select",
                        "Trend ingestion provider.",
                        default="rss",
                        options=[
                            {"label": "RSS", "value": "rss"},
                            {"label": "NewsAPI", "value": "newsapi"},
                            {"label": "GDELT", "value": "gdelt"},
                        ],
                    ),
                    field("enforce_profile_quality", "Enforce Profile Quality", "checkbox", "Require ready profiles before generation.", default=False),
                    field("profile_quality_min_score", "Profile Quality Min Score", "number", "Minimum score threshold when enforcement is enabled.", default=60),
                    field("disable_trends", "Disable Trends", "checkbox", "Skip trend ingestion and generate from scholar corpus only.", default=False),
                    field("dry_run", "Dry Run", "checkbox", "Run the suite without live content generation.", default=False),
                    field("no_llm", "Disable LLM", "checkbox", "Bypass LLM generation where supported.", default=False),
                ],
            },
        ]

        for spec in curated_specs:
            script = script_map.get(spec["script_id"])
            if not script:
                continue
            used_scripts.add(script.id)
            modules.append(
                ScriptModule(
                    id=spec["id"],
                    script_id=script.id,
                    title=spec["title"],
                    subtitle=spec["subtitle"],
                    summary=spec["summary"],
                    category=spec["category"],
                    scope=spec["scope"],
                    risk=spec["risk"],
                    mode="structured",
                    featured=spec["featured"],
                    run_label=spec["run_label"],
                    success_message=spec["success_message"],
                    caution_message=spec["caution_message"],
                    indications=spec["indications"],
                    fields=spec["fields"],
                    usage_examples=script.usage_examples,
                    script_filename=script.filename,
                    script_path=script.path,
                )
            )

        for script in scripts:
            if script.id in used_scripts:
                continue
            modules.append(
                ScriptModule(
                    id=f"generic_{script.id}",
                    script_id=script.id,
                    title=script.title,
                    subtitle="Advanced raw CLI launcher.",
                    summary=script.summary,
                    category=script.category,
                    scope=script.scope,
                    risk=script.risk,
                    mode="raw",
                    featured=False,
                    run_label="Run advanced script",
                    success_message="Script started. Use the live console to follow progress and verify outcomes.",
                    caution_message="This module exposes a raw argument box because no curated admin form has been defined yet.",
                    indications=[
                        f"Backed directly by `{script.filename}`.",
                        "Use the embedded examples or exact CLI flags from the script when needed.",
                        "This is the fallback surface for scripts that are runnable but not yet modeled with a custom form.",
                    ],
                    fields=[
                        field(
                            "raw_args",
                            "Raw CLI Arguments",
                            "textarea",
                            "Arguments appended after the script path.",
                            placeholder="--limit 10 --dry-run",
                        )
                    ],
                    usage_examples=script.usage_examples,
                    script_filename=script.filename,
                    script_path=script.path,
                )
            )

        modules.sort(key=lambda item: (not item.featured, item.category, item.risk, item.title.lower()))
        return modules

    def get_script(self, script_id: str) -> ScriptDefinition:
        for script in self.catalog():
            if script.id == script_id:
                return script
        raise KeyError(f"Unknown script id: {script_id}")

    def get_module(self, module_id: str) -> ScriptModule:
        for module in self.modules():
            if module.id == module_id:
                return module
        raise KeyError(f"Unknown module id: {module_id}")

    def list_jobs(self) -> List[Dict[str, object]]:
        return execution_service.list_jobs()

    def get_job(self, job_id: str) -> RuntimeJob:
        return execution_service.get_job(job_id)

    def _build_command(self, script: ScriptDefinition, args: List[str]) -> List[str]:
        script_path = WORKSPACE_ROOT / script.path
        return [sys.executable, str(script_path), *args]

    def run_script(
        self,
        script_id: str,
        raw_args: str,
        *,
        schedule_id: Optional[str] = None,
        pre_approved: bool = False,
        module_id: Optional[str] = None,
        module_values: Optional[Mapping[str, Any]] = None,
        risk_override: Optional[str] = None,
    ) -> Dict[str, object]:
        args = shlex.split(raw_args or "", posix=True)
        return self.run_script_args(
            script_id,
            args,
            raw_args_display=raw_args,
            schedule_id=schedule_id,
            pre_approved=pre_approved,
            module_id=module_id,
            module_values=module_values,
            risk_override=risk_override,
        )

    def run_script_args(
        self,
        script_id: str,
        args: List[str],
        *,
        raw_args_display: Optional[str] = None,
        schedule_id: Optional[str] = None,
        pre_approved: bool = False,
        module_id: Optional[str] = None,
        module_values: Optional[Mapping[str, Any]] = None,
        risk_override: Optional[str] = None,
    ) -> Dict[str, object]:
        script = self.get_script(script_id)
        command = self._build_command(script, args)
        display_args = raw_args_display if raw_args_display is not None else subprocess.list2cmdline(args)
        risk = risk_override or script.risk
        needs_approval = approval_required_for_risk(risk) and not pre_approved
        job = RuntimeJob(
            id=uuid.uuid4().hex,
            script_id=script.id,
            script_filename=script.filename,
            command=command,
            raw_args=display_args,
            module_id=module_id,
            module_values=dict(module_values or {}),
            risk=risk,
            approval_required=needs_approval,
            status="pending_approval" if needs_approval else "queued",
            schedule_id=schedule_id,
        )
        job.append_log(f"[queued] {utc_now_iso()} {subprocess.list2cmdline(command)}")
        return execution_service.register_job(job, enqueue=not needs_approval)

    def run_module(
        self,
        module_id: str,
        values: Mapping[str, Any],
        *,
        schedule_id: Optional[str] = None,
        pre_approved: bool = False,
    ) -> Dict[str, object]:
        module = self.get_module(module_id)
        if module.mode == "raw":
            raw_args = clean_string(values.get("raw_args"))
            return self.run_script(
                module.script_id,
                raw_args,
                schedule_id=schedule_id,
                pre_approved=pre_approved,
                module_id=module.id,
                module_values=values,
                risk_override=module.risk,
            )

        builder = MODULE_BUILDERS.get(module.id)
        if not builder:
            raise ValueError(f"No argument builder is registered for module {module.id}")
        args = builder(values)
        return self.run_script_args(
            module.script_id,
            args,
            raw_args_display=subprocess.list2cmdline(args),
            schedule_id=schedule_id,
            pre_approved=pre_approved,
            module_id=module.id,
            module_values=values,
            risk_override=module.risk,
        )

    def launch_scheduled_module(self, module_id: str, values: Mapping[str, Any], schedule_id: str) -> Dict[str, object]:
        return self.run_module(module_id, values, schedule_id=schedule_id, pre_approved=True)

    def approve_job(self, job_id: str) -> Dict[str, object]:
        return execution_service.approve_job(job_id)

    def reject_job(self, job_id: str, *, reason: str = "Rejected before execution.") -> Dict[str, object]:
        return execution_service.reject_job(job_id, reason=reason)

    def terminate_job(self, job_id: str) -> Dict[str, object]:
        return execution_service.terminate_job(job_id)

    def stream(self, job_id: str) -> Iterable[str]:
        return execution_service.stream(job_id)

    def platform_status(self) -> Dict[str, object]:
        return execution_service.platform_status()


runner = AdminScriptRunner()
