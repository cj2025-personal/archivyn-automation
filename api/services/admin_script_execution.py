from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional

from api.services.admin_script_store import AdminScriptStore, store


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
MAX_LOG_LINES = int(os.getenv("ADMIN_JOB_MAX_LOG_LINES", "20000"))
STREAM_POLL_SECONDS = float(os.getenv("ADMIN_STREAM_POLL_SECONDS", "0.35"))
LOG_FLUSH_BATCH = int(os.getenv("ADMIN_JOB_LOG_FLUSH_LINES", "25"))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class RuntimeJob:
    id: str
    script_id: str
    script_filename: str
    command: List[str]
    raw_args: str
    module_id: Optional[str] = None
    module_values: Optional[Dict[str, Any]] = None
    risk: str = "safe"
    approval_required: bool = False
    status: str = "queued"
    created_at: str = field(default_factory=utc_now_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    exit_code: Optional[int] = None
    logs: Deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    log_version: int = 0
    process: Optional[subprocess.Popen[str]] = None
    error: Optional[str] = None
    schedule_id: Optional[str] = None
    _pending_log_flush: List[str] = field(default_factory=list, repr=False)

    def append_log(self, line: str) -> None:
        self.logs.append(line.rstrip("\n"))
        self.log_version += 1
        self._pending_log_flush.append(line.rstrip("\n"))

    def snapshot(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "script_id": self.script_id,
            "script_filename": self.script_filename,
            "command": self.command,
            "raw_args": self.raw_args,
            "module_id": self.module_id,
            "risk": self.risk,
            "approval_required": self.approval_required,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "log_count": len(self.logs),
            "log_version": self.log_version,
            "logs": list(self.logs),
            "error": self.error,
            "schedule_id": self.schedule_id,
        }

    def to_document(self) -> Dict[str, Any]:
        doc = self.snapshot()
        doc["module_values"] = self.module_values or {}
        doc["logs"] = list(self.logs)
        return doc

    @classmethod
    def from_document(cls, doc: Dict[str, Any]) -> "RuntimeJob":
        job = cls(
            id=doc["id"],
            script_id=doc["script_id"],
            script_filename=doc["script_filename"],
            command=list(doc.get("command") or []),
            raw_args=str(doc.get("raw_args") or ""),
            module_id=doc.get("module_id"),
            module_values=dict(doc.get("module_values") or {}),
            risk=str(doc.get("risk") or "safe"),
            approval_required=bool(doc.get("approval_required")),
            status=str(doc.get("status") or "queued"),
            created_at=str(doc.get("created_at") or utc_now_iso()),
            started_at=doc.get("started_at"),
            finished_at=doc.get("finished_at"),
            exit_code=doc.get("exit_code"),
            error=doc.get("error"),
            schedule_id=doc.get("schedule_id"),
        )
        job.log_version = int(doc.get("log_version") or 0)
        for line in doc.get("logs") or []:
            job.logs.append(str(line))
        return job


class AdminScriptExecutionService:
    """Queued worker pool with Mongo-backed persistence and audit hooks."""

    def __init__(self, mongo_store: AdminScriptStore) -> None:
        self._store = mongo_store
        self._lock = threading.Lock()
        self._jobs: Dict[str, RuntimeJob] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._max_concurrent = _env_int("ADMIN_MAX_CONCURRENT_JOBS", 2)
        self._active_workers = 0
        self._worker_threads: List[threading.Thread] = []
        self._started = False
        self._audit_callback: Optional[Callable[..., None]] = None

    def set_audit_callback(self, callback: Callable[..., None]) -> None:
        self._audit_callback = callback

    def _audit(self, event_type: str, **kwargs: Any) -> None:
        try:
            self._store.record_audit(event_type, **kwargs)
        except Exception:
            pass
        if self._audit_callback:
            try:
                self._audit_callback(event_type, **kwargs)
            except Exception:
                pass

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            for index in range(self._max_concurrent):
                thread = threading.Thread(target=self._worker_loop, name=f"admin-script-worker-{index}", daemon=True)
                thread.start()
                self._worker_threads.append(thread)
            self._recover_jobs()

    def stop(self) -> None:
        with self._lock:
            self._started = False

    def platform_status(self) -> Dict[str, Any]:
        with self._lock:
            running = sum(1 for job in self._jobs.values() if job.status == "running")
            queued = sum(1 for job in self._jobs.values() if job.status == "queued")
            pending = sum(1 for job in self._jobs.values() if job.status == "pending_approval")
        return {
            "max_concurrent_jobs": self._max_concurrent,
            "active_workers": self._active_workers,
            "running_jobs": running,
            "queued_jobs": queued,
            "pending_approval_jobs": pending,
            "queue_depth": self._queue.qsize(),
        }

    def _recover_jobs(self) -> None:
        try:
            docs = self._store.list_non_terminal_jobs()
        except Exception:
            return
        for doc in docs:
            job = RuntimeJob.from_document(doc)
            with self._lock:
                self._jobs[job.id] = job
            if job.status == "running":
                job.status = "failed"
                job.finished_at = utc_now_iso()
                job.error = "Interrupted by platform restart."
                job.append_log(f"[error] {job.finished_at} {job.error}")
                self._persist_job(job)
                self._audit(
                    "job_interrupted",
                    job_id=job.id,
                    module_id=job.module_id,
                    script_id=job.script_id,
                    risk=job.risk,
                    status=job.status,
                    message=job.error,
                )
            elif job.status == "queued":
                self._queue.put(job.id)

    def register_job(self, job: RuntimeJob, *, enqueue: bool) -> Dict[str, object]:
        with self._lock:
            self._jobs[job.id] = job
        self._persist_job(job)
        self._audit(
            "job_created",
            job_id=job.id,
            module_id=job.module_id,
            script_id=job.script_id,
            risk=job.risk,
            status=job.status,
            payload={"raw_args": job.raw_args, "approval_required": job.approval_required},
        )
        if enqueue:
            self._queue.put(job.id)
            self._audit("job_queued", job_id=job.id, module_id=job.module_id, script_id=job.script_id, risk=job.risk, status="queued")
        elif job.status == "pending_approval":
            self._audit(
                "job_pending_approval",
                job_id=job.id,
                module_id=job.module_id,
                script_id=job.script_id,
                risk=job.risk,
                status=job.status,
                message="High-impact script requires explicit approval before execution.",
            )
        return job.snapshot()

    def approve_job(self, job_id: str) -> Dict[str, object]:
        job = self.get_job(job_id)
        if job.status != "pending_approval":
            raise ValueError(f"Job {job_id} is not awaiting approval.")
        job.status = "queued"
        job.approval_required = False
        self._persist_job(job)
        self._audit("job_approved", job_id=job.id, module_id=job.module_id, script_id=job.script_id, risk=job.risk, status=job.status)
        self._queue.put(job.id)
        self._audit("job_queued", job_id=job.id, module_id=job.module_id, script_id=job.script_id, risk=job.risk, status="queued")
        return job.snapshot()

    def reject_job(self, job_id: str, *, reason: str = "Rejected before execution.") -> Dict[str, object]:
        job = self.get_job(job_id)
        if job.status != "pending_approval":
            raise ValueError(f"Job {job_id} is not awaiting approval.")
        job.status = "rejected"
        job.finished_at = utc_now_iso()
        job.error = reason
        job.append_log(f"[rejected] {job.finished_at} {reason}")
        self._persist_job(job)
        self._audit(
            "job_rejected",
            job_id=job.id,
            module_id=job.module_id,
            script_id=job.script_id,
            risk=job.risk,
            status=job.status,
            message=reason,
        )
        return job.snapshot()

    def list_jobs(self) -> List[Dict[str, object]]:
        try:
            docs = self._store.list_jobs()
            snapshots = []
            for doc in docs:
                with self._lock:
                    live = self._jobs.get(doc["id"])
                snapshots.append(live.snapshot() if live else self._public_snapshot(doc))
            return snapshots
        except Exception:
            with self._lock:
                jobs = sorted(self._jobs.values(), key=lambda item: (item.created_at, item.id), reverse=True)
                return [job.snapshot() for job in jobs]

    def get_job(self, job_id: str) -> RuntimeJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job:
            return job
        doc = self._store.get_job(job_id)
        if not doc:
            raise KeyError(f"Unknown job id: {job_id}")
        job = RuntimeJob.from_document(doc)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def terminate_job(self, job_id: str) -> Dict[str, object]:
        job = self.get_job(job_id)
        process = job.process
        if process and process.poll() is None:
            process.terminate()
            job.status = "terminating"
            job.append_log(f"[terminating] {utc_now_iso()}")
            self._persist_job(job)
            self._audit("job_terminate_requested", job_id=job.id, module_id=job.module_id, script_id=job.script_id, risk=job.risk, status=job.status)
        return job.snapshot()

    def stream(self, job_id: str) -> Iterable[str]:
        import json

        job = self.get_job(job_id)
        seen_version = -1
        while True:
            snapshot = job.snapshot()
            if snapshot["log_version"] != seen_version:
                payload = {
                    "job": {k: v for k, v in snapshot.items() if k != "logs"},
                    "logs": snapshot["logs"],
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                seen_version = int(snapshot["log_version"])

            if snapshot["status"] in {"completed", "failed", "rejected"}:
                break

            yield ": keep-alive\n\n"
            time.sleep(STREAM_POLL_SECONDS)

    def _public_snapshot(self, doc: Dict[str, Any]) -> Dict[str, object]:
        logs = doc.get("logs") or []
        if len(logs) > MAX_LOG_LINES:
            logs = logs[-MAX_LOG_LINES:]
        return {
            "id": doc["id"],
            "script_id": doc["script_id"],
            "script_filename": doc["script_filename"],
            "command": doc.get("command") or [],
            "raw_args": doc.get("raw_args") or "",
            "module_id": doc.get("module_id"),
            "risk": doc.get("risk") or "safe",
            "approval_required": bool(doc.get("approval_required")),
            "status": doc.get("status") or "queued",
            "created_at": doc.get("created_at"),
            "started_at": doc.get("started_at"),
            "finished_at": doc.get("finished_at"),
            "exit_code": doc.get("exit_code"),
            "log_count": len(logs),
            "log_version": int(doc.get("log_version") or 0),
            "logs": list(logs),
            "error": doc.get("error"),
            "schedule_id": doc.get("schedule_id"),
        }

    def _persist_job(self, job: RuntimeJob) -> None:
        try:
            self._store.save_job(job.to_document())
            job._pending_log_flush.clear()
        except Exception:
            pass

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                with self._lock:
                    self._active_workers += 1
                self._execute_job(job_id)
            finally:
                with self._lock:
                    self._active_workers = max(0, self._active_workers - 1)
                self._queue.task_done()

    def _execute_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job.status not in {"queued"}:
            return
        try:
            job.status = "running"
            job.started_at = utc_now_iso()
            job.append_log(f"[started] {job.started_at}")
            self._persist_job(job)
            self._audit("job_started", job_id=job.id, module_id=job.module_id, script_id=job.script_id, risk=job.risk, status=job.status)

            process = subprocess.Popen(
                job.command,
                cwd=str(WORKSPACE_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
            job.process = process
            assert process.stdout is not None
            lines_since_flush = 0
            for line in process.stdout:
                job.append_log(line.rstrip("\n"))
                lines_since_flush += 1
                if lines_since_flush >= LOG_FLUSH_BATCH:
                    self._persist_job(job)
                    lines_since_flush = 0
            process.wait()
            job.exit_code = process.returncode
            job.finished_at = utc_now_iso()
            job.status = "completed" if process.returncode == 0 else "failed"
            job.append_log(f"[finished] {job.finished_at} exit_code={process.returncode}")
            self._persist_job(job)
            self._audit(
                "job_finished" if job.status == "completed" else "job_failed",
                job_id=job.id,
                module_id=job.module_id,
                script_id=job.script_id,
                risk=job.risk,
                status=job.status,
                message=f"exit_code={process.returncode}",
            )
        except Exception as exc:
            job.finished_at = utc_now_iso()
            job.status = "failed"
            job.error = str(exc)
            job.append_log(f"[error] {job.finished_at} {exc}")
            self._persist_job(job)
            self._audit(
                "job_failed",
                job_id=job.id,
                module_id=job.module_id,
                script_id=job.script_id,
                risk=job.risk,
                status=job.status,
                message=str(exc),
            )
        finally:
            job.log_version += 1


execution_service = AdminScriptExecutionService(store)
