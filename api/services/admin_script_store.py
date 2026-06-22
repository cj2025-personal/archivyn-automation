from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config.mongodb_utils import create_mongo_client, resolve_mongo_db_name


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


class AdminScriptStore:
  """MongoDB persistence for admin script jobs, audit events, and schedules."""

  def __init__(self) -> None:
    self._jobs_collection = os.getenv("ADMIN_SCRIPT_JOBS_COLLECTION", "admin_script_jobs")
    self._audit_collection = os.getenv("ADMIN_SCRIPT_AUDIT_COLLECTION", "admin_script_audit_events")
    self._schedules_collection = os.getenv("ADMIN_SCRIPT_SCHEDULES_COLLECTION", "admin_script_schedules")
    self._job_list_limit = _env_int("ADMIN_JOB_LIST_LIMIT", 200)
    self._audit_list_limit = _env_int("ADMIN_AUDIT_LIST_LIMIT", 500)
    self._client = None
    self._db = None

  def _require_db(self):
    if self._db is not None:
      return self._db
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
      raise RuntimeError("MONGODB_URI is required for admin script persistence.")
    self._client = create_mongo_client(mongodb_uri)
    db_name = resolve_mongo_db_name(mongodb_uri)
    self._db = self._client[db_name]
    self._ensure_indexes()
    return self._db

  def _ensure_indexes(self) -> None:
    jobs = self._db[self._jobs_collection]
    audit = self._db[self._audit_collection]
    schedules = self._db[self._schedules_collection]
    jobs.create_index("created_at")
    jobs.create_index("status")
    jobs.create_index([("status", 1), ("created_at", -1)])
    audit.create_index("created_at")
    audit.create_index("event_type")
    audit.create_index("job_id")
    schedules.create_index("enabled")
    schedules.create_index("next_run_at")

  def close(self) -> None:
    if self._client is not None:
      try:
        self._client.close()
      except Exception:
        pass
    self._client = None
    self._db = None

  def save_job(self, job_doc: Dict[str, Any]) -> Dict[str, Any]:
    db = self._require_db()
    db[self._jobs_collection].replace_one({"id": job_doc["id"]}, job_doc, upsert=True)
    return job_doc

  def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
    db = self._require_db()
    return db[self._jobs_collection].find_one({"id": job_id}, {"_id": 0})

  def list_jobs(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    db = self._require_db()
    cap = limit or self._job_list_limit
    cursor = db[self._jobs_collection].find({}, {"_id": 0}).sort("created_at", -1).limit(cap)
    return list(cursor)

  def append_job_logs(self, job_id: str, lines: List[str], *, status: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    if not lines and status is None and not extra:
      return
    db = self._require_db()
    update: Dict[str, Any] = {"$inc": {"log_version": 1}}
    set_fields: Dict[str, Any] = {}
    if lines:
      update["$push"] = {"logs": {"$each": lines}}
    if status is not None:
      set_fields["status"] = status
    if extra:
      set_fields.update(extra)
    if set_fields:
      update["$set"] = set_fields
    db[self._jobs_collection].update_one({"id": job_id}, update)

  def update_job(self, job_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    db = self._require_db()
    db[self._jobs_collection].update_one({"id": job_id}, {"$set": fields})
    return self.get_job(job_id)

  def record_audit(
    self,
    event_type: str,
    *,
    job_id: Optional[str] = None,
    module_id: Optional[str] = None,
    script_id: Optional[str] = None,
    risk: Optional[str] = None,
    status: Optional[str] = None,
    message: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    db = self._require_db()
    doc = {
      "id": uuid.uuid4().hex,
      "event_type": event_type,
      "created_at": utc_now_iso(),
      "job_id": job_id,
      "module_id": module_id,
      "script_id": script_id,
      "risk": risk,
      "status": status,
      "message": message,
      "payload": payload or {},
    }
    db[self._audit_collection].insert_one(doc)
    doc.pop("_id", None)
    return doc

  def list_audit_events(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    db = self._require_db()
    cap = limit or self._audit_list_limit
    cursor = db[self._audit_collection].find({}, {"_id": 0}).sort("created_at", -1).limit(cap)
    return list(cursor)

  def save_schedule(self, schedule_doc: Dict[str, Any]) -> Dict[str, Any]:
    db = self._require_db()
    db[self._schedules_collection].replace_one({"id": schedule_doc["id"]}, schedule_doc, upsert=True)
    return schedule_doc

  def get_schedule(self, schedule_id: str) -> Optional[Dict[str, Any]]:
    db = self._require_db()
    return db[self._schedules_collection].find_one({"id": schedule_id}, {"_id": 0})

  def list_schedules(self) -> List[Dict[str, Any]]:
    db = self._require_db()
    cursor = db[self._schedules_collection].find({}, {"_id": 0}).sort("created_at", -1)
    return list(cursor)

  def update_schedule(self, schedule_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    db = self._require_db()
    db[self._schedules_collection].update_one({"id": schedule_id}, {"$set": fields})
    return self.get_schedule(schedule_id)

  def delete_schedule(self, schedule_id: str) -> bool:
    db = self._require_db()
    result = db[self._schedules_collection].delete_one({"id": schedule_id})
    return result.deleted_count > 0

  def list_non_terminal_jobs(self) -> List[Dict[str, Any]]:
    db = self._require_db()
    statuses = ["queued", "running", "terminating", "pending_approval"]
    cursor = db[self._jobs_collection].find({"status": {"$in": statuses}}, {"_id": 0})
    return list(cursor)


store = AdminScriptStore()
