from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from api.services.admin_script_store import store, utc_now_iso


class AdminScriptScheduler:
    """Cron-based schedule runner for curated admin modules."""

    def __init__(self) -> None:
        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._launch_callback: Optional[Callable[[str, Dict[str, Any], str], Dict[str, object]]] = None
        self._started = False

    def set_launch_callback(self, callback: Callable[[str, Dict[str, Any], str], Dict[str, object]]) -> None:
        self._launch_callback = callback

    def start(self) -> None:
        if self._started:
            return
        self._scheduler.start()
        self._started = True
        self.reload_schedules()

    def stop(self) -> None:
        if not self._started:
            return
        self._scheduler.shutdown(wait=False)
        self._started = False

    def reload_schedules(self) -> None:
        if not self._started:
            return
        self._scheduler.remove_all_jobs()
        try:
            schedules = store.list_schedules()
        except Exception:
            return
        for schedule in schedules:
            if not schedule.get("enabled", True):
                continue
            cron = str(schedule.get("cron_expression") or "").strip()
            if not cron:
                continue
            try:
                trigger = CronTrigger.from_crontab(cron, timezone="UTC")
            except Exception:
                continue
            self._scheduler.add_job(
                self._run_schedule,
                trigger=trigger,
                id=schedule["id"],
                kwargs={"schedule_id": schedule["id"]},
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )

    def list_schedules(self) -> List[Dict[str, Any]]:
        return store.list_schedules()

    def create_schedule(
        self,
        *,
        name: str,
        module_id: str,
        values: Dict[str, Any],
        cron_expression: str,
        enabled: bool = True,
        timezone_name: str = "UTC",
    ) -> Dict[str, Any]:
        schedule_id = uuid.uuid4().hex
        now = utc_now_iso()
        doc = {
            "id": schedule_id,
            "name": name.strip() or module_id,
            "module_id": module_id,
            "values": values,
            "cron_expression": cron_expression.strip(),
            "timezone": timezone_name,
            "enabled": enabled,
            "created_at": now,
            "updated_at": now,
            "last_run_at": None,
            "last_job_id": None,
            "last_status": None,
        }
        store.save_schedule(doc)
        store.record_audit(
            "schedule_created",
            module_id=module_id,
            status="enabled" if enabled else "disabled",
            message=name,
            payload={"cron_expression": cron_expression},
        )
        self.reload_schedules()
        return doc

    def update_schedule(self, schedule_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        existing = store.get_schedule(schedule_id)
        if not existing:
            raise KeyError(f"Unknown schedule id: {schedule_id}")
        existing.update(fields)
        existing["updated_at"] = utc_now_iso()
        store.save_schedule(existing)
        store.record_audit(
            "schedule_updated",
            module_id=existing.get("module_id"),
            status="enabled" if existing.get("enabled", True) else "disabled",
            payload={"schedule_id": schedule_id, **fields},
        )
        self.reload_schedules()
        return existing

    def delete_schedule(self, schedule_id: str) -> None:
        existing = store.get_schedule(schedule_id)
        if not existing:
            raise KeyError(f"Unknown schedule id: {schedule_id}")
        store.delete_schedule(schedule_id)
        store.record_audit(
            "schedule_deleted",
            module_id=existing.get("module_id"),
            payload={"schedule_id": schedule_id},
        )
        self.reload_schedules()

    def _run_schedule(self, schedule_id: str) -> None:
        schedule = store.get_schedule(schedule_id)
        if not schedule or not schedule.get("enabled", True):
            return
        if not self._launch_callback:
            return
        now = utc_now_iso()
        try:
            snapshot = self._launch_callback(
                schedule["module_id"],
                dict(schedule.get("values") or {}),
                schedule_id,
            )
            store.update_schedule(schedule_id, {
                "last_run_at": now,
                "last_job_id": snapshot.get("id"),
                "last_status": snapshot.get("status"),
                "updated_at": now,
            })
            store.record_audit(
                "schedule_triggered",
                job_id=str(snapshot.get("id") or ""),
                module_id=schedule.get("module_id"),
                status=str(snapshot.get("status") or ""),
                payload={"schedule_id": schedule_id},
            )
        except Exception as exc:
            store.update_schedule(schedule_id, {
                "last_run_at": now,
                "last_status": "failed",
                "updated_at": now,
            })
            store.record_audit(
                "schedule_trigger_failed",
                module_id=schedule.get("module_id"),
                status="failed",
                message=str(exc),
                payload={"schedule_id": schedule_id},
            )


scheduler_service = AdminScriptScheduler()
