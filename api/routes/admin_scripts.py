from __future__ import annotations

import os
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from api.middleware.archivyn_auth import require_archivyn_auth
from api.services.admin_script_runner import runner
from api.services.admin_script_scheduler import scheduler_service
from api.services.admin_script_store import store


router = APIRouter(
    prefix="/api/admin/scripts",
    tags=["admin-scripts"],
    dependencies=[Depends(require_archivyn_auth)],
)
portal_router = APIRouter(prefix="/admin", tags=["admin-portal"])


class ScriptRunRequest(BaseModel):
    script_id: str = Field(..., description="Catalog script id.")
    raw_args: str = Field(default="", description="Raw CLI args to append after the script path.")


class ModuleRunRequest(BaseModel):
    module_id: str = Field(..., description="Admin script module id.")
    values: Dict[str, object] = Field(default_factory=dict, description="Structured form values for the selected module.")


class JobRejectRequest(BaseModel):
    reason: str = Field(default="Rejected before execution.")


class ScheduleCreateRequest(BaseModel):
    name: str = Field(..., description="Human-readable schedule name.")
    module_id: str = Field(..., description="Module to launch on schedule.")
    values: Dict[str, object] = Field(default_factory=dict, description="Module form values.")
    cron_expression: str = Field(..., description="Standard 5-field cron expression in UTC.")
    enabled: bool = Field(default=True)


class ScheduleUpdateRequest(BaseModel):
    name: Optional[str] = None
    values: Optional[Dict[str, object]] = None
    cron_expression: Optional[str] = None
    enabled: Optional[bool] = None


@router.get("/catalog")
async def get_script_catalog() -> Dict[str, List[Dict[str, object]]]:
    scripts = [script.as_dict() for script in runner.catalog()]
    return {"scripts": scripts}


@router.get("/modules")
async def get_script_modules() -> Dict[str, List[Dict[str, object]]]:
    modules = [module.as_dict() for module in runner.modules()]
    return {"modules": modules}


@router.post("/catalog/refresh")
async def refresh_script_catalog() -> Dict[str, List[Dict[str, object]]]:
    scripts = [script.as_dict() for script in runner.refresh_catalog()]
    return {"scripts": scripts}


@router.post("/modules/refresh")
async def refresh_script_modules() -> Dict[str, List[Dict[str, object]]]:
    modules = [module.as_dict() for module in runner.refresh_modules()]
    return {"modules": modules}


@router.get("/platform/status")
async def get_platform_status() -> Dict[str, object]:
    return {
        "platform": runner.platform_status(),
        "approval_required_risks": os.getenv("ADMIN_APPROVAL_REQUIRED_RISKS", "danger"),
        "max_concurrent_jobs": os.getenv("ADMIN_MAX_CONCURRENT_JOBS", "2"),
    }


@router.get("/audit")
async def list_audit_events() -> Dict[str, List[Dict[str, object]]]:
    try:
        return {"events": store.list_audit_events()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Audit store unavailable: {exc}") from exc


@router.get("/schedules")
async def list_schedules() -> Dict[str, List[Dict[str, object]]]:
    try:
        return {"schedules": scheduler_service.list_schedules()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Schedule store unavailable: {exc}") from exc


@router.post("/schedules")
async def create_schedule(request: ScheduleCreateRequest) -> Dict[str, object]:
    try:
        runner.get_module(request.module_id)
        schedule = scheduler_service.create_schedule(
            name=request.name,
            module_id=request.module_id,
            values=request.values,
            cron_expression=request.cron_expression,
            enabled=request.enabled,
        )
        return {"schedule": schedule}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/schedules/{schedule_id}")
async def update_schedule(schedule_id: str, request: ScheduleUpdateRequest) -> Dict[str, object]:
    fields = {key: value for key, value in request.model_dump().items() if value is not None}
    try:
        schedule = scheduler_service.update_schedule(schedule_id, fields)
        return {"schedule": schedule}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/schedules/{schedule_id}")
async def delete_schedule(schedule_id: str) -> Dict[str, object]:
    try:
        scheduler_service.delete_schedule(schedule_id)
        return {"deleted": True, "schedule_id": schedule_id}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/jobs")
async def list_script_jobs() -> Dict[str, List[Dict[str, object]]]:
    return {"jobs": runner.list_jobs()}


@router.get("/jobs/{job_id}")
async def get_script_job(job_id: str) -> Dict[str, object]:
    try:
        return {"job": runner.get_job(job_id).snapshot()}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/jobs")
async def run_script(request: ScriptRunRequest) -> Dict[str, object]:
    try:
        snapshot = runner.run_script(request.script_id, request.raw_args)
        return {"job": snapshot}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/jobs/from-module")
async def run_script_module(request: ModuleRunRequest) -> Dict[str, object]:
    try:
        snapshot = runner.run_module(request.module_id, request.values)
        return {"job": snapshot}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/approve")
async def approve_script_job(job_id: str) -> Dict[str, object]:
    try:
        return {"job": runner.approve_job(job_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/reject")
async def reject_script_job(job_id: str, request: JobRejectRequest) -> Dict[str, object]:
    try:
        return {"job": runner.reject_job(job_id, reason=request.reason)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/terminate")
async def terminate_script_job(job_id: str) -> Dict[str, object]:
    try:
        return {"job": runner.terminate_job(job_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/jobs/{job_id}/stream")
async def stream_script_job(job_id: str) -> StreamingResponse:
    try:
        runner.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(
        runner.stream(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@portal_router.get("/automation", response_class=HTMLResponse)
async def automation_module_shell() -> HTMLResponse:
    iframe_url = os.getenv("ADMIN_PORTAL_UI_URL", "http://localhost:3000/automation")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Automation Console</title>
  <style>
    html,body{{margin:0;height:100%;background:#101820;color:#fff;font-family:Segoe UI,system-ui,sans-serif}}
    .frame{{border:0;width:100%;height:100%}}
    .fallback{{display:flex;align-items:center;justify-content:center;height:100%;padding:32px}}
    a{{color:#9bf6ff}}
  </style>
</head>
<body>
  <iframe class="frame" src="{iframe_url}" title="Automation Console"></iframe>
  <noscript>
    <div class="fallback">
      Open the automation console directly:
      <a href="{iframe_url}" target="_blank" rel="noreferrer">{iframe_url}</a>
    </div>
  </noscript>
</body>
</html>"""
    return HTMLResponse(content=html)
