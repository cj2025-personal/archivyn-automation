from __future__ import annotations

from api.services.admin_script_execution import execution_service
from api.services.admin_script_runner import runner
from api.services.admin_script_scheduler import scheduler_service


def start_admin_script_platform() -> None:
    execution_service.start()
    scheduler_service.set_launch_callback(runner.launch_scheduled_module)
    scheduler_service.start()


def stop_admin_script_platform() -> None:
    scheduler_service.stop()
    execution_service.stop()
