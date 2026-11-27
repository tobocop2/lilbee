"""Enrol a launch in a swap manager the way ``start`` does, log file included."""

from __future__ import annotations

from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.base import EngineAdapter
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.readback import engine_log_of
from lilbee.providers.fleet.swap_manager import SwapManager


def enrol_launch(
    manager: SwapManager, launch: InstanceLaunch, engine: EngineAdapter = LLAMA
) -> None:
    """Record *launch* and the log file its engine writes, or none when it writes none."""
    manager._launch_by_model = {launch.model_id: launch}
    log_path = engine_log_of(engine, launch, manager.log_path.parent)
    manager._engine_logs = {} if log_path is None else {launch.model_id: log_path}
